//===- AffineToStandard.cpp - Lower affine constructs to primitives -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers affine constructs (If and For statements, AffineApply
// operations) within a function into their standard If and For equivalent ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static llvm::cl::opt<bool>
    clUseNestedPatterns("use-nested-patterns",
                        llvm::cl::desc("use nested patterns in gemm matching"),
                        llvm::cl::init(false));

namespace {


// Class for matching a top-level `affine.for` Op in the MLIR pattern matching
// framework.
class MatmulMatcher : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;


  /// Returns the AffineForOp perfectly nested in the given Op. Perfectly nested
  /// means it is the only non-terminator operation in the first region of `op`.
  /// If the nested Op is imperfectly nested, or not an AffineForOp, returns
  /// nullptr.
  AffineForOp getPerfectlyNestedLoop(Operation *op) const {
    if (!op)
      return nullptr;

    assert(op->getNumRegions() > 0 && "expected Op with regions");

    Region &bodyRegion = op->getRegion(0);
    assert(bodyRegion.getBlocks().size() == 1 && "expecte single-block region");

    // Check for perfect-nestedness.
    Block &body = bodyRegion.front();
    if (std::distance(body.begin(), body.end()) != 2)
      return nullptr;

    assert(!body.back().isKnownNonTerminator() &&
           "unexpected non-terminator at the end of the block");
    return dyn_cast<AffineForOp>(body.front());
  }

  /// Checks if the given `op` is an affine load/store operation accessing a 2D
  /// memref using the subscript [d1][d2].
  template <typename OpTy>
  bool is2DAccess(OpTy op, Value *d1, Value *d2) const {
    static_assert(std::is_same<OpTy, AffineLoadOp>::value ||
                      std::is_same<OpTy, AffineStoreOp>::value,
                  "expected affine load or store");

    Value *operand1 = *op.getMapOperands().begin();
    Value *operand2 = *(op.getMapOperands().begin() + 1);

    return (op.getAffineMap().isIdentity() &&
            op.getAffineMap().getNumResults() == 2 && d1 == operand1 &&
            d2 == operand2);
  }

  /// Checks if the given `op` is an affine load/store operation accessing a 2D
  /// memref using as subscripts any pair of values in `indices`. If so, assigns
  /// those indices to `first` and `second` and returns `true`. Returs `false`
  /// otherwise. Expects `op`, `first` a `second` to be non-null and `indices`
  /// to contain exactly three values.
  template <typename OpTy>
  bool anyPermutation(OpTy op, SmallVector<Value *, 3> indices,
                      Value **first, Value **second) const {
    assert(first);
    assert(second);

    if (!op)
      return false;

    // TODO: This is a hack to obtain all pairs of 2 out of 3 possible values
    // using next permutation. We need a more robust way of obtaining a subset,
    // on which we can call next permutation.
    assert(indices.size() == 3);
    // SSA values are pointer-comparable so we can just sort and iterate through
    // all value pointers.
    std::sort(indices.begin(), indices.end());
    while (std::next_permutation(indices.begin(), indices.end())) {
      if (is2DAccess(op, indices[0], indices[1])) {
        *first = indices[0];
        *second = indices[1];
        return true;
      }
       if (is2DAccess(op, indices[1], indices[0])) {
        *first = indices[1];
        *second = indices[0];
        return true;
      }
    }

    return false;
  }

  /// Match the region (assumed to be the body of the inntermost loop) to be
  /// a matrix multiplication kernel.
  PatternMatchResult matchBody(Region &body, Value *i, Value *j,
                               Value *k) const {
    // We proceed in the reverse order to leverage use-def chains.
    // TODO: a better mathcher-like syntax could be beneficial in either C++ or
    // Tablegen, or both.

    // Check that the last operation before we exit the body (i.e. before the
    // mandatory terminator) is a store.
    Value *storeD1, *storeD2;
    auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));
    if (!anyPermutation(store, {i, j, k}, &storeD1, &storeD2))
      return matchFailure();

    // Check that the value we store was computed as the result of an floating
    // point addition (the second operation in the matmul kernel).
    auto add =
        dyn_cast_or_null<AddFOp>(store.getValueToStore()->getDefiningOp());
    if (!add)
      return matchFailure();

    // Check that the left operand of the add comes from a load that uses the
    // same array and the same subscripts as the store.
    // TODO: relax the order of operands
    // TODO: introduce the notion of placeholders instead of using explicit
    // value matching here.
    auto loadC = dyn_cast_or_null<AffineLoadOp>(add.lhs()->getDefiningOp());
    Value *loadCD1, *loadCD2;
    if (!loadC || !anyPermutation(loadC, {i, j, k}, &loadCD1, &loadCD2) ||
        loadC.getMemRef() != store.getMemRef() || loadCD1 != storeD1 ||
        loadCD2 != storeD2)
      return matchFailure();

    // Check that the right operand of the add comes from a floating point
    // multiplication (the first operation in the matmul kernel).
    // TODO: relax the order of operands
    auto mul = dyn_cast_or_null<MulFOp>(add.rhs()->getDefiningOp());
    if (!mul)
      return matchFailure();

    // Check that the left and right operands of a multiplication come from
    // loads.
    auto loadA = dyn_cast_or_null<AffineLoadOp>(mul.lhs()->getDefiningOp());
    auto loadB = dyn_cast_or_null<AffineLoadOp>(mul.rhs()->getDefiningOp());
    if (!loadA || !loadB)
      return matchFailure();

    // Given loads A(a1,a2), B(b1,b2), C(c1,c2) [defined above], check that the
    // subscripts match as in a1=c1, b2=c2, a2=b1.
    // TODO: support transposed matrix multiplications here.
    Value *loadAD1, *loadAD2, *loadBD1, *loadBD2;
    if (!anyPermutation(loadA, {i, j, k}, &loadAD1, &loadAD2) ||
        !anyPermutation(loadB, {i, j, k}, &loadBD1, &loadBD2) ||
        loadAD1 != loadCD1 || loadBD2 != loadCD2 || loadAD2 != loadBD1)
      return matchFailure();

    // Check that only the operations that we've seen already are in the region.
    // In particular, there must be 3 loads, 1 store, 1 mul, 1 add and 1
    // implicit terminator, totalling 7 operations.
    // TODO: we may be lenient to operations without side-effects, but they
    // should have been removed by DCE beforehand.
    if (std::distance(body.front().begin(), body.front().end()) != 7)
      return matchFailure();

    return matchSuccess();
  }

  /// Main rewriting function. Doesn't do actual rewriting here, just checks
  /// that the matching works.
  PatternMatchResult matchAndRewrite(AffineForOp op,
                                     PatternRewriter &rewriter) const override {
    if (clUseNestedPatterns)
      return matchAndRewriteNestedPattern(op, rewriter);
    else
      return matchAndRewriteRegionTraversal(op, rewriter);
  }

  PatternMatchResult
  matchAndRewriteRegionTraversal(AffineForOp op,
                                 PatternRewriter &rewriter) const {
    AffineForOp nested1 = getPerfectlyNestedLoop(op);
    AffineForOp nested2 = getPerfectlyNestedLoop(nested1);

    if (!nested1 || !nested2)
      return matchFailure();

    Value *i = op.getInductionVar();
    Value *j = nested1.getInductionVar();
    Value *k = nested2.getInductionVar();

    if (!matchBody(nested2.getLoopBody(), i, j, k))
      return matchFailure();

    llvm::outs() << "match succeeded\n";

    rewriter.eraseOp(op);

    return matchSuccess();
  }

  // Same as above but using the new nested matcher infrastructure. The body
  // part is still based on the use-def chain.
  PatternMatchResult
  matchAndRewriteNestedPattern(Operation *op, PatternRewriter &rewriter) const {
    auto body = [this](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      Value *k = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      Value *j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      Value *i = parent.getInductionVar();
      return matchBody(loop.getLoopBody(), i, j, k).hasValue();
    };

    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(body)));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return matchFailure();
    }

    llvm::outs() << "match succeeded\n";

    rewriter.eraseOp(op);

    return matchSuccess();
  }
};
}

namespace {
// Visit affine expressions recursively and build the sequence of operations
// that correspond to it.  Visitation functions return an Value of the
// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value *> {
public:
  // This internal class expects arguments to be non-null, checks must be
  // performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ArrayRef<Value *> dimValues,
                      ArrayRef<Value *> symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy> Value *buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value *visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value *visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
  }

  // Euclidean modulo operation: negative RHS is not allowed.
  // Remainder of the euclidean integer division is always non-negative.
  //
  // Implemented as
  //
  //     a mod b =
  //         let remainder = srem a, b;
  //             negative = a < 0 in
  //         select negative, remainder + b, remainder.
  Value *visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *remainder = builder.create<RemISOp>(loc, lhs, rhs);
    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *isRemainderNegative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value *correctedRemainder = builder.create<AddIOp>(loc, remainder, rhs);
    Value *result = builder.create<SelectOp>(loc, isRemainderNegative,
                                             correctedRemainder, remainder);
    return result;
  }

  // Floor division operation (rounds towards negative infinity).
  //
  // For positive divisors, it can be implemented without branching and with a
  // single division operation as
  //
  //        a floordiv b =
  //            let negative = a < 0 in
  //            let absolute = negative ? -a - 1 : a in
  //            let quotient = absolute / b in
  //                negative ? -quotient - 1 : quotient
  Value *visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *noneCst = builder.create<ConstantIndexOp>(loc, -1);
    Value *negative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, zeroCst);
    Value *negatedDecremented = builder.create<SubIOp>(loc, noneCst, lhs);
    Value *dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value *quotient = builder.create<DivISOp>(loc, dividend, rhs);
    Value *correctedQuotient = builder.create<SubIOp>(loc, noneCst, quotient);
    Value *result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
    return result;
  }

  // Ceiling division operation (rounds towards positive infinity).
  //
  // For positive divisors, it can be implemented without branching and with a
  // single division operation as
  //
  //     a ceildiv b =
  //         let negative = a <= 0 in
  //         let absolute = negative ? -a : a - 1 in
  //         let quotient = absolute / b in
  //             negative ? -quotient : quotient + 1
  Value *visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *oneCst = builder.create<ConstantIndexOp>(loc, 1);
    Value *nonPositive =
        builder.create<CmpIOp>(loc, CmpIPredicate::sle, lhs, zeroCst);
    Value *negated = builder.create<SubIOp>(loc, zeroCst, lhs);
    Value *decremented = builder.create<SubIOp>(loc, lhs, oneCst);
    Value *dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value *quotient = builder.create<DivISOp>(loc, dividend, rhs);
    Value *negatedQuotient = builder.create<SubIOp>(loc, zeroCst, quotient);
    Value *incrementedQuotient = builder.create<AddIOp>(loc, quotient, oneCst);
    Value *result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                             incrementedQuotient);
    return result;
  }

  Value *visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, builder.getIndexType(), valueAttr);
    return op.getResult();
  }

  Value *visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value *visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ArrayRef<Value *> dimValues;
  ArrayRef<Value *> symbolValues;

  Location loc;
};
} // namespace

// Create a sequence of operations that implement the `expr` applied to the
// given dimension and symbol values.
mlir::Value *mlir::expandAffineExpr(OpBuilder &builder, Location loc,
                                    AffineExpr expr,
                                    ArrayRef<Value *> dimValues,
                                    ArrayRef<Value *> symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

// Create a sequence of operations that implement the `affineMap` applied to
// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value *, 8>> static expandAffineMap(
    OpBuilder &builder, Location loc, AffineMap affineMap,
    ArrayRef<Value *> operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, &builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value *v) { return v; }))
    return expanded;
  return None;
}

// Given a range of values, emit the code that reduces them with "min" or "max"
// depending on the provided comparison predicate.  The predicate defines which
// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
// `cmpi` operation followed by the `select` operation:
//
//   %cond   = cmpi "predicate" %v0, %v1
//   %result = select %cond, %v0, %v1
//
// Multiple values are scanned in a linear sequence.  This creates a data
// dependences that wouldn't exist in a tree reduction, but is easier to
// recognize as a reduction by the subsequent passes.
static Value *buildMinMaxReductionSeq(Location loc, CmpIPredicate predicate,
                                      ArrayRef<Value *> values,
                                      OpBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value *value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<SelectOp>(loc, cmpOp.getResult(), value, *valueIt);
  }

  return value;
}

// Emit instructions that correspond to the affine map in the lower bound
// applied to the respective operands, and compute the maximum value across
// the results.
Value *mlir::lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value *, 8> boundOperands(op.getLowerBoundOperands());
  auto lbValues = expandAffineMap(builder, op.getLoc(), op.getLowerBoundMap(),
                                  boundOperands);
  if (!lbValues)
    return nullptr;
  return buildMinMaxReductionSeq(op.getLoc(), CmpIPredicate::sgt, *lbValues,
                                 builder);
}

// Emit instructions that correspond to the affine map in the upper bound
// applied to the respective operands, and compute the minimum value across
// the results.
Value *mlir::lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value *, 8> boundOperands(op.getUpperBoundOperands());
  auto ubValues = expandAffineMap(builder, op.getLoc(), op.getUpperBoundMap(),
                                  boundOperands);
  if (!ubValues)
    return nullptr;
  return buildMinMaxReductionSeq(op.getLoc(), CmpIPredicate::slt, *ubValues,
                                 builder);
}

namespace {
// Affine terminators are removed.
class AffineTerminatorLowering : public OpRewritePattern<AffineTerminatorOp> {
public:
  using OpRewritePattern<AffineTerminatorOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineTerminatorOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<loop::TerminatorOp>(op);
    return matchSuccess();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineForOp op,
                                     PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value *lowerBound = lowerAffineLowerBound(op, rewriter);
    Value *upperBound = lowerAffineUpperBound(op, rewriter);
    Value *step = rewriter.create<ConstantIndexOp>(loc, op.getStep());
    auto f = rewriter.create<loop::ForOp>(loc, lowerBound, upperBound, step);
    f.region().getBlocks().clear();
    rewriter.inlineRegionBefore(op.region(), f.region(), f.region().end());
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineIfOp op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value *zeroConstant = rewriter.create<ConstantIndexOp>(loc, 0);
    SmallVector<Value *, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value *cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value *affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                          operandsRef.take_front(numDims),
                                          operandsRef.drop_front(numDims));
      if (!affResult)
        return matchFailure();
      auto pred = isEquality ? CmpIPredicate::eq : CmpIPredicate::sge;
      Value *cmpVal =
          rewriter.create<CmpIOp>(loc, pred, affResult, zeroConstant);
      cond =
          cond ? rewriter.create<AndOp>(loc, cond, cmpVal).getResult() : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<ConstantIntOp>(loc, /*value=*/1, /*width=*/1);

    bool hasElseRegion = !op.elseRegion().empty();
    auto ifOp = rewriter.create<loop::IfOp>(loc, cond, hasElseRegion);
    rewriter.inlineRegionBefore(op.thenRegion(), &ifOp.thenRegion().back());
    ifOp.thenRegion().back().erase();
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.elseRegion(), &ifOp.elseRegion().back());
      ifOp.elseRegion().back().erase();
    }

    // Ok, we're done!
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Convert an "affine.apply" operation into a sequence of arithmetic
// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineApplyOp op,
                                     PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return matchFailure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.load' operation to its operands, and
// feed the results to a newly created 'std.load' operation (which replaces the
// original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineLoadOp op,
                                     PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value *, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return matchFailure();

    // Build std.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<LoadOp>(op, op.getMemRef(), *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.store' operation to its operands, and
// feed the results to a newly created 'std.store' operation (which replaces the
// original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineStoreOp op,
                                     PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value *, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return matchFailure();

    // Build std.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValueToStore(),
                                         op.getMemRef(), *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine maps from an 'affine.dma_start' operation to each of their
// respective map operands, and feed the results to a newly created
// 'std.dma_start' operation (which replaces the original 'affine.dma_start').
class AffineDmaStartLowering : public OpRewritePattern<AffineDmaStartOp> {
public:
  using OpRewritePattern<AffineDmaStartOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineDmaStartOp op,
                                     PatternRewriter &rewriter) const override {
    SmallVector<Value *, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Expand affine map for DMA source memref.
    auto maybeExpandedSrcMap = expandAffineMap(
        rewriter, op.getLoc(), op.getSrcMap(),
        operandsRef.drop_front(op.getSrcMemRefOperandIndex() + 1));
    if (!maybeExpandedSrcMap)
      return matchFailure();
    // Expand affine map for DMA destination memref.
    auto maybeExpandedDstMap = expandAffineMap(
        rewriter, op.getLoc(), op.getDstMap(),
        operandsRef.drop_front(op.getDstMemRefOperandIndex() + 1));
    if (!maybeExpandedDstMap)
      return matchFailure();
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap = expandAffineMap(
        rewriter, op.getLoc(), op.getTagMap(),
        operandsRef.drop_front(op.getTagMemRefOperandIndex() + 1));
    if (!maybeExpandedTagMap)
      return matchFailure();

    // Build std.dma_start operation with affine map results.
    rewriter.replaceOpWithNewOp<DmaStartOp>(
        op, op.getSrcMemRef(), *maybeExpandedSrcMap, op.getDstMemRef(),
        *maybeExpandedDstMap, op.getNumElements(), op.getTagMemRef(),
        *maybeExpandedTagMap, op.getStride(), op.getNumElementsPerStride());
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.dma_wait' operation tag memref,
// and feed the results to a newly created 'std.dma_wait' operation (which
// replaces the original 'affine.dma_wait').
class AffineDmaWaitLowering : public OpRewritePattern<AffineDmaWaitOp> {
public:
  using OpRewritePattern<AffineDmaWaitOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineDmaWaitOp op,
                                     PatternRewriter &rewriter) const override {
    // Expand affine map for DMA tag memref.
    SmallVector<Value *, 8> indices(op.getTagIndices());
    auto maybeExpandedTagMap =
        expandAffineMap(rewriter, op.getLoc(), op.getTagMap(), indices);
    if (!maybeExpandedTagMap)
      return matchFailure();

    // Build std.dma_wait operation with affine map results.
    rewriter.replaceOpWithNewOp<DmaWaitOp>(
        op, op.getTagMemRef(), *maybeExpandedTagMap, op.getNumElements());
    return matchSuccess();
  }
};

} // end namespace

void mlir::populateAffineToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns
      .insert<MatmulMatcher>(
          ctx);
}

namespace {
class LowerAffinePass : public FunctionPass<LowerAffinePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<loop::LoopOpsDialect, StandardOpsDialect>();
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
std::unique_ptr<OpPassBase<FuncOp>> mlir::createLowerAffinePass() {
  return std::make_unique<LowerAffinePass>();
}

static PassRegistration<LowerAffinePass>
    pass("lower-affine",
         "Lower If, For, AffineApply operations to primitive equivalents");
