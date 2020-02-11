 // RUN: ./../../../../bin/mlir-pet /llvm-project/mlir/tools/mlir-pet/inputs/gemm.c | FileCheck %s
// CHECK:  func @scop_entry([[arg2:[%][a-z0-9]*]]: memref<1024x1024xf32>, [[arg3:[%][a-z0-9]*]]: memref<1024x1024xf32>, [[arg4:[%][a-z0-9]*]]: memref<1024x1024xf32>, [[arg5:[%][a-z0-9]*]]: f32, [[arg6:[%][a-z0-9]*]]: f32) {
// CHECK-NEXT:			      "affine.for"() ( {
// CHECK-NEXT:    [[b0:[\^][a-z0-9]*]]([[arg0:[%][a-z0-9]*]]: index):	// no predecessors
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       "affine.for"() ( {
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       [[b0]]([[arg1:[%][a-z0-9]*]]: index):	// no predecessors
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         %0 = "affine.load"([[arg4]], [[arg0]], [[arg1]]) {map = [[map0:[#][a-z0-9]*]]} : (memref<1024x1024xf32>, index, index) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         %1 = "std.mulf"([[arg6]], %0) : (f32, f32) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         "affine.store"(%1, [[arg4]], [[arg0]], [[arg1]]) {map = [[map0]]} : (f32, memref<1024x1024xf32>, index, index) -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         "affine.terminator"() : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       }) {lower_bound = #map1, step = 1 : index, upper_bound = [[map2:[#][a-z0-9]*]]} : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       "affine.for"() ( {
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       [[b0]]([[arg1]]: index):	// no predecessors
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         "affine.for"() ( {
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         [[b0]](%arg7: index):	// no predecessors
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %0 = "affine.load"([[arg2]], [[arg0]], [[arg1]]) {map = [[map0]]} : (memref<1024x1024xf32>, index, index) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %1 = "std.mulf"([[arg5]], %0) : (f32, f32) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %2 = "affine.load"([[arg3]], [[arg1]], %arg7) {map = [[map0]]} : (memref<1024x1024xf32>, index, index) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %3 = "std.mulf"(%1, %2) : (f32, f32) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %4 = "affine.load"([[arg4]], [[arg0]], %arg7) {map = [[map0]]} : (memref<1024x1024xf32>, index, index) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           %5 = "std.mulf"(%3, %4) : (f32, f32) -> f32
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           "affine.store"(%5, [[arg4]], [[arg0]], %arg7) {map = [[map0]]} : (f32, memref<1024x1024xf32>, index, index) -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:           "affine.terminator"() : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         }) {lower_bound = #map3, step = 1 : index, upper_bound = [[map2]]} : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:         "affine.terminator"() : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       }) {lower_bound = [[map4:[#][a-z0-9]*]], step = 1 : index, upper_bound = [[map2]]} : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:       "affine.terminator"() : () -> ()
// CHECK-NEXT:     }) {lower_bound = [[map4]], step = 1 : index, upper_bound = [[map2]]} : () -> ()
// CHECK-NOT:        {{[^ ]+}}
// CHECK-NEXT:     "std.return"() : () -> ()


