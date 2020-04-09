// RUN: mlir-pet %S/Inputs/gemver.c | FileCheck %s
// CHECK:  func @scop_entry(%arg0: memref<1024x1024xf32>, %arg1: f32, %arg2: f32, %arg3: memref<1024xf32>, %arg4: memref<1024xf32>, %arg5: memref<1024xf32>, %arg6: memref<1024xf32>, %arg7: memref<1024xf32>, %arg8: memref<1024xf32>, %arg9: memref<1024xf32>, %arg10: memref<1024xf32>) {
// CHECK-NEXT:    "affine.for"() ( {
// CHECK-NEXT:    ^bb0(%arg11: index):	// no predecessors
// CHECK-NEXT:      "affine.for"() ( {
// CHECK-NEXT:      ^bb0(%arg12: index):	// no predecessors
 // CHECK-NEXT:       %0 = "affine.load"(%arg0, %arg11, %arg12) {map = #map0} : (memref<1024x1024xf32>, index, index) -> f32
 // CHECK-NEXT:       %1 = "affine.load"(%arg3, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %2 = "affine.load"(%arg5, %arg12) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %3 = "std.mulf"(%1, %2) : (f32, f32) -> f32
 // CHECK-NEXT:       %4 = "std.mulf"(%0, %3) : (f32, f32) -> f32
 // CHECK-NEXT:       %5 = "affine.load"(%arg4, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %6 = "affine.load"(%arg6, %arg12) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %7 = "std.mulf"(%5, %6) : (f32, f32) -> f32
 // CHECK-NEXT:       %8 = "std.mulf"(%4, %7) : (f32, f32) -> f32
 // CHECK-NEXT:       "affine.store"(%8, %arg0, %arg11, %arg12) {map = #map0} : (f32, memref<1024x1024xf32>, index, index) -> ()
 // CHECK-NEXT:       "affine.terminator"() : () -> ()
 // CHECK-NEXT:     }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:     "affine.terminator"() : () -> ()
 // CHECK-NEXT:   }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:   "affine.for"() ( {
 // CHECK-NEXT:   ^bb0(%arg11: index):	// no predecessors
 // CHECK-NEXT:     "affine.for"() ( {
 // CHECK-NEXT:     ^bb0(%arg12: index):	// no predecessors
 // CHECK-NEXT:       %0 = "affine.load"(%arg8, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %1 = "affine.load"(%arg0, %arg12, %arg11) {map = #map0} : (memref<1024x1024xf32>, index, index) -> f32
 // CHECK-NEXT:       %2 = "std.mulf"(%arg2, %1) : (f32, f32) -> f32
 // CHECK-NEXT:       %3 = "affine.load"(%arg9, %arg12) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %4 = "std.mulf"(%2, %3) : (f32, f32) -> f32
 // CHECK-NEXT:       %5 = "std.mulf"(%0, %4) : (f32, f32) -> f32
 // CHECK-NEXT:       "affine.store"(%5, %arg8, %arg11) {map = #map1} : (f32, memref<1024xf32>, index) -> ()
 // CHECK-NEXT:       "affine.terminator"() : () -> ()
 // CHECK-NEXT:     }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:     "affine.terminator"() : () -> ()
 // CHECK-NEXT:   }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:   "affine.for"() ( {
 // CHECK-NEXT:   ^bb0(%arg11: index):	// no predecessors
 // CHECK-NEXT:     %0 = "affine.load"(%arg8, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:     %1 = "affine.load"(%arg10, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:     %2 = "std.mulf"(%0, %1) : (f32, f32) -> f32
 // CHECK-NEXT:     "affine.store"(%2, %arg8, %arg11) {map = #map1} : (f32, memref<1024xf32>, index) -> ()
 // CHECK-NEXT:     "affine.terminator"() : () -> ()
 // CHECK-NEXT:   }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:   "affine.for"() ( {
 // CHECK-NEXT:   ^bb0(%arg11: index):	// no predecessors
 // CHECK-NEXT:     "affine.for"() ( {
 // CHECK-NEXT:     ^bb0(%arg12: index):	// no predecessors
 // CHECK-NEXT:       %0 = "affine.load"(%arg7, %arg11) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %1 = "affine.load"(%arg0, %arg11, %arg12) {map = #map0} : (memref<1024x1024xf32>, index, index) -> f32
 // CHECK-NEXT:       %2 = "std.mulf"(%arg1, %1) : (f32, f32) -> f32
 // CHECK-NEXT:       %3 = "affine.load"(%arg8, %arg12) {map = #map1} : (memref<1024xf32>, index) -> f32
 // CHECK-NEXT:       %4 = "std.mulf"(%2, %3) : (f32, f32) -> f32
 // CHECK-NEXT:       %5 = "std.mulf"(%0, %4) : (f32, f32) -> f32
 // CHECK-NEXT:       "affine.store"(%5, %arg7, %arg11) {map = #map1} : (f32, memref<1024xf32>, index) -> ()
 // CHECK-NEXT:       "affine.terminator"() : () -> ()
 // CHECK-NEXT:     }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:     "affine.terminator"() : () -> ()
 // CHECK-NEXT:   }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
 // CHECK-NEXT:   "std.return"() : () -> ()
