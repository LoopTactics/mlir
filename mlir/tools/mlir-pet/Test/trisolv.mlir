// RUN: mlir-pet %S/Inputs/trisolv.c | FileCheck %s
// CHECK:#map0 = affine_map<(d0) -> (d0)>
// CHECK:#map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:#map2 = affine_map<() -> (0)>
// CHECK:#map3 = affine_map<() -> (2000)>

// CHECK:module {
// CHECK:  func @scop_entry(%arg0: memref<2000x2000xf32>, %arg1: memref<2000xf32>, %arg2: memref<2000xf32>) {
// CHECK:    affine.for %arg3 = 0 to 2000 {
// CHECK:      %0 = affine.load %arg1[%arg3] : memref<2000xf32>
// CHECK:      affine.store %0, %arg2[%arg3] : memref<2000xf32>
// CHECK:      affine.for %arg4 = 0 to #map0(%arg3) {
// CHECK:        %4 = affine.load %arg2[%arg3] : memref<2000xf32>
// CHECK:        %5 = affine.load %arg0[%arg3, %arg4] : memref<2000x2000xf32>
// CHECK:        %6 = affine.load %arg2[%arg4] : memref<2000xf32>
// CHECK:        %7 = mulf %5, %6 : f32
// CHECK:        %8 = subf %4, %7 : f32
// CHECK:        affine.store %8, %arg2[%arg3] : memref<2000xf32>
// CHECK:      }
// CHECK:      %1 = affine.load %arg2[%arg3] : memref<2000xf32>
// CHECK:      %2 = affine.load %arg0[%arg3, %arg3] : memref<2000x2000xf32>
// CHECK:      %3 = divf %1, %2 : f32
// CHECK:      affine.store %3, %arg2[%arg3] : memref<2000xf32>
// CHECK:    }
// CHECK:    return
// CHECK:  }
// CHECK:}