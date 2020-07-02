// RUN: mlir-pet %S/Inputs/negLoop.c | FileCheck %s
// CHECK:#map0 = affine_map<(d0) -> (d0)>
// CHECK:#map1 = affine_map<(d0, d1) -> (-d1)>
// CHECK:#map2 = affine_map<(d0) -> (-d0)>
// CHECK:#map3 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:#map4 = affine_map<() -> (-200)>
// CHECK:#map5 = affine_map<() -> (-99)>
// CHECK:#map6 = affine_map<() -> (101)>
// CHECK:#map7 = affine_map<() -> (201)>


// CHECK:module {
// CHECK:  func @scop_entry(%arg0: memref<100x100xf32>, %arg1: memref<1xf32>) {
// CHECK:    %cst = constant 0.000000e+00 : f32
// CHECK:    %c0 = constant 0 : index
// CHECK:    affine.store %cst, %arg1[%c0] : memref<1xf32>
// CHECK:    affine.for %arg2 = -200 to -99 {
// CHECK:      affine.for %arg3 = -200 to -99 {
// CHECK:        %0 = affine.apply #map1(%arg2, %arg3)
// CHECK:        %1 = affine.apply #map2(%arg2)
// CHECK:        %c0_0 = constant 0 : index
// CHECK:        %2 = affine.load %arg1[%c0_0] : memref<1xf32>
// CHECK:        %cst_1 = constant 5.000000e+00 : f32
// CHECK:        %3 = addf %2, %cst_1 : f32
// CHECK:        %c0_2 = constant 0 : index
// CHECK:        affine.store %3, %arg1[%c0_2] : memref<1xf32>
// CHECK:        %4 = affine.apply #map1(%arg2, %arg3)
// CHECK:        %5 = affine.apply #map2(%arg2)
// CHECK:        %cst_3 = constant 2.000000e+00 : f32
// CHECK:        affine.store %cst_3, %arg0[%4, %5] : memref<100x100xf32>
// CHECK:      }
// CHECK:      affine.for %arg3 = 101 to 201 {
// CHECK:        %0 = affine.apply #map2(%arg2)
// CHECK:        %c0_0 = constant 0 : index
// CHECK:        %1 = affine.load %arg1[%c0_0] : memref<1xf32>
// CHECK:        %cst_1 = constant 5.000000e+00 : f32
// CHECK:        %2 = addf %1, %cst_1 : f32
// CHECK:        %c0_2 = constant 0 : index
// CHECK:        affine.store %2, %arg1[%c0_2] : memref<1xf32>
// CHECK:        %3 = affine.apply #map2(%arg2)
// CHECK:        %cst_3 = constant 2.000000e+00 : f32
// CHECK:        affine.store %cst_3, %arg0[%3, %arg3] : memref<100x100xf32>
// CHECK:      }
// CHECK:    }
// CHECK:    return
// CHECK:  }
// CHECK:}