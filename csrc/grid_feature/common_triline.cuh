// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRID_FEATURE_COMMON_TRILINE_CUH
#define GRID_FEATURE_COMMON_TRILINE_CUH

#include <helper_math.h>

namespace grid_feature {

namespace triline {

__device__ inline
int2 select_locations(const int i, 
                      const float3 xyz0, 
                      const float3 xyz1) {
  if (i == 0) {
    return make_int2(xyz0.x, xyz1.x);
  } else if (i == 1) {
    return make_int2(xyz0.y, xyz1.y);
  } else {
    return make_int2(xyz0.z, xyz1.z);
  }
}

__device__ inline
float2 select_coefficients(const int i, 
                           const float3 pqr0, 
                           const float3 pqr1) {
  if (i == 0) {
    return make_float2(pqr0.x, pqr1.x);
  } else if (i == 1) {
    return make_float2(pqr0.y, pqr1.y);
  } else {
    return make_float2(pqr0.z, pqr1.z);
  }
}

__device__ inline
float select_grad_coefficient(const int i, 
                              const float3 gpqr0) {
                              
  if (i == 0) {
    return gpqr0.x;
  } else if (i == 1) {
    return gpqr0.y;
  } else {
    return gpqr0.z;
  }
}

__device__ inline
float select_location(const int l, 
                      const float3 xyz0) {
  if (l == 0) {
    return xyz0.x;
  } else if (l == 1) {
    return xyz0.y;
  } else {
    return xyz0.z;
  }
}

__device__ inline
float select_scale(const int i, 
                   const float3 scales) {
  if (i == 0) {
    return scales.x;
  } else if (i == 1) {
    return scales.y;
  } else {
    return scales.z;
  }
}

__device__ inline
float select_ggu(const int i, 
                   const float3 gg_xyz) {
  if (i == 0) {
    return gg_xyz.x;
  } else if (i == 1) {
    return gg_xyz.y;
  } else {
    return gg_xyz.z;
  }
}

}

}

#endif