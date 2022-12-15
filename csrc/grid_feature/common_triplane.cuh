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

#ifndef GRID_FEATURE_COMMON_TRIPLANE_CUH
#define GRID_FEATURE_COMMON_TRIPLANE_CUH

#include <helper_math.h>

namespace grid_feature {

namespace triplane {

__device__ inline
int4 select_locations(const int i, 
                      const float3 xyz0, 
                      const float3 xyz1) {
  if (i == 0) {
    return make_int4(xyz0.x, xyz1.x, xyz0.y, xyz1.y);
  } else if (i == 1) {
    return make_int4(xyz0.y, xyz1.y, xyz0.z, xyz1.z);
  } else {
    return make_int4(xyz0.z, xyz1.z, xyz0.x, xyz1.x);
  }
}

__device__ inline
float2 select_locations(const int l, 
                      const float3 xyz0) {
  if (l == 0) {
    return make_float2(xyz0.x, xyz0.y);
  } else if (l == 1) {
    return make_float2(xyz0.y, xyz0.z);
  } else {
    return make_float2(xyz0.z, xyz0.x);
  }
}

__device__ inline
float4 select_coefficients(const int i, 
                           const float3 pqr0, 
                           const float3 pqr1) {
  if (i == 0) {
    return make_float4(pqr0.x, pqr1.x, pqr0.y, pqr1.y);
  } else if (i == 1) {
    return make_float4(pqr0.y, pqr1.y, pqr0.z, pqr1.z);
  } else {
    return make_float4(pqr0.z, pqr1.z, pqr0.x, pqr1.x);
  }
}


__device__ inline
float2 select_scales(const int i, 
                     const float3 scales) {
  if (i == 0) {
    return make_float2(scales.x, scales.y);
  } else if (i == 1) {
    return make_float2(scales.y, scales.z);
  } else {
    return make_float2(scales.z, scales.x);
  }
}

__device__ inline
int2 select_shift(const int i) {
  if (i == 0) {
    return make_int2(0, 1);
  } else if (i == 1) {
    return make_int2(1, 2);
  } else {
    return make_int2(2, 0);
  }
}

__device__ inline
float2 select_grad_coefficients(const int i, 
                                const float3 gpqr0) {
  if (i == 0) {
    return make_float2(gpqr0.x, gpqr0.y);
  } else if (i == 1) {
    return make_float2(gpqr0.y, gpqr0.z);
  } else {
    return make_float2(gpqr0.z, gpqr0.x);
  }
}

__device__ inline
float2 select_gguv(const int i, 
                   const float3 gg_xyz) {
  if (i == 0) {
    return make_float2(gg_xyz.x, gg_xyz.y);
  } else if (i == 1) {
    return make_float2(gg_xyz.y, gg_xyz.z);
  } else {
    return make_float2(gg_xyz.z, gg_xyz.x);
  }
}

}

}

#endif