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

#ifndef GRID_FEATURE_COMMON_CUH
#define GRID_FEATURE_COMMON_CUH

#include <helper_math.h>

namespace grid_feature {

__global__
void kernel_zero(int N, float *data) {
  NBLA_CUDA_KERNEL_LOOP(n, N) {
    data[n] = 0.f;
  }
}

__device__ inline
int3 flat_to_3d(const int n, const int D) {
  auto b = n / (D * 3);
  auto d = (n - b * (D * 3)) / 3;
  auto i = n - b * (D * 3) - d * 3;
  return make_int3(b, d, i);
}

__device__ inline
uint2 flat_to_2d(int n, int B) {
  auto l = n / B;
  auto b = n - l * B;
  return make_uint2(l, b);
}

__device__ inline
float3 cos(float3 data) {
  return make_float3(cosf(data.x), cosf(data.y), cosf(data.z));
}

__device__ inline
float3 sin(float3 data) {
  return make_float3(sinf(data.x), sinf(data.y), sinf(data.z));
}

__device__ inline
float sinc(float x) {
  if (x == 0.f)
    return 1.0;
  return sinf(x) / x;
}


__device__ inline
float lanczos(float x, int a) {
  auto z = M_PI * x;
  auto u = sinc(z);
  auto v = sinc(z / a);
  auto y = u * v;
  return y;
}

__device__ inline
int3 to_int3(float3 data) {
  return make_int3(data.x, data.y, data.z);
}

__device__ inline
float3 to_float3(int3 data) {
  return make_float3(data.x, data.y, data.z);
}


__device__ inline
float grad_coefficient(float x, int a) {
  if (x == 0.f)
    return 0.0;

  auto z0 = M_PI * x;
  auto z1 = M_PI * x / a;
  auto sinc_z0 = sinc(z0);
  auto sinc_z1 = sinc(z1);

  auto t0 = (cosf(z0) - sinc_z0) * sinc_z1;
  auto t1 = (cosf(z1) - sinc_z1) * sinc_z0;

  auto gc = (t0 + t1) / x;
  return gc;
}


} // grid_feature

#endif