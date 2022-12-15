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

#ifndef GRID_FEATURE_COMMON_VOXEL_HASH_CUH
#define GRID_FEATURE_COMMON_VOXEL_HASH_CUH

#include <helper_math.h>

namespace grid_feature {

namespace voxel_hash {

__device__ inline __host__
int force_align(int size, int mod = 8) {
  auto reminder = size % mod;
  return size + reminder;
}


__device__ inline __host__
int compute_grid_size(int G0, float growth_factor, int level) {
  auto Gf = floor(G0 * pow(growth_factor, level));
  return int(Gf);
}

__device__ inline __host__
int compute_table_size(int G, int T0) {
  // address the case: G**3 > 2**31 - 1
  float Gf = G;
  auto T = min(Gf * Gf * Gf, float(T0));
  return min(int(T), int(T0));
}

__device__ inline __host__
int compute_num_params(int G0, float growth_factor, int T0, int L, int D, int mod = 8) {
  auto n_params = 0;
  for (int l = 0; l < L; l++) {
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto n_params_l = force_align(T * D, mod);
    n_params += n_params_l;
  }
  return n_params;
}

}

}

#endif