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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_common.cuh>
#include <helper_math.h>


namespace py = pybind11;


namespace ray_aabb_intersection {


__device__ 
bool intersect(const float3 &x, const float &t, 
               const float3 &min, const float3 &max) {
  auto cond = true;
  cond &= (t >= 0.f);
  cond &= (x.x >= min.x) && (x.x <= max.x);
  cond &= (x.y >= min.y) && (x.y <= max.y);
  cond &= (x.z >= min.z) && (x.z <= max.z);
  return cond;
}


__device__
void compute_t_list_and_x_list(float *t_list, float3 *x_list, 
                               const float3 &camloc, const float3 &raydir, 
                               const float3 &min, const float3 &max) {
  auto inv_raydir = 1.f / raydir;
  auto t_list_max = (max - camloc) * inv_raydir;
  auto t_list_min = (min - camloc) * inv_raydir;
  t_list[0] = t_list_max.x;
  t_list[1] = t_list_max.y;
  t_list[2] = t_list_max.z;
  t_list[3] = t_list_min.x;
  t_list[4] = t_list_min.y;
  t_list[5] = t_list_min.z;

  x_list[0] = camloc + t_list[0] * raydir;
  x_list[1] = camloc + t_list[1] * raydir;
  x_list[2] = camloc + t_list[2] * raydir;
  x_list[3] = camloc + t_list[3] * raydir;
  x_list[4] = camloc + t_list[4] * raydir;
  x_list[5] = camloc + t_list[5] * raydir;

  // explicitly correct due to numerical error
  x_list[0].x = max.x;
  x_list[1].y = max.y;
  x_list[2].z = max.z;
  x_list[3].x = min.x;
  x_list[4].y = min.y;
  x_list[5].z = min.z;
}


__global__
void kernel_ray_aabb_intersection(int N, float *t_near, float *t_far, float *n_hits, 
                                  const float *camloc, const float *raydir, 
                                  const int B, const int R, 
                                  const float3 min, const float3 max) {
  // TODO: bias

  // t_near: (B, R, 1)
  // t_far: (B, R, 1)
  // n_hits: (B, R, 1)
  // camloc: (B, 3)
  // raydir: (B, R, 3)

  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = B * R
    auto b = n / R;

    auto camloc_n = *(float3*)(camloc + b * 3);
    auto raydir_n = *(float3*)(raydir + n * 3);
    
    auto n_hits_n = 0;
    auto index_near_or_far = make_int2(0, 0);
    float t_list[6];
    float3 x_list[6];
    compute_t_list_and_x_list(t_list, x_list, camloc_n, raydir_n, 
                              min, max);

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
      auto t = t_list[i];
      auto x = x_list[i];

      // parallel case
      if (isinf(t))
        continue;

      // intersect but out-of-bounds
      auto cond = intersect(x, t, min, max);
      if (!cond) 
        continue;

      // if intersection is on vertex, multiple conditions are met, 
      // then overwrites the last one.
      if (n_hits_n == 0)
        index_near_or_far.x = i;
      else
        index_near_or_far.y = i;

      n_hits_n++;
    }

    *(n_hits + n) = n_hits_n;
    auto t_near_n = 0.f;
    auto t_far_n = 0.f;
    
    if (n_hits_n >= 2) {
      auto ix = index_near_or_far.x;
      auto iy = index_near_or_far.y;
      if (t_list[ix] <= t_list[iy]) {
        t_near_n = t_list[ix];
        t_far_n = t_list[iy];
      } else {
        t_near_n = t_list[iy];
        t_far_n = t_list[ix];
      }
    } else if (n_hits_n == 1) { // ray casted inside AABB
      t_far_n = t_list[index_near_or_far.x];
    }

    *(float*)(t_near + n) = t_near_n;
    *(float*)(t_far + n) = t_far_n;
  }

}


void ray_aabb_intersection(int N, int64_t t_near_ptr, int64_t t_far_ptr, int64_t n_hits_ptr, 
                           int64_t camloc_ptr, int64_t raydir_ptr, 
                           const int B, const int R, 
                           const std::vector<float> min, const std::vector<float> max) {

  auto t_near = reinterpret_cast<float*>(t_near_ptr);
  auto t_far = reinterpret_cast<float*>(t_far_ptr);
  auto hist_mask = reinterpret_cast<float*>(n_hits_ptr);
  auto camloc = reinterpret_cast<float*>(camloc_ptr);
  auto raydir = reinterpret_cast<float*>(raydir_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_ray_aabb_intersection, N, 
                                 t_near, t_far, hist_mask, 
                                 camloc, raydir, 
                                 B, R, 
                                 make_float3(min[0], min[1], min[2]), 
                                 make_float3(max[0], max[1], max[2]));
}

} // namespace ray_aabb_intersection

PYBIND11_MODULE(ray_aabb_intersection_cuda, m) {
  m.doc() = "Ray-AABB Intersection";
  m.def("ray_aabb_intersection", &ray_aabb_intersection::ray_aabb_intersection, "Ray-AABB intersection.");
}
