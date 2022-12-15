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


namespace ray_sphere_intersection {

__global__
void kernel_ray_sphere_intersection(int N, float *t_near, float *t_far, float *n_hits, 
                                  const float *camloc, const float *raydir, 
                                  const int B, const int R, 
                                  const float radius = 1.f) {
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

    // t = {X -+ sqrt(Y)} / Z
    // X = -cv
    // Y = cv**2 - vv(cc - radius**2)
    // Z = vv
    auto r2 = radius * radius;
    auto cv = dot(camloc_n, raydir_n);
    auto vv = dot(raydir_n, raydir_n);
    auto cc = dot(camloc_n, camloc_n);
    auto X = -cv;
    auto Y = cv * cv - vv * (cc - r2);
    auto Z = vv;
    
    auto n_hits_n = 0;
    auto t_near_n = 0.f;
    auto t_far_n = 0.f;
    auto Z_inv = 1.f / Z;
    if (Y > 0) {
        auto Y_sqrt = sqrt(Y);
        t_near_n = (X - Y_sqrt) * Z_inv;
        t_far_n = (X + Y_sqrt) * Z_inv;
        auto pos_mask = int(t_near_n >= 0);
        t_near_n = pos_mask * t_near_n;
        n_hits_n = 2 - (1 - pos_mask);
    } else if (Y == 0) {
        n_hits_n = 1;
        t_near_n = X * Z_inv;
        t_far_n = X * Z_inv;
    }
    
    *(n_hits + n) = n_hits_n;
    *(float*)(t_near + n) = t_near_n;
    *(float*)(t_far + n) = t_far_n;
  }
}


void ray_sphere_intersection(int N, int64_t t_near_ptr, int64_t t_far_ptr, int64_t n_hits_ptr, 
                           int64_t camloc_ptr, int64_t raydir_ptr, 
                           const int B, const int R, 
                           const float radius = 1.f) {

  auto t_near = reinterpret_cast<float*>(t_near_ptr);
  auto t_far = reinterpret_cast<float*>(t_far_ptr);
  auto hist_mask = reinterpret_cast<float*>(n_hits_ptr);
  auto camloc = reinterpret_cast<float*>(camloc_ptr);
  auto raydir = reinterpret_cast<float*>(raydir_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_ray_sphere_intersection, N, 
                                 t_near, t_far, hist_mask, 
                                 camloc, raydir, 
                                 B, R, 
                                 radius);
}

} // namespace ray_sphere_intersection

PYBIND11_MODULE(ray_sphere_intersection_cuda, m) {
  m.doc() = "Ray-sphere Intersection";
  m.def("ray_sphere_intersection", &ray_sphere_intersection::ray_sphere_intersection, "Ray-sphere intersection.");
}
