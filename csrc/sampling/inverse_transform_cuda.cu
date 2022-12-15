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
#include <iostream>

#include <helper_math.h>
#include <cuda_common.cuh>

#include <pybind11/stl.h>


namespace py = pybind11;


namespace sampling {


__global__
void kernel_sample_uniform_directions(int size,
                                      float *light_dirs,
                                      const float *normal, 
                                      const float *cdf_the,
                                      const float *cdf_phi,
                                      const int batch_size, const int n_lights, 
                                      const int n_thes, const int n_phis, const float eps) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {  // (BR, M)
    auto b = s / n_lights;
    auto m = s - b * n_lights;
  
    auto m_the = m / n_phis;
    auto m_phi = m - m_the * n_phis;
    
    auto cdf_the_bm = cdf_the[b * n_thes + m_the];
    auto cdf_phi_bm = cdf_phi[b * n_phis + m_phi];
    
    auto phi = 2.f * M_PI * cdf_phi_bm;
    auto cos_the = cdf_the_bm;
    auto sin_the = sqrtf(1.f - cos_the * cos_the);

    // assume mathematical coordinate system below
    auto x = sin_the * cosf(phi);
    auto y = sin_the * sinf(phi);
    auto z = cos_the;
    auto xyz = make_float3(x, y, z);
    
    auto normal_b = *(float3*)(normal + (b * 3)) + eps;
    auto z_axis = normalize(normal_b);
    auto x_axis = normalize(make_float3(-normal_b.y, normal_b.x, 0.f));
    auto y_axis = cross(z_axis, x_axis);
    auto x_oriented = dot(xyz, make_float3(x_axis.x, y_axis.x, z_axis.x));
    auto y_oriented = dot(xyz, make_float3(x_axis.y, y_axis.y, z_axis.y));
    auto z_oriented = dot(xyz, make_float3(x_axis.z, y_axis.z, z_axis.z));
    auto xyz_oriented = make_float3(x_oriented, y_oriented, z_oriented);

    *(float3*)(light_dirs + s * 3) = xyz_oriented;
  }
}


void sample_uniform_directions(int size,
                               int64_t light_dirs_ptr, 
                               int64_t normal_ptr, 
                               int64_t cdf_the_ptr, 
                               int64_t cdf_phi_ptr, 
                               const int batch_size, const int n_lights, 
                               const int n_thes, const int n_phis, const float eps) {

  auto light_dirs = reinterpret_cast<float*>(light_dirs_ptr);
  auto normal = reinterpret_cast<float*>(normal_ptr);
  auto cdf_the = reinterpret_cast<float*>(cdf_the_ptr);
  auto cdf_phi = reinterpret_cast<float*>(cdf_phi_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_sample_uniform_directions, size,
                                 light_dirs, 
                                 normal, cdf_the, cdf_phi, 
                                 batch_size, n_lights, 
                                 n_thes, n_phis, eps);
}


__global__
void kernel_sample_importance_directions(int size,
                                         float *light_dirs,
                                         const float *normal, 
                                         const float *cdf_the,
                                         const float *cdf_phi,
                                         const float *alpha, 
                                         const int batch_size, const int n_lights, 
                                         const int n_thes, const int n_phis, 
                                         const float eps) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {  // (BR, M)
    auto b = s / n_lights;
    auto m = s - b * n_lights;
  
    auto m_the = m / n_phis;
    auto m_phi = m - m_the * n_phis;
    
    auto cdf_the_bm = cdf_the[b * n_thes + m_the];
    auto cdf_phi_bm = cdf_phi[b * n_phis + m_phi];
    
    auto alpha_b = *(alpha + b);
    auto alpha_b2 = alpha_b * alpha_b;
    auto phi = 2.f * M_PI * cdf_phi_bm;
    auto cos_the = sqrtf((1.f - cdf_the_bm) / ((alpha_b2 - 1.f) * cdf_the_bm + 1.f));
    auto sin_the = sqrtf(1.f - cos_the * cos_the);

    // assume mathematical coordinate system below
    auto x = sin_the * cosf(phi);
    auto y = sin_the * sinf(phi);
    auto z = cos_the;
    auto xyz = make_float3(x, y, z);
    
    auto normal_b = *(float3*)(normal + (b * 3)) + eps;
    auto z_axis = normalize(normal_b);
    auto x_axis = normalize(make_float3(-normal_b.y, normal_b.x, 0.f));
    auto y_axis = cross(z_axis, x_axis);
    auto x_oriented = dot(xyz, make_float3(x_axis.x, y_axis.x, z_axis.x));
    auto y_oriented = dot(xyz, make_float3(x_axis.y, y_axis.y, z_axis.y));
    auto z_oriented = dot(xyz, make_float3(x_axis.z, y_axis.z, z_axis.z));
    auto xyz_oriented = make_float3(x_oriented, y_oriented, z_oriented);

    *(float3*)(light_dirs + s * 3) = xyz_oriented;
  }
}


void sample_importance_directions(int size,
                                  int64_t light_dirs_ptr, 
                                  int64_t normal_ptr, 
                                  int64_t cdf_the_ptr, 
                                  int64_t cdf_phi_ptr,
                                  int64_t alpha_ptr,  
                                  const int batch_size, const int n_lights, 
                                  const int n_thes, const int n_phis, const float eps) {

  auto light_dirs = reinterpret_cast<float*>(light_dirs_ptr);
  auto normal = reinterpret_cast<float*>(normal_ptr);
  auto cdf_the = reinterpret_cast<float*>(cdf_the_ptr);
  auto cdf_phi = reinterpret_cast<float*>(cdf_phi_ptr);
  auto alpha = reinterpret_cast<float*>(alpha_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_sample_importance_directions, size,
                                 light_dirs, 
                                 normal, cdf_the, cdf_phi, alpha, 
                                 batch_size, n_lights, 
                                 n_thes, n_phis, eps);
}


} // sampling


PYBIND11_MODULE(inverse_transform_cuda, m) {
  m.doc() = "Inverse Transform";
  m.def("sample_uniform_directions", &sampling::sample_uniform_directions, "");
  m.def("sample_importance_directions", &sampling::sample_importance_directions, "");
}