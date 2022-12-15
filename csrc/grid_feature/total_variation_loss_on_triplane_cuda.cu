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
#include <grid_feature/common.cuh>
#include <grid_feature/common_triplane.cuh>

namespace py = pybind11;

using namespace grid_feature;
using namespace grid_feature::triplane;

namespace total_variation_loss {

__global__
void kernel_tv_loss_on_triplane_forward(int N, float *output, const float *query, const float *feature,
                              int G, int D, 
                              float3 min, float3 max, 
                              bool boundary_check) {
                                 
  auto stride_u = G * D;
  auto stride_v = D;
  auto G1 = G - 1.f;
  auto grid_sizes1 = make_float3(G1, G1, G1);

  NBLA_CUDA_KERNEL_LOOP(n, N) {
    auto n_idx = flat_to_3d(n, D);
    auto b = n_idx.x, d = n_idx.y, i = n_idx.z;

    auto querys = *(float3*)(query + b * 3);
  
    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
    xyz0 = fminf(xyz0, grid_sizes1);
    auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);
    
    // grid features
    auto feature_index = [&](const uint u, const uint v) {
      return (u * stride_u) + (v * stride_v) + d;
    };

    auto locs = select_locations(i, xyz0, xyz1);
    auto u0 = locs.x, u1 = locs.y, v0 = locs.z, v1 = locs.w;

    auto feature_i = feature + i * (G * G * D);
    auto f00 = feature_i[feature_index(u0, v0)];
    auto f01 = feature_i[feature_index(u0, v1)];
    auto f10 = feature_i[feature_index(u1, v0)];
    
    // TV loss
    auto delta_u = (f10 - f00), delta_v = (f01 - f00);
    auto delta_u2 = delta_u * delta_u;
    auto delta_v2 = delta_v * delta_v;
        
    auto f = sqrt(delta_u2 + delta_v2);

    output[n] = f;
  }
}


void tv_loss_on_triplane_forward(int N, int64_t output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                              int G, int D, 
                              std::vector<float> min, std::vector<float> max, 
                              bool boundary_check) {
  auto output_buff = reinterpret_cast<float*>(output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_tv_loss_on_triplane_forward, N, 
                                 output_buff, query_buff, feature_buff, 
                                 G, D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


template<bool sym_backward = false>
__global__
void kernel_tv_loss_on_triplane_backward(int N, float *grad_feature, 
                               const float *grad_output, 
                               const float *query, const float *feature,
                               int G, int D, 
                               float3 min, float3 max, 
                               bool boundary_check) {
  auto stride_u = G * D;
  auto stride_v = D;
  auto G1 = G - 1.f;
  auto grid_sizes1 = make_float3(G1, G1, G1);
  
  NBLA_CUDA_KERNEL_LOOP(n, N) {
    auto n_idx = flat_to_3d(n, D);
    auto b = n_idx.x, d = n_idx.y, i = n_idx.z;

    auto querys = *(float3*)(query + b * 3);
  
    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
    xyz0 = fminf(xyz0, grid_sizes1);
    auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);
    
    // grid features
    auto feature_index = [&](const uint u, const uint v) {
      return (u * stride_u) + (v * stride_v) + d;
    };

    auto locs = select_locations(i, xyz0, xyz1);
    auto u0 = locs.x, u1 = locs.y, v0 = locs.z, v1 = locs.w;

    auto feature_i = feature + i * (G * G * D);
    auto f00 = feature_i[feature_index(u0, v0)];
    auto f01 = feature_i[feature_index(u0, v1)];
    auto f10 = feature_i[feature_index(u1, v0)];
    
    // TV loss
    auto delta_u = (f10 - f00), delta_v = (f01 - f00);
    auto delta_u2 = delta_u * delta_u;
    auto delta_v2 = delta_v * delta_v;
        
    auto ograd = grad_output[n];
    auto gf_common = ograd * rsqrt(delta_u2 + delta_v2 + 1e-12);
    auto gf10 = gf_common * (delta_u);
    auto gf01 = gf_common * (delta_v);

    auto grad_feature_i = grad_feature + i * (G * G * D);
    atomicAdd(grad_feature_i + feature_index(u1, v0), gf10);
    atomicAdd(grad_feature_i + feature_index(u0, v1), gf01);

    if (sym_backward) {
      auto grad = - (gf10 + gf01);
      atomicAdd(grad_feature_i + feature_index(u0, v0), grad);
    }    
  }
}

void tv_loss_on_triplane_backward(int N, int64_t grad_feature_ptr, 
                               int64_t grad_output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                               int G, int D, 
                               std::vector<float> min, std::vector<float> max, 
                               bool sym_backward, bool boundary_check, bool accum) {
  auto grad_feature_buff = reinterpret_cast<float*>(grad_feature_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  auto kernel = sym_backward \
    ? kernel_tv_loss_on_triplane_backward<true> \
    : kernel_tv_loss_on_triplane_backward<false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
                                 grad_feature_buff, 
                                 grad_output_buff, query_buff, feature_buff, 
                                 G, D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}

} // total_variation_loss

PYBIND11_MODULE(total_variation_loss_on_triplane_cuda, m) {
  m.doc() = "Total variation losses";
  
  // forward
  m.def("tv_loss_on_triplane", &total_variation_loss::tv_loss_on_triplane_forward, "");
  
  // backward
  m.def("tv_loss_on_triplane_backward", &total_variation_loss::tv_loss_on_triplane_backward, "");
}
