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

namespace py = pybind11;

using namespace grid_feature;

/***
    Forward
 ***/

namespace total_variation_loss {

__global__
void kernel_tv_loss_on_voxel_forward(int N, float *output, const float *query, const float *feature,
                              int3 grid_sizes, int D, 
                              float3 min, float3 max, 
                              bool boundary_check) {
                                 
  auto Gy0 = grid_sizes.y;
  auto Gz0 = grid_sizes.z;
  auto stride_x = Gy0 * Gz0 * D;
  auto stride_y = Gz0 * D;
  auto stride_z = D;
  auto grid_sizes1 = to_float3(grid_sizes) - 1.f;
  
  NBLA_CUDA_KERNEL_LOOP(n, N) {
    auto b = n / D;
    auto d = n - b * D;

    auto querys = *(float3*)(query + b * 3);
  
    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
    xyz0 = fminf(xyz0, grid_sizes1);
    auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);
    
    // scalars
    uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
    uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;

    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z) {
      return (x * stride_x) + (y * stride_y) + (z * stride_z + d);
    };
    
    auto f000 = feature[feature_index(x0, y0, z0)];
    auto f001 = feature[feature_index(x0, y0, z1)];
    auto f010 = feature[feature_index(x0, y1, z0)];
    auto f100 = feature[feature_index(x1, y0, z0)];

    // TV loss
    auto delta_x = (f100 - f000), delta_y = (f010 - f000), delta_z = (f001 - f000);
    auto delta_x2 = delta_x * delta_x;
    auto delta_y2 = delta_y * delta_y;
    auto delta_z2 = delta_z * delta_z;
        
    auto f = sqrt(delta_x2 + delta_y2 + delta_z2);

    output[n] = f;
  }
}


void tv_loss_on_voxel_forward(int N, int64_t output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                              std::vector<int> grid_sizes, int D, 
                              std::vector<float> min, std::vector<float> max, 
                              bool boundary_check) {
  auto output_buff = reinterpret_cast<float*>(output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_tv_loss_on_voxel_forward, N, 
                                 output_buff, query_buff, feature_buff, 
                                 make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


/***
    Backward
 ***/

template<bool sym_backward = false>
__global__
void kernel_tv_loss_on_voxel_backward(int N, float *grad_feature, 
                               const float *grad_output, 
                               const float *query, const float *feature,
                               int3 grid_sizes, int D, 
                               float3 min, float3 max, 
                               bool boundary_check) {
  auto Gy0 = grid_sizes.y;
  auto Gz0 = grid_sizes.z;
  auto stride_x = Gy0 * Gz0 * D;
  auto stride_y = Gz0 * D;
  auto stride_z = D;
  auto grid_sizes1 = to_float3(grid_sizes) - 1.f;
  
  NBLA_CUDA_KERNEL_LOOP(n, N) {
    auto b = n / D;
    auto d = n - b * D;

    auto querys = *(float3*)(query + b * 3);
  
    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
    xyz0 = fminf(xyz0, grid_sizes1);
    auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);
    
    // scalars
    uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
    uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;

    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z) {
      return (x * stride_x) + (y * stride_y) + (z * stride_z + d);
    };
    
    auto f000 = feature[feature_index(x0, y0, z0)];
    auto f001 = feature[feature_index(x0, y0, z1)];
    auto f010 = feature[feature_index(x0, y1, z0)];
    auto f100 = feature[feature_index(x1, y0, z0)];

    // Grad wrt feature
    auto delta_x = (f100 - f000), delta_y = (f010 - f000), delta_z = (f001 - f000);
    auto delta_x2 = delta_x * delta_x;
    auto delta_y2 = delta_y * delta_y;
    auto delta_z2 = delta_z * delta_z;
    
    auto ograd = grad_output[n];
    auto gf_common = ograd * rsqrt(delta_x2 + delta_y2 + delta_z2 + 1e-12);
    auto gf100 = gf_common * (delta_x);
    auto gf010 = gf_common * (delta_y);
    auto gf001 = gf_common * (delta_z);

    atomicAdd(grad_feature + feature_index(x1, y0, z0), gf100);
    atomicAdd(grad_feature + feature_index(x0, y1, z0), gf010);
    atomicAdd(grad_feature + feature_index(x0, y0, z1), gf001);

    if (sym_backward) {
      auto grad = - (gf100 + gf010 + gf001);
      atomicAdd(grad_feature + feature_index(x0, y0, z0), grad);
    }
  }
}

void tv_loss_on_voxel_backward(int N, int64_t grad_feature_ptr, 
                               int64_t grad_output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                               std::vector<int> grid_sizes, int D, 
                               std::vector<float> min, std::vector<float> max, 
                               bool sym_backward, bool boundary_check, bool accum) {
  auto grad_feature_buff = reinterpret_cast<float*>(grad_feature_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  auto kernel = sym_backward \
    ? kernel_tv_loss_on_voxel_backward<true> \
    : kernel_tv_loss_on_voxel_backward<false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
                                 grad_feature_buff, 
                                 grad_output_buff, query_buff, feature_buff, 
                                 make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}

} // total_variation_loss

PYBIND11_MODULE(total_variation_loss_cuda, m) {
  m.doc() = "Total variation losses";
  
  // forward
  m.def("tv_loss_on_voxel", &total_variation_loss::tv_loss_on_voxel_forward, "");
  
  // backward
  m.def("tv_loss_on_voxel_backward", &total_variation_loss::tv_loss_on_voxel_backward, "");
}
