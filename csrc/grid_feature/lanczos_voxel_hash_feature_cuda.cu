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
#include <grid_feature/common_voxel_hash.cuh>

namespace py = pybind11;

using namespace grid_feature;
using namespace grid_feature::voxel_hash;

/***
    Forward
 ***/


namespace voxel_hash_feature {

// borrowed from
// https://github.com/NVlabs/tiny-cuda-nn/blob/4e0905311f1245c60b9fcc3a1c271ebb0ba50218/include/tiny-cuda-nn/encodings/grid.h#L77
__device__ inline
uint hash(uint x, uint y, uint z, int T) {
  constexpr uint32_t primes[3] = {1u, 2654435761u, 805459861u};  
	uint32_t result = 0;
	
  result ^= x * primes[0];
  result ^= y * primes[1];
  result ^= z * primes[2];

  auto index = result % T;
	return index;
}

/***
 hash index
 ***/

__global__
void kernel_hash_index(int N, float *output, const float *query, 
                       int T, 
                       bool boundary_check) {

  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = B
    auto b = n;
    auto querys = *(float3*)(query + b * 3);
    uint x0 = querys.x, y0 = querys.y, z0 = querys.z;
    auto idx = hash(x0, y0, z0, T);
    *(output + b) = idx;
  }
}


void hash_index(int N, int64_t output_ptr, int64_t query_ptr, 
                int T, 
                bool boundary_check) {
  auto output_buff = reinterpret_cast<float*>(output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_hash_index, N, 
                                 output_buff, query_buff, 
                                 T, 
                                 boundary_check);
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

/***
 forward
 ***/

template<int window_size = 2>
__global__
void kernel_voxel_hash_feature(int N, float *output, const float *query, const float *feature,
                               int G0, float growth_factor, int T0, int L, int D, 
                               float3 min, float3 max, 
                               bool boundary_check) {

  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = L * (B * R * N)
    // query
    auto B = N / L;
    auto lb = flat_to_2d(n, B);
    auto l = lb.x, b = lb.y;
    auto querys = *(float3*)(query + b * 3);

    // variable feature hash table (T_l, D)
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto grid_sizes1 = make_float3(G, G, G) - 1.f;
    auto gx1 = grid_sizes1.x, gy1 = grid_sizes1.y, gz1 = grid_sizes1.z;

    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;
    auto x = xyz.x, y = xyz.y, z = xyz.z;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;

    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z, const uint d) {
      auto hindex = hash(x, y, z, T);
      return hindex * D + d;
    };

    auto num_params = compute_num_params(G0, growth_factor, T0, l, D);
    auto feature_l = feature + num_params;

    for (int d = 0; d < D; ++d) {
      auto f = 0.f;
      auto w = window_size;

      for (auto i = -w + 1; i <= w; ++i) {
        auto xi = clamp(x0 + i, 0.f, gx1);
        auto ci = lanczos(x - xi, w);

        for (auto j = -w + 1; j <= w; ++j) {
          auto yj = clamp(y0 + j, 0.f, gy1);
          auto cj = lanczos(y - yj, w);

          for (auto k = -w + 1; k <= w; ++k) {
            auto zk = clamp(z0 + k, 0.f, gz1);
            auto ck = lanczos(z - zk, w);

            auto f_ijk = feature_l[feature_index(xi, yj, zk, d)];
            auto c_ijk = ci * cj * ck;
            f += c_ijk * f_ijk;
          }
        }
      }
      
      auto o_idx = d * (L * B) + l * (B) + b;
      output[o_idx] = f;
    }
  }
}


void voxel_hash_feature(int N, int64_t output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                        int G0, float growth_factor, int T0, int L, int D, 
                        std::vector<float> min, std::vector<float> max, 
                        bool boundary_check) {
  auto output_buff = reinterpret_cast<float*>(output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_voxel_hash_feature, N, 
                                 output_buff, query_buff, feature_buff, 
                                 G0, growth_factor, T0, L, D,
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


/***
    Backward (1st-order)
    1. wrt query
    2. wrt feature
 ***/

template<int window_size = 2>
__global__
void kernel_grad_query(int N, float *grad_query, 
                       const float *grad_output, 
                       const float *query, const float *feature,
                       int G0, float growth_factor, int T0, int L, int D, 
                       float3 min, float3 max, 
                       bool boundary_check) {
  
  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = L * (B * R * N)
    // query
    auto B = N / L;
    auto lb = flat_to_2d(n, B);
    auto l = lb.x, b = lb.y;
    auto querys = *(float3*)(query + b * 3);

    // variable feature hash table (T_l, D)
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto grid_sizes1 = make_float3(G, G, G) - 1.f;
    auto gx1 = grid_sizes1.x, gy1 = grid_sizes1.y, gz1 = grid_sizes1.z;

    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto sx = scales.x, sy = scales.y, sz = scales.z;
    auto xyz = (querys - min) * scales;
    auto x = xyz.x, y = xyz.y, z = xyz.z;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;

    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z, const uint d) {
      auto hindex = hash(x, y, z, T);
      return hindex * D + d;
    };
    
    auto num_params = compute_num_params(G0, growth_factor, T0, l, D);
    auto feature_l = feature + num_params;

    for (int d = 0; d < D; ++d) {
      auto gx = 0.f, gy = 0.f, gz = 0.f;
      auto w = window_size;

      for (auto i = -w + 1; i <= w; ++i) {
        auto xi = clamp(x0 + i, 0.f, gx1);
        auto ci = lanczos(x - xi, w);
        auto gci = grad_coefficient(x - xi, w);

        for (auto j = -w + 1; j <= w; ++j) {
          auto yj = clamp(y0 + j, 0.f, gy1);
          auto cj = lanczos(y - yj, w);
          auto gcj = grad_coefficient(y - yj, w);

          for (auto k = -w + 1; k <= w; ++k) {
            auto zk = clamp(z0 + k, 0.f, gz1);
            auto ck = lanczos(z - zk, w);
            auto gck = grad_coefficient(z - zk, w);

            auto f_ijk = feature_l[feature_index(xi, yj, zk, d)];

            gx += sx * gci * cj * ck * f_ijk;
            gy += sy * ci * gcj * ck * f_ijk;
            gz += sz * ci * cj * gck * f_ijk;
          }
        }
      }
      
      auto o_idx = d * (L * B) + l * (B) + b;
      auto ograd = grad_output[o_idx];
      gx = ograd * gx;
      gy = ograd * gy;
      gz = ograd * gz;
      atomicAdd(grad_query + b * 3, gx);
      atomicAdd(grad_query + b * 3 + 1, gy);
      atomicAdd(grad_query + b * 3 + 2, gz);
    }
  }
}

void grad_query(int N, int64_t grad_query_ptr, 
                int64_t grad_output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                int G0, float growth_factor, int T0, int L, int D, 
                std::vector<float> min, std::vector<float> max, 
                bool boundary_check, bool accum) {
  auto grad_query_buff = reinterpret_cast<float*>(grad_query_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  if (!accum) {
    auto size = N / L * 3;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_zero, size, grad_query_buff);
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_grad_query, N, 
                                 grad_query_buff, 
                                 grad_output_buff, query_buff, feature_buff, 
                                 G0, growth_factor, T0, L, D,
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


template<int window_size = 2>
__global__
void kernel_grad_feature(int N, float *grad_feature, 
                         const float *grad_output, 
                         const float *query, 
                         int G0, float growth_factor, int T0, int L, int D, 
                         float3 min, float3 max, 
                         bool boundary_check) {

  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = L * (B * R * N)
    // query
    auto B = N / L;
    auto lb = flat_to_2d(n, B);
    auto l = lb.x, b = lb.y;
    auto querys = *(float3*)(query + b * 3);

    // variable feature hash table (T_l, D)
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto grid_sizes1 = make_float3(G, G, G) - 1.f;
    auto gx1 = grid_sizes1.x, gy1 = grid_sizes1.y, gz1 = grid_sizes1.z;

    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto xyz = (querys - min) * scales;
    auto x = xyz.x, y = xyz.y, z = xyz.z;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;

    // gradients
    auto feature_index = [&](const uint x, const uint y, const uint z, const uint d) {
      auto hindex = hash(x, y, z, T);
      return hindex * D + d;
    };

    auto num_params = compute_num_params(G0, growth_factor, T0, l, D);
    auto grad_feature_l = grad_feature + num_params;

    auto compute_grad = [&](const float ograd, const uint x, const uint y, const uint z, const uint d, 
                            const float p, const float q, const float r) {
      auto f_idx = feature_index(x, y, z, d);
      atomicAdd(grad_feature_l + f_idx, ograd * p * q * r);
    };

    for (int d = 0; d < D; ++d) {
      auto w = window_size;
      
      auto o_idx = d * (L * B) + l * (B) + b;
      auto ograd = grad_output[o_idx];

      for (auto i = -w + 1; i <= w; ++i) {
        auto xi = clamp(x0 + i, 0.f, gx1);
        auto ci = lanczos(x - xi, w);

        for (auto j = -w + 1; j <= w; ++j) {
          auto yj = clamp(y0 + j, 0.f, gy1);
          auto cj = lanczos(y - yj, w);

          for (auto k = -w + 1; k <= w; ++k) {
            auto zk = clamp(z0 + k, 0.f, gz1);
            auto ck = lanczos(z - zk, w);
            compute_grad(ograd, xi, yj, zk, d, ci, cj, ck);
          }
        }
      }
    }
  }
}


void grad_feature(int N, int64_t grad_feature_ptr, 
                  int64_t grad_output_ptr, int64_t query_ptr, 
                  int G0, float growth_factor, int T0, int L, int D, 
                  std::vector<float> min, std::vector<float> max, 
                  bool boundary_check, bool accum) {
  auto grad_feature_buff = reinterpret_cast<float*>(grad_feature_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  
  if (!accum) {
    auto size = compute_num_params(G0, growth_factor, T0, L, D);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_zero, size, grad_feature_buff);
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_grad_feature, N, 
                                 grad_feature_buff, 
                                 grad_output_buff, query_buff, 
                                 G0, growth_factor, T0, L, D,
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


/***
    Backward (2nd-order)
    1-1. grad_query wrt grad_output
    // 1-2. grad_query wrt query
    1-3. grad_query wrt feature
    // 2-1. grad_feature wrt grad_output
    // 2-2. grad_feature wrt query

    naming rule, kernel_<backward_function_name>_grad_<wrt_input>
    
 ***/


// 1-1. grad_query wrt grad_output
template<bool accum = false, int window_size = 2>
__global__
void kernel_grad_query_grad_grad_output(int N, float *grad_grad_output, 
                                        const float *grad_grad_query, 
                                        const float *query, const float *feature,
                                        int G0, float growth_factor, int T0, int L, int D, 
                                        float3 min, float3 max, 
                                        bool boundary_check) {

  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = L * (B * R * N)
    // query
    auto B = N / L;
    auto lb = flat_to_2d(n, B);
    auto l = lb.x, b = lb.y;
    auto querys = *(float3*)(query + b * 3);

    // variable feature hash table (T_l, D)
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto grid_sizes1 = make_float3(G, G, G) - 1.f;
    auto gx1 = grid_sizes1.x, gy1 = grid_sizes1.y, gz1 = grid_sizes1.z;

    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto sx = scales.x, sy = scales.y, sz = scales.z;
    auto xyz = (querys - min) * scales;
    auto x = xyz.x, y = xyz.y, z = xyz.z;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;

    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z, const uint d) {
      auto hindex = hash(x, y, z, T);
      return hindex * D + d;
    };

    auto num_params = compute_num_params(G0, growth_factor, T0, l, D);
    auto feature_l = feature + num_params;

    auto ggq = *(float3*)(grad_grad_query + b * 3);

    for (int d = 0; d < D; ++d) {

      auto gx = 0.f, gy = 0.f, gz = 0.f;
      auto w = window_size;

      for (auto i = -w + 1; i <= w; ++i) {
        auto xi = clamp(x0 + i, 0.f, gx1);
        auto ci = lanczos(x - xi, w);
        auto gci = grad_coefficient(x - xi, w);

        for (auto j = -w + 1; j <= w; ++j) {
          auto yj = clamp(y0 + j, 0.f, gy1);
          auto cj = lanczos(y - yj, w);
          auto gcj = grad_coefficient(y - yj, w);

          for (auto k = -w + 1; k <= w; ++k) {
            auto zk = clamp(z0 + k, 0.f, gz1);
            auto ck = lanczos(z - zk, w);
            auto gck = grad_coefficient(z - zk, w);

            auto f_ijk = feature_l[feature_index(xi, yj, zk, d)];

            gx += sx * gci * cj * ck * f_ijk;
            gy += sy * ci * gcj * ck * f_ijk;
            gz += sz * ci * cj * gck * f_ijk;
          }
        }
      }
      auto o_idx = d * (L * B) + l * (B) + b;
      auto ggo = ggq.x * gx + ggq.y * gy + ggq.z * gz;
      grad_grad_output[o_idx] = accum ? grad_grad_output[o_idx] + ggo : ggo;
    }
  }
}


void grad_query_grad_grad_output(int N, int64_t grad_grad_output_ptr, 
                                 int64_t grad_grad_query_ptr, 
                                 int64_t query_ptr, int64_t feature_ptr, 
                                 int G0, float growth_factor, int T0, int L, int D, 
                                 std::vector<float> min, std::vector<float> max, 
                                 bool boundary_check, bool accum) {
  auto grad_grad_output_buff = reinterpret_cast<float*>(grad_grad_output_ptr);
  auto grad_grad_query_buff = reinterpret_cast<float*>(grad_grad_query_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  auto kernel = accum 
    ? kernel_grad_query_grad_grad_output<true> 
    : kernel_grad_query_grad_grad_output<false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
                                 grad_grad_output_buff, 
                                 grad_grad_query_buff, 
                                 query_buff, feature_buff, 
                                 G0, growth_factor, T0, L, D,
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}

// // 1-2. grad_query wrt query
// __global__
// void kernel_grad_query_grad_query(int N, float *grad_query, 
//                                   const float *grad_grad_query, 
//                                   const float *grad_output, 
//                                   const float *query, const float *feature,
//                                   int3 grid_sizes, int D, 
//                                   float3 min, float3 max, 
//                                   bool boundary_check) {
//   auto Gy0 = grid_sizes.y;
//   auto Gz0 = grid_sizes.z;
//   auto stride_x = Gy0 * Gz0 * D;
//   auto stride_y = Gz0 * D;
//   auto stride_z = D;
//   auto grid_sizes1 = to_float3(grid_sizes) - 1.f;
  
//   NBLA_CUDA_KERNEL_LOOP(n, N) {

//     auto b = n / D;
//     auto d = n - b * D;

//     auto querys = *(float3*)(query + b * 3);
  
//     // continuous point   
//     auto scales = grid_sizes1 / (max - min);
//     auto xyz = (querys - min) * scales;

//     // discrete points
//     auto xyz0 = floorf(xyz);    
//     xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
//     xyz0 = fminf(xyz0, grid_sizes1);
//     auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);

//     // coefficients
//     auto pqr0 = xyz1 - xyz;
//     auto pqr1 = 1.f - pqr0;

//     // scalars
//     uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
//     uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
//     auto p0 = pqr0.x, q0 = pqr0.y, r0 = pqr0.z;
//     auto p1 = pqr1.x, q1 = pqr1.y, r1 = pqr1.z;
//     auto sx = scales.x, sy = scales.y, sz = scales.z;

//     // grid features
//     auto feature_index = [&](const uint x, const uint y, const uint z) {
//       return x * stride_x + y * stride_y + z * stride_z + d;
//     };
    
//     auto f000 = feature[feature_index(x0, y0, z0)];
//     auto f001 = feature[feature_index(x0, y0, z1)];
//     auto f010 = feature[feature_index(x0, y1, z0)];
//     auto f011 = feature[feature_index(x0, y1, z1)];
//     auto f100 = feature[feature_index(x1, y0, z0)];
//     auto f101 = feature[feature_index(x1, y0, z1)];
//     auto f110 = feature[feature_index(x1, y1, z0)];
//     auto f111 = feature[feature_index(x1, y1, z1)];


//     // gradients
//     auto gg_xyz = *(float3*)(grad_grad_query + b * 3);
//     auto ggx = gg_xyz.x;
//     auto ggy = gg_xyz.y;
//     auto ggz = gg_xyz.z;
//     auto go = grad_output[n];
    
//     auto ti = go * sy * sz 
//       * (p0 * (f000 - f001 - f010 + f011) + p1 * (f100 - f101 - f110 + f111));
//     auto tj = go * sx * sz
//       * (q0 * (f000 - f001 - f100 + f101) + q1 * (f010 - f011 - f110 + f111));
//     auto tk = go * sx * sy
//       * (r0 * (f000 - f010 - f100 + f110) + r1 * (f001 - f011 - f101 + f111));

//     auto gx = ggy * tk + ggz * tj;
//     auto gy = ggz * ti + ggx * tk;
//     auto gz = ggx * tj + ggy * ti;

//     atomicAdd(grad_query + b * 3, gx);
//     atomicAdd(grad_query + b * 3 + 1, gy);
//     atomicAdd(grad_query + b * 3 + 2, gz);
//   }
// }


// void grad_query_grad_query(int N, int64_t grad_query_ptr, 
//                            int64_t grad_grad_query_ptr, 
//                            int64_t grad_output_ptr, 
//                            int64_t query_ptr, int64_t feature_ptr, 
//                            std::vector<int> grid_sizes, int D, 
//                            std::vector<float> min, std::vector<float> max, 
//                            bool boundary_check, bool accum) {
//   auto grad_query_buff = reinterpret_cast<float*>(grad_query_ptr);
//   auto grad_grad_query_buff = reinterpret_cast<float*>(grad_grad_query_ptr);
//   auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
//   auto query_buff = reinterpret_cast<float*>(query_ptr);
//   auto feature_buff = reinterpret_cast<float*>(feature_ptr);

//   NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_grad_query_grad_query, N, 
//                                  grad_query_buff,
//                                  grad_grad_query_buff,
//                                  grad_output_buff, 
//                                  query_buff, feature_buff, 
//                                  make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
//                                  make_float3(min[0], min[1], min[2]),
//                                  make_float3(max[0], max[1], max[2]),
//                                  boundary_check);
// }
  
// 1-3. grad_query wrt feature
template<int window_size = 2>
__global__
void kernel_grad_query_grad_feature(int N, float *grad_feature, 
                                    const float *grad_grad_query, 
                                    const float *grad_output, 
                                    const float *query, 
                                    int G0, float growth_factor, int T0, int L, int D, 
                                    float3 min, float3 max, 
                                    bool boundary_check) {
  NBLA_CUDA_KERNEL_LOOP(n, N) { // N = L * (B * R * N)
    // query
    auto B = N / L;
    auto lb = flat_to_2d(n, B);
    auto l = lb.x, b = lb.y;
    auto querys = *(float3*)(query + b * 3);

    // variable feature hash table (T_l, D)
    auto G = compute_grid_size(G0, growth_factor, l);
    auto T = compute_table_size(G, T0);
    auto grid_sizes1 = make_float3(G, G, G) - 1.f;
    auto gx1 = grid_sizes1.x, gy1 = grid_sizes1.y, gz1 = grid_sizes1.z;

    // continuous point
    auto scales = grid_sizes1 / (max - min);
    auto sx = scales.x, sy = scales.y, sz = scales.z;
    auto xyz = (querys - min) * scales;
    auto x = xyz.x, y = xyz.y, z = xyz.z;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;

    // gradients
    auto gg_xyz = *(float3*)(grad_grad_query + b * 3);
    auto ggx = gg_xyz.x;
    auto ggy = gg_xyz.y;
    auto ggz = gg_xyz.z;

    auto num_params = compute_num_params(G0, growth_factor, T0, l, D);
    auto grad_feature_l = grad_feature + num_params;

    auto feature_index = [&](const uint x, const uint y, const uint z, const uint d) {
      auto hindex = hash(x, y, z, T);
      return hindex * D + d;
    };    

    auto compute_grad = [&](const float ograd, const uint x, const uint y, const uint z, const uint d, 
                            const float a, const float b, const float c) {
      auto f_idx = feature_index(x, y, z, d);
      atomicAdd(grad_feature_l + f_idx, ograd * (ggx * sx * a + ggy * sy * b + ggz * sz * c));
    };

    for (int d = 0; d < D; ++d) {
      auto o_idx = d * (L * B) + l * (B) + b;
      auto ograd = grad_output[o_idx];

      auto w = window_size;

      for (auto i = -w + 1; i <= w; ++i) {
        auto xi = clamp(x0 + i, 0.f, gx1);
        auto ci = lanczos(x - xi, w);
        auto gci = grad_coefficient(x - xi, w);

        for (auto j = -w + 1; j <= w; ++j) {
          auto yj = clamp(y0 + j, 0.f, gy1);
          auto cj = lanczos(y - yj, w);
          auto gcj = grad_coefficient(y - yj, w);

          for (auto k = -w + 1; k <= w; ++k) {
            auto zk = clamp(z0 + k, 0.f, gz1);
            auto ck = lanczos(z - zk, w);
            auto gck = grad_coefficient(z - zk, w);
            compute_grad(ograd, xi, yj, zk, d, 
                         gci * cj * ck, ci * gcj * ck, ci * cj * gck);
          }
        }
      }
    }
  }

}

void grad_query_grad_feature(int N, int64_t grad_feature_ptr, 
                             int64_t grad_grad_query_ptr, 
                             int64_t grad_output_ptr, 
                             int64_t query_ptr, 
                             int G0, float growth_factor, int T0, int L, int D, 
                             std::vector<float> min, std::vector<float> max, 
                             bool boundary_check, bool accum) {
  auto grad_feature_buff = reinterpret_cast<float*>(grad_feature_ptr);
  auto grad_grad_query_buff = reinterpret_cast<float*>(grad_grad_query_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_grad_query_grad_feature, N, 
                                 grad_feature_buff,
                                 grad_grad_query_buff,
                                 grad_output_buff, 
                                 query_buff, 
                                 G0, growth_factor, T0, L, D,
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}

// // 2-1. grad_feature wrt grad_output
// template<bool accum = false>
// __global__
// void kernel_grad_feature_grad_grad_output(int N, float *grad_grad_output, 
//                                           const float *grad_grad_feature, 
//                                           const float *query,
//                                           int3 grid_sizes, int D, 
//                                           float3 min, float3 max, 
//                                           bool boundary_check) {
//   auto Gy0 = grid_sizes.y;
//   auto Gz0 = grid_sizes.z;
//   auto stride_x = Gy0 * Gz0 * D;
//   auto stride_y = Gz0 * D;
//   auto stride_z = D;
//   auto grid_sizes1 = to_float3(grid_sizes) - 1.f;
  
//   NBLA_CUDA_KERNEL_LOOP(n, N) {
//     auto b = n / D;
//     auto d = n - b * D;

//     auto querys = *(float3*)(query + b * 3);
  
//     // continuous point   
//     auto scales = grid_sizes1 / (max - min);
//     auto xyz = (querys - min) * scales;

//     // discrete points
//     auto xyz0 = floorf(xyz);    
//     xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
//     xyz0 = fminf(xyz0, grid_sizes1);
//     auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);

//     // coefficients
//     auto pqr0 = xyz1 - xyz;
//     auto pqr1 = 1.f - pqr0;

//     // scalars
//     uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
//     uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
//     auto p0 = pqr0.x, q0 = pqr0.y, r0 = pqr0.z;
//     auto p1 = pqr1.x, q1 = pqr1.y, r1 = pqr1.z;
    
//     // grid features
//     auto feature_index = [&](const uint x, const uint y, const uint z) {
//       return (x * stride_x) + (y * stride_y) + (z * stride_z + d);
//     };
    
//     auto ggf000 = grad_grad_feature[feature_index(x0, y0, z0)];
//     auto ggf001 = grad_grad_feature[feature_index(x0, y0, z1)];
//     auto ggf010 = grad_grad_feature[feature_index(x0, y1, z0)];
//     auto ggf011 = grad_grad_feature[feature_index(x0, y1, z1)];
//     auto ggf100 = grad_grad_feature[feature_index(x1, y0, z0)];
//     auto ggf101 = grad_grad_feature[feature_index(x1, y0, z1)];
//     auto ggf110 = grad_grad_feature[feature_index(x1, y1, z0)];
//     auto ggf111 = grad_grad_feature[feature_index(x1, y1, z1)];
    
//     // linear interpolation
//     auto ggo = p0 * q0 * r0 * ggf000
//       + p0 * q0 * r1 * ggf001
//       + p0 * q1 * r0 * ggf010
//       + p0 * q1 * r1 * ggf011
//       + p1 * q0 * r0 * ggf100
//       + p1 * q0 * r1 * ggf101
//       + p1 * q1 * r0 * ggf110
//       + p1 * q1 * r1 * ggf111;

//     grad_grad_output[n] = accum ? grad_grad_output[n] + ggo : ggo;
//   }
// }

// void grad_feature_grad_grad_output(int N, int64_t grad_grad_output_ptr, 
//                                    int64_t grad_grad_feature_ptr, 
//                                    int64_t query_ptr, 
//                                    std::vector<int> grid_sizes, int D, 
//                                    std::vector<float> min, std::vector<float> max, 
//                                    bool boundary_check, bool accum) {
//   auto grad_grad_output_buff = reinterpret_cast<float*>(grad_grad_output_ptr);
//   auto grad_grad_feature_buff = reinterpret_cast<float*>(grad_grad_feature_ptr);
//   auto query_buff = reinterpret_cast<float*>(query_ptr);

//   auto kernel = accum 
//     ? kernel_grad_feature_grad_grad_output<true> 
//     : kernel_grad_feature_grad_grad_output<false>;
//   NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
//                                  grad_grad_output_buff,
//                                  grad_grad_feature_buff,
//                                  query_buff, 
//                                  make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
//                                  make_float3(min[0], min[1], min[2]),
//                                  make_float3(max[0], max[1], max[2]),
//                                  boundary_check);
// }

// // 2-2. grad_feature wrt query
// __global__
// void kernel_grad_feature_grad_query(int N, float *grad_query, 
//                                     const float *grad_grad_feature, 
//                                     const float *grad_output, 
//                                     const float *query,
//                                     int3 grid_sizes, int D, 
//                                     float3 min, float3 max, 
//                                     bool boundary_check) {
//   auto Gy0 = grid_sizes.y;
//   auto Gz0 = grid_sizes.z;
//   auto stride_x = Gy0 * Gz0 * D;
//   auto stride_y = Gz0 * D;
//   auto stride_z = D;
//   auto grid_sizes1 = to_float3(grid_sizes) - 1.f;
  
//   NBLA_CUDA_KERNEL_LOOP(n, N) {
//     auto b = n / D;
//     auto d = n - b * D;

//     auto querys = *(float3*)(query + b * 3);
  
//     // continuous point   
//     auto scales = grid_sizes1 / (max - min);
//     auto xyz = (querys - min) * scales;

//     // discrete points
//     auto xyz0 = floorf(xyz);    
//     xyz0 = fmaxf(xyz0, make_float3(0.f, 0.f, 0.f));
//     xyz0 = fminf(xyz0, grid_sizes1);
//     auto xyz1 = fminf(xyz0 + 1.f, grid_sizes1);

//     // coefficients
//     auto pqr0 = xyz1 - xyz;
//     auto pqr1 = 1.f - pqr0;

//     // scalars
//     uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
//     uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
//     auto p0 = pqr0.x, q0 = pqr0.y, r0 = pqr0.z;
//     auto p1 = pqr1.x, q1 = pqr1.y, r1 = pqr1.z;
//     auto sx = scales.x, sy = scales.y, sz = scales.z;

//     // grid features
//     auto feature_index = [&](const uint x, const uint y, const uint z) {
//       return x * stride_x + y * stride_y + z * stride_z + d;
//     };
    
//     auto ggf000 = grad_grad_feature[feature_index(x0, y0, z0)];
//     auto ggf001 = grad_grad_feature[feature_index(x0, y0, z1)];
//     auto ggf010 = grad_grad_feature[feature_index(x0, y1, z0)];
//     auto ggf011 = grad_grad_feature[feature_index(x0, y1, z1)];
//     auto ggf100 = grad_grad_feature[feature_index(x1, y0, z0)];
//     auto ggf101 = grad_grad_feature[feature_index(x1, y0, z1)];
//     auto ggf110 = grad_grad_feature[feature_index(x1, y1, z0)];
//     auto ggf111 = grad_grad_feature[feature_index(x1, y1, z1)];

//     // gradients
//     auto ograd = grad_output[n];
//     auto compute_grad = [&](float scale, float a0, float a1, float b0, float b1, 
//                             float d00, float d01, float d10, float d11) {
//       return ograd * scale * (a0 * b0 * d00 + a0 * b1 * d01 + a1 * b0 * d10 + a1 * b1 * d11);
//     };
//     auto gx = compute_grad(sx, q0, q1, r0, r1, 
//                            (ggf100 - ggf000), 
//                            (ggf101 - ggf001), 
//                            (ggf110 - ggf010), 
//                            (ggf111 - ggf011));
//     auto gy = compute_grad(sy, p0, p1, r0, r1, 
//                            (ggf010 - ggf000), 
//                            (ggf011 - ggf001), 
//                            (ggf110 - ggf100), 
//                            (ggf111 - ggf101));
//     auto gz = compute_grad(sz, p0, p1, q0, q1, 
//                            (ggf001 - ggf000), 
//                            (ggf011 - ggf010), 
//                            (ggf101 - ggf100), 
//                            (ggf111 - ggf110));
//     atomicAdd(grad_query + b * 3, gx);
//     atomicAdd(grad_query + b * 3 + 1, gy);
//     atomicAdd(grad_query + b * 3 + 2, gz);
//   }  
// }


// void grad_feature_grad_query(int N, int64_t grad_query_ptr, 
//                              int64_t grad_grad_feature_ptr, 
//                              int64_t grad_output_ptr, 
//                              int64_t query_ptr, 
//                              std::vector<int> grid_sizes, int D, 
//                              std::vector<float> min, std::vector<float> max, 
//                              bool boundary_check, bool accum) {
//   auto grad_query_buff = reinterpret_cast<float*>(grad_query_ptr);
//   auto grad_grad_feature_buff = reinterpret_cast<float*>(grad_grad_feature_ptr);
//   auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
//   auto query_buff = reinterpret_cast<float*>(query_ptr);

//   NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_grad_feature_grad_query, N, 
//                                  grad_query_buff,
//                                  grad_grad_feature_buff,
//                                  grad_output_buff, 
//                                  query_buff, 
//                                  make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
//                                  make_float3(min[0], min[1], min[2]),
//                                  make_float3(max[0], max[1], max[2]),
//                                  boundary_check);
// }

} // voxel_hash_feature

PYBIND11_MODULE(lanczos_voxel_hash_feature_cuda, m) {
  m.doc() = "Interpolation by query on voxel hash";

  m.def("hash_index", &voxel_hash_feature::hash_index, "");

  // forward
  m.def("voxel_hash_feature", &voxel_hash_feature::voxel_hash_feature, "");

  // 1st-order gradient
  m.def("grad_query", &voxel_hash_feature::grad_query, "");
  m.def("grad_feature", &voxel_hash_feature::grad_feature, "");

  // 2nd-order gradient of 1st-order gradient wrt query
  m.def("grad_query_grad_grad_output", 
        &voxel_hash_feature::grad_query_grad_grad_output, "");
  // m.def("grad_query_grad_query", 
  //       &voxel_hash_feature::grad_query_grad_query, "");
  m.def("grad_query_grad_feature", 
        &voxel_hash_feature::grad_query_grad_feature, "");

  // // 2nd-order gradient of 1st-order gradient wrt feature
  // m.def("grad_feature_grad_grad_output", 
  //       &voxel_hash_feature::grad_feature_grad_grad_output, "");
  // m.def("grad_feature_grad_query", 
  //       &voxel_hash_feature::grad_feature_grad_query, "");
}
