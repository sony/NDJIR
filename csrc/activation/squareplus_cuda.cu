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


namespace squareplus {


__global__
void kernel_forward(int size,
                    float *output,
                    const float *input,
                    const float b) {
  auto inv_b = 1.f / b;
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    auto x = input[s];
    output[s] = 0.5f * (x + sqrtf(x * x + b));
  }
}


template<bool accum = false>
__global__
void kernel_backward(int size,
                     float *dinput,
                     const float *doutput,
                     const float *input,
                     const float b) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    auto x = input[s];
    auto dy = doutput[s];
    auto dx = dy * 0.5f * (1.f + x * rsqrtf(x * x + b));
    if (accum)
      dinput[s] += dx;
    else
      dinput[s] = dx;
  }
}


void forward(int size,            
             int64_t output_ptr, 
             int64_t input_ptr, 
             const float b) {

  auto output = reinterpret_cast<float*>(output_ptr);
  auto input = reinterpret_cast<float*>(input_ptr);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_forward, size,
                                 output, input, b);
                                 
}


void backward(int size,            
              int64_t dinput_ptr, 
              int64_t doutput_ptr, 
              int64_t input_ptr, 
              const float b, 
              const bool accum) {

  auto dinput = reinterpret_cast<float*>(dinput_ptr);
  auto doutput = reinterpret_cast<float*>(doutput_ptr);
  auto input = reinterpret_cast<float*>(input_ptr);

  auto kernel = accum ? kernel_backward<true> : kernel_backward<false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size,
                                 dinput, doutput, input, b);
}

}


PYBIND11_MODULE(squareplus_cuda, m) {
  m.doc() = "Squareplus";
  m.def("forward", &squareplus::forward, "");
  m.def("backward", &squareplus::backward, "");
}

