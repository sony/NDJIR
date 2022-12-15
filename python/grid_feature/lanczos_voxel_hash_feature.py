# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lanczos_voxel_hash_feature_cuda
import nnabla as nn
import nnabla.backward_functions
import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.function import PythonFunction
from nnabla.parametric_functions import parametric_function_api


def force_align(size, mod=8):
    reminder = size % mod
    return size + reminder

def compute_grid_size(G0, growth_factor, T0, level):
    G = int(G0 * growth_factor ** level)
    return G


def compute_table_size(G, T0):
    Gf = float(G)
    T = min(Gf * Gf * Gf, T0)
    return int(T)

def compute_num_params(G0, growth_factor, T0, D, levels):
    num_params = 0

    for l in range(levels):
        G = compute_grid_size(G0, growth_factor, T0, l)
        T = compute_table_size(G, T0)
        num_params_l = force_align(T * D)
        num_params += num_params_l

    return num_params


def compute_params_boundary(G0, growth_factor, T0, D, levels):
    num_params0 = 0
    num_params1 = 0

    for l in range(levels + 1):
        G = compute_grid_size(G0, growth_factor, T0, l)
        T = compute_table_size(G, T0)
        num_params_l = force_align(T * D)
        num_params0 = num_params1
        num_params1 += num_params_l
    
    return num_params0, num_params1


# Forward
class LanczosQueryOnVoxelHash(PythonFunction):
    """
    Given a query (B, 3) and grid feature (L, T_l, D), 
    the function returns interpolations of features (B, D).

    The feature parameter size is given by the equation:

    G_l = floor(G0 * growth_fator ** l)
    param_size_l = align(min(G_l ** 3, T) * D), 8)
    param_size = sum_l (param_size_l)

    """

    def __init__(self, ctx, 
                 G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                 min_=[-1, -1, -1], max_=[1, 1, 1],
                 boundary_check=False):
        super(LanczosQueryOnVoxelHash, self).__init__(ctx)

        self._G0 = G0
        self._growth_factor = growth_factor
        self._T0 = T0
        self._L = L
        self._D = D
        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check

    @property            
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(G0=self._G0,
                    growth_factor=self._growth_factor,
                    T0=self._T0,
                    L=self._L, 
                    D=self._D, 
                    min_=self._min,
                    max_=self._max,
                    boundary_check=self._boundary_check)
        return data

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[0]
        feature = inputs[1]

        assert len(query.shape) > 1, "Query shape must be greater than 1, e.g., (B1, ..., Bn, 3)."
        assert query.shape[-1] == 3, "Query shape[-1] must be 3."
        
        batch_sizes = query.shape[:-1]
        L = self._L
        D = self._D

        outputs[0].reset_shape(batch_sizes + (D * L, ), True)

    def forward_impl(self, inputs, outputs):
        query = inputs[0]
        feature = inputs[1]
        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        G0 = self._G0
        growth_factor = self._growth_factor
        T0 = self._T0
        L = self._L
        D = self._D
        N = L * B

        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)
        output_ptr = output.data.data_ptr(np.float32, self.ctx)

        lanczos_voxel_hash_feature_cuda.voxel_hash_feature(
            N,
            output_ptr, query_ptr, feature_ptr, 
            G0, growth_factor, T0, L, D, 
            self._min, self._max,
            self._boundary_check)

        # M = D * L * B
        # voxel_hash_feature_cuda.transpose_inplace(M, output_ptr, B, L, D)
        out = F.reshape(output.data, (D * L, B))
        F.transpose(out, (1, 0), outputs=[out])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        query = inputs[0]
        feature = inputs[1]
        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        G0 = self._G0
        growth_factor = self._growth_factor
        T0 = self._T0
        L = self._L
        D = self._D
        N = L * B

        grad_output = output.grad
        grad_output = F.reshape(grad_output, (B, D * L))
        F.transpose(grad_output, (1, 0), outputs=[grad_output])

        grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        grad_feature_ptr = feature.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = output.grad.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)

        # Do not call for optimization
        # if propagate_down[0]:
        #     lanczos_voxel_hash_feature_cuda.grad_query(
        #         N,
        #         grad_query_ptr, 
        #         grad_output_ptr, query_ptr, feature_ptr, 
        #         G0, growth_factor, T0, L, D, 
        #         self._min, self._max,
        #         self._boundary_check, accum[0])

        if propagate_down[1]:
            lanczos_voxel_hash_feature_cuda.grad_feature(
                N,
                grad_feature_ptr, 
                grad_output_ptr, query_ptr,
                G0, growth_factor, T0, L, D, 
                self._min, self._max,
                self._boundary_check, accum[1])

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0:
            return True
        if i == 1 and j == 0:
            return True
        return False

    
def query_on_voxel_hash(query, feature, 
                         G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                         min_=[-1., -1., -1.], max_=[1., 1., 1.], 
                         boundary_check=False, ctx=None):
    func = LanczosQueryOnVoxelHash(ctx, G0, growth_factor, T0, L, D, min_, max_, boundary_check)
    return func(query, feature)

F.lanczos_query_on_voxel_hash = query_on_voxel_hash

@parametric_function_api("voxel_hash_feature", [
    ('F', 'Voxel hash faeture', '[L, T_l, D]', True),
])
def _query_on_voxel_hash(x, 
                          G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                          min_=[-1, -1, -1], max_=[1, 1, 1],
                          f_init=None, 
                          fix_parameters=False, rng=None):
    """
    """
    eps = 1e-3
    f_init = f_init if f_init is not None else I.NormalInitializer(eps)

    n_params = compute_num_params(G0, growth_factor, T0, D, L)
    feature = nn.parameter.get_parameter_or_create("F",
                                                   (n_params, ), 
                                                   f_init,
                                                   True, not fix_parameters)
    h = F.lanczos_query_on_voxel_hash(x, feature, G0, growth_factor, T0, L, D, min_, max_)
    return h

PF.lanczos_query_on_voxel_hash = _query_on_voxel_hash


# Backward
class LanczosQueryOnVoxelHashGradQuery(PythonFunction):
    """
    Gradient wrt query of QueryOnVoxelHash

    Given inputs of grad_output (B, 3), query (B, 3), feature (L, T_l, D),
    the function outupts grad_query (B, 3).
    """

    def __init__(self, ctx, 
                 G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                 min_=[-1, -1, -1], max_=[1, 1, 1], boundary_check=False):
        super(LanczosQueryOnVoxelHashGradQuery, self).__init__(ctx)

        self._G0 = G0
        self._growth_factor = growth_factor
        self._T0 = T0
        self._L = L
        self._D = D
        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check
        
    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[1]
        outputs[0].reset_shape(query.shape, True)

    def forward_impl(self, inputs, outputs):
        grad_query = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]
        feature = inputs[2]

        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        G0 = self._G0
        growth_factor = self._growth_factor
        T0 = self._T0
        L = self._L
        D = self._D
        N = L * B

        grad_output_data = F.reshape(grad_output.data, (B, D * L))
        grad_output_data = F.transpose(grad_output_data, (1, 0))

        grad_query_ptr = grad_query.data.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output_data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)
        
        lanczos_voxel_hash_feature_cuda.grad_query(N, grad_query_ptr,
                                     grad_output_ptr, query_ptr, feature_ptr,
                                     G0, growth_factor, T0, L, D, 
                                     self._min, self._max,
                                     self._boundary_check, False)

        

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        grad_query = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]
        feature = inputs[2]

        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        G0 = self._G0
        growth_factor = self._growth_factor
        T0 = self._T0
        L = self._L
        D = self._D
        N = L * B

        grad_output_data = grad_output.data
        grad_output_data = F.reshape(grad_output_data, (B, D * L))
        grad_output_data = F.transpose(grad_output_data, (1, 0))

        grad_grad_query_ptr = grad_query.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output_data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)

        grad_grad_output_ptr = grad_output.grad.data_ptr(np.float32, self.ctx)
        grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        grad_feature_ptr = feature.grad.data_ptr(np.float32, self.ctx)

        if propagate_down[0]:
            lanczos_voxel_hash_feature_cuda.grad_query_grad_grad_output(N, grad_grad_output_ptr,
                                                          grad_grad_query_ptr,
                                                          query_ptr, feature_ptr,
                                                          G0, growth_factor, T0, L, D, 
                                                          self._min, self._max,
                                                          self._boundary_check, accum[0])
            grad_grad_output = grad_output.grad
            grad_grad_output = F.reshape(grad_grad_output, (D * L, B))
            F.transpose(grad_grad_output, (1, 0), outputs=[grad_grad_output])

        # if propagate_down[1]:
        #     voxel_hash_feature_cuda.grad_query_grad_query(N, grad_query_ptr,
        #                                             grad_grad_query_ptr,
        #                                             grad_output_ptr,
        #                                             query_ptr, feature_ptr,
        #                                             G0, growth_factor, T0, L, D, 
        #                                             self._min, self._max,
        #                                             self._boundary_check, accum[1])
        if propagate_down[2]:
            lanczos_voxel_hash_feature_cuda.grad_query_grad_feature(N, grad_feature_ptr, 
                                                      grad_grad_query_ptr,
                                                      grad_output_ptr,
                                                      query_ptr, 
                                                      G0, growth_factor, T0, L, D, 
                                                      self._min, self._max,
                                                      self._boundary_check, accum[2])
            

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0 and (j == 1 or j == 2):
            return True
        if i == 1:
            return True
        if i == 2 and (j == 0 or j == 1):
            return True
        return False


def grad_query(grad_output, query, feature, 
               G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
               min_=[-1, -1, -1], max_=[1, 1, 1],
               boundary_check=False, ctx=None):
    func = LanczosQueryOnVoxelHashGradQuery(ctx, G0, growth_factor, T0, L, D, 
                                     min_, max_, boundary_check)
    return func(grad_output, query, feature)


# class LanczosQueryOnVoxelHashGradFeature(PythonFunction):
#     """
#     Gradient wrt query of QueryOnVoxelHash

#     Given inputs of grad_output (B, 3), query (B, 3), feature (L, T_l, D),
#     the function outupts grad_query (B, L*D).
#     """

#     def __init__(self, ctx, min_, max_, boundary_check=False, grid_sizes=None):
#         super(LanczosQueryOnVoxelHashGradFeature, self).__init__(ctx)

#         self._min = min_
#         self._max = max_
#         self._boundary_check = boundary_check
#         self._grid_sizes = grid_sizes
        
#     @property
#     def name(self):
#         return self.__class__.__name__

#     def min_outputs(self):
#         return 1

#     def setup_impl(self, inputs, outputs):
#         grad_output = inputs[0]

#         D = grad_output.shape[-1]
#         outputs[0].reset_shape(self._grid_sizes + (D, ), True)

#     def forward_impl(self, inputs, outputs):
#         grad_feature = outputs[0]
#         grad_output = inputs[0]
#         query = inputs[1]

#         batch_sizes = query.shape[:-1]
#         B = np.prod(batch_sizes)
#         D = grad_output.shape[-1]

#         grad_feature_ptr = grad_feature.data.data_ptr(np.float32, self.ctx)
#         grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
#         query_ptr = query.data.data_ptr(np.float32, self.ctx)

#         voxel_hash_feature_cuda.grad_feature(B * D, grad_feature_ptr,
#                                        grad_output_ptr, query_ptr, 
#                                        self._grid_sizes, D,
#                                        self._min, self._max,
#                                        self._boundary_check, False)
        
#     def backward_impl(self, inputs, outputs, propagate_down, accum):
#         grad_feature = outputs[0]
#         grad_output = inputs[0]
#         query = inputs[1]

#         batch_sizes = query.shape[:-1]
#         B = np.prod(batch_sizes)
#         G = max(self._grid_sizes) - 1
#         D = grad_output.shape[-1]

#         grad_grad_feature_ptr = grad_feature.grad.data_ptr(np.float32, self.ctx)
#         grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
#         query_ptr = query.data.data_ptr(np.float32, self.ctx)

#         grad_grad_output_ptr = grad_output.grad.data_ptr(np.float32, self.ctx)
#         grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        
#         if propagate_down[0]:
#             voxel_hash_feature_cuda.grad_feature_grad_grad_output(B * D, grad_grad_output_ptr,
#                                                             grad_grad_feature_ptr,
#                                                             query_ptr,
#                                                             self._grid_sizes, D,
#                                                             self._min, self._max,
#                                                             self._boundary_check, accum[0])
#         if propagate_down[1]:
#             voxel_hash_feature_cuda.grad_feature_grad_query(B * D, grad_query_ptr,
#                                                       grad_grad_feature_ptr,
#                                                       grad_output_ptr,
#                                                       query_ptr, 
#                                                       self._grid_sizes, D,
#                                                       self._min, self._max,
#                                                       self._boundary_check, accum[1])

#     def grad_depends_output_data(self, i, o):
#         return False

#     def grad_depends_input_data(self, i, j):
#         if i == 0 and j == 1:
#             return True
#         if i == 1:
#             return True
#         return False
            

# def grad_feature(grad_output, query, min_=[-1, -1, -1], max_=[1, 1, 1],
#                  boundary_check=False, grid_sizes=None, ctx=None):
#     func = LanczosQueryOnVoxelHashGradFeature(ctx, min_, max_, boundary_check, grid_sizes)
#     return func(grad_output, query)


def voxel_hash_feature_backward(inputs, 
                                G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                                min_=[-1, -1, -1], max_=[1, 1, 1],
                                boundary_check=False):
    grad_output = inputs[0]
    query = inputs[1]
    feature = inputs[2]

    # Directional gradients
    gq = grad_query(grad_output, query, feature, 
                    G0, growth_factor, T0, L, D, 
                    min_, max_, boundary_check)
    # gf = grad_feature(grad_output, query, min_, max_, boundary_check, grid_sizes)

    return gq, None

    # # STE
    # return None, None

nnabla.backward_functions.register("LanczosQueryOnVoxelHash", voxel_hash_feature_backward)
