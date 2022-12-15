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

import nnabla.functions as F
import numpy as np
import total_variation_loss_on_voxel_hash_cuda
from nnabla.function import PythonFunction


# Forward
class TVLossOnVoxelHash(PythonFunction):
    """
    Given a query (B, 3) and grid feature (G, G, G, D), 
    the function returns interpolations of features (B, D).
    """

    def __init__(self, ctx, 
                 G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                 min_=[-1, -1, -1], max_=[1, 1, 1],
                 boundary_check=False):
        super(TVLossOnVoxelHash, self).__init__(ctx)

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

        total_variation_loss_on_voxel_hash_cuda.tv_loss_on_voxel_hash(
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

        if propagate_down[1]:
            total_variation_loss_on_voxel_hash_cuda.tv_loss_on_voxel_hash_backward(
                N,
                grad_feature_ptr, 
                grad_output_ptr, 
                query_ptr, feature_ptr, 
                G0, growth_factor, T0, L, D, 
                self._min, self._max,
                self._boundary_check)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 1:
            return True
        return False

    
def tv_loss_on_voxel_hash(query, feature, 
                          G0=16, growth_factor=1.5, T0=2**15, L=16, D=2, 
                          min_=[-1, -1, -1], max_=[1, 1, 1], boundary_check=False, ctx=None):
    func = TVLossOnVoxelHash(ctx, G0, growth_factor, T0, L, D, 
                             min_, max_, boundary_check)
    return func(query, feature)


F.tv_loss_on_voxel_hash = tv_loss_on_voxel_hash
