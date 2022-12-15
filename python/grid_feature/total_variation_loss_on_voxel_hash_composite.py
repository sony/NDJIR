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
import voxel_hash_feature_cuda
from nnabla.function import PythonFunction

from grid_feature.voxel_hash_feature import (compute_grid_size,
                                             compute_params_boundary,
                                             compute_table_size)


class HashIndex(PythonFunction):
    """
    Given a query (B, 3) and some parameters 
    the function returns hash index of voxel hash feature.
    """

    def __init__(self, ctx, G, T, min_=[-1, -1, -1], max_=[1, 1, 1],
                 boundary_check=False):
        super(HashIndex, self).__init__(ctx)

        self._G = G
        self._T = T
        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check
        
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(G=self._G,
                    T=self._T, 
                    min_=self._min,
                    max_=self._max,
                    boundary_check=self._boundary_check)
        return data
                
    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[0]

        assert len(query.shape) > 1, "Query shape must be greater than 1, e.g., (B1, ..., Bn, 3)."
        assert query.shape[-1] == 3, "Query shape[-1] must be 3."
        
        batch_sizes = query.shape[:-1]
        
        outputs[0].reset_shape(batch_sizes + (8, ), True)

    def forward_impl(self, inputs, outputs):
        query = inputs[0]

        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        G = self._G
        T = self._T

        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        output_ptr = output.data.data_ptr(np.float32, self.ctx)

        voxel_hash_feature_cuda.hash_index(
            B,
            output_ptr, query_ptr, 
            G, T, 
            self._min, self._max,
            self._boundary_check)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        pass        

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0:
            return True
        if i == 1 and j == 0:
            return True
        return False

    
def hash_index(query, G, T, 
               min_=[-1, -1, -1], max_=[1, 1, 1], boundary_check=False, ctx=None):
    func = HashIndex(ctx, G, T, min_, max_, boundary_check)
    return func(query)


def tv_loss_on_voxel_hash(query, voxel_hash_feature, 
                          G0, growth_factor, T0, L, D, 
                          min_=-1, max_=1):
    """
    query: (B, 3)
    voxel_hash_feature: (L * T_l * D, ) which is 1D feature since T_l is variable-length
    """
    feats = []
    for l in range(L):
        G = compute_grid_size(G0, growth_factor, T0, l)
        T = compute_table_size(G, T0)
        num_params0, num_params1 = compute_params_boundary(G0, growth_factor, T0, D, l)
        voxel_hash_feature_l = voxel_hash_feature[num_params0:num_params1]
        hash_index_l = hash_index(query, G, T, [min_] * 3, [max_] * 3)
        feat_l = tv_loss_on_voxel_hash_at_level(query, voxel_hash_feature_l, hash_index_l, G, D, min_, max_)
        feats.append(feat_l)

    if len(feats) == 1:
        return feats[0]

    B = query.shape[0]
    feats = F.concatenate(*feats)
    feats = F.reshape(feats, (B, L, D))
    feats = F.transpose(feats, (0, 2, 1))
    feats = F.reshape(feats, (B, D * L))

    return feats


def tv_loss_on_voxel_hash_at_level(query, voxel_hash_feature_l, hash_index_l, G, D, min_, max_):
    
    B = query.shape[0]

    # continuous point, discrete points, coefficients
    scale = (G - 1) / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    point1 = point0 + 1

    # Corresponding voxel features and coefficients
    idx000 = hash_index_l[:, 0]
    idx001 = hash_index_l[:, 1]
    idx010 = hash_index_l[:, 2]
    idx100 = hash_index_l[:, 4]

    idx000 = F.reshape(idx000, (1, B))
    idx001 = F.reshape(idx001, (1, B))
    idx010 = F.reshape(idx010, (1, B))
    idx100 = F.reshape(idx100, (1, B))

    voxel_hash_feature_l = F.reshape(voxel_hash_feature_l, (-1, D))
    
    # nnabla bug: two sequential advanced indexing causes an expected error, 
    # so here call F.gather_nd directly
    f000 = F.gather_nd(voxel_hash_feature_l, idx000).apply(need_grad=False)
    f001 = F.gather_nd(voxel_hash_feature_l, idx001)
    f010 = F.gather_nd(voxel_hash_feature_l, idx010)
    f100 = F.gather_nd(voxel_hash_feature_l, idx100)

    delta_x = f100 - f000
    delta_y = f010 - f000
    delta_z = f001 - f000
    delta_x2 = delta_x * delta_x
    delta_y2 = delta_y * delta_y
    delta_z2 = delta_z * delta_z

    f = (delta_x2 + delta_y2 + delta_z2) ** 0.5
    
    return f

#F.tv_loss_on_voxel_hash = tv_loss_on_voxel_hash
