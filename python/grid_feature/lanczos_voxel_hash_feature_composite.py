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
import nnabla.functions as F
import numpy as np
from nnabla.function import PythonFunction

from grid_feature.voxel_hash_feature import (compute_grid_size,
                                             compute_params_boundary,
                                             compute_table_size)


class HashIndex(PythonFunction):
    """
    Given a query (B, 3) and some parameters 
    the function returns hash index of voxel hash feature.
    """

    def __init__(self, ctx, T, 
                 boundary_check=False):
        super(HashIndex, self).__init__(ctx)

        self._T = T
        self._boundary_check = boundary_check
        
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(T=self._T, 
                    boundary_check=self._boundary_check)
        return data
                
    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[0]

        assert len(query.shape) > 1, "Query shape must be greater than 1, e.g., (B1, ..., Bn, 3)."
        assert query.shape[-1] == 3, "Query shape[-1] must be 3."
        
        batch_sizes = query.shape[:-1]
        
        outputs[0].reset_shape(batch_sizes + (1, ), True)

    def forward_impl(self, inputs, outputs):
        query = inputs[0]

        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        T = self._T

        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        output_ptr = output.data.data_ptr(np.float32, self.ctx)

        lanczos_voxel_hash_feature_cuda.hash_index(
            B,
            output_ptr, query_ptr, 
            T, 
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

    
def hash_index(query, T, boundary_check=False, ctx=None):
    func = HashIndex(ctx, T, boundary_check)
    return func(query)


def query_on_voxel_hash(query, voxel_hash_feature, 
                         G0, growth_factor, T0, L, D, 
                         min_=-1, max_=1, window_size=2):
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
        voxel_hash_feature_l = F.reshape(voxel_hash_feature_l, (-1, D))
        feat_l = query_on_voxel_hash_at_level(query, voxel_hash_feature_l, G, T, min_, max_, window_size)
        feats.append(feat_l)

    if len(feats) == 1:
        return feats[0]

    B = query.shape[0]
    feats = F.concatenate(*feats)
    feats = F.reshape(feats, (B, L, D))
    feats = F.transpose(feats, (0, 2, 1))
    feats = F.reshape(feats, (B, D * L))

    return feats


def lanczos(x, a):
    z = np.pi * x
    u = F.sinc(z)
    v = F.sinc(z / a)
    y = u * v
    return y

def clamp(x, a, b):
    y = F.greater_equal_scalar(x, a)
    z = F.less_equal_scalar(y, b)
    return z


def query_on_voxel_hash_at_level(query, voxel_hash_feature_l, G, T, min_, max_, window_size=2):
    
    B = query.shape[0]
    G1 = G - 1

    # continuous point, discrete points, coefficients
    scale = G1 / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    x0, y0, z0 = point0[:, 0], point0[:, 1], point0[:, 2]
    x, y, z = pointf[:, 0], pointf[:, 1], pointf[:, 2]
    x0, y0, z0 = x0.reshape((B, 1)), y0.reshape((B, 1)), z0.reshape((B, 1))
    x, y, z = x.reshape((B, 1)), y.reshape((B, 1)), z.reshape((B, 1))
    
    f = 0
    w = window_size
    for i in range(-w + 1, w + 1):
        xi = F.clip_by_value(x0 + i, 0, G1).reshape((B, 1))
        ci = lanczos(x - xi, w)
        
        for j in range(-w + 1, w + 1):
            yj = F.clip_by_value(y0 + j, 0, G1).reshape((B, 1))
            cj = lanczos(y - yj, w)

            for k in range(-w + 1, w + 1):
                zk = F.clip_by_value(z0 + k, 0, G1).reshape((B, 1))
                ck = lanczos(z - zk, w)
                
                xyz = F.concatenate(*[xi, yj, zk])
                idx = hash_index(xyz, T)
                idx = F.reshape(idx, (1, B)).apply(need_grad=False)

                f_ijk = F.gather_nd(voxel_hash_feature_l, idx)
                c_ijk = ci * cj * ck
                f = f + f_ijk * c_ijk

    return f
