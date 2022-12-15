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


def lanczos(x, a):
    z = np.pi * x
    u = F.sinc(z)
    v = F.sinc(z / a)
    y = u * v
    return y

def query_on_triplane(query, triplane_feature, min_, max_, window_size=2):
    B = query.shape[0]
    G = triplane_feature.shape[1]
    G1 = G - 1
    D = triplane_feature.shape[-1]

    # continuous point, discrete points, coefficients
    scale = G1 / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    x0, y0, z0 = point0[:, 0], point0[:, 1], point0[:, 2]
    x, y, z = pointf[:, 0], pointf[:, 1], pointf[:, 2]
    
    w = window_size
    def interpolate(l, u, v, u0, v0):
        f = 0
        idx_l = F.constant(l, (B, ))
        
        for i in range(-w + 1, w + 1):
            ui = F.clip_by_value(u0 + i, 0, G1)
            ci = lanczos(u - ui, w).reshape((B, 1))
            
            for j in range(-w + 1, w + 1):
                vj = F.clip_by_value(v0 + j, 0, G1)
                cj = lanczos(v - vj, w).reshape((B, 1))

                f_ij = triplane_feature[idx_l, ui, vj]
                c_ij = ci * cj
                f = f + f_ij * c_ij

        return f
   
    fxy = interpolate(0, x, y, x0, y0)
    fyz = interpolate(1, y, z, y0, z0)
    fzx = interpolate(2, z, x, z0, x0)
    fxy = F.reshape(fxy, (B, D, 1))
    fyz = F.reshape(fyz, (B, D, 1))
    fzx = F.reshape(fzx, (B, D, 1))
    f = F.concatenate(*[fxy, fyz, fzx], axis=-1)
    f = F.reshape(f, (B, D * 3))

    return f

# F.query_on_tripilne = query_on_triplane
