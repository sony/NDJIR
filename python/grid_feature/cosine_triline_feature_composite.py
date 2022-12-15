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


def query_on_triline(query, triline_feature, min_, max_):
    B = query.shape[0]
    G = triline_feature.shape[1]
    D = triline_feature.shape[-1]

    # continuous point, discrete points, coefficients
    scale = (G - 1) / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    point1 = point0 + 1
    coeff0 = 0.5 * F.cos(np.pi * (pointf - point0)) + 0.5
    coeff1 = 1 - coeff0
    x0 = point0[:, 0]
    x1 = point1[:, 0]
    y0 = point0[:, 1]
    y1 = point1[:, 1]
    z0 = point0[:, 2]
    z1 = point1[:, 2]

    # Corresponding voxel features and coefficients
    i0 = F.constant(0, (B, ))
    fx_0 = triline_feature[i0, x0]
    fx_1 = triline_feature[i0, x1]

    i1 = F.constant(1, (B, ))
    fy_0 = triline_feature[i1, y0]
    fy_1 = triline_feature[i1, y1]
    
    i2 = F.constant(2, (B, ))
    fz_0 = triline_feature[i2, z0]
    fz_1 = triline_feature[i2, z1]

    p0 = coeff0[:, 0].reshape((B, 1))
    p1 = coeff1[:, 0].reshape((B, 1))
    q0 = coeff0[:, 1].reshape((B, 1))
    q1 = coeff1[:, 1].reshape((B, 1))
    r0 = coeff0[:, 2].reshape((B, 1))
    r1 = coeff1[:, 2].reshape((B, 1))

    # Linear interpolation
    def interpolate(f0, f1, u0, u1):
        return u0 * f0 + u1 * f1
            
    fx = interpolate(fx_0, fx_1, p0, p1)
    fy = interpolate(fy_0, fy_1, q0, q1)
    fz = interpolate(fz_0, fz_1, r0, r1)
   
    fx = F.reshape(fx, (B, D, 1))
    fy = F.reshape(fy, (B, D, 1))
    fz = F.reshape(fz, (B, D, 1))
    f = F.concatenate(*[fx, fy, fz], axis=-1)
    f = F.reshape(f, (B, D * 3))

    return f
