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


def query_on_triplane(query, triplane_feature, min_, max_):
    B = query.shape[0]
    G = triplane_feature.shape[1]
    D = triplane_feature.shape[-1]

    # continuous point, discrete points, coefficients
    scale = (G - 1) / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    point1 = point0 + 1
    coeff0 = point1 - pointf
    coeff1 = pointf - point0
    x0 = point0[:, 0]
    x1 = point1[:, 0]
    y0 = point0[:, 1]
    y1 = point1[:, 1]
    z0 = point0[:, 2]
    z1 = point1[:, 2]

    # Corresponding voxel features and coefficients
    i0 = F.constant(0, (B, ))
    fxy_00 = triplane_feature[i0, x0, y0]
    fxy_01 = triplane_feature[i0, x0, y1]
    fxy_10 = triplane_feature[i0, x1, y0]
    fxy_11 = triplane_feature[i0, x1, y1]
    
    i1 = F.constant(1, (B, ))
    fyz_00 = triplane_feature[i1, y0, z0]
    fyz_01 = triplane_feature[i1, y0, z1]
    fyz_10 = triplane_feature[i1, y1, z0]
    fyz_11 = triplane_feature[i1, y1, z1]
    
    i2 = F.constant(2, (B, ))
    fzx_00 = triplane_feature[i2, z0, x0]
    fzx_01 = triplane_feature[i2, z0, x1]
    fzx_10 = triplane_feature[i2, z1, x0]
    fzx_11 = triplane_feature[i2, z1, x1]

    p0 = coeff0[:, 0].reshape((B, 1))
    p1 = coeff1[:, 0].reshape((B, 1))
    q0 = coeff0[:, 1].reshape((B, 1))
    q1 = coeff1[:, 1].reshape((B, 1))
    r0 = coeff0[:, 2].reshape((B, 1))
    r1 = coeff1[:, 2].reshape((B, 1))

    # Linear interpolation
    def interpolate(f00, f01, f10, f11, u0, u1, v0, v1):
        return u0 * v0 * f00 + u0 * v1 * f01 + u1 * v0 * f10 + u1 * v1 * f11
            
    fxy = interpolate(fxy_00, fxy_01, fxy_10, fxy_11, p0, p1, q0, q1)
    fyz = interpolate(fyz_00, fyz_01, fyz_10, fyz_11, q0, q1, r0, r1)
    fzx = interpolate(fzx_00, fzx_01, fzx_10, fzx_11, r0, r1, p0, p1)
   
    fxy = F.reshape(fxy, (B, D, 1))
    fyz = F.reshape(fyz, (B, D, 1))
    fzx = F.reshape(fzx, (B, D, 1))
    f = F.concatenate(*[fxy, fyz, fzx], axis=-1)
    f = F.reshape(f, (B, D * 3))

    return f
