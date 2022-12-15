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


def tv_loss_on_triplane(query, triplane_feature, min_, max_, sym_backward):
    B = query.shape[0]
    G = triplane_feature.shape[1]
    D = triplane_feature.shape[-1]

    # continuous point, discrete points
    scale = (G - 1) / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    point1 = F.clip_by_value(point0 + 1, 0, G - 1)
    x0 = point0[:, 0]
    x1 = point1[:, 0]
    y0 = point0[:, 1]
    y1 = point1[:, 1]
    z0 = point0[:, 2]
    z1 = point1[:, 2]

    # Corresponding triplane features
    def tv_loss(l, u0, u1, v0, v1):
        idx_l = F.constant(l, (B, ))
        fuv_00 = triplane_feature[idx_l, u0, v0].apply(need_grad=sym_backward)
        fuv_01 = triplane_feature[idx_l, u0, v1]
        fuv_10 = triplane_feature[idx_l, u1, v0]
        delta_u = fuv_10 - fuv_00
        delta_v = fuv_01 - fuv_00
        delta_u2 = delta_u * delta_u
        delta_v2 = delta_v * delta_v
        fuv = (delta_u2 + delta_v2) ** 0.5
        return fuv

    fxy = tv_loss(0, x0, x1, y0, y1)
    fyz = tv_loss(1, y0, y1, z0, z1)
    fzx = tv_loss(2, z0, z1, x0, x1)

    fxy = F.reshape(fxy, (B, D, 1))
    fyz = F.reshape(fyz, (B, D, 1))
    fzx = F.reshape(fzx, (B, D, 1))
    f = F.concatenate(*[fxy, fyz, fzx], axis=-1)
    f = F.reshape(f, (B, D * 3))

    return f
