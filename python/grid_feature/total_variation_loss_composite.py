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


def tv_loss_on_voxel(query, voxel_feature, min_, max_, sym_backward):
    
    B = query.shape[0]
    G = voxel_feature.shape[0]

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

    # Corresponding voxel features
    f000 = voxel_feature[x0, y0, z0].apply(need_grad=sym_backward)
    f001 = voxel_feature[x0, y0, z1]
    f010 = voxel_feature[x0, y1, z0]
    f100 = voxel_feature[x1, y0, z0]

    # TV loss
    delta_x = f100 - f000
    delta_y = f010 - f000
    delta_z = f001 - f000

    delta_x2 = delta_x * delta_x
    delta_y2 = delta_y * delta_y
    delta_z2 = delta_z * delta_z
    
    f = (delta_x2 + delta_y2 + delta_z2) ** 0.5
        
    return f
