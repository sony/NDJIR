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

def clamp(x, a, b):
    y = F.greater_equal_scalar(x, a)
    z = F.less_equal_scalar(y, b)
    return z

def query_on_voxel(query, voxel_feature, min_, max_, window_size=2):
    
    B = query.shape[0]
    G = voxel_feature.shape[0]
    G1 = G - 1

    # continuous point, discrete points, coefficients
    scale = G1 / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    x0, y0, z0 = point0[:, 0], point0[:, 1], point0[:, 2]
    x, y, z = pointf[:, 0], pointf[:, 1], pointf[:, 2]
    

    f = 0
    w = window_size
    for i in range(-w + 1, w + 1):
        xi = F.clip_by_value(x0 + i, 0, G1)
        cx = lanczos(x - xi, w).reshape((B, 1))
        
        for j in range(-w + 1, w + 1):
            yj = F.clip_by_value(y0 + j, 0, G1)
            cy = lanczos(y - yj, w).reshape((B, 1))

            for k in range(-w + 1, w + 1):
                zk = F.clip_by_value(z0 + k, 0, G1)
                cz = lanczos(z - zk, w).reshape((B, 1))

                fxyz = voxel_feature[xi, yj, zk]
                cxyz = cx * cy * cz
                f = f + fxyz * cxyz

    return f
