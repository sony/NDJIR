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


def query_on_voxel(query, voxel_feature, min_, max_, all_depth=True):
    B, R, N, _ = query.shape
    query = F.reshape(query, (B * R * N, 3))
    
    G = voxel_feature.shape[0]
    assert G % 2 == 1, "Grid size must be the power of 2 + 1"
    D = voxel_feature.shape[-1]
    L = int(np.log2(G - 1)) + 1 if all_depth else 1
    outputs = []
    for l in range(L):
        rG = 2 ** l
        out = _query_on_voxel(query, voxel_feature, min_, max_, rG)
        outputs.append(out)
    if len(outputs) != 1:
        out = F.stack(*outputs, axis=1)
        _, L, D = out.shape
        out = out.reshape((B, R, N, L, D))
    else:
        out = out.reshape((B, R, N, 1, D))
    return out


def _query_on_voxel(query, voxel_feature, min_, max_, rG):
    B = query.shape[0]
    G = voxel_feature.shape[0]

    # continuous point, discrete points, coefficients
    scale = (G - 1) / rG / (max_ - min_)
    pointf = (query - min_) * scale 
    point0 = F.floor(pointf).apply(need_grad=False)
    point1 = point0 + 1
    coeff0 = point1 - pointf
    coeff1 = pointf - point0
    x0 = point0[:, 0] * rG
    x1 = point1[:, 0] * rG
    y0 = point0[:, 1] * rG
    y1 = point1[:, 1] * rG
    z0 = point0[:, 2] * rG
    z1 = point1[:, 2] * rG

    # Corresponding voxel features and coefficients
    f000 = voxel_feature[x0, y0, z0]
    f001 = voxel_feature[x0, y0, z1]
    f010 = voxel_feature[x0, y1, z0]
    f011 = voxel_feature[x0, y1, z1]
    f100 = voxel_feature[x1, y0, z0]
    f101 = voxel_feature[x1, y0, z1]
    f110 = voxel_feature[x1, y1, z0]
    f111 = voxel_feature[x1, y1, z1]

    p0 = coeff0[:, 0].reshape((B, 1))
    p1 = coeff1[:, 0].reshape((B, 1))
    q0 = coeff0[:, 1].reshape((B, 1))
    q1 = coeff1[:, 1].reshape((B, 1))
    r0 = coeff0[:, 2].reshape((B, 1))
    r1 = coeff1[:, 2].reshape((B, 1))

    # Linear interpolation
    f = (p0 * q0 * r0) * f000 \
      + (p0 * q0 * r1) * f001 \
      + (p0 * q1 * r0) * f010 \
      + (p0 * q1 * r1) * f011 \
      + (p1 * q0 * r0) * f100 \
      + (p1 * q0 * r1) * f101 \
      + (p1 * q1 * r0) * f110 \
      + (p1 * q1 * r1) * f111
    
    return f
