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


import nnabla as nn
import nnabla.backward_functions
import numpy as np
import pytest
from intersection.ray_aabb_intersection import ray_aabb_intersection
from nnabla.ext_utils import get_extension_context


def ray_aabb_intersection_python(camloc, raydir, size=1.0):
    """
    AABB outscribes r-radius sphere.

    camloc: (B, 3)
    raydir: (B, R, 3)
    """
    B, R, _ = raydir.shape
    camloc = np.broadcast_to(camloc.reshape((B, 1, 3)), (B, R, 3))
    camloc = camloc.reshape((B * R, 3))
    raydir = raydir.reshape((B * R, 3))
    normal = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                        [-1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32)
    normal = normal.transpose((1, 0))
    nc = np.matmul(camloc, normal)
    nd = np.matmul(raydir, normal)
    t = (size - nc) / nd

    t_nears = []
    t_fars = []
    n_hits = np.zeros((B * R, 1))

    def correct_numerical_error(x_bi, i):
        if i == 0:
            x_bi[0] = size
        elif i == 1:
            x_bi[1] = size
        elif i == 2:
            x_bi[2] = size
        elif i == 3:
            x_bi[0] = -size
        elif i == 4:
            x_bi[1] = -size
        else:
            x_bi[2] = -size

    for b in range(0, B * R):
        camloc_b = camloc[b, :]
        raydir_b = raydir[b, :]
        t_b = t[b, :]
        aabb_intersection_indices = []
        
        for i in range(0, 6):
            t_bi = t_b[i:i+1]
            if np.isinf(t_bi): # parallel
                continue

            x_bi = camloc_b + t_bi * raydir_b
            correct_numerical_error(x_bi, i)
            greater_cond = x_bi >= -size
            less_cond = x_bi <= size
            
            cond = (t_bi >= 0) * np.prod(greater_cond * less_cond, axis=-1)            
            if not cond:
                continue

            aabb_intersection_indices.append(i)

        t_near = 0
        t_far = 0
        n_hits[b, :] = len(aabb_intersection_indices)
        if len(aabb_intersection_indices) >= 2:
            k = aabb_intersection_indices[0]
            l = aabb_intersection_indices[1]
            t_bik = t_b[k]
            t_bil = t_b[l]
            if t_bik < t_bil:
                t_near = t_bik
                t_far = t_bil
            else:
                t_near = t_bil
                t_far = t_bik
            t_nears.append(t_near)
            t_fars.append(t_far)
        elif len(aabb_intersection_indices) == 1:
            k = aabb_intersection_indices[0]
            t_far = t_b[k]
            t_nears.append(t_near)
            t_fars.append(t_far)
        else:
            t_nears.append(t_near)
            t_fars.append(t_far)

    t_nears = np.asarray(t_nears)
    t_fars = np.asarray(t_fars)
    
    return t_nears, t_fars, n_hits


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("B, R", [(2, 3)])
@pytest.mark.parametrize("radius, size", [(3, 1), (3, 1.5), 
                                          (1, 2) # camloc is inside AABB
                                          ])
def test_ray_aabb_intersection(seed, B, R, radius, size):

    # Sample camera location on r-radius sphere
    rng = np.random.RandomState(seed)
    camloc = rng.randn(B, 3)
    camloc /= np.linalg.norm(camloc, ord=2, axis=-1, keepdims=True)
    camloc *= radius
    
    # Sample raydir towards near origin from camera location
    raydir = rng.rand(B, R, 3) * size * 2 - size
    raydir = raydir - camloc.reshape((B, 1, 3))
    raydir /= np.linalg.norm(raydir, ord=2, axis=-1, keepdims=True)

    camloc_data = camloc.astype(np.float32)
    raydir_data = raydir.astype(np.float32)

    ctx = get_extension_context("cudnn", device_id="0")
    nn.set_default_context(ctx)
    camloc = nn.Variable.from_numpy_array(camloc_data)
    raydir = nn.Variable.from_numpy_array(raydir_data)
    min = [-size, -size, -size]
    max = [size, size, size]
    with nn.auto_forward():
        t_near0, t_far0, n_hits0 = ray_aabb_intersection(camloc, raydir, min, max)
    
    t_near1, t_far1, n_hits1 = ray_aabb_intersection_python(camloc_data, raydir_data, size)
    
    np.testing.assert_allclose(t_near0.d.flatten(), t_near1.flatten(), atol=1e-6)
    np.testing.assert_allclose(t_far0.d.flatten(), t_far1.flatten(), atol=1e-6)
    np.testing.assert_allclose(n_hits0.d.flatten(), n_hits1.flatten())
