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
from intersection.ray_sphere_intersection import (ray_sphere_intersection,
                                                  sample_inside_sphere)
from nnabla.ext_utils import get_extension_context


def ray_sphere_intersection_python(camloc, raydir, radius=1.0):
    """
    sphere outscribes r-radius sphere.

    camloc: (B, 3)
    raydir: (B, R, 3)
    """
    B, R, _ = raydir.shape
    camloc = np.broadcast_to(camloc.reshape((B, 1, 3)), (B, R, 3))
    camloc = camloc.reshape((B * R, 3))
    raydir = raydir.reshape((B * R, 3))
    # t = {X -+ sqrt(Y)} / Z
    # X = -cv
    # Y = cv**2 - vv(cc - radius**2)
    # Z = vv
    r2 = radius * radius
    cv = np.matmul(camloc.reshape(B * R, 1, 3), raydir.reshape(B * R, 3, 1))
    vv = np.matmul(raydir.reshape(B * R, 1, 3), raydir.reshape(B * R, 3, 1))
    cc = np.matmul(camloc.reshape(B * R, 1, 3), camloc.reshape(B * R, 3, 1))
    X = -cv
    Y = cv * cv - vv * (cc - r2)
    Z = vv
    Z_inv = 1. / Z
    t_nears = []
    t_fars = []
    n_hits = np.zeros((B * R, 1))
    for b in range(0, B * R):
        t_near = 0
        t_far = 0
        n_hit = 0
        x = float(X[b])
        y = float(Y[b])
        z_inv = float(Z_inv[b])
        if y > 0:
            y_sqrt = y ** 0.5
            t_near = (x - y_sqrt) * z_inv
            t_far = (x + y_sqrt) * z_inv
            pos_mask = int(t_near >= 0)
            t_near = pos_mask * t_near
            n_hit = 2 - (1 - pos_mask)
        elif y == 0:
            t_near = x * z_inv
            t_far = x * z_inv
            n_hit = 1
        n_hits[b, :] = n_hit
        t_nears.append(t_near)
        t_fars.append(t_far)

    t_nears = np.asarray(t_nears)
    t_fars = np.asarray(t_fars)
    
    return t_nears, t_fars, n_hits


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("B, R", [(2, 3)])
@pytest.mark.parametrize("radius, ratio", [(1, 2), (1.5, 2), 
                                           (1.0, 0.5)
                                           ])
def test_ray_sphere_intersection(seed, B, R, radius, ratio):

    # Sample camera location on 2 x r-radius sphere
    rng = np.random.RandomState(seed)
    camloc = rng.randn(B, 3)
    camloc /= np.linalg.norm(camloc, ord=2, axis=-1, keepdims=True)
    camloc *= (radius * ratio)
    
    # Sample raydir towards near origin from camera location
    raydir = sample_inside_sphere(B, R, radius, rng) - camloc.reshape((B, 1, 3))
    raydir /= np.linalg.norm(raydir, ord=2, axis=-1, keepdims=True)

    camloc_data = camloc.astype(np.float32)
    raydir_data = raydir.astype(np.float32)

    ctx = get_extension_context("cudnn", device_id="0")
    nn.set_default_context(ctx)
    camloc = nn.Variable.from_numpy_array(camloc_data)
    raydir = nn.Variable.from_numpy_array(raydir_data)
    radius = 1.0
    with nn.auto_forward():
        t_near0, t_far0, n_hits0 = ray_sphere_intersection(camloc, raydir, radius)
    
    t_near1, t_far1, n_hits1 = ray_sphere_intersection_python(camloc_data, raydir_data, radius)
    
    np.testing.assert_allclose(t_near0.d.flatten(), t_near1.flatten(), atol=1e-6)
    np.testing.assert_allclose(t_far0.d.flatten(), t_far1.flatten(), atol=1e-6)
    np.testing.assert_allclose(n_hits0.d.flatten(), n_hits1.flatten())
