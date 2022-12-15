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
import numpy as np
import pytest
from nnabla.ext_utils import get_extension_context

from sampler import sample_importance_directions, sample_uniform_directions


def sample_directions_numpy(normal, cdf_the, cdf_phi, alpha=None):
    B, R, _ = normal.shape
    normal = normal.reshape((B * R, 3))
    cdf_the = cdf_the.reshape((B * R, -1))
    cdf_phi = cdf_phi.reshape((B * R, -1))
    M = cdf_the.shape[-1] * cdf_phi.shape[-1]

    light_dirs = []
    for b in range(B * R):
        # cdf_the_b, cdf_phi_b = np.meshgrid(cdf_phi[b], cdf_the[b])
        cdf_phi_b, cdf_the_b = np.meshgrid(cdf_phi[b], cdf_the[b])
        cdf_the_b = cdf_the_b.flatten()
        cdf_phi_b = cdf_phi_b.flatten()

        phi = 2 * np.pi * cdf_phi_b
        if alpha is None:
            cos_the = cdf_the_b
        else:
            alpha = alpha.reshape((B * R, 1))
            alpha_b = alpha[b]
            alpha_b2 = alpha_b * alpha_b
            cos_the = np.sqrt((1.0 - cdf_the_b) / ((alpha_b2 - 1.0) * cdf_the_b + 1.0))
        sin_the = np.sqrt(1.0 - cos_the * cos_the)
        x = sin_the * np.cos(phi)
        y = sin_the * np.sin(phi)
        z = cos_the
        xyz = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)

        normal_b = normal[b]
        normal_b = normal_b / np.linalg.norm(normal_b, ord=2, keepdims=True)
        z_axis = normal_b
        x_axis = np.asarray([-normal_b[1], normal_b[0], 0])
        x_axis = x_axis / np.linalg.norm(x_axis, ord=2, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis, ord=2, keepdims=True)
        rot = np.asarray([x_axis, y_axis, z_axis])
        rot = rot.T

        rot = rot.reshape((1, 3, 3))
        xyz = xyz.reshape((xyz.shape[0], 3, 1))
        xyz = rot @ xyz
        xyz = xyz.reshape((xyz.shape[0], 3))
        
        light_dirs.append(xyz)


    light_dirs = np.asarray(light_dirs).reshape((B, R, M, 3))
    return light_dirs


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("batch_size, n_rays", [
                        (1, 1), 
                        (2, 4), 
                        ])
@pytest.mark.parametrize("n_thetas", [1, 4])
@pytest.mark.parametrize("type", ["uniform", "importance"])
@pytest.mark.parametrize("eps", [0.0, 1e-12])
def test_sample_uniform_direction(seed, batch_size, n_rays, n_thetas, type, eps):
    nn.clear_parameters()
    
    ctx = get_extension_context('cudnn', device_id="0")
    nn.set_default_context(ctx)
    
    # common
    B = batch_size
    R = n_rays
    rng = np.random.RandomState(seed)

    normal_data = rng.randn(B, R, 3).astype(np.float32)
    normal_data = normal_data / np.linalg.norm(normal_data, ord=2, axis=-1, keepdims=True)
    cdf_the_data = rng.rand(B, R, n_thetas).astype(np.float32)
    cdf_phi_data = rng.rand(B, R, 2 * n_thetas).astype(np.float32)

    normal = nn.Variable.from_numpy_array(normal_data)
    cdf_the = nn.Variable.from_numpy_array(cdf_the_data)
    cdf_phi = nn.Variable.from_numpy_array(cdf_phi_data)

    if type == "uniform":
        with nn.auto_forward(True):
            light_dirs = sample_uniform_directions(normal, cdf_the, cdf_phi, eps)
        light_dirs_data = sample_directions_numpy(normal_data, cdf_the_data, cdf_phi_data)
    elif type == "importance":
        alpha_data = rng.randn(B, R, 1).astype(np.float32)
        alpha = nn.Variable.from_numpy_array(alpha_data)
        with nn.auto_forward(True):
            light_dirs = sample_importance_directions(normal, cdf_the, cdf_phi, alpha, eps)
        light_dirs_data = sample_directions_numpy(normal_data, cdf_the_data, cdf_phi_data, alpha_data)
    np.testing.assert_allclose(light_dirs_data, light_dirs.d, atol=1e-5)

