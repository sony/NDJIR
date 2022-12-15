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

import grid_feature.total_variation_loss_on_voxel_hash
import nnabla as nn
import nnabla.functions as F
import numpy as np
import pytest
from grid_feature.total_variation_loss_on_voxel_hash_composite import \
    tv_loss_on_voxel_hash as tv_loss_on_voxel_hash_composite
from grid_feature.voxel_hash_feature import compute_num_params
from nnabla.ext_utils import get_extension_context


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("grid_size_base", [2, 4])
@pytest.mark.parametrize("growth_factor", [1.5])
@pytest.mark.parametrize("table_size_base", [2**10])
@pytest.mark.parametrize("n_levels", [1, 4])
@pytest.mark.parametrize("feature_size", [2])
@pytest.mark.parametrize("m, M", [(-1, 1)])
def test_tv_loss_on_voxel_hash_forward_backward(seed, batch_size, 
                                               grid_size_base, growth_factor, table_size_base, n_levels, 
                                               feature_size, 
                                               m, M):
    nn.clear_parameters()
    
    ctx = get_extension_context('cudnn', device_id="0")
    nn.set_default_context(ctx)
    
    # common
    B = batch_size
    G0 = grid_size_base
    T0 = table_size_base
    L = n_levels
    D = feature_size
    rng = np.random.RandomState(seed)
    
    query_data = m + rng.rand(batch_size, 3) * (M - m)
    n_params = compute_num_params(G0, growth_factor, T0, D, L)
    initializer_data = rng.randn(n_params) * 0.01

    # composite
    query_data0 = query_data.astype(np.float32)
    initializer_data0 = initializer_data.astype(np.float32)
    query0 = nn.Variable.from_numpy_array(query_data0).apply(need_grad=True)
    feature0 = nn.parameter.get_parameter_or_create("F0", (n_params, ),
                                                          initializer_data0)
    output0 = tv_loss_on_voxel_hash_composite(query0, feature0, 
                                              G0, growth_factor, T0, L, D, 
                                              m, M)

    # monolithic
    query_data1 = query_data.astype(np.float32)
    initializer_data1 = initializer_data.astype(np.float32)
    query1 = nn.Variable.from_numpy_array(query_data1).apply(need_grad=True)
    feature1 = nn.parameter.get_parameter_or_create("F1", (n_params, ),
                                                          initializer_data1)
    output1 = F.tv_loss_on_voxel_hash(query1, feature1, 
                                      G0, growth_factor, T0, L, D, 
                                      [m] * 3, [M] * 3)

    # forward check
    output0.forward(clear_no_need_grad=True)
    output1.forward(clear_no_need_grad=True)
    np.testing.assert_allclose(output0.d, output1.d, atol=1e-6)

    # 1st-order backward check
    feature0.grad.fill(0)
    feature1.grad.fill(0)

    ograd = rng.randn(*output0.shape).astype(np.float32)
    output0.backward(ograd, clear_buffer=True)
    output1.backward(ograd, clear_buffer=True)

    np.testing.assert_allclose(feature0.g, feature1.g, atol=1e-6)

