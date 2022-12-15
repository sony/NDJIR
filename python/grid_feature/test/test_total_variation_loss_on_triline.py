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

import grid_feature.total_variation_loss_on_triline
import nnabla as nn
import nnabla.functions as F
import numpy as np
import pytest
from grid_feature.total_variation_loss_on_triline_composite import \
    tv_loss_on_triline as tv_loss_on_triline_composite
from nnabla.ext_utils import get_extension_context


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("batch_size", [2, 16])
@pytest.mark.parametrize("grid_size", [2, 8])
@pytest.mark.parametrize("feature_size", [4])
@pytest.mark.parametrize("m, M", [(-1, 1)])
@pytest.mark.parametrize("sym_backward", [False, True])
def test_tv_loss_on_triline_forward_backward(seed, batch_size, grid_size, feature_size, m, M, sym_backward):
    nn.clear_parameters()
    
    ctx = get_extension_context('cudnn', device_id="0")
    nn.set_default_context(ctx)
    
    # common
    B = batch_size
    G = grid_size
    D = feature_size
    rng = np.random.RandomState(seed)
    
    query_data = m + rng.rand(batch_size, 3) * (M - m)
    initializer_data = rng.randn(3, G, D) * 0.01

    # composite
    query_data0 = query_data.astype(np.float32)
    initializer_data0 = initializer_data.astype(np.float32)
    query0 = nn.Variable.from_numpy_array(query_data0).apply(need_grad=True)
    feature0 = nn.parameter.get_parameter_or_create("F0", (3, G, D),
                                                          initializer_data0)
    output0 = tv_loss_on_triline_composite(query0, feature0, m, M, sym_backward) 

    # monolithic
    query_data1 = query_data.astype(np.float32)
    initializer_data1 = initializer_data.astype(np.float32)
    query1 = nn.Variable.from_numpy_array(query_data1).apply(need_grad=True)
    feature1 = nn.parameter.get_parameter_or_create("F1", (3, G, D),
                                                          initializer_data1)
    output1 = F.tv_loss_on_triline(query1, feature1, [m] * 3, [M] * 3, sym_backward=sym_backward)

    # forward check
    output0.forward(clear_no_need_grad=True)
    output1.forward(clear_no_need_grad=True)
    np.testing.assert_allclose(output0.d, output1.d, atol=1e-6)

    # backward check
    feature0.grad.fill(0)
    feature1.grad.fill(0)

    ograd = rng.randn(*output0.shape).astype(np.float32)
    output0.backward(ograd, clear_buffer=True)
    output1.backward(ograd, clear_buffer=True)

    np.testing.assert_allclose(feature0.g, feature1.g, atol=1e-4)
