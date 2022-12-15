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

import grid_feature.triplane_feature
import nnabla as nn
import nnabla.functions as F
import numpy as np
import pytest
from grid_feature.triplane_feature_composite import \
    query_on_triplane as query_on_triplane_composite
from nnabla.ext_utils import get_extension_context


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("batch_size", [2, 16])
@pytest.mark.parametrize("grid_size", [2, 8])
@pytest.mark.parametrize("feature_size", [4])
@pytest.mark.parametrize("m, M", [(-1, 1)])
def test_query_on_triplane_forward_backward(seed, batch_size, grid_size, feature_size, m, M):
    nn.clear_parameters()
    
    ctx = get_extension_context('cudnn', device_id="0")
    nn.set_default_context(ctx)
    
    # common
    B = batch_size
    G = grid_size
    D = feature_size
    rng = np.random.RandomState(seed)
    
    query_data = m + rng.rand(batch_size, 3) * (M - m)
    initializer_data = rng.randn(3, G, G, D) * 0.01

    # composite
    query_data0 = query_data.astype(np.float32)
    initializer_data0 = initializer_data.astype(np.float32)
    query0 = nn.Variable.from_numpy_array(query_data0).apply(need_grad=True)
    feature0 = nn.parameter.get_parameter_or_create("F0", (3, G, G, D),
                                                          initializer_data0)
    output0 = query_on_triplane_composite(query0, feature0, m, M)

    # monolithic
    query_data1 = query_data.astype(np.float32)
    initializer_data1 = initializer_data.astype(np.float32)
    query1 = nn.Variable.from_numpy_array(query_data1).apply(need_grad=True)
    feature1 = nn.parameter.get_parameter_or_create("F1", (3, G, G, D),
                                                          initializer_data1)
    output1 = F.query_on_triplane(query1, feature1, [m] * 3, [M] * 3)

    # forward check
    output0.forward(clear_no_need_grad=True)
    output1.forward(clear_no_need_grad=True)
    np.testing.assert_allclose(output0.d, output1.d, atol=1e-6)

    # 1st-order backward check
    query0.grad.fill(0)
    query1.grad.fill(0)
    feature0.grad.fill(0)
    feature1.grad.fill(0)

    ograd = rng.randn(*output0.shape).astype(np.float32)
    output0.backward(ograd, clear_buffer=True)
    output1.backward(ograd, clear_buffer=True)

    # np.testing.assert_allclose(query0.g, query1.g, atol=1e-6)
    np.testing.assert_allclose(feature0.g, feature1.g, atol=1e-6)


@pytest.mark.parametrize("seed", [412])
@pytest.mark.parametrize("batch_size", [2, 16])
@pytest.mark.parametrize("grid_size", [2, 8])
@pytest.mark.parametrize("feature_size", [4])
@pytest.mark.parametrize("m, M", [(-1.0, 1.0)]) 
def test_query_on_triplane_double_backward(seed, batch_size, grid_size, feature_size, m, M):
    nn.clear_parameters()
    
    ctx = get_extension_context('cudnn', device_id="0")
    nn.set_default_context(ctx)

    # common
    B = batch_size
    G = grid_size
    D = feature_size
    rng = np.random.RandomState(seed)
    
    query_data = m + rng.rand(batch_size, 3) * (M - m)
    initializer_data = rng.randn(3, G, G, D) * 0.01

    # composite
    query_data0 = query_data.astype(np.float32)
    initializer_data0 = initializer_data.astype(np.float32)
    query0 = nn.Variable.from_numpy_array(query_data0).apply(need_grad=True)
    feature0 = nn.parameter.get_parameter_or_create("F0", (3, G, G, D),
                                                          initializer_data0)
    output0 = query_on_triplane_composite(query0, feature0, m, M)

    # monolithic
    query_data1 = query_data.astype(np.float32)
    initializer_data1 = initializer_data.astype(np.float32)
    query1 = nn.Variable.from_numpy_array(query_data1).apply(need_grad=True)
    feature1 = nn.parameter.get_parameter_or_create("F1", (3, G, G, D),
                                                          initializer_data1)
    output1 = F.query_on_triplane(query1, feature1, [m] * 3, [M] * 3)

    # 1st-order grads
    ograd = rng.randn(*output0.shape).astype(np.float32)
    ograd0 = nn.Variable.from_numpy_array(ograd).apply(need_grad=True, persistent=True)
    ograd1 = nn.Variable.from_numpy_array(ograd).apply(need_grad=True, persistent=True)
    # grad_query0, grad_feature0 = nn.grad([output0], [query0, feature0], grad_outputs=[ograd0])
    # grad_query1, grad_feature1 = nn.grad([output1], [query1, feature1], grad_outputs=[ograd1])
    grad_query0 = nn.grad([output0], [query0], grad_outputs=[ograd0])[0]
    grad_query1 = nn.grad([output1], [query1], grad_outputs=[ograd1])[0]
    
    # F.sink(*[grad_query0, grad_feature0]).forward(clear_no_need_grad=True)
    # F.sink(*[grad_query1, grad_feature1]).forward(clear_no_need_grad=True)
    F.sink(*[grad_query0]).forward(clear_no_need_grad=True)
    F.sink(*[grad_query1]).forward(clear_no_need_grad=True)
    np.testing.assert_allclose(grad_query0.d, grad_query1.d, atol=1e-6)
    # np.testing.assert_allclose(grad_feature0.d, grad_feature1.d, atol=1e-6)

    # 2nd-order grads (grad_query)
    ograd0.grad.fill(0)
    ograd1.grad.fill(0)
    query0.grad.fill(0)
    query1.grad.fill(0)
    feature0.grad.fill(0)
    feature1.grad.fill(0)
    
    o0 = F.sum(grad_query0 ** 2)
    o1 = F.sum(grad_query1 ** 2)
    o0.forward(clear_no_need_grad=True)
    o1.forward(clear_no_need_grad=True)
    ograd = rng.randn()
    o0.backward(ograd, clear_buffer=True)
    o1.backward(ograd, clear_buffer=True)

    np.testing.assert_allclose(ograd0.g, ograd1.g, atol=1e-4)
    # np.testing.assert_allclose(query0.g, query1.g, atol=1e-6)
    np.testing.assert_allclose(feature0.g, feature1.g, atol=8e-3)


    # # 2nd-order grads (grad_feature)
    # ograd0.grad.fill(0)
    # ograd1.grad.fill(0)
    # query0.grad.fill(0)
    # query1.grad.fill(0)
    # feature0.grad.fill(0)
    # feature1.grad.fill(0)
    
    # o0 = F.sum(grad_feature0 ** 2)
    # o1 = F.sum(grad_feature1 ** 2)
    # o0.forward(clear_no_need_grad=True)
    # o1.forward(clear_no_need_grad=True)
    # ograd = rng.randn()
    # o0.backward(ograd, clear_buffer=True)
    # o1.backward(ograd, clear_buffer=True)

    # np.testing.assert_allclose(query0.g, query1.g, atol=1e-6)
    # np.testing.assert_allclose(feature0.g, feature1.g, atol=1e-6)
