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

import lanczos_triline_feature_cuda
import nnabla as nn
import nnabla.backward_functions
import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.function import PythonFunction
from nnabla.parametric_functions import parametric_function_api


# Forward
class LanczosQueryOnTriplane(PythonFunction):
    """
    Given a query (B, 3) and grid feature (3, G, G, D), 
    the function returns interpolations of features (B, D).
    """

    def __init__(self, ctx, min_=[-1, -1, -1], max_=[1, 1, 1],
                 use_ste=False,
                 boundary_check=False):
        super(LanczosQueryOnTriplane, self).__init__(ctx)

        self._min = min_
        self._max = max_
        self._use_ste = use_ste
        self._boundary_check = boundary_check
        
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(min_=self._min,
                    max_=self._max,
                    use_ste=self._use_ste, 
                    boundary_check=self._boundary_check)
        return data
                
    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[0]
        feature = inputs[1]

        assert len(query.shape) > 1, "Query shape must be greater than 1, e.g., (B1, ..., Bn, 3)."
        assert query.shape[-1] == 3, "Query shape[-1] must be 3."
        assert len(feature.shape) > 1 and len(feature.shape) < 5, "Feature must be either 1D, 2D, 3D features"
        
        batch_sizes = query.shape[:-1]
        D = feature.shape[-1]
        
        outputs[0].reset_shape(batch_sizes + (D * 3, ), True)

    def forward_impl(self, inputs, outputs):
        query = inputs[0]
        feature = inputs[1]
        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        G = feature.shape[1]
        B = np.prod(batch_sizes)
        D = feature.shape[-1]

        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)
        output_ptr = output.data.data_ptr(np.float32, self.ctx)

        lanczos_triline_feature_cuda.query_on_triline(
            B * D * 3,
            output_ptr, query_ptr, feature_ptr, 
            G, D,
            self._min, self._max,
            self._boundary_check)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        query = inputs[0]
        feature = inputs[1]
        output = outputs[0]
        
        batch_sizes = query.shape[:-1]
        G = feature.shape[1]
        B = np.prod(batch_sizes)
        D = feature.shape[-1]

        grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        grad_feature_ptr = feature.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = output.grad.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)

        # Do not call for optimization
        # if propagate_down[0]:
        #     lanczos_triline_feature_cuda.grad_query(
        #         B * D * 3,
        #         grad_query_ptr, 
        #         grad_output_ptr, query_ptr, feature_ptr, 
        #         G, D, 
        #         self._min, self._max,
        #         self._boundary_check, accum[0])

        if propagate_down[1]:
            lanczos_triline_feature_cuda.grad_feature(
                B * D * 3,
                grad_feature_ptr, 
                grad_output_ptr, query_ptr,
                G, D, 
                self._min, self._max,
                self._boundary_check, accum[1])

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0:
            return True
        if i == 1 and j == 0:
            return True
        return False

    
def query_on_triline(query, feature, min_, max_, use_ste=False, boundary_check=False, ctx=None):
    func = LanczosQueryOnTriplane(ctx, min_, max_, use_ste, boundary_check)
    return func(query, feature)

F.lanczos_query_on_triline = query_on_triline

@parametric_function_api("triline_feature", [
    ('F', 'Grid faeture', '[3, G, G, D]', True),
])
def _query_on_triline(x, G, feature_size,
                         min_=[-1, -1, -1], max_=[1, 1, 1],
                         use_ste=False, 
                         f_init=None, 
                         fix_parameters=False, rng=None):
    """
    """
    # eps = 1.0 / feature_size ** 2
    # f_init = f_init if f_init is not None else I.NormalInitializer(eps)

    f_init = f_init if f_init is not None else I.NormalInitializer(1e-3)

    shape = [3, G, G, feature_size]
    feature = nn.parameter.get_parameter_or_create("F",
                                                   shape, 
                                                   f_init,
                                                   True, not fix_parameters)
    h = F.lanczos_query_on_triline(x, feature, min_, max_, use_ste)
    return h

PF.lanczos_query_on_triline = _query_on_triline


# Backward
class QueryOnTriplaneGradQuery(PythonFunction):
    """
    Gradient wrt query of QueryOnTriplane

    Given inputs of grad_output (B, 3), query (B, 3), feature (3, G, G, D),
    the function outupts grad_query (B, 3).
    """

    def __init__(self, ctx, min_, max_, boundary_check=False):
        super(QueryOnTriplaneGradQuery, self).__init__(ctx)

        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check
        
    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        query = inputs[1]
        outputs[0].reset_shape(query.shape, True)

    def forward_impl(self, inputs, outputs):
        grad_query = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]
        feature = inputs[2]

        batch_sizes = query.shape[:-1]
        G = feature.shape[1]
        B = np.prod(batch_sizes)
        D = feature.shape[-1]

        grad_query_ptr = grad_query.data.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)
        
        lanczos_triline_feature_cuda.grad_query(B * D * 3, grad_query_ptr,
                                     grad_output_ptr, query_ptr, feature_ptr,
                                     G, D,
                                     self._min, self._max,
                                     self._boundary_check, False)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        grad_query = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]
        feature = inputs[2]

        batch_sizes = query.shape[:-1]
        G = feature.shape[1]
        B = np.prod(batch_sizes)
        D = feature.shape[-1]

        grad_grad_query_ptr = grad_query.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)

        grad_grad_output_ptr = grad_output.grad.data_ptr(np.float32, self.ctx)
        grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        grad_feature_ptr = feature.grad.data_ptr(np.float32, self.ctx)

        if propagate_down[0]:
            lanczos_triline_feature_cuda.grad_query_grad_grad_output(B * D * 3, grad_grad_output_ptr,
                                                          grad_grad_query_ptr,
                                                          query_ptr, feature_ptr,
                                                          G, D,
                                                          self._min, self._max,
                                                          self._boundary_check, accum[0])
        # if propagate_down[1]:
        #     lanczos_triline_feature_cuda.grad_query_grad_query(B * D * 3, grad_query_ptr,
        #                                             grad_grad_query_ptr,
        #                                             grad_output_ptr,
        #                                             query_ptr, feature_ptr,
        #                                             G, D,
        #                                             self._min, self._max,
        #                                             self._boundary_check, accum[1])

        if propagate_down[2]:
            lanczos_triline_feature_cuda.grad_query_grad_feature(B * D * 3, grad_feature_ptr, 
                                                      grad_grad_query_ptr,
                                                      grad_output_ptr,
                                                      query_ptr, 
                                                      G, D,
                                                      self._min, self._max,
                                                      self._boundary_check, accum[2])
            

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0 and (j == 1 or j == 2):
            return True
        if i == 1:
            return True
        if i == 2 and (j == 0 or j == 1):
            return True
        return False


def grad_query(grad_output, query, feature, 
               min_, max_, boundary_check=False, ctx=None):
    func = QueryOnTriplaneGradQuery(ctx, min_, max_, boundary_check)
    return func(grad_output, query, feature)


class QueryOnTriplaneGradFeature(PythonFunction):
    """
    Gradient wrt query of QueryOnTriplane

    Given inputs of grad_output (B, 3), query (B, 3), feature (3, G, G, D),
    the function outupts grad_query (3, G, G, D).
    """

    def __init__(self, ctx, min_, max_, boundary_check=False, G=None):
        super(QueryOnTriplaneGradFeature, self).__init__(ctx)

        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check
        self._G = G
        
    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        grad_output = inputs[0]

        D = grad_output.shape[-1]
        outputs[0].reset_shape(self._G + (D, ), True)

    def forward_impl(self, inputs, outputs):
        grad_feature = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]

        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        D = grad_output.shape[-1]

        grad_feature_ptr = grad_feature.data.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)

        lanczos_triline_feature_cuda.grad_feature(B * D * 3, grad_feature_ptr,
                                       grad_output_ptr, query_ptr, 
                                       self._G, D,
                                       self._min, self._max,
                                       self._boundary_check, False)
        
    def backward_impl(self, inputs, outputs, propagate_down, accum):
        grad_feature = outputs[0]
        grad_output = inputs[0]
        query = inputs[1]

        batch_sizes = query.shape[:-1]
        B = np.prod(batch_sizes)
        D = grad_output.shape[-1]

        grad_grad_feature_ptr = grad_feature.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = grad_output.data.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)

        grad_grad_output_ptr = grad_output.grad.data_ptr(np.float32, self.ctx)
        grad_query_ptr = query.grad.data_ptr(np.float32, self.ctx)
        
        if propagate_down[0]:
            lanczos_triline_feature_cuda.grad_feature_grad_grad_output(B * D * 3, grad_grad_output_ptr,
                                                            grad_grad_feature_ptr,
                                                            query_ptr,
                                                            self._G, D,
                                                            self._min, self._max,
                                                            self._boundary_check, accum[0])
        if propagate_down[1]:
            lanczos_triline_feature_cuda.grad_feature_grad_query(B * D * 3, grad_query_ptr,
                                                      grad_grad_feature_ptr,
                                                      grad_output_ptr,
                                                      query_ptr, 
                                                      self._G, D,
                                                      self._min, self._max,
                                                      self._boundary_check, accum[1])

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 0 and j == 1:
            return True
        if i == 1:
            return True
        return False
            

def grad_feature(grad_output, query, min_=[-1, -1, -1], max_=[1, 1, 1],
                 boundary_check=False, G=None, ctx=None):
    func = QueryOnTriplaneGradFeature(ctx, min_, max_, boundary_check, G)
    return func(grad_output, query)


def query_on_triline_backward(inputs, min_=[-1, -1, -1], max_=[1, 1, 1], use_ste=False, 
                                  boundary_check=False):
    grad_output = inputs[0]
    query = inputs[1]
    feature = inputs[2]
    G = feature.shape[:-1]

    if use_ste: 
        return None, None

    # Directional gradients
    gq = grad_query(grad_output, query, feature, min_, max_, boundary_check)
    # gf = grad_feature(grad_output, query, min_, max_, boundary_check, G)

    return gq, None

nnabla.backward_functions.register("LanczosQueryOnTriplane", query_on_triline_backward)
