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
import total_variation_loss_on_triplane_cuda
from nnabla.function import PythonFunction


# Forward
class TVLossOnTriplane(PythonFunction):
    """
    Given a query (B, 3) and grid feature (3, G, G, D), 
    the function returns interpolations of features (B, D*3).
    """

    def __init__(self, ctx, min_=[-1, -1, -1], max_=[1, 1, 1],
                 sym_backward=False, boundary_check=False):
        super(TVLossOnTriplane, self).__init__(ctx)

        self._min = min_
        self._max = max_
        self._boundary_check = boundary_check
        self._sym_backward = sym_backward
        
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(min_=self._min,
                    max_=self._max,
                    sym_backward=self._sym_backward, 
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

        total_variation_loss_on_triplane_cuda.tv_loss_on_triplane(
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

        grad_feature_ptr = feature.grad.data_ptr(np.float32, self.ctx)
        grad_output_ptr = output.grad.data_ptr(np.float32, self.ctx)
        query_ptr = query.data.data_ptr(np.float32, self.ctx)
        feature_ptr = feature.data.data_ptr(np.float32, self.ctx)

        if propagate_down[0]:
            pass

        if propagate_down[1]:
            total_variation_loss_on_triplane_cuda.tv_loss_on_triplane_backward(
                B * D * 3,
                grad_feature_ptr, 
                grad_output_ptr, query_ptr, feature_ptr, 
                G, D, 
                self._min, self._max,
                self._sym_backward, 
                self._boundary_check, accum[1])

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        if i == 1:
            return True
        return False

    
def tv_loss_on_triplane(query, feature, min_=[-1, -1, -1], max_=[1, 1, 1], sym_backward=False, 
                        boundary_check=False, ctx=None):
    func = TVLossOnTriplane(ctx, min_, max_, sym_backward, boundary_check)
    return func(query, feature)


F.tv_loss_on_triplane = tv_loss_on_triplane
