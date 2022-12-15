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

import numpy as np
import squareplus_cuda
from nnabla.function import PythonFunction


class Squareplus(PythonFunction):

    def __init__(self, ctx, b=4):

        super(Squareplus, self).__init__(ctx)

        self.b = b

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(b=self.b)
        return data
                
    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        """
        """
        x = inputs[0]
        outputs[0].reset_shape(x.shape, True);

    def forward_impl(self, inputs, outputs):
        x = inputs[0]
        y = outputs[0]

        size = x.size
        x_ptr = x.data.data_ptr(np.float32, self.ctx)
        y_ptr = y.data.data_ptr(np.float32, self.ctx)

        squareplus_cuda.forward(size, y_ptr, x_ptr, self.b)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        x = inputs[0]
        y = outputs[0]

        size = x.size

        dx_ptr = x.grad.data_ptr(np.float32, self.ctx)
        dy_ptr = y.grad.data_ptr(np.float32, self.ctx)
        x_ptr = x.data.data_ptr(np.float32, self.ctx)

        if propagate_down[0]:
            squareplus_cuda.backward(size, dx_ptr, dy_ptr, x_ptr, self.b, accum[0])
                            
    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True
    

def squareplus(x, b=4, ctx=None):
    func = Squareplus(ctx, b)
    return func(x)


def squareplus_backward(inputs, b=1.0):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.
    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    dx0 = dy * 0.5 * (1.0 + x0 * (x0 * x0 + b) ** -0.5)
    return dx0


from nnabla.backward_functions import register

register("Squareplus", squareplus_backward)

