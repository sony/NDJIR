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
import nnabla.functions as F
import numpy as np
import ray_sphere_intersection_cuda
from nnabla.ext_utils import get_extension_context
from nnabla.function import PythonFunction


# Forward
class RaySphereIntersection(PythonFunction):
    """
    Ray Sphere Intersection.

    Assume that rays are casted outside sphere. Normally, there are two intersections or no intersection.
    The hit condition is (t >= 0.0) and the term in the square root is positive or 0.

    If a ray is casted inside the sphere, then we have 1 intersection.

    When we reduce rays to ones each of which has more than two intersections, use

    ```
        mask = F.greater_scalar(n_hits, 1)
    ```

    """

    def __init__(self, ctx, radius=1.0):
        super(RaySphereIntersection, self).__init__(ctx)
        self.radius = radius

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        data = dict(radius=self.radius)
        return data
                
    def min_outputs(self):
        return 3

    def setup_impl(self, inputs, outputs):
        """
        inputs: 
            camloc: (B, 3) 
            raydir: (B, R, 3) assumed to be unit-vector normalized
        outputs:
            t_near: (B, R, 1)
            t_far: (B, R, 1)
            n_hits: (B, R, 1)
        """
        camloc = inputs[0]
        raydir = inputs[1]
        assert len(camloc.shape) == 2, "length of camloc.shape must be 2."
        assert len(raydir.shape) == 3, "length of raydir.shape must be 3."

        B, R, _ = raydir.shape
        outputs[0].reset_shape((B, R, 1), True)
        outputs[1].reset_shape((B, R, 1), True)
        outputs[2].reset_shape((B, R, 1), True)

    def forward_impl(self, inputs, outputs):
        camloc = inputs[0]
        raydir = inputs[1]
        t_near = outputs[0]
        t_far = outputs[1]
        n_hits = outputs[2]

        camloc_ptr = camloc.data.data_ptr(np.float32, self.ctx)
        raydir_ptr = raydir.data.data_ptr(np.float32, self.ctx)
        t_near_ptr = t_near.data.data_ptr(np.float32, self.ctx)
        t_far_ptr = t_far.data.data_ptr(np.float32, self.ctx)
        n_hits_ptr = n_hits.data.data_ptr(np.float32, self.ctx)

        B, R, _ = raydir.shape
        N = B * R
        
        ray_sphere_intersection_cuda.ray_sphere_intersection(
            N, t_near_ptr, t_far_ptr, n_hits_ptr, 
            camloc_ptr, raydir_ptr, 
            B, R, self.radius)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        pass

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False

    
def ray_sphere_intersection(camloc, raydir, radius=1.0, ctx=None):
    func = RaySphereIntersection(ctx, radius)
    return func(camloc, raydir)


def sample_inside_sphere(B, R, radius=1, rng=None):
    rng = np.random.RandomState(412) if not rng else rng
    phi = rng.rand(B, R) * 2 * np.pi
    costheta = rng.rand(B, R) * 2 - 1
    u = rng.rand(B, R)

    theta = np.arccos(costheta)
    r = radius * u ** (1. / 3.)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x = x.reshape(x.shape + (1, ))
    y = y.reshape(y.shape + (1, ))
    z = z.reshape(z.shape + (1, ))
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz


if __name__ == "__main__":
    # Generate intersection points for visibility check
    rng = np.random.RandomState(412)

    radius = 1.0

    camloc = radius * 2.0 + rng.randn(3)
    camloc = camloc.reshape((1, 3))
    
    R = 10000
    raydir = sample_inside_sphere(1, R, radius, rng)
    raydir = raydir - camloc.reshape((1, 1, 3))
    raydir = raydir / np.linalg.norm(raydir, ord=2, axis=-1, keepdims=True)
    
    ctx = get_extension_context("cudnn", device_id="0")
    nn.set_default_context(ctx)    

    camloc = nn.Variable.from_numpy_array(camloc)
    raydir = nn.Variable.from_numpy_array(raydir)

    with nn.auto_forward():
        t_near, t_far, n_hits = ray_sphere_intersection(camloc, raydir, radius)
        mask = F.greater_scalar(n_hits, 1)
        camloc = F.reshape(camloc, (1, 1, 3))
        x_near = camloc + t_near * raydir
        x_far = camloc + t_far * raydir
        print(f"#hits = {np.sum(mask.d.flatten() == 1.0)}")
    
    mask = n_hits.d.reshape((1, R))
    x_near = x_near.d[mask.astype(bool)]
    x_far = x_far.d[mask.astype(bool)]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.concatenate([x_near, 
                        x_far, 
                        camloc.d.reshape((1, 3))], axis=0))
    pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate([np.asarray([1, 0, 0] * x_near.shape[0]).reshape(x_near.shape), 
                        np.asarray([0, 0, 1] * x_far.shape[0]).reshape(x_far.shape), 
                        np.asarray([[1, 0.5, 0.25]])], axis=0))
    o3d.io.write_point_cloud("ray_sphere_intersection.ply", pcd)
