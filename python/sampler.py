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

import inverse_transform_cuda
import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.function import PythonFunction

from network import geometric_network


class SamplePoints(PythonFunction):
    """
    """

    def __init__(self, ctx, conf):
        super(SamplePoints, self).__init__(ctx)

        self.conf = conf

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return {}
                
    def min_outputs(self):
        return 5
        
    def setup_impl(self, inputs, outputs):
        """
        inputs: 
            camloc: (B, 3)
            raydir: (B, R, 3)
            stratified_sample: (B, R, N, 1)
            background_sample: (B, R, M + 1, 1)
        outputs:
            x_fg: (B, R, N + M * U, 3)
            ts_fg: (B, R, N + M * U + 1, 1)
            x_bg: (B, R, Nb, 3)
            ts_bg: (B, R, Nb + 1, 1)
            mask: (B, R, 1, 1)
        """
        B, R, _ = inputs[1].shape
        N = self.conf.renderer.n_samples0
        M = self.conf.renderer.n_samples1
        U = self.conf.renderer.n_upsamples
        outputs[0].reset_shape((B, R, N + M * U, 3), True)
        outputs[1].reset_shape((B, R, N + M * U + 1, 1), True)

        Nb = self.conf.renderer.n_bg_samples
        outputs[2].reset_shape((B, R, Nb, 4), True)
        outputs[3].reset_shape((B, R, Nb + 1, 1), True)

        outputs[4].reset_shape((B, R, 1, 1), True)
        
    def t_near_far(self, camloc, raydir):
        if self.conf.renderer.t_near_far_method == "intersect_with_r_sphere":
            t_near, t_far, mask = self._intersect_with_r_sphere(camloc, raydir)
        elif self.conf.renderer.t_near_far_method == "intersect_with_aabb":
            t_near, t_far, mask = self._intersect_with_aabb(camloc, raydir)
        elif self.conf.renderer.t_near_far_method == "intersect_with_midpoint":
            t_near, t_far, mask = self._intersect_with_midpoint(camloc, raydir)
        elif self.conf.renderer.t_near_far_method == "intersect_with_camloc_dists":
            t_near, t_far, mask = self._intersect_with_camloc_dists(camloc, raydir)
        else:
            raise ValueError(f"{self.conf.renderer.t_near_far_method} is not supported.")
        return t_near, t_far, mask

    def _intersect_with_r_sphere(self, camloc, raydir):
        from intersection.ray_sphere_intersection import \
            ray_sphere_intersection
        radius = self.conf.renderer.bounding_sphere_radius
        t_near, t_far, n_hits = ray_sphere_intersection(camloc, raydir, radius)
                
        mask = F.greater_scalar(n_hits, 1.0)
        return t_near, t_far, mask

    def _intersect_with_aabb(self, camloc, raydir):
        from intersection.ray_aabb_intersection import ray_aabb_intersection
        radius = self.conf.renderer.bounding_sphere_radius
        min = [-radius, -radius, -radius]
        max = [radius, radius, radius]
        t_near, t_far, n_hits = ray_aabb_intersection(camloc, raydir, min, max)
        
        mask = F.greater_scalar(n_hits, 1.0)
        return t_near, t_far, mask

    def _intersect_with_midpoint(self, camloc, raydir):
        radius = self.conf.renderer.bounding_sphere_radius
        
        B, R, _, _ = raydir.shape
        BR = B * R

        camloc = F.broadcast(F.reshape(camloc, (B, 1, 3)), (B, R, 3))

        a = 1.0  # raydir is already normalized
        b = 2.0 * F.batch_matmul(F.reshape(camloc, (BR, 1, 3)),
                                 F.reshape(raydir, (BR, 3, 1)))
        midpoint = - b / (2 * a)
        t_near = midpoint - self.conf.renderer.bounding_sphere_radius
        t_near = F.maximum_scalar(t_near, 0)        
        t_near = F.reshape(t_near, (B, R, 1))
        t_far = midpoint + self.conf.renderer.bounding_sphere_radius
        t_far = F.reshape(t_far, (B, R, 1))

        mask = F.constant(1.0, (B, R, 1))
        return t_near, t_far, mask

    def _intersect_with_camloc_dists(self, camloc, raydir):
        radius = self.conf.renderer.bounding_sphere_radius
        
        B, R, _ = raydir.shape

        camloc_dists = F.norm(camloc, axis=camloc.ndim - 1, keepdims=True)
        t_near = camloc_dists - self.conf.renderer.bounding_sphere_radius
        t_near = F.reshape(t_near, (B, 1, 1))
        t_near = F.broadcast(t_near, (B, R, 1))
        t_far = camloc_dists + self.conf.renderer.bounding_sphere_radius
        t_far = F.reshape(t_far, (B, 1, 1))
        t_far = F.broadcast(t_far, (B, R, 1))

        mask = F.constant(1.0, (B, R, 1)).data
        return t_near, t_far, mask

    def sample_stratified_dists(self, t_near, t_far, stratified_sample):
        """
        Stratified sampling produces the following distances first, 
        
            |--------| ... |--------|--------|
        i = 0        1            N - 1      N
        t = tn     tn + s         tf - s     tf

        where
        t = tn + (tf - tn) / N * i
        s = (tf - tn) / N

        Then, sample one distance uniformly between adjacent intervals.
        """
        B, R, _ = t_far.shape
        t_near = F.reshape(t_near, (B, R, 1, 1))
        t_far = F.reshape(t_far, (B, R, 1, 1))
        N = self.conf.renderer.n_samples0

        step = (t_far - t_near) / N
        t = F.arange(0, N)
        t = F.reshape(t, (1, 1, N, 1))
        # t = t_near + step * (t + F.rand(0, 1, (B, R, N, 1)))
        t = t_near + step * (t + stratified_sample)

        return t

    def sample_importance_dists(self, camloc, raydir, t_near, t_far, t):
        B, R, N, _ = t.shape
        M = self.conf.renderer.n_samples1
        U = self.conf.renderer.n_upsamples

        camloc = F.reshape(camloc, (B, 1, 1, 3))
        raydir = F.reshape(raydir, (B, R, 1, 3))
        t_near = F.reshape(t_near, (B, R, 1, 1))
        t_far = F.reshape(t_far, (B, R, 1, 1))
        t = F.reshape(t, (B, R, N, 1))
        sampling_sigmoid_gain = self.conf.renderer.sampling_sigmoid_gain
        
        def compute_u():
            if self.conf.renderer.deterministic:
                # [0, 1) is must since F.searchsorted returns N if u = 1.
                u = F.arange(0, M) / (M - 1 + 1 / M)
                u = F.reshape(u, (1, 1, M))
                u = F.broadcast(u, (B, R, M))
                return u
            else:
                return F.rand(0, 1, (B, R, M))

        for u in range(U):

            eps = self.conf.renderer.eps
            x = camloc + t * raydir
            sdf, _, _ = geometric_network(x, self.conf)

            N_ts = t.shape[2]
            ts_end = t[:, :, N_ts-1:N_ts, :]

            # robust sampling
            sdf0, sdf1 = sdf[:, :, :-1, :], sdf[:, :, 1:, :]
            t0, t1 = t[:, :, :-1, :], t[:, :, 1:, :]
            sdfm = (sdf0 + sdf1) * 0.5
            cos_val1 = (sdf1 - sdf0) / (t1 - t0 + 1e-5)
            cos_val0 = F.concatenate(*[F.constant(1, (B, R, 1, 1)), cos_val1[:, :, :-1, :]], axis=2)
            cos_val = F.stack(*[cos_val0, cos_val1], axis=-1)
            cos_val = F.min(cos_val, axis=-1, keepdims=False)
            cos_val = F.clip_by_value(cos_val, -1e3, 0.0)
            
            dist = t1 - t0
            sdf0 = sdfm - cos_val * dist * 0.5
            sdf1 = sdfm + cos_val * dist * 0.5

            # weights construction
            gain = sampling_sigmoid_gain * 2 ** u
            cdf0 = F.sigmoid(sdf0 * gain)
            cdf1 = F.sigmoid(sdf1 * gain)
            alpha = F.clip_by_value((cdf0 - cdf1 + 1e-5) / (cdf0 + 1e-5), 0.0, 1.0)
            weights = alpha * F.cumprod(1 - alpha, axis=2, exclusive=True)
            weights = F.reshape(weights, weights.shape[:-1])
                
            # inverse transform sampling
            weights = weights / F.sum(weights, axis=2, keepdims=True)
            cumsum_weights = F.cumsum(weights, axis=2)
            u = compute_u()
            idx = F.searchsorted(cumsum_weights, u)
            cumsum_weights = F.concatenate(*[F.constant(0, (B, R, 1)), cumsum_weights])
            denorm = F.gather(weights, idx, axis=2, batch_dims=2)
            lower = F.gather(cumsum_weights, idx, axis=2, batch_dims=2)
            raito = (u - lower) / denorm
            ratio = F.reshape(raito, (B, R, M, 1))

            steps = t[:, :, 1:, :] - t[:, :, :-1, :]
            steps = F.concatenate(*[steps, t_far - ts_end], axis=2)
            steps_idx = F.gather(steps, idx, axis=2, batch_dims=2)
            ts_idx = F.gather(t, idx, axis=2, batch_dims=2)

            t_prev = t
            t = ts_idx + steps_idx * ratio
            t = F.clip_by_value(t, t_near, t_far) 
            t = F.concatenate(*[t_prev, t], axis=2)
            t = F.sort(t, axis=2)

        return t

    def sample_outside_dists(self, t_base, background_sample):
        B, R, _ = t_base.shape
        M = self.conf.renderer.n_bg_samples
        
        t_base = F.reshape(t_base, (B, R, 1, 1))
        # u = F.rand(1e-5, 1, (B, R, M + 1, 1))
        u = background_sample
        t = t_base / u
        t = F.sort(t, axis=2)

        return t
        
    def forward_impl(self, inputs, outputs):
        ## ctx0 = nn.get_current_context()
        ## ctx = get_extension_context("cudnn", device_id=ctx0.device_id, type_config="half")
        ## with nn.auto_forward(), nn.context_scope(ctx):
        ##     self._forward_impl(inputs, outputs)

        with nn.auto_forward():
            self._forward_impl(inputs, outputs)

    def _forward_impl(self, inputs, outputs):
        camloc = inputs[0].data
        raydir = inputs[1].data
        stratified_sample = inputs[2].data
        background_sample = inputs[3].data

        B, R, _ = raydir.shape

        t_near, t_far, mask = self.t_near_far(camloc, raydir)
        t = self.sample_stratified_dists(t_near, t_far, stratified_sample)
        t = self.sample_importance_dists(camloc, raydir, t_near, t_far, t)
        x = F.reshape(camloc, (B, 1, 1, 3)) + t * F.reshape(raydir, (B, R, 1, 3))
        outputs[0].data.copy_from(x)

        t = F.concatenate(*[t, F.reshape(t_far, (B, R, 1, 1))], axis=2)
        outputs[1].data.copy_from(t)

        if self.conf.background_modeling:
            t_near_bg, t_far_bg, _ = self._intersect_with_camloc_dists(camloc, raydir)
            t_base = t_far * mask + t_near_bg * (1 - mask)
            # t_base = t_far * mask + t_far_bg * (1 - mask)
            t = self.sample_outside_dists(t_base, background_sample)
            x_bg = F.reshape(camloc, (B, 1, 1, 3)) + t[:, :, :-1, :] * F.reshape(raydir, (B, R, 1, 3))
            dists = F.norm(x_bg, axis=3, keepdims=True) + 1e-6
            x_bg = F.concatenate(*[x_bg / dists, 1.0 / dists], axis=-1)
            outputs[2].data.copy_from(x_bg)
            outputs[3].data.copy_from(t)

        else:
            # dummy
            M = self.conf.renderer.n_bg_samples
            outputs[2].data.copy_from(F.constant(1.0, (B, R, M, 4)).data)
            outputs[3].data.copy_from(F.constant(1.0, (B, R, M + 1, 1)).data)

        outputs[4].data.copy_from(mask)            

    def backward_impl(self, inputs, outputs, propergate_down, accum):
        pass

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def sample_points(camloc, raydir, stratified_sample, background_sample, conf, 
                  ctx=None):
    func = SamplePoints(ctx, conf)
    return func(camloc, raydir, stratified_sample, background_sample)


class SampleDirections(PythonFunction):
    """
    """

    def __init__(self, ctx, eps):
        super(SampleDirections, self).__init__(ctx)
        self._eps = eps

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return dict(eps=self._eps)
                
    def min_outputs(self):
        return 1
        
    def setup_impl(self, inputs, outputs):
        """
        inputs: 
            normal: (B, R, 3)
            cdf_the: (B, R, n_thes)
            cdf_phi: (B, R, n_phis)
            alpha (optional): (B, R, 1)
        outputs:
            light_dirs: (B, R, M, 3)
        """
        B, R, _ = inputs[0].shape
        n_thes = inputs[1].shape[-1]
        n_phis = inputs[2].shape[-1]
        M = n_thes * n_phis
        outputs[0].reset_shape((B, R, M, 3), True)
        
    def forward_impl(self, inputs, outputs):
        with nn.auto_forward():
            self._forward_impl(inputs, outputs)

    def _forward_impl(self, inputs, outputs):
        B, R, _ = inputs[0].shape
        n_thes = inputs[1].shape[-1]
        n_phis = inputs[2].shape[-1]

        light_dirs = outputs[0]
        normal = inputs[0]
        cdf_the = inputs[1]
        cdf_phi = inputs[2]
        
        M = n_thes * n_phis
        size = B * R * M
        light_dirs_ptr = light_dirs.data.data_ptr(np.float32, self.ctx)
        normal_ptr = normal.data.data_ptr(np.float32, self.ctx)
        cdf_the_ptr = cdf_the.data.data_ptr(np.float32, self.ctx)
        cdf_phi_ptr = cdf_phi.data.data_ptr(np.float32, self.ctx)
        batch_size = B * R
        n_lights = M
        eps = self._eps

        if len(inputs) == 3:
            inverse_transform_cuda.sample_uniform_directions(
                                            size, light_dirs_ptr, 
                                            normal_ptr, cdf_the_ptr, cdf_phi_ptr, 
                                            batch_size, n_lights, n_thes, n_phis, eps)
        elif len(inputs) == 4:
            alpha = inputs[3]
            alpha_ptr = alpha.data.data_ptr(np.float32, self.ctx)
            inverse_transform_cuda.sample_importance_directions(
                                            size, light_dirs_ptr, 
                                            normal_ptr, cdf_the_ptr, cdf_phi_ptr, alpha_ptr, 
                                            batch_size, n_lights, n_thes, n_phis, eps)
        else:
            raise ValueError("Input lenght must be either 3 or 4")

    def backward_impl(self, inputs, outputs, propergate_down, accum):
        pass

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False


def sample_uniform_directions(normal, cdf_the, cdf_phi, eps=0.0, ctx=None):
    func = SampleDirections(ctx, eps)
    return func(normal, cdf_the, cdf_phi)


def sample_importance_directions(normal, cdf_the, cdf_phi, alpha, eps=0.0, ctx=None):
    func = SampleDirections(ctx, eps)
    return func(normal, cdf_the, cdf_phi, alpha)
