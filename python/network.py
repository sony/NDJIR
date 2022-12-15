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

from functools import partial

import nnabla as nn
import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.random import prng

# these imports add functions to F and/or PF namespaces
import grid_feature.cosine_triline_feature
import grid_feature.cosine_triplane_feature
import grid_feature.cosine_voxel_feature
import grid_feature.lanczos_triline_feature
import grid_feature.lanczos_triplane_feature
import grid_feature.lanczos_voxel_feature
import grid_feature.triline_feature
import grid_feature.triplane_feature
import grid_feature.voxel_feature


class GeometricInitializer(I.BaseInitializer):
    """
    """

    def __init__(self, Di, Do, sigma, zero_start=None, last=False):
        self.Di = Di
        self.Do = Do
        self.sigma = sigma
        self.zero_start = zero_start
        self.last = last
        
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__)

    def __call__(self, shape):
        w_init = np.sqrt(self.sigma) * prng.randn(*[self.Di, self.Do])
        if self.zero_start is not None:
            w_init[self.zero_start:, :] = 0.0
        if self.last:
            w_init[:, 0] = np.sqrt(np.pi / self.Di) * np.ones([self.Di]) + prng.randn(self.Di) * 1e-4
        return w_init
    

def weight_normalization_backward(inputs, dim=1, eps=1e-12):
    # We do not need the gradients wrt weight and scale by nn.grad,
    # since nn.grad is performed wrt the intersection point.
    ## dw_wn = inputs[0]
    ## w0 = inputs[1]
    ## g0 = inputs[2]
    return None, None


nn.backward_functions.register(
    "WeightNormalization", weight_normalization_backward)


def norm_normalization_backward(inputs, p, axes, eps=1e-12):
    # We do not need the gradients wrt norm and scale by nn.grad,
    # since nn.grad is performed wrt the intersection point.
    ## dw_wn = inputs[0]
    ## w0 = inputs[1]
    return None


nn.backward_functions.register(
    "NormNormalization", norm_normalization_backward)


def norm(x, axis):
    return F.sum(x ** 2, axis, keepdims=True) ** 0.5


def affine(h, D, use_wn, w_init=None, b_init=None, name=None):
    with nn.parameter_scope(name):
        apply_w = partial(PF.weight_normalization, dim=1) if use_wn else None
        h = PF.affine(h, D, base_axis=h.ndim - 1, apply_w=apply_w, 
                      w_init=w_init, b_init=b_init, with_bias=True)# if b_init else False)
    return h


def positional_encoding(x, M=6, include_input=True):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      M: Number of bands
    """

    base = 2.0

    bands = (base ** np.arange(0, M))
    data_holder = nn.Variable if isinstance(x, nn.Variable) else nn.NdArray
    bands = data_holder.from_numpy_array(bands)
    bands = F.reshape(bands, tuple([1] * x.ndim) + (M, )) \
        * F.reshape(x, x.shape + (1, ))
    bands = F.reshape(bands, bands.shape[:-2] + (-1, ))
    cos_x = F.cos(bands)
    sin_x = F.sin(bands)

    gamma = [x, cos_x, sin_x] if include_input else [cos_x, sin_x]
    gamma = F.concatenate(*gamma, axis=-1)

    return gamma


def query_on_grid(x, G, D, use_ste, type):
    if type == "none":
        return None

    if type == "triplaneline":
        feat0 = PF.query_on_triplane(x, G, D, use_ste=use_ste)
        feat1 = PF.query_on_triline(x, G, D, use_ste=use_ste)
        return F.concatenate(*[feat0, feat1], axis=-1)

    if type == "cosine_triplaneline":
        feat0 = PF.cosine_query_on_triplane(x, G, D, use_ste=use_ste)
        feat1 = PF.cosine_query_on_triline(x, G, D, use_ste=use_ste)
        return F.concatenate(*[feat0, feat1], axis=-1)

    if type == "lanczos_triplaneline":
        feat0 = PF.lanczos_query_on_triplane(x, G, D, use_ste=use_ste)
        feat1 = PF.lanczos_query_on_triline(x, G, D, use_ste=use_ste)
        return F.concatenate(*[feat0, feat1], axis=-1)

    func_dict = dict(
        voxel=PF.query_on_voxel,
        triplane=PF.query_on_triplane,
        triline=PF.query_on_triline,
        cosine_voxel=PF.cosine_query_on_voxel,
        cosine_triplane=PF.cosine_query_on_triplane,
        cosine_triline=PF.cosine_query_on_triline,
        lanczos_voxel=PF.lanczos_query_on_voxel, 
        lanczos_triplane=PF.lanczos_query_on_triplane, 
        lanczos_triline=PF.lanczos_query_on_triline
    )
    func = func_dict[type]
    return func(x, G, D, use_ste=use_ste)


@nn.parameter_scope("geometric-network")
def geometric_network(x, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
    """
    D = conf.geometric_network.feature_size
    L = conf.geometric_network.layers
    M = conf.geometric_network.pe_bands
    act = conf.geometric_network.act
    act_map = dict(relu=F.relu,
                   softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn

    r = conf.renderer.bounding_sphere_radius
    skip_layers = conf.geometric_network.skip_layers

    type = conf.geometric_network.voxel.type
    use_ste = conf.geometric_network.voxel.use_ste
    G = conf.geometric_network.voxel.grid_size
    D0 = conf.geometric_network.voxel.feature_size

    pe_x = positional_encoding(x, M) if M > 0 else x
    vfeat = query_on_grid(x, G, D0, use_ste, type)
    inputs = [pe_x]
    inputs = inputs + [vfeat] if vfeat is not None else inputs
    inputs = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else inputs[0]
    h = inputs

    if not conf.geometric_network.geometric_init:
        for l in range(L - 1):
            h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
            h = F.concatenate(*[h, pe_x]) if l in skip_layers else h
            h = act(h)
        h = affine(h, D + 1, use_wn=use_wn, name=f"affine-{L - 1:02d}")
        sdf, feature = h[..., 0:1], h[..., 1:]
    else:
        initial_sphere_radius = conf.geometric_network.initial_sphere_radius
        Dx = x.shape[-1]
        Dinputs = inputs.shape[-1]
        for l in range(L):
            # First
            if l == 0:
                Dh = h.shape[-1]
                w_init = GeometricInitializer(Dh, D, 2 / D, Dx)
                h = affine(h, D, use_wn=use_wn, w_init=w_init, name=f"affine-{l:02d}")
                h = act(h)
            # Skip
            elif l in skip_layers:
                w_init = GeometricInitializer(D, D, 2 / (D - Dinputs), -Dinputs)
                h = affine(h, D, use_wn=use_wn, w_init=w_init, name=f"affine-{l:02d}")
                h = act(h)
            # Last (scalar + feature_size)
            elif l == L - 1:
                Do = 1 + D
                w_init = GeometricInitializer(D, Do, 2 / Do, last=True)
                b_init = I.ConstantInitializer(-initial_sphere_radius)
                h = affine(h, Do, use_wn=use_wn,
                           w_init=w_init, b_init=b_init, 
                           name=f"affine-last")
            # Intermediate
            else:
                Do = D - Dinputs if l + 1 in skip_layers else D
                w_init = GeometricInitializer(h.shape[-1], Do, 2 / Do)
                h = affine(h, Do, use_wn=use_wn, w_init=w_init, name=f"affine-{l:02d}")
                h = act(h)
                if conf.geometric_network.use_inv_square: 
                    h = F.concatenate(*[h, inputs]) / np.sqrt(2) if l + 1 in skip_layers else h
                else:
                    h = F.concatenate(*[h, inputs]) if l + 1 in skip_layers else h
        sdf, feature = h[..., 0:1], h[..., 1:]
            
    gain = nn.parameter.get_parameter_or_create("gain", (1, ),
                                                np.asarray([conf.train.sigmoid_gain]),
                                                True, True)
    gain = F.exp(gain * 10)
    gain = F.clip_by_value(gain, 1e-6, 5e4)
    return sdf, feature, gain


@nn.parameter_scope("base-color-network")
def base_color_network(x, feature, normal, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      feature: Geometric feature (B, R, N, .) or (B, 3)
      normal: Normal of geometry (B, R, N, 3) or (B, 3)
    """
    D = conf.base_color_network.feature_size
    L = conf.base_color_network.layers
    act = conf.base_color_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    use_geometric_feature = conf.base_color_network.use_geometric_feature
    use_normal = conf.base_color_network.use_normal

    inputs = [x]
    inputs = inputs + [feature] if use_geometric_feature else inputs
    inputs = inputs + [normal] if use_normal else inputs
    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x

    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
        h = act(h)
    h = affine(h, 3, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    base_color = F.sigmoid(h)

    return base_color


@nn.parameter_scope("environment-light-network")
def environment_light_network(light_dirs, conf):
    """
    Args:
      light_dirs: Light direction (B, R, M, 3) or (B, 3)
    """
    D = conf.environment_light_network.feature_size
    L = conf.environment_light_network.layers
    M0 = conf.environment_light_network.pe_bands
    act = conf.environment_light_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    Do = conf.environment_light_network.channels

    pe_light_dirs = positional_encoding(light_dirs, M0) if M0 > 0 else light_dirs
    inputs = pe_light_dirs
    h = inputs

    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
        h = act(h)
    h = affine(h, Do, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    act = dict(softplus=lambda x: F.softplus(x, conf.environment_light_network.inverse_black_degree), 
               relu=F.relu,
               sigmoid=F.sigmoid)
    environment_light_intensity = act[conf.environment_light_network.act_last](h)
    upper_bound = conf.environment_light_network.upper_bound
    if upper_bound > 0: 
        environment_light_intensity = F.clip_by_value(environment_light_intensity, 0.0, upper_bound)

    return environment_light_intensity


@nn.parameter_scope("implicit-illumination-network")
def implicit_illumination_network(x, feature, normal, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      feature: Geometric feature (B, R, N, .) or (B, 3)
      normal: Normal of geometry (B, R, M, 3) or (B, 3)
    """
    if not conf.implicit_illumination_network.use_me:
        return F.constant(0.0, x.shape[:-1] + (1, ))

    D = conf.implicit_illumination_network.feature_size
    L = conf.implicit_illumination_network.layers
    act = conf.implicit_illumination_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    Do = conf.implicit_illumination_network.channels

    use_geometric_feature = conf.implicit_illumination_network.use_geometric_feature
    use_normal = conf.implicit_illumination_network.use_normal

    inputs = [x]
    inputs = inputs + [feature] if use_geometric_feature else inputs
    inputs = inputs + [normal] if use_normal else inputs
    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x

    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
        h = act(h)
    h = affine(h, Do, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    act = dict(softplus=lambda x: F.softplus(x, conf.implicit_illumination_network.inverse_black_degree), 
               relu=F.relu,
               sigmoid=F.sigmoid)
    implicit_illumination = act[conf.implicit_illumination_network.act_last](h)
    
    return implicit_illumination


@nn.parameter_scope("soft-visibility-light-network")
def soft_visibility_light_network(x, light_dirs, feature, normal, conf):
    """
    Args:
      x: Input (B, R, M, 3) or (B, 3)
      light_dirs: Light direction (B, R, M, 3) or (B, 3)
      feature: Geometric feature (B, R, M, .) or (B, 3)
      normal: Normal of geometry (B, R, M, 3) or (B, 3)
    """
    D = conf.soft_visibility_light_network.feature_size
    L = conf.soft_visibility_light_network.layers
    M0 = conf.soft_visibility_light_network.pe_bands
    act = conf.soft_visibility_light_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    Do = conf.soft_visibility_light_network.channels

    use_geometric_feature = conf.soft_visibility_light_network.use_geometric_feature
    use_normal = conf.soft_visibility_light_network.use_normal

    pe_light_dirs = positional_encoding(light_dirs, M0) if M0 > 0 else light_dirs

    inputs = [x]
    inputs = inputs + [pe_light_dirs]
    inputs = inputs + [feature] if use_geometric_feature else inputs
    inputs = inputs + [normal] if use_normal else inputs
    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x

    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
        h = act(h)
    h = affine(h, Do, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    act = dict(softplus=lambda x: F.softplus(x, conf.soft_visibility_light_network.inverse_black_degree), 
               relu=F.relu,
               sigmoid=F.sigmoid)
    light_intensity = act[conf.soft_visibility_light_network.act_last](h)
    
    return light_intensity


@nn.parameter_scope("photogrammetric-light-network")
def photogrammetric_light_network(x, camloc, view, feature, normal, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      camloc: Camera location (B, 1, 3)
      view: Viewing direction (B, R, 1, 3)
      feature: Geometric feature (B, R, N, .) or (B, 3)
      normal: Normal of geometry (B, R, N, 3) or (B, 3)
    """
    D = conf.photogrammetric_light_network.feature_size
    L = conf.photogrammetric_light_network.layers
    M = conf.photogrammetric_light_network.pe_bands
    act = conf.photogrammetric_light_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    use_inverse_distance = conf.photogrammetric_light_network.use_inverse_distance
    Do = conf.photogrammetric_light_network.channels

    B, R, N, _ = x.shape
    view = F.broadcast(view, (B, R, N, 3))
    pe_view = positional_encoding(view, M) if M > 0 else view
    inputs = [x]
    inputs = inputs + [pe_view]
    inputs = inputs + [feature]
    inputs = inputs + [normal]
    camloc = F.reshape(camloc, (B, 1, 1, 3))
    dist2 = F.norm(x - camloc, axis=x.ndim - 1, keepdims=True) ** 2
    inv_dist2 = 1.0 / (dist2 + 1e-5)
    inputs = inputs + [inv_dist2] if use_inverse_distance else inputs
    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x

    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l:02d}")
        h = act(h)
    h = affine(h, Do, use_wn=use_wn, name=f"affine-{L - 1:02d}")

    gain = nn.parameter.get_parameter_or_create("gain", (1, ),
                                                np.asarray([conf.train.sigmoid_gain_lv_start]),
                                                False, False)
    gain = F.reshape(gain, [1] * h.ndim)
    light_visibility = F.sigmoid(gain * h)
        
    return light_visibility


@nn.parameter_scope("roughness-network")
def roughness_network(x, feature, normal, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      feature: Geometric feature (B, R, N, .) or (B, 3)
      normal: Normal of geometry (B, R, N, 3) or (B, 3)
    """
    D = conf.roughness_network.feature_size
    L = conf.roughness_network.layers
    act = conf.roughness_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn

    use_geometric_feature = conf.roughness_network.use_geometric_feature
    use_normal = conf.roughness_network.use_normal

    inputs = [x]
    inputs = inputs + [feature] if use_geometric_feature else inputs
    inputs = inputs + [normal] if use_normal else inputs

    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x
    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l - 1:02d}")
        h = act(h)

    h = affine(h, 2, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    h0, h1 = h[..., 0:1], h[..., 1:2]
    h = h0
    std = F.softplus(h1)

    h = F.sigmoid(h)
    if conf.specular_brdf.model == "filament" and conf.specular_brdf.remap:
        h = h ** 2
    h = F.clip_by_value(h, conf.roughness_network.lower_bound, 1.0)

    return h, std


@nn.parameter_scope("specular-reflectance-network")
def specular_reflectance_network(x, feature, normal, conf):
    """
    Args:
      x: Input (B, R, N, 3) or (B, 3)
      feature: Geometric feature (B, R, N, .) or (B, 3)
      normal: Normal of geometry (B, R, N, 3) or (B, 3)
    """
    if conf.specular_reflectance_network.fixme:
        return F.constant(0.04, x.shape[:-1] + (conf.specular_reflectance_network.channels, ))

    D = conf.specular_reflectance_network.feature_size
    L = conf.specular_reflectance_network.layers
    act = conf.specular_reflectance_network.act
    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    act = act_map[act]
    use_wn = conf.use_wn
    Do = conf.specular_reflectance_network.channels

    use_geometric_feature = conf.specular_reflectance_network.use_geometric_feature
    use_normal = conf.specular_reflectance_network.use_normal

    inputs = [x]
    inputs = inputs + [feature] if use_geometric_feature else inputs
    inputs = inputs + [normal] if use_normal else inputs

    h = F.concatenate(*inputs, axis=-1) if len(inputs) > 1 else x
    for l in range(L - 1):
        h = affine(h, D, use_wn=use_wn, name=f"affine-{l - 1:02d}")
        h = act(h)

    h = affine(h, Do * 2, use_wn=use_wn, name=f"affine-{L - 1:02d}")
    h0, h1 = h[..., :-Do], h[..., Do:]
    h = h0
    std = F.softplus(h1)

    h = F.sigmoid(h)
    if conf.specular_brdf.model == "filament" and conf.specular_brdf.remap:
        h = 0.16 * (h ** 2)
    else:
        h = conf.specular_reflectance_network.upper_bound_scale * h

    return h, std


@nn.parameter_scope("background-network")
def background_network(x, view, delta, conf):
    """
    Args:
      x: Input (B, R, N, 4). 4 means (x, y, z) unit-vector normalized and distance t
      view: Viewing direction (B, R, N, 3)
      delta: Difference between distances of adjacent points (B, R, N, 1)
    """
    B, R, N, _ = x.shape
    D0 = conf.background_network.feature_size0
    D1 = conf.background_network.feature_size1
    L0 = conf.background_network.layers0
    L1 = conf.background_network.layers1
    M0 = conf.background_network.pe_bands0
    M1 = conf.background_network.pe_bands1
    
    use_wn = conf.use_wn
    act_map = dict(relu=F.relu,
                softplus=partial(F.softplus, beta=100))
    act = conf.background_network.act                
    act = act_map[act]

    # geometric network
    with nn.parameter_scope("geometric-network"):
        pe_x = positional_encoding(x, M0) if M0 > 0 else x
        h = pe_x
             
        for l in range(L0 - 1):
            h = affine(h, D0, use_wn=use_wn, name=f"affine-{l:02d}")
            h = act(h)
        h = affine(h, D0 + 1, use_wn=use_wn, name=f"affine-{L0 - 1:02d}")
        # density, feature = F.relu(h[..., 0:1]), h[..., 1:]
        density, feature = F.softplus(h[..., 0:1], beta=100), h[..., 1:]
        alpha = 1 - F.exp(-density * delta)

    # lighting network
    with nn.parameter_scope("lighting-network"):
        view = F.broadcast(view, (B, R, N, 3))
        if M1 > 0:
            pe_view = positional_encoding(view, M1)
            h = F.concatenate(*[x, feature, view, pe_view], axis=-1)
        else:
            h = F.concatenate(*[x, feature, view], axis=-1)
        for l in range(L1 - 1):
            h = affine(h, D1, use_wn=use_wn, name=f"affine-{l:02d}")
            h = act(h)
        h = affine(h, 3, use_wn=use_wn, name=f"affine-{L1 - 1:02d}")
        color = F.sigmoid(h)
    
    return alpha, color
