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
import nnabla.functions as F
import numpy as np
from tqdm import tqdm

from helper import generate_all_pixels, generate_raydir_camloc
from network import (background_network, base_color_network,
                     environment_light_network, geometric_network,
                     implicit_illumination_network,
                     photogrammetric_light_network, roughness_network,
                     soft_visibility_light_network,
                     specular_reflectance_network)
from sampler import (sample_importance_directions, sample_points,
                     sample_uniform_directions)
from specular_brdf import dot, specular_brdf_model


def pb_render(x_fg, t_fg, x_bg, t_bg, camloc, raydir, mask, cos_anneal_ratio, conf):
    """
    Args:
      x_fg: Points on rays for foreground scene (B, R, N, 3)
      t_fg: Points on rays for foreground scene (B, R, N, 1)
      x_bg: Points on rays for background scene (B, R, M, 4)
      t_bg: Points on rays for foreground scene (B, R, M, 1)
      camloc: Camera location (B, 1, 3)
      raydir: Ray direction (B, R, 3)
      mask: Mask of ray-<sphere, AABB, boundary> intersection (B, R, 1, 1)
      cos_anneal_ratio: (1, )
    """

    B, R, N, _ = x_fg.shape
    raydir = F.reshape(raydir, (B, R, 1, 3))
    view_dir = -raydir
    eps_normal = conf.renderer.eps_normal

    # Geometric network
    sdf_x_fg, feature_x_fg, gain = geometric_network(x_fg, conf)
    grad_x_fg = nn.grad([sdf_x_fg], [x_fg])[0]
    
    # Foreground alpha
    cos_anneal_ratio = F.reshape(cos_anneal_ratio, [1] * x_fg.ndim)
    true_cos = F.sum(raydir * grad_x_fg, axis=-1, keepdims=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    delta_t_fg = t_fg[:, :, 1:, :] - t_fg[:, :, :-1, :]
    sdf_x_fg1 = sdf_x_fg + iter_cos * delta_t_fg * 0.5
    sdf_x_fg0 = sdf_x_fg - iter_cos * delta_t_fg * 0.5

    gain = gain.reshape([1] * sdf_x_fg.ndim)
    cdf0 = F.sigmoid(gain * sdf_x_fg0)
    cdf1 = F.sigmoid(gain * sdf_x_fg1)
    alpha = F.clip_by_value((cdf0 - cdf1 + 1e-5) / (cdf0 + 1e-5), 0.0, 1.0)
    alpha_fg = alpha

    # Background alpha
    if conf.background_modeling:
        delta_bg = t_bg[:, :, 1:, :] - t_bg[:, :, :-1, :]
        delta_bg = delta_bg.apply(need_grad=False)
        alpha_bg, color_bg = background_network(x_bg, view_dir, delta_bg, conf)
    else:
        alpha_bg, color_bg = F.constant(1, (B, R, 1, 1)), \
                            F.constant(conf.background_color, (B, R, 1, 3))

    # Wegiths
    alpha = F.concatenate(*[alpha_fg * mask, alpha_bg], axis=2)
    trans = F.cumprod(1 - alpha, axis=2, exclusive=True)
    weights = alpha * trans
    trans_fg = trans[:, :, :N, :]
    weights_fg = weights[:, :, :N, :]
    weights_bg = weights[:, :, N:, :]

    def VR(x, weights=weights_fg, axis=2):
        return F.sum(weights * x, axis=axis)

    # Normal
    grad_x_fg_pixel = VR(grad_x_fg) + eps_normal
    normal_pixel = grad_x_fg_pixel / F.norm(grad_x_fg_pixel, axis=-1, keepdims=True)

    # Light conf
    n_thetas = conf.renderer.n_thetas
    n_phis = 2 * conf.renderer.n_thetas

    cdf_the = F.rand(0.0, 1.0, (B, R, n_thetas), seed=conf.renderer.diffuse_cdf_the_seed)
    cdf_phi = F.rand(0.0, 1.0, (B, R, n_phis), seed=conf.renderer.diffuse_cdf_phi_seed)
    M = n_thetas * n_phis

    D = feature_x_fg.shape[-1]
    x_fg_pixel = F.broadcast(F.reshape(VR(x_fg), (B, R, 1, 3)), (B, R, M, 3))
    feature_x_fg_pixel = F.broadcast(F.reshape(VR(feature_x_fg), (B, R, 1, D)), (B, R, M, D))

    # Direct light
    uniform_light_dir = sample_uniform_directions(normal_pixel, cdf_the, cdf_phi)
    environment_light_intensity = environment_light_network(uniform_light_dir, conf)

    # Direct ligt visibility
    soft_vis = soft_visibility_light_network(x_fg_pixel, uniform_light_dir, feature_x_fg_pixel, F.broadcast(normal_pixel[:, :, None, :], (B, R, M, 3)), conf)
    
    # Implicit light
    implicit_light_intensity = implicit_illumination_network(x_fg, feature_x_fg, grad_x_fg, conf)
    implicit_light_intensity_pixel = VR(implicit_light_intensity)

    # Diffuse color
    cos = dot(F.broadcast(normal_pixel[:, :, None, :], (B, R, M, 3)), uniform_light_dir)
    environment_light_intensity = F.mean(soft_vis * environment_light_intensity * cos, axis=2)
    diffuse_light_intensity_pixel = environment_light_intensity + implicit_light_intensity_pixel
    base_color = base_color_network(x_fg, feature_x_fg, grad_x_fg, conf)

    # Roughness
    roughness, std_roughness = roughness_network(x_fg, feature_x_fg, grad_x_fg, conf)
    roughness_pixel = VR(roughness)

    # Specular reflectance
    specular_reflectance, std_specular_reflectance = specular_reflectance_network(x_fg, feature_x_fg, grad_x_fg, conf)
    specular_reflectance_pixel = VR(specular_reflectance)

    # Specular color
    cdf_the = F.rand(0.0, 1.0, (B, R, n_thetas), seed=conf.renderer.specular_cdf_the_seed)
    cdf_phi = F.rand(0.0, 1.0, (B, R, n_phis), seed=conf.renderer.specular_cdf_phi_seed)
    if conf.specular_brdf.sampling == "importance":
        importance_light_dir = sample_importance_directions(normal_pixel, cdf_the, cdf_phi, roughness_pixel)
        sBRDF, cos = specular_brdf_model(normal_pixel, view_dir, importance_light_dir, \
                                         roughness_pixel, specular_reflectance_pixel, conf)
    elif conf.specular_brdf.sampling == "uniform":
        importance_light_dir = sample_uniform_directions(normal_pixel, cdf_the, cdf_phi)
        sBRDF, cos = specular_brdf_model(normal_pixel, view_dir, importance_light_dir, \
                                         roughness_pixel, specular_reflectance_pixel, conf)
        
    environment_light_intensity = environment_light_network(importance_light_dir, conf)
    soft_vis = soft_visibility_light_network(x_fg_pixel, importance_light_dir, feature_x_fg_pixel, 
                                             F.broadcast(normal_pixel[:, :, None, :], (B, R, M, 3)), conf)
    
    if conf.specular_brdf.use_split_sum: 
        specular_color_pixel = F.mean(soft_vis * environment_light_intensity, axis=2) \
            * F.mean(sBRDF * cos, axis=2)
    else:
        specular_color_pixel = \
            F.mean(sBRDF * soft_vis * environment_light_intensity * cos, axis=2)

    use_me_on_specular = conf.implicit_illumination_network.use_me \
        and conf.implicit_illumination_network.use_me_on_specular
    if use_me_on_specular:
        specular_color_pixel = specular_color_pixel \
            + F.mean(sBRDF * implicit_light_intensity_pixel[:, :, :, None], axis=2)

    # Diffuse + specular BRDF
    specular_color_pixel = conf.specular_brdf.weight * specular_color_pixel
    if conf.photogrammetric_light_network.use_me: 
        # Light photogrammetry
        photogrammetric_light_intensity = photogrammetric_light_network(x_fg, camloc, view_dir, feature_x_fg, grad_x_fg, conf)
        photogrammetric_light_intensity_pixel = VR(photogrammetric_light_intensity)
        
        if conf.diffuse_brdf.entangle: 
            diffuse_color_pixel = VR(base_color * photogrammetric_light_intensity)
            color_fg_pixel = diffuse_color_pixel * diffuse_light_intensity_pixel
            color_fg_pixel = color_fg_pixel + photogrammetric_light_intensity_pixel * specular_color_pixel
        else:
            diffuse_color_pixel = VR(base_color)
            diffuse_color_pixel = diffuse_color_pixel * diffuse_light_intensity_pixel
            color_fg_pixel = photogrammetric_light_intensity_pixel * (diffuse_color_pixel + specular_color_pixel)
    else:
        diffuse_color_pixel = VR(base_color)
        color_fg_pixel = diffuse_color_pixel + specular_color_pixel

    # Foreground + background
    color_bg_pixel = VR(color_bg, weights_bg)
    color_pixel =  color_fg_pixel + color_bg_pixel

    # Mask
    obj_mask_pred = nn.Variable.from_numpy_array(0.0)
    if conf.train.mask_weight > 0.0:
        obj_mask_pred = F.sum(alpha_fg * trans_fg, axis=2)

    # Diffuse color perturbation
    G = conf.geometric_network.voxel.grid_size
    seed = conf.train.base_color_perturb_seed
    r = conf.renderer.bounding_sphere_radius
    x_fg_ptb = x_fg + F.randn(0.0, 1.0, x_fg.shape, seed=seed) * (np.sqrt(3) * 2 * r / G)
    _, feature_x_fg_ptb, _ = geometric_network(x_fg_ptb, conf)
    base_color_ptb = base_color_network(x_fg_ptb, feature_x_fg_ptb, None, conf) # normal is not used

    res = dict(
        color_pixel=color_pixel,
        sdf_x_fg=sdf_x_fg,
        grad_x_fg=grad_x_fg,
        alpha_fg=alpha_fg,
        trans_fg=trans_fg,
        obj_mask_pred=obj_mask_pred,
        base_color=base_color,
        base_color_ptb=base_color_ptb,
        roughness=roughness,
        specular_reflectance=specular_reflectance,
        std_roughness=std_roughness,
        std_specular_reflectance=std_specular_reflectance
    )
    return res
    

def render_image(pose, intrinsic, resolution, conf):
    """
    Args:
        pose: Camera pose (1, 4, 4)
        intrinsic: Camera intrinsic (1, 3, 3)
        resolution: Image width and height
    """
    import nnabla_ext.cuda
    nnabla_ext.cuda.clear_memory_cache()
    
    scale = 1.0 / 2 ** conf.valid.n_down_samples
    W, H = resolution
    W, H = int(W * scale), int(H * scale)
    P = conf.valid.n_rays

    intrinsic = intrinsic.copy()
    intrinsic[:, 0, 0] = intrinsic[:, 0, 0] * scale
    intrinsic[:, 1, 1] = intrinsic[:, 1, 1] * scale
    intrinsic[:, 0, 2] = intrinsic[:, 0, 2] * scale
    intrinsic[:, 1, 2] = intrinsic[:, 1, 2] * scale
    intrinsic[:, 0, 1] = intrinsic[:, 0, 1] * scale

    xy = generate_all_pixels(W, H)
    xy = xy.reshape((1, H * W, 2))

    _, m = divmod((W * H), P)
    P_prev = P
    P = P - m
    nn.logger.info(
        f"P is changed due to reminder ({m} = mod({W} x {H}, {P_prev})): {P_prev} --> {P}")

    camloc = nn.Variable([1, 3])
    raydir = nn.Variable([1, P, 3])

    with nn.auto_forward(False):
        # RGB
        N = conf.renderer.n_samples0
        M = conf.renderer.n_bg_samples
        stratified_sample = F.rand(0, 1, (1, P, N, 1), conf.renderer.stratified_sample_seed)
        background_sample = F.rand(1e-5, 1, (1, P, M + 1, 1), conf.renderer.background_sample_seed)
        x_fg, t_fg, x_bg, t_bg, mask = sample_points(camloc, raydir, stratified_sample, background_sample, conf)
        cos_anneal_ratio = F.constant(1.0, [1] * x_fg.ndim)
        x_fg.apply(need_grad=True)
        res = pb_render(x_fg, t_fg, x_bg, t_bg, camloc, raydir, mask, cos_anneal_ratio, conf)
        color_pixel = res["color_pixel"]
        color_pixel = color_pixel.reshape((1, P, 3))

    rimage = np.ndarray([1, H * W, 3])
    for p in tqdm(range(0, H * W, P), desc="Rendering images"):
        xy_b = xy[:, p:p+P, :].reshape((1, P, 2))
        raydir.d, camloc.d = generate_raydir_camloc(pose, intrinsic, xy_b)
        
        color_pixel.forward(clear_buffer=True)
        rimage[0, p:p+P, :] = color_pixel.d.copy()

    rimage = rimage.reshape((1, H, W, 3)).transpose((0, 3, 1, 2))  # NCHW
    rimage = np.clip(rimage, 0.0, 1.0)

    nnabla_ext.cuda.clear_memory_cache()
    
    return rimage
