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

from renderer import pb_render
from sampler import sample_points

import grid_feature.total_variation_loss
import grid_feature.total_variation_loss_on_triline
import grid_feature.total_variation_loss_on_triplane
import grid_feature.total_variation_loss_on_voxel_hash


def total_loss(camloc, raydir, color_gt, obj_mask, cos_anneal_ratio, conf):
    """
    Args:
      camloc: Camera location (B, 1, 3)
      raydir: Ray direction (B, R, 3)
      color_gt: Color ground truth (B, R, 3)
    """

    B, R, _ = color_gt.shape
    N = conf.renderer.n_samples0
    M = conf.renderer.n_bg_samples

    # Points on ray
    stratified_sample = F.rand(0, 1, (B, R, N, 1), conf.renderer.stratified_sample_seed)
    background_sample = F.rand(1e-5, 1, (B, R, M + 1, 1), conf.renderer.background_sample_seed)
    x_fg, t_fg, x_bg, t_bg, mask = sample_points(camloc, raydir, stratified_sample, background_sample, conf)
    x_fg.apply(need_grad=True)
    
    res = pb_render(x_fg, t_fg, x_bg, t_bg, camloc, raydir, mask, cos_anneal_ratio, conf)
    color = res["color_pixel"]
    sdf_x_fg = res["sdf_x_fg"]
    grad_x_fg = res["grad_x_fg"]
    alpha_fg = res["alpha_fg"]
    trans_fg = res["trans_fg"]
    obj_mask_pred = res["obj_mask_pred"]
    base_color = res["base_color"]
    base_color_ptb = res["base_color_ptb"]
    roughness = res["roughness"]
    specular_reflectance = res["specular_reflectance"]
    std_roughness = res["std_roughness"]
    std_specular_reflectance = res["std_specular_reflectance"]

    # RGB loss
    rgb_loss_map = dict(l1=F.absolute_error, l2=F.squared_error)
    rgb_loss = rgb_loss_map[conf.train.rgb_loss]
    loss_rgb = rgb_loss(color, color_gt)
    obj_mask = obj_mask if conf.train.mask_weight > 0.0 else 1.0
    denorm = F.sum(obj_mask) + 1e-5 if conf.train.mask_weight > 0.0 else B * R
    loss_rgb = F.sum(loss_rgb * obj_mask) / denorm
    loss_rgb = loss_rgb.apply(persistent=True)

    # Eikonal loss
    loss_eikonal = nn.Variable.from_numpy_array(0.0)
    if conf.train.eikonal_weight > 0.0:
        grad_eikonal = grad_x_fg
        gn = F.norm(grad_eikonal, keepdims=True, axis=grad_eikonal.ndim - 1)
        _, _, N, _ = grad_eikonal.shape
        denorm = F.sum(mask) * N + 1e-5
        loss_eikonal = F.sum(((gn - 1) * mask) ** 2.0) / denorm
        loss_eikonal = loss_eikonal.apply(persistent=True)

    # TV loss
    loss_tv = nn.Variable.from_numpy_array(0.0)
    tv_loss_map = dict(voxel_feature=F.tv_loss_on_voxel, 
                       voxel_hash_feature=F.tv_loss_on_voxel_hash, 
                       triplane_feature=F.tv_loss_on_triplane, 
                       triline_feature=F.tv_loss_on_triline
                      )
    if conf.geometric_network.voxel.type != "none" and conf.train.tv_weight > 0.0:
        def compute_tv_loss():
            loss_tv = nn.Variable.from_numpy_array(0.0)
            params = nn.get_parameters()
            for param_name, param in params.items():
                if not param_name.endswith("feature/F"):
                    continue

                feature = params[param_name]
                _, _, N, _ = x_fg.shape
                denorm = F.sum(mask) * N + 1e-5
                feature_name = param_name.split("/")[-2]
                tv_loss_on_voxel = tv_loss_map[feature_name]
                sym_backward = conf.train.tv_sym_backward
                loss_tv0 = tv_loss_on_voxel(x_fg, feature, sym_backward=sym_backward)
                loss_tv0 = F.sum(loss_tv0 * mask) / denorm
                loss_tv = loss_tv + loss_tv0
            return loss_tv

        loss_tv = compute_tv_loss()
        loss_tv = loss_tv.apply(persistent=True)

    # Mask loss
    loss_mask = nn.Variable.from_numpy_array(0.0)
    if conf.train.mask_weight > 0.0:
        denorm = F.sum(mask) + 1e-5
        obj_mask_pred = F.clip_by_value(obj_mask_pred, 1e-3, 1.0 - 1e-3)
        loss_mask = F.binary_cross_entropy(obj_mask_pred, obj_mask)
        # loss_mask = F.sum(loss_mask * mask[:, :, 0]) / denorm
        loss_mask = F.sum(loss_mask) / denorm
        loss_mask = loss_mask.apply(persistent=True)

    # denorm of priors
    denorm = F.sum(mask) * N + 1e-5

    # Base color prior
    prior_base_color = nn.Variable.from_numpy_array(0.0)
    if conf.train.base_color_prior_weight > 0.0:
        sym = conf.train.base_color_prior_sym_backward
        base_color = F.identity(base_color)
        base_color = base_color.apply(need_grad=sym)
        prior_base_color = F.absolute_error(base_color, base_color_ptb)
        prior_base_color = F.sum(prior_base_color * mask)
        prior_base_color = prior_base_color / denorm
        prior_base_color = prior_base_color.apply(persistent=True)
    
    # Roughness prior
    prior_roughness = nn.Variable.from_numpy_array(0.0)
    if conf.train.roughness_prior_weight > 0.0:
        roughness_prior_value = F.constant(conf.roughness_network.prior_value, roughness.shape)
        prior_roughness = F.absolute_error(roughness, roughness_prior_value)
        prior_roughness = prior_roughness / std_roughness
        prior_roughness = F.sum(prior_roughness * mask)
        prior_roughness = prior_roughness / denorm
        prior_roughness = prior_roughness.apply(persistent=True)

    # Roughness std reg
    reg_std_roughness = nn.Variable.from_numpy_array(0.0)
    if conf.train.roughness_prior_weight > 0.0:
        reg_std_roughness = F.clip_by_value(F.log(std_roughness), 1e-5, 1e5)
        reg_std_roughness = F.sum(reg_std_roughness * mask)
        reg_std_roughness = reg_std_roughness / denorm
        reg_std_roughness = reg_std_roughness.apply(persistent=True)

    # Specular reflectance prior
    prior_specular_reflectance = nn.Variable.from_numpy_array(0.0)
    if conf.train.specular_reflectance_prior_weight > 0.0:
        specular_reflectance_prior_value = F.constant(conf.specular_reflectance_network.prior_value, specular_reflectance.shape)
        prior_specular_reflectance = F.absolute_error(specular_reflectance, specular_reflectance_prior_value)
        prior_specular_reflectance = prior_specular_reflectance / std_specular_reflectance
        prior_specular_reflectance = F.sum(prior_specular_reflectance * mask)
        prior_specular_reflectance = prior_specular_reflectance / denorm
        prior_specular_reflectance = prior_specular_reflectance.apply(persistent=True)

    # Specular reflectance std reg
    reg_std_specular_reflectance = nn.Variable.from_numpy_array(0.0)
    if conf.train.specular_reflectance_prior_weight > 0.0:
        reg_std_specular_reflectance = F.clip_by_value(F.log(std_specular_reflectance), 1e-5, 1e5)
        reg_std_specular_reflectance = F.sum(reg_std_specular_reflectance * mask)
        reg_std_specular_reflectance = reg_std_specular_reflectance / denorm
        reg_std_specular_reflectance = reg_std_specular_reflectance.apply(persistent=True)
    
    # Total
    loss = loss_rgb \
        + conf.train.eikonal_weight * loss_eikonal\
        + conf.train.tv_weight * loss_tv \
        + conf.train.mask_weight * loss_mask \
        + conf.train.base_color_prior_weight * prior_base_color \
        + conf.train.roughness_prior_weight * prior_roughness \
        + conf.train.specular_reflectance_prior_weight * prior_specular_reflectance \
        + conf.train.roughness_prior_weight * reg_std_roughness \
        + conf.train.specular_reflectance_prior_weight * reg_std_specular_reflectance

    loss = loss.apply(persistent=True)

    losses = dict(
        loss=loss,
        loss_rgb=loss_rgb,
        loss_eikonal=loss_eikonal,
        loss_tv=loss_tv,
        loss_mask=loss_mask,
        prior_base_color=prior_base_color,
        prior_roughness=prior_roughness,
        prior_specular_reflectance=prior_specular_reflectance,
        reg_std_roughness=reg_std_roughness,
        reg_std_specular_reflectance=reg_std_specular_reflectance
    )
    return losses
