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

import hydra
import nnabla as nn
import numpy as np
from nnabla.monitor import (Monitor, MonitorImage, MonitorSeries,
                            MonitorTimeElapsed)
from omegaconf import DictConfig

from dataset import IDRDataSource, data_iterator_idr
from evaluate_chamfer_dtumvs import evaluate_by_chamfer
from evaluate_image import psnr
from extract_by_mc import extract
from helper import (check_dtu_data, generate_raydir_camloc, resize_image,
                    setup_system)
from loss import total_loss
from renderer import render_image
from solver import Solvers


def main(conf: DictConfig):
    # System setup
    setup_system(conf, train=True)
    
    # Setting
    B = conf.train.batch_size
    R = conf.train.n_rays

    # Dataset
    ds = IDRDataSource(shuffle=True, conf=conf)
    di = data_iterator_idr(ds, B)
    W, H = ds._W, ds._H
    dn_scale = 2 ** conf.valid.n_down_samples
    Wl, Hl = W // dn_scale, H // dn_scale

    camloc = nn.Variable([B, 3])
    raydir = nn.Variable([B, R, 3])
    color_gt = nn.Variable([B, R, 3])
    obj_mask = nn.Variable([B, R, 1])

    # Monitor
    interval = 1
    monitor_path = conf.monitor_path
    monitor = Monitor(monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=interval)
    monitor_rgb_loss = MonitorSeries("RGB loss", monitor, interval=interval)
    monitor_eikonal_loss = MonitorSeries("Eikonal loss", monitor, interval=interval)
    monitor_tv_loss = MonitorSeries("TV loss", monitor, interval=interval)
    monitor_mask_loss = MonitorSeries("Mask loss", monitor, interval=interval)
    monitor_base_color_prior = MonitorSeries("Base color prior", monitor, interval=interval)
    monitor_roughness_prior = MonitorSeries("Roughness prior", monitor, interval=interval)
    monitor_specular_reflectance_prior = MonitorSeries("Specular reflectance prior", monitor, interval=interval)
    monitor_std_roughness_reg = MonitorSeries("Std roughness reg", monitor, interval=interval)
    monitor_std_specular_reflectance_reg = MonitorSeries("Std specular reflectance reg", monitor, interval=interval)
    monitor_gain = MonitorSeries("Gain", monitor, interval=interval)
    monitor_psnr = MonitorSeries(f"PSNR {Wl}x{Hl} {conf.valid.index:03d}", monitor, interval=interval)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=interval)
    normalize_method = lambda x: x
    monitor_image = MonitorImage(f"Rendered image {Wl}x{Hl}", monitor, interval=1, 
                                 normalize_method=normalize_method)

    # Loss
    cos_anneal_ratio = nn.parameter.get_parameter_or_create("cos_anneal_ratio", (1, ),
                                            np.asarray([0.0]),
                                            False, False)
    losses = total_loss(camloc, raydir, color_gt, obj_mask, cos_anneal_ratio, conf)
    loss = losses["loss"]
    loss_rgb = losses["loss_rgb"]
    loss_eikonal = losses["loss_eikonal"]
    loss_tv = losses["loss_tv"]
    loss_mask = losses["loss_mask"]
    prior_base_color = losses["prior_base_color"]
    prior_roughness = losses["prior_roughness"]
    prior_specular_reflectance = losses["prior_specular_reflectance"]
    reg_std_roughness = losses["reg_std_roughness"]
    reg_std_specular_reflectance = losses["reg_std_specular_reflectance"]
    
    # Solver
    solvers = Solvers(conf)
    solvers.set_parameters()

    # Training loop
    iters_per_epoch = di.size // B
    for i in range(conf.train.epoch):

        # Validate
        def validate(i, train=True):
            fname = f"model_{i:05d}"
            nn.save_parameters(f"{monitor_path}/{fname}.h5") if not train else None
            
            pose_ = ds.poses[conf.valid.index:conf.valid.index+1, ...]
            intrinsic_ = ds.intrinsics[conf.valid.index:conf.valid.index+1, ...]
            image_ = ds._images[conf.valid.index, ...]
            image_gt = resize_image(image_, conf)
            rimage = render_image(pose_, intrinsic_, (W, H), conf)

            monitor_image.add(i, rimage)
            monitor_psnr.add(i, psnr(rimage, image_gt))

            fpath = extract(monitor_path, fname, ds, conf, train=train)
            
            if check_dtu_data(conf.data_path):
                conf.valid.dtumvs.mesh_path = fpath
                conf.valid.dtumvs.scan = conf.data_path.split("/")[-1]
                conf.valid.dtumvs.vis_out_dir = monitor_path
                evaluate_by_chamfer(conf) if not train else None

        if i != 0 and i % conf.valid.epoch_interval == 0 and not conf.valid.skip:
            validate(i)
            
        # Train
        for j in range(iters_per_epoch):
            # Feed data
            color_, obj_mask_, intrinsic_, pose_, xy_ = di.next()
            color_gt.d = color_
            obj_mask.d = obj_mask_
            
            raydir_, camloc_ = generate_raydir_camloc(pose_, intrinsic_, xy_)
            raydir.d = raydir_
            camloc.d = camloc_

            # Network
            loss.forward(clear_no_need_grad=True)
            solvers.zero_grad()
            solvers.weight_decay()
            solvers.clip_grad_by_norm()

            loss.backward(1, clear_buffer=True)
            if solvers.check_inf_or_nan_grad():
                nn.logger.info(f"Inf or nan grad epoch={i}, iter={j}")
                continue
            if np.any(np.isnan(loss.d)):
                nn.logger.info(f"Inf or nan loss epoch={i}, iter={j}")
                continue

            solvers.update()

        monitor_loss.add(i, loss.d)
        monitor_rgb_loss.add(i, loss_rgb.d)
        monitor_eikonal_loss.add(i, loss_eikonal.d)
        monitor_tv_loss.add(i, loss_tv.d)
        monitor_mask_loss.add(i, loss_mask.d)
        monitor_base_color_prior.add(i, prior_base_color.d)
        monitor_roughness_prior.add(i, prior_roughness.d)
        monitor_specular_reflectance_prior.add(i, prior_specular_reflectance.d)
        monitor_std_roughness_reg.add(i, reg_std_roughness.d)
        monitor_std_specular_reflectance_reg.add(i, reg_std_specular_reflectance.d)
        monitor_gain.add(i, nn.get_parameters()["geometric-network/gain"].d)
        monitor_time.add(i)

        solvers.update_learning_rate(i)

    validate(i, False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train NDJIR for decomposing \
                                      geometry, lights, and materials")
    parser.add_argument("--config-path", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="default")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    hydra_main = hydra.main(config_path=args.config_path, config_name=args.config_name)
    hydra_main(main)()
