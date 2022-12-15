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
from nnabla.monitor import Monitor, MonitorImage
from omegaconf import DictConfig

from dataset import IDRDataSource
from helper import (resize_image, setup_system)
from renderer import render_image


def main(conf: DictConfig):
    # System setup
    setup_system(conf)
    
    # Dataset
    ds = IDRDataSource(shuffle=False, conf=conf)
    W, H = ds._W, ds._H
    dn_scale = 2 ** conf.valid.n_down_samples
    Wl, Hl = W // dn_scale, H // dn_scale

    # Monitor
    monitor_path = "/".join(conf.model_load_path.split("/")[0:-1])
    monitor = Monitor(monitor_path)
    monitor_image = MonitorImage(f"Eval rendered image {Wl}x{Hl}", monitor, interval=1)
    monitor_masked_image = MonitorImage(f"Eval masked rendered image {Wl}x{Hl}", monitor, interval=1)

    # Load model
    nn.load_parameters(conf.model_load_path)

    # Render
    for i, elms in enumerate(zip(ds.images, ds.poses, ds.intrinsics, ds.masks)):
        image_gt, pose, intrinsic, mask_obj = elms
        rendered_image = render_image(pose[np.newaxis, ...],
                                intrinsic[np.newaxis, ...],
                                (ds._W, ds._H), conf)
        image_gt = resize_image(image_gt, conf)
        mask_obj = resize_image(mask_obj, conf)

        monitor_image.add(i, rendered_image)
        monitor_masked_image.add(i, rendered_image * mask_obj)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Neural-render images.")
    parser.add_argument("--config-path", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="default")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    hydra_main = hydra.main(config_path=args.config_path, config_name=args.config_name)
    hydra_main(main)()
