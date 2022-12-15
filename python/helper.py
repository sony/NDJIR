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

import os
import shutil
import time

import cv2
import hydra
import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context
from omegaconf import OmegaConf


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]  # intrinsic matrix
    R = out[1]  # world-to-camera matrix
    c = out[2]  # camera location

    K = K / K[2, 2]
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)  # [camera-to-world | camera location]
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (c[:3] / c[3])[:, 0]

    return intrinsic, pose


def generate_raydir_camloc(pose, intrinsic, xy):
    """
    pose: [B, 4, 4]
    intrinsic: [B, 3, 3]
    xy: [B, R, 2]

    x_p = K [R | t] x_w
    => x_p = K (R x_w + t)
    => x_w = R_inv K_inv x_p - R_inv t
    => x_w = normalize(x_w)
    """
    B, R, _ = xy.shape

    # Align dimensions
    R_c2w = pose[:, np.newaxis, :3, :3]
    camloc = pose[:, np.newaxis, :3, 3:4]
    K_inv = np.linalg.inv(intrinsic[:, np.newaxis, :, :])

    # Transform pixel --> camera --> world
    z = np.ones([B, R, 1])
    xyz_pixel = np.concatenate([xy, z], axis=-1)[:, :, :, np.newaxis]
    xyz_camera = np.matmul(K_inv, xyz_pixel)
    xyz_world = np.matmul(R_c2w, xyz_camera)

    # Normalize
    xyz_world = xyz_world.reshape((B, R, 3))
    raydir = xyz_world / \
        np.sqrt(np.sum(xyz_world ** 2, axis=-1, keepdims=True))

    return raydir, camloc.reshape((B, 3))


def generate_all_pixels(W, H):
    x = np.arange(0, W)
    y = np.arange(0, H)
    xx, yy = np.meshgrid(x, y)
    xy = np.asarray([xx.flatten(), yy.flatten()]).T
    return xy


def resize_image(image, conf):
    if conf.valid.n_down_samples == 0:
        return image[np.newaxis, ...].transpose((0, 3, 1, 2))
    
    H, W, _ = image.shape
    dn_scale = 2 ** conf.valid.n_down_samples
    Wl, Hl = W // dn_scale, H // dn_scale
            
    rimage = cv2.resize(image, (Wl, Hl), interpolation=cv2.INTER_LINEAR)
    if rimage.ndim == 2:
        rimage = rimage[..., np.newaxis]
    rimage = rimage[np.newaxis, ...].transpose((0, 3, 1, 2))
    return rimage


def setup_system(conf, train=False):
    # Hydra workaround
    owd = hydra.utils.get_original_cwd()
    os.chdir(owd)
    # shutil.rmtree("outputs", ignore_errors=True)

    # Create monitor path when in training and copy config and scripts
    monitor_base_path = conf.monitor_base_path
    scene_name = conf.data_path.split("/")[-1]
    monitor_path = f"{monitor_base_path}_{scene_name}"
    conf.monitor_path = monitor_path
    
    if train:
        os.makedirs(monitor_path, exist_ok=True)
        conf_str = OmegaConf.to_yaml(conf)
        with open(os.path.join(monitor_path, "config.yaml"), "w") as fp:
            fp.write(conf_str)
        
        shutil.copytree("python", os.path.join(monitor_path, "python"), dirs_exist_ok=True)
        shutil.copytree("csrc", os.path.join(monitor_path, "csrc"), dirs_exist_ok=True)

    # Set context
    ctx = get_extension_context('cudnn', device_id=conf.device_id, type_config=conf.type_config)
    nn.set_default_context(ctx)

    nn.logger.info(f"Monitor Path: {monitor_path}")


def watch_etime(func):
    def wrapper(*args, **kwargs):
        st = time.perf_counter()
        ret = func(*args, **kwargs)
        et = time.perf_counter()
        print(f"Elapsed times ({func.__name__}) = {(et - st):03f} [s]")
        return ret
    return wrapper


def check_dtu_data(data_path):
    return data_path.split("/")[-2].startswith("DTU")
