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

import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


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


def create_o3d_camera_parameter(cp_path, world_mat, scale_mat, image_path):
    ipath = Path(image_path)
    idx_view = int(ipath.stem)

    P = world_mat @ scale_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(P)
    K = intrinsics[:3, :3]

    # Read extrinsics and intrinsics    
    R_c2w = pose[:3, :3]
    camloc = pose[:3, 3]

    R_w2c = np.linalg.inv(R_c2w)
    trans = -R_w2c @ camloc

    # Set extrinsics and intrinsics    
    with open("scripts/ScreenCamera_2022-00-00-00-00-00.tmpl.json") as fp:
        data_tmpl = json.load(fp)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R_w2c
    Rt[:3, 3] = trans
    Rt[3, 3] = 1.0

    data_tmpl["intrinsic"]["width"] = int(K[0, 2] * 2)
    data_tmpl["intrinsic"]["height"] = int(K[1, 2] * 2)

    # Order of array in json format seems like column-major while the loaded matrix is row-major
    for i, r in enumerate(Rt.T.flatten()):
        data_tmpl["extrinsic"][i] = r

    for i, k in enumerate(K.T.flatten()):
        data_tmpl["intrinsic"]["intrinsic_matrix"][i] = k

    # Save camera parameter
    opath = f"{cp_path}/ScreenCamera_2022-00-00-00-00-00.{idx_view:02d}.json"
    with open(opath, "w") as fp:
        json.dump(data_tmpl, fp)


def main(args):
    path = Path(args.fpath)
    cp_path = f"{path.parent.absolute()}/o3d_camera_params_from_npz"
    if os.path.exists(cp_path):
        shutil.rmtree(cp_path)
    os.makedirs(cp_path)

    image_paths = sorted(glob.glob(f"{path.parent.absolute()}/image/*"))
    camera_params = np.load(args.fpath)

    world_mat_names = [name for name in camera_params.files \
        if (name.startswith("world_mat_") and "inv" not in name)]
    world_mats = [camera_params[name] for name in world_mat_names]
    scale_mat_names = [name for name in camera_params.files \
        if (name.startswith("scale_mat_") and "inv" not in name)]
    scale_mats = [camera_params[name] for name in scale_mat_names]
    
    for world_mat, scale_mat, image_path in zip(world_mats, scale_mats, image_paths):
        print(f"O3D camera param of {image_path} is being created...")
        create_o3d_camera_parameter(cp_path, world_mat, scale_mat, image_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create camera parameter format of Open3D based on cameras.npz.")
    parser.add_argument("-f", "--fpath", required=True, 
                        help="Path to cameras.npz. Assumed <scene_name>/cameras.npz.")
    args = parser.parse_args()

    main(args)
