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

import argparse

import cv2
import numpy as np
import open3d as o3d


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


def compute_raydirs(poses, intrinsics, W, H):
    raydirs = []

    for i in range(len(poses)):
        R_c2w = poses[i, :3, :3]
        K_inv = np.linalg.inv(intrinsics[i, :, :])
        xyz = np.asarray([W / 2.0, H / 2.0, 1.0])
        raydir = R_c2w @ K_inv @ xyz
        raydir = raydir / np.linalg.norm(raydir, ord=2)
        raydirs.append(raydir)
    
    raydirs = np.asarray(raydirs)
    return raydirs


def main(args):

    camera_params = np.load(args.fpath)

    world_mat_names = [name for name in camera_params.files \
        if (name.startswith("world_mat_") and "inv" not in name)]
    world_mats = [camera_params[name] for name in world_mat_names]
    scale_mat_names = [name for name in camera_params.files \
        if (name.startswith("scale_mat_") and "inv" not in name)]
    scale_mats = [camera_params[name] for name in scale_mat_names]

    intrinsics, poses = [], []
    for W, S in zip(world_mats, scale_mats):
        P = W @ S
        P = P[:3, :4]
        intrinsic, pose = load_K_Rt_from_P(P)
        intrinsics.append(intrinsic[:3, :3])
        poses.append(pose)

    poses = np.asarray(poses)
    intrinsics = np.asarray(intrinsics)
    print("Distance to the origin")
    dists = np.linalg.norm(poses[:, :3, 3], ord=2, axis=1)
    print(dists)
    print(f"mean distance: {np.mean(dists)}")

    o3d.visualization.gui.Application.instance.initialize()
    w = o3d.visualization.O3DVisualizer(title="Sparse Point Cloud")

    # Camera locations
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(([1, 0, 0] * len(poses))).reshape(len(poses), 3))
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    w.add_geometry("camera locations", pcd)

    # Points per ray
    W, H = int(intrinsics[0, 0, 2] * 2), int(intrinsics[0, 1, 2] * 2)
    raydirs = compute_raydirs(poses, intrinsics, W, H)
    B, _ = raydirs.shape
    raydirs = raydirs.reshape((B, 1, 3))
    t = np.linspace(args.t_near, args.t_far, args.n_points)
    t = t.reshape((1, len(t), 1))
    camlocs = poses[:, :3, 3:4]
    camlocs = camlocs.reshape((B, 1, 3))
    points = camlocs + t * raydirs
    points = points.reshape((-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(([0, 1, 0] * len(points))).reshape(len(points), 3))
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    w.add_geometry("points per ray", pcd)

    # Unit sphere
    unit_sphere = o3d.geometry.TriangleMesh.create_sphere()
    unit_sphere = o3d.t.geometry.TriangleMesh.from_legacy(unit_sphere)
    w.add_geometry("unit sphere", unit_sphere)
    
    o3d.visualization.gui.Application.instance.add_window(w)
    o3d.visualization.gui.Application.instance.run()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display cameras.npy in 3D")
    parser.add_argument("-f", "--fpath", required=True)
    parser.add_argument("--t_near", type=float, default=0.1)
    parser.add_argument("--t_far", type=float, default=3.5)
    parser.add_argument("--n_points", type=int, default=100)
    args = parser.parse_args()
    main(args)
