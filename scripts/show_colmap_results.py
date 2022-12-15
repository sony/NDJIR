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

import numpy as np
import open3d as o3d

from convert_colmap_to_npz import (qvec2rotmat, read_cameras, read_images,
                                   read_points3d)


def compute_camlocs(quats, trans, indices):
    camlocs = []
    for i in indices:
        R = qvec2rotmat(quats[i])
        camloc = - R.T @ trans[i]
        camlocs.append(camloc)
    camlocs = np.asarray(camlocs)
    return camlocs


def compute_intrinsics(cameras, indices):
    intrinsics = []
    
    for i in indices:
        j = 0 if len(cameras.focal_length_xs) == 1 else i
        fx = cameras.focal_length_xs[j]
        fy = cameras.focal_length_ys[j]
        cx = cameras.principal_point_xs[j]
        cy = cameras.principal_point_ys[j]
        k = np.asarray([[fx,  0.0,  cx], 
                        [0.0, fy,   cy],
                        [0.0, 0.0, 1.0]])
        intrinsics.append(k)

    intrinsics = np.asarray(intrinsics)
    return intrinsics


def compute_raydirs(quats, intrinsics, W, H, indices):
    raydirs = []

    for i in indices:
        R_w2c = qvec2rotmat(quats[i])
        R_c2w = R_w2c.T
        K = intrinsics[i]
        K_inv = np.linalg.inv(K)
        xyz = np.asarray([W / 2.0, H / 2.0, 1.0])
        raydir = R_c2w @ K_inv @ xyz
        raydir = raydir / np.linalg.norm(raydir, ord=2)
        raydirs.append(raydir)
    
    raydirs = np.asarray(raydirs)
    return raydirs


def main(args):
    o3d.visualization.gui.Application.instance.initialize()
    w = o3d.visualization.O3DVisualizer(title="Colmap Results")
    
    # points3D
    fpath = f"{args.dpath}/points3D.txt"
    point3d = read_points3d(fpath)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d.points)
    pcd.colors = o3d.utility.Vector3dVector(point3d.colors)
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    w.add_geometry("point3D", pcd)

    # images
    fpath = f"{args.dpath}/images.txt"
    images = read_images(fpath)
    camlocs = compute_camlocs(images.quats, images.trans, np.argsort(images.names))
    print("Distance to the origin")
    print(np.linalg.norm(camlocs, ord=2, axis=1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camlocs)
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(([1, 0, 0] * len(camlocs))).reshape(len(camlocs), 3))
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    w.add_geometry("camera locations", pcd)

    # points per ray
    fpath = f"{args.dpath}/cameras.txt"
    cameras = read_cameras(fpath)
    intrinsics = compute_intrinsics(cameras, np.argsort(images.names))
    W, H = cameras.widths[0], cameras.heights[0]
    raydirs = compute_raydirs(images.quats, intrinsics, W, H, np.argsort(images.names))
    B, _ = raydirs.shape
    raydirs = raydirs.reshape((B, 1, 3))
    t = np.linspace(args.t_near, args.t_far, args.n_points)
    t = t.reshape((1, len(t), 1))
    camlocs = camlocs.reshape((B, 1, 3))
    points = camlocs + t * raydirs
    # points = points[0:1, :, :]
    points = points.reshape((-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(([0, 1, 0] * len(points))).reshape(len(points), 3))
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    w.add_geometry("points per ray", pcd)

    o3d.visualization.gui.Application.instance.add_window(w)
    o3d.visualization.gui.Application.instance.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Display COLMAP camera parameters estimation results.")
    parser.add_argument("-d", "--dpath", help="Path to directory contraining points3D.txt, images.txt, and cameras.txt")
    parser.add_argument("--t_near", type=float, default=0.1)
    parser.add_argument("--t_far", type=float, default=3.5)
    parser.add_argument("--n_points", type=int, default=100)

    args = parser.parse_args()
    main(args)
