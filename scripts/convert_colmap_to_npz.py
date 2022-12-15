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
import glob
import os
from collections import namedtuple

import numpy as np


# cameras.txt
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
def read_cameras(fpath):
    camera_ids = []
    widths = []
    heights = []
    focal_length_xs = []
    focal_length_ys = []
    principal_point_xs = []
    principal_point_ys = []
    with open(fpath, "r") as fp:
        for i, l in enumerate(fp):
            if l.startswith("#"):
                continue

            data = l.split()
            camera_model = data[1]
            if camera_model not in ("RADIAL", "SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE"):
                assert 'Use "RADIAL", "SIMPLE_RADIAL", "SIMPLE_PINHOLE", or "PINHOLE" in colmap.'
            
            camera_ids.append(int(data[0]))
            widths.append(int(data[2]))
            heights.append(int(data[3]))
            focal_length_xs.append(float(data[4]))
            if camera_model.startswith("SIMPLE"):
                focal_length_ys.append(float(data[4]))
                principal_point_xs.append(int(data[5]))
                principal_point_ys.append(int(data[6]))
            else:
                focal_length_ys.append(float(data[5]))
                principal_point_xs.append(int(data[6]))
                principal_point_ys.append(int(data[7]))

    camera_ids = np.asarray(camera_ids)
    widths = np.asarray(widths)
    heights = np.asarray(heights)
    focal_length_xs = np.asarray(focal_length_xs)
    focal_length_ys = np.asarray(focal_length_ys)
    principal_point_xs = np.asarray(principal_point_xs)
    principal_point_ys = np.asarray(principal_point_ys)
    Cameras = namedtuple("Cameras", 
                        ["camera_ids", "widths", "heights", 
                        "focal_length_xs", "focal_length_ys",
                        "principal_point_xs", "principal_point_ys"])
    return Cameras(camera_ids, widths, heights, 
                   focal_length_xs, focal_length_ys, 
                   principal_point_xs, principal_point_ys)

    
# images.txt format
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
def read_images(fpath):
    image_ids = []
    quats = []
    trans = []
    camera_ids = []
    names = []
    points2d = []
    with open(fpath, "r") as fp:
        for i, l in enumerate(fp):
            if l.startswith("#"):
                continue

            data = l.split()

            if i % 2 == 0:
                image_ids.append(int(data[0]))
                quats.append(list(map(float, data[1:5])))
                trans.append(list(map(float, data[5:8])))
                camera_ids.append(int(data[8]))
                names.append(data[9])
            else:
                for j in range(0, len(data), 3):
                    points2d.append([float(data[j]), float(data[j+1]), int(data[j+2])])

    image_ids = np.asarray(image_ids)
    quats = np.asarray(quats)
    trans = np.asarray(trans)
    camera_ids = np.asarray(camera_ids)
    Images = namedtuple("Images", ["image_ids", "quats", "trans", 
                        "camera_ids", "names", "points2d"])
    return Images(image_ids, quats, trans, camera_ids, names, points2d)


# points3D.txt format
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
def read_points3d(fpath):
    points = []
    colors = []

    with open(fpath, "r") as fp:
        for i, l in enumerate(fp):
            if l.startswith("#"):
                continue

            data = l.split()
            points.append(list(map(float, data[1:4])))
            colors.append(list(map(float, data[4:7])))
            
    points = np.asarray(points)
    colors = np.asarray(colors) / 255.0
    Points3D = namedtuple("Points3D", ["points", "colors"])
    return Points3D(points, colors)


# Borrowed from https://github.com/Fyusion/LLFF/blob/master/llff/poses/colmap_read_model.py#L272, 
# but this is common conversion and Hamilton convention
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def main(args):
    print("Converting COLMAP camera estimation to npz format...")

    base_path = f"{args.ipath}/sparse/0"
    cameras = read_cameras(f"{base_path}/cameras.txt")
    images = read_images(f"{base_path}/images.txt")
    # points3d = read_points3d(f"{base_path}/points3d.txt")

    # Most logic comes from https://github.com/Totoro97/NeuS/issues/8#issuecomment-943098737
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    indices = np.argsort(images.names)
    camera_params = {}
    for o, i in enumerate(indices):
        r = qvec2rotmat(images.quats[i])
        t = images.trans[i].reshape([3, 1])
        w2c = np.concatenate([np.concatenate([r, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)

        j = 0 if len(cameras.focal_length_xs) == 1 else i
        fx = cameras.focal_length_xs[j]
        fy = cameras.focal_length_ys[j]
        cx = cameras.principal_point_xs[j]
        cy = cameras.principal_point_ys[j]
        k = np.asarray([[fx,  0.0,  cx], 
                        [0.0, fy,   cy],
                        [0.0, 0.0, 1.0]])

        r = c2w[:3, :3]
        r = r.T  # because of the load_K_Rt_from_P() function implemented in dataset.py
        # where the decomposed rotation matrix is transposed
        c = c2w[:3, 3]
        c = -c # -t because of the opencv projection
        # matrix decomposition function implementation
        # https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix/69556782#69556782

        wm = np.eye(4)
        wm[:3,:3] = k @ r
        wm[:3,3] = k @ r @ c

        camera_params['world_mat_%d' % o] = wm
        camera_params['scale_mat_%d' % o] = np.eye(4)

    np.savez(f"{args.ipath}/cameras.npz", **camera_params)

    # Remove dropped views
    image_paths = sorted(glob.glob(f"{args.ipath}/image/*"))
    mask_paths = sorted(glob.glob(f"{args.ipath}/mask/*"))

    for i, paths in enumerate(zip(image_paths, mask_paths)):
        image_path, mask_path = paths

        key = 'world_mat_%d' % i
        if key in camera_params:
            continue
        
        print(f"COLMAP camera pose estimation is not good...")
        print(f"Remove {image_path} and {mask_path}!")
        os.remove(image_path)
        os.remove(mask_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert camera parameters estimation of COLMAP as txt to npz format. Note that COLMAP camera estimation somtimes is not perfect with the given number of views, it drops unconfident views. Accordingly, this script removes the corresponding images and masks.")
    parser.add_argument("-i", "--ipath", 
                        help="Path to colmap project directory containing cameras.txt, images.txt, and points3d.txt. E.g, custom_dataset/<scene_name> which contains 'sparse/0'.")

    args = parser.parse_args()
    main(args)
