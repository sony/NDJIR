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
from collections import namedtuple

import cv2 as cv
import hydra
import nnabla as nn
import numpy as np
import open3d as o3d
import trimesh
from nnabla.ext_utils import get_extension_context
from omegaconf import DictConfig
from skimage import measure
from tqdm import tqdm

from dataset import IDRDataSource
from helper import check_dtu_data, setup_system, watch_etime
from network import (base_color_network, environment_light_network,
                     geometric_network, implicit_illumination_network,
                     roughness_network, specular_reflectance_network)


@watch_etime
def create_mesh_from_volume(volume, level, mins, maxs, G, gradient_direction="ascent"):
    verts, faces, normals, values = measure.marching_cubes(volume,
                                                           level, 
                                                           gradient_direction=gradient_direction)
    verts = verts * (maxs - mins) / (G - 1) + mins
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)
    return mesh
    

@watch_etime
def compute_pts_vol(mins, maxs, grid_size, conf):
    x = np.linspace(mins[0], maxs[0], grid_size).astype(np.float32)
    y = np.linspace(mins[1], maxs[1], grid_size).astype(np.float32)
    z = np.linspace(mins[2], maxs[2], grid_size).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z)
    x_size = x.size
    y_size = y.size
    z_size = z.size
    print(f"Grid shape for (x, y, z) = {x_size, y_size, z_size}")

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    B = conf.extraction.batch_size

    vol = []
    for b in tqdm(range(0, pts.shape[0], B),
                  desc="compute-volume"):
        p = pts[b:b+B, :]
        v = geometric_network(nn.NdArray.from_numpy_array(p), conf)[0]
        v = v.data.copy().reshape(-1)
        vol.append(v)
    pts = pts.reshape((-1, 3))

    vol = np.concatenate(vol).reshape((y_size, x_size, z_size)).transpose((1, 0, 2))
    return pts, vol


# Following two functions borrowed from https://gist.github.com/Totoro97/43664cfc28110a469d88a158af040014
def clean_points_by_mask(points, ds, conf):
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(len(ds._masks)):
        print(f"Procesing with {i}-th mask")
        pose = ds._poses[i:i+1, :, :]
        R = np.linalg.inv(pose[:, :3, :3])
        t = -R @ pose[:, :3, 3:]
        K = ds._intrinsics[i:i+1, :, :]
        pts_image = np.matmul(K, np.matmul(R, points[:, :, None]) + t).squeeze()
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1
        
        mask_image = ds._masks[i]
        pixel_margin = 50
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (pixel_margin * 2 + 1, pixel_margin * 2 + 1))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image >= 0.5).astype(np.int32)
        
        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask


@watch_etime
def create_trimmed_meshes(old_mesh, ds, conf):
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    old_vertex_colors = old_mesh.visual.vertex_colors[:]
    mask = clean_points_by_mask(old_vertices, ds, conf)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]
    new_vertex_colors = old_vertex_colors[np.where(mask)]
    new_mesh = trimesh.Trimesh(new_vertices, new_faces, vertex_colors=new_vertex_colors)

    meshes = new_mesh.split(only_watertight=False)
    idx_mesh_sorted = np.argsort([len(mesh.faces) for mesh in meshes])[::-1]
    new_mesh = meshes[idx_mesh_sorted]

    return new_mesh


def create_largest_meshes(mesh, top_k=3):
    meshes = mesh.split(only_watertight=False)
    areas = np.array([m.area for m in meshes], dtype=np.float)
    print("Areas of rough meshes")
    print(areas)
    idx_areas_sorted = np.argsort(areas)[::-1]
    meshes_large = []
    for i in idx_areas_sorted[:top_k]:
        meshes_large.append(meshes[i])
    return meshes_large


@watch_etime
def compute_vertex_colors(mesh, conf, network, out_index):
    B = conf.extraction.batch_size
    vertices = np.asarray(mesh.vertices)
    vertex_colors = []
    for b in tqdm(range(0, vertices.shape[0], B),
                   desc="compute-colors"):
        v0 = vertices[b:b+B, :]
        v0 = nn.Variable.from_numpy_array(v0).apply(need_grad=True)
        with nn.auto_forward(False):
            sdf, feature, _ = geometric_network(v0, conf)
            camloc = None
            view = None
            normal = nn.grad([sdf], [v0])[0]
            vc = network(v0, feature, normal, conf)
            if out_index is not None:
                vc = vc[out_index]
        vc.forward(clear_buffer=True)
        vertex_colors.append(vc.d)

    vertex_colors = np.concatenate(vertex_colors)
    return vertex_colors


def convert_to_o3d_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    vertex_normals = np.asarray(mesh.vertex_normals)
    vertex_colors = np.asarray(mesh.visual.vertex_colors)

    vertices = o3d.utility.Vector3dVector(vertices.copy())
    triangles = o3d.utility.Vector3iVector(faces.copy())
    vertex_normals = o3d.utility.Vector3dVector(vertex_normals.copy())
    vertex_colors = o3d.utility.Vector3dVector(vertex_colors.copy())

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = vertices
    mesh.triangles = triangles
    mesh.vertex_normals = vertex_normals
    mesh.vertex_colors = vertex_colors

    return mesh
    

def create_rgb_color(color, dim):
    B = color.shape[0]
    rgb_color = np.zeros((B, 3))
    if dim == -1:
        rgb_color = np.clip(color, 0.0, 1.0)
    else:
        rgb_color[:, dim:dim+1] = np.clip(color, 0.0, 1.0)
    return rgb_color


def save_attributed_mesh(dirname, fname, mesh, mins, train, type, idx, conf):
    ExtractConf = namedtuple("ExtractConf", ["texture_name", "network", "fill_index", "out_index"])
    econf0 = ExtractConf("base_color", base_color_network, -1, None)
    econf1 = ExtractConf("implicit_illumination", implicit_illumination_network, 
                        2 if conf.implicit_illumination_network.channels == 1 else -1, None)
    econf2 = ExtractConf("roughness", roughness_network, 1, 0)
    econf3 = ExtractConf("specular_reflectance", specular_reflectance_network, 
                        0 if conf.specular_reflectance_network.channels == 1 else -1, 0)
    econf4 = ExtractConf("roughness_std", roughness_network, 1, 1)
    econf5 = ExtractConf("specular_reflectance_std", specular_reflectance_network, 
                        0 if conf.specular_reflectance_network.channels == 1 else -1, 1)
    econfs = [econf0, econf1, econf2, econf3, econf4, econf5]

    G = conf.extraction.rough_grid_size if train else conf.extraction.grid_size

    for econf in econfs:
        vertex_colors = compute_vertex_colors(mesh, conf, econf.network, econf.out_index)
        vertex_colors = create_rgb_color(vertex_colors, econf.fill_index)
        if econf.out_index == 1:
            vertex_colors = vertex_colors / vertex_colors.max()
        mesh.visual.vertex_colors = vertex_colors
        fpath = f"{dirname}/{fname}_{G}grid_{type}_{econf.texture_name}_mesh{idx:02d}.obj"
        mesh.export(fpath, "obj")
        print(f"#vertices = {len(mesh.vertices)}")
        print(f"#triangles = {len(mesh.faces)}")

    return fpath


def extract_environment_map(dirname, conf):
    H = 256
    W = 2 * H
    thetas = np.linspace(0, np.pi, H)
    phis = np.linspace(-np.pi, np.pi, W)
    the, phi = np.meshgrid(phis, thetas)

    x = np.cos(phi) * np.sin(the)
    y = np.sin(phi) * np.sin(the)
    z = np.cos(the)
    
    light_dir_data = np.stack([x, y, z], axis=-1).reshape((1, 1, H * W, 3))
    light_dir = nn.Variable.from_numpy_array(light_dir_data)
    environment_light_intensity = environment_light_network(light_dir, conf).apply(persistent=True)
    environment_light_intensity.forward(clear_buffer=True)
    environment_light_intensity_data = environment_light_intensity.d
    M = environment_light_intensity_data.max()
    m = environment_light_intensity_data.min()
    if conf.environment_light_network.act_last == "sigmoid":
        environment_light_intensity_data = environment_light_intensity_data * 255.0
    elif m != M:
        environment_light_intensity_data = environment_light_intensity_data / M * 255.0
    else: # m = M = upper_bound
        environment_light_intensity_data = 255.0 * np.ones(environment_light_intensity_data.shape)
    channels = environment_light_intensity_data.shape[-1]
    shape = (H, W, 3) if channels == 3 else (H, W)
    environment_light_intensity_data = environment_light_intensity_data.reshape(shape)
    environment_light_intensity_data = np.clip(environment_light_intensity_data, 0.0, 255.0)
    environment_light_intensity_data = environment_light_intensity_data.astype(np.uint8)
    environment_light_intensity_data = environment_light_intensity_data[..., ::-1]

    path = os.path.join(dirname, "environment_map.png")
    cv.imwrite(path, environment_light_intensity_data)
    path = os.path.join(dirname, "environment_map_min_max.txt")
    with open(path, "w") as fp:
        fp.write(f"min, max = {m}, {M}")

def extract(dirname, fname, ds, conf, train=False):
    # Direct illumination
    extract_environment_map(dirname, conf)

    ## Take care that Marching Cubes typically
    ## 1. starts from the origin 0, so add min of the domain of SDF when computing color
    ## 2. spaces with 1 as default, so change it to the grid step
    
    nn.logger.info("Extracting mesh")
    radius = conf.renderer.bounding_sphere_radius
    mins = np.asarray([-radius] * 3)
    maxs = np.asarray([+radius] * 3)
    G = conf.extraction.rough_grid_size if train else conf.extraction.grid_size
    pts, vol = compute_pts_vol(mins, maxs, G, conf)
    mesh_raw = create_mesh_from_volume(vol, conf.extraction.level, 
                                       mins, maxs, G, conf.extraction.gradient_direction)
    fpath = save_attributed_mesh(dirname, fname, mesh_raw, mins, train, "raw", 0, conf)

    if not train and check_dtu_data(conf.data_path):
        nn.logger.info("Trimming mesh by mask")
        meshes_trimmed = create_trimmed_meshes(mesh_raw, ds, conf)
        for k in range(min(len(meshes_trimmed), 5) - 1, -1, -1):
            mesh_trimmed = meshes_trimmed[k]
            fpath = save_attributed_mesh(dirname, fname, mesh_trimmed, mins, train, "trimmed", k, conf)

    return fpath

def main(conf: DictConfig):
    # System setup
    setup_system(conf)

    # Context
    ctx = get_extension_context("cudnn", device_id=conf.device_id)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Load model parameters
    nn.load_parameters(conf.model_load_path)

    # Data Source
    ds = IDRDataSource(conf=conf)

    # File prefix
    names = conf.model_load_path.split("/")
    fname = os.path.splitext(names[-1])[0]
    dirname = "/".join(names[:-1])

    # Extract rough mesh by marching cube
    extract(dirname, fname, ds, conf, train=False)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract mesh by MatchingCubes algorithm.")
    parser.add_argument("--config-path", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="default")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    hydra_main = hydra.main(config_path=args.config_path, config_name=args.config_name)
    hydra_main(main)()
