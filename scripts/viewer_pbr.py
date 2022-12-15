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
import json
import os
import shutil
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm


class SaveCameraParameter:

    def __init__(self, base_color_path):
        path = Path(base_color_path)
        base_path = path.parent.absolute()
        dpath = f"{base_path}/o3d_camera_parameters_interactive"
        self.dpath = dpath

    def __call__(self, o3dvis):
        os.makedirs(self.dpath, exist_ok=True)

        cp = self.create_camera_parameters(o3dvis)
        if cp is None:
            return
        
        with open("scripts/ScreenCamera_2022-00-00-00-00-00.tmpl.json") as fp:
            data_tmpl = json.load(fp)

        data_tmpl["intrinsic"]["width"] = cp.intrinsic.width
        data_tmpl["intrinsic"]["height"] = cp.intrinsic.height

        # Order of array in json format seems like column-major while the loaded matrix is row-major
        Rt = cp.extrinsic
        for i, r in enumerate(Rt.T.flatten()):
            data_tmpl["extrinsic"][i] = r

        K = cp.intrinsic.intrinsic_matrix
        for i, k in enumerate(K.T.flatten()):
            data_tmpl["intrinsic"]["intrinsic_matrix"][i] = k

        # Save camera parameter
        json_files = sorted(glob.glob(f"{self.dpath}/*.json"))
        idx = 0 if json_files == [] else len(json_files)
        opath = f"{self.dpath}/ScreenCamera_2022-00-00-00-00-00.{idx}.json"
        with open(opath, "w") as fp:
            json.dump(data_tmpl, fp)

        print(f"{opath} was saved.")

    def create_camera_parameters(self, o3dvis):
        if o3dvis.show_settings == True:
            # We could not control the projection_matrix values 
            # using o3dvis.{size, os_frame, content_rect}.
            o3dvis.show_message_box("", 
                        "Hide settings first for correct focal length of width.")
            return

        fov = o3dvis.scene.camera.get_field_of_view()
        vm = o3dvis.scene.camera.get_view_matrix()
        pm = o3dvis.scene.camera.get_projection_matrix()

        # OpneGL --> OpenCV
        rx, ry, rz = vm[0, :3], vm[1, :3], vm[2, :3]
        R_w2c = np.stack([rx, -ry, -rz], axis=0)
        t = np.asarray([vm[0, 3], -vm[1, 3], -vm[2, 3]])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t

        width = o3dvis.content_rect.width
        height = o3dvis.content_rect.height
        fx = pm[0, 0] * width / 2
        fy = pm[1, 1] * height / 2
        cx = width / 2
        cy = height / 2
        # tan_fov2 = np.tan(fov / 2 * np.pi / 180)
        # fx = width / 2 / tan_fov2
        # fy = height / 2 / tan_fov2

        cp = o3d.camera.PinholeCameraParameters()
        cp.extrinsic = extrinsic
        cp.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        cp.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        print("Camera parameter: intrinsic")
        print(width, height)
        print(cp.intrinsic.intrinsic_matrix)
        print("Camera parameter: extrinsic")
        print(extrinsic)

        return cp


def interpolate_poses(extrinsic0, extrinsic1, ratio):
    # interpolate matrices
    R_w2c0 = extrinsic0[:3, :3]
    R_w2c1 = extrinsic1[:3, :3]
    R_c2w0 = np.linalg.inv(R_w2c0)
    R_c2w1 = np.linalg.inv(R_w2c1)
    rots = R.from_matrix(np.stack([R_c2w0, R_c2w1]))
    slerp = Slerp([0, 1], rots)
    rot = slerp(ratio)
    R_c2w = rot.as_matrix()
    R_w2c = np.linalg.inv(R_c2w)

    # interpolate tranlations
    trans0 = extrinsic0[:3, 3]
    trans1 = extrinsic1[:3, 3]
    trans = (1 - ratio) * trans0 + ratio * trans1

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_w2c
    extrinsic[:3, 3] = trans

    return extrinsic


def smooth(mesh, args):
    if args.filter_iters > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=args.filter_iters)
        mesh.compute_vertex_normals()
    return mesh


def load_mesh(fpath, args):
    mesh = o3d.io.read_triangle_mesh(fpath)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors) * args.scale + args.trans, 0, 1))
    
    print(f"Surface area = {mesh.get_surface_area()}")
    print(f"#vertices = {len(mesh.vertices)}")
    print(f"#triangles = {len(mesh.triangles)}")
    print(f"min(vertices), max(vertices) = {np.min(np.asarray(mesh.vertices), axis=0)}, {np.max(np.asarray(mesh.vertices), axis=0)}")
    if len(np.asarray(mesh.vertex_colors)) != 0:
        print(f"min(vertex_colors), max(vertex_colors) = {np.min(np.asarray(mesh.vertex_colors), axis=0)}, {np.max(np.asarray(mesh.vertex_colors), axis=0)}")
        print(f"mean(vertex_colors) = {np.mean(np.asarray(mesh.vertex_colors), axis=0)}")

    mesh.compute_vertex_normals()
    mesh = smooth(mesh, args)
    return mesh


def main(args):
    # Colored mesh
    mesh_base_color = load_mesh(args.fpath_base_color, args)
    if args.fpath_uvs != "":
        print("Loading triangle_uvs...")
        triangle_uvs = np.load(args.fpath_uvs)
        mesh_base_color.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_base_color)
    
    # Material
    material = rendering.MaterialRecord()
    material.shader = args.shader
    if args.fpath_roughness_texture != "":
        print("Loading roughness...")
        roughness = o3d.t.io.read_image(args.fpath_roughness_texture)
        np.asarray(roughness)[:, :, :] = np.asarray(roughness)[:, :, 1:2]
        # Open3D's specular term is somehow too strong compared to Blender, 
        # so transform here if necessary
        remapped_roughness = np.asarray(roughness)[:, :, :] / 255.0 * (args.roughness_max - args.roughness_min) + args.roughness_min
        np.asarray(roughness)[:, :, :] = remapped_roughness.astype(np.uint8)
        roughness = roughness.to_legacy()
        material.roughness_img = roughness
    if args.fpath_specular_reflectance_texture != "":
        print("Loading specular reflectance...")
        reflectance = o3d.t.io.read_image(args.fpath_specular_reflectance_texture)
        reflectance = reflectance.to_legacy()
        material.reflectance_img = reflectance

    material.base_metallic = args.base_metallic

    # Camera parameter is set
    if args.cpath != "":
        # Renderer and Scene
        path = Path(args.cpath)
        cpaths = sorted(glob.glob(f"{path}/*.json")) if path.is_dir() else [str(path)]
        cp = o3d.io.read_pinhole_camera_parameters(cpaths[0])
        W, H = cp.intrinsic.width, cp.intrinsic.height
        
        if int("".join(o3d.__version__.split("."))) < 160:
            # in windows, we have to use <0.16.0
            headless = args.server # Change depending on server or desktop
            renderer = rendering.OffscreenRenderer(W, H, headless=headless)
        else:
            renderer = rendering.OffscreenRenderer(W, H)
        renderer.scene.add_geometry("object", mesh, material)
        renderer.scene.set_background(np.asarray(args.background_color))
        
        renderer.scene.scene.set_sun_light(args.sun_light_direction, args.sun_light_color, 
                                         args.sun_light_intensity)
        renderer.scene.scene.enable_sun_light(True)

        ibl_resource_path = o3d.visualization.gui.Application.instance.resource_path
        renderer.scene.scene.set_indirect_light(f"{ibl_resource_path}/{args.envmap}")
        renderer.scene.scene.enable_indirect_light(True)
        renderer.scene.scene.set_indirect_light_intensity(args.indirect_light_intensity)
        renderer.scene.show_skybox(args.show_skybox)

        # Save video
        if args.video_name != "":
            cpaths0, cpaths1 = cpaths[:-1], cpaths[1:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            path = Path(args.fpath_base_color)
            video_path = f"{path.parent.absolute()}/{args.video_name}.mp4"
            writer = cv2.VideoWriter(video_path, fourcc, args.fps, (W, H))

            print(f"Writing video to {video_path}...")
            n_views = len(cpaths)
            for i, cpath01 in enumerate(tqdm(zip(cpaths0, cpaths1), total=n_views)):
                cpath0, cpath1 = cpath01
                cp0 = o3d.io.read_pinhole_camera_parameters(cpath0)
                intrinsic, extrinsic_mat0 = cp0.intrinsic.intrinsic_matrix, cp0.extrinsic
                cp1 = o3d.io.read_pinhole_camera_parameters(cpath1)
                extrinsic_mat1 = cp1.extrinsic

                for i in range(args.n_frames):
                    ratio = np.sin(((i / args.n_frames) - 0.5) * np.pi) * 0.5 + 0.5
                    extrinsic_mat = interpolate_poses(extrinsic_mat0, extrinsic_mat1, ratio)
                    renderer.setup_camera(intrinsic, extrinsic_mat, W, H)
                    img = renderer.render_to_image()
                    img = np.asarray(img)[:, :, ::-1]
                    writer.write(img)

            writer.release()
            return

        # Save images
        for cpath in cpaths:
            print(f"Saving image at the view of {cpath}...")
            
            cp = o3d.io.read_pinhole_camera_parameters(cpath)
            intrinsic = cp.intrinsic.intrinsic_matrix
            extrinsic_mat = cp.extrinsic
            view_id = cpath.split(".")[-2]
            view_id = f"_{view_id}"

            renderer.setup_camera(intrinsic, extrinsic_mat, W, H)
            img_path, _ = os.path.splitext(args.fpath_base_color)
            suffix = ""
            if args.base_metallic:
                suffix += f"metallic{args.base_metallic}"
            if args.reflectance_aware_metallic:
                suffix += "reflectance_aware"
            img_path = f"{img_path}_{args.shader}_{args.envmap}{view_id}{suffix}.png"
            img = renderer.render_to_image()
            o3d.io.write_image(img_path, img, 9)

            print(f"Image was saved in {img_path}")
        return

    # GUI
    if args.web:
        o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.gui.Application.instance.initialize()
    w = o3d.visualization.O3DVisualizer(title="PBR Visualizer")
    w.add_action("Save camera parameter", SaveCameraParameter(args.fpath_base_color))
    w.add_geometry("object", mesh, material)

    o3d.visualization.gui.Application.instance.add_window(w)
    o3d.visualization.gui.Application.instance.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Viewer for PBR mesh")
    parser.add_argument("-fc", "--fpath_base_color", required=True, 
                        help="Path to a mesh fpath.")
    parser.add_argument("-fuv", "--fpath_uvs", default="", 
                        help="Path to a uv-attr fpath.")
    parser.add_argument("-fr", "--fpath_roughness_texture", default="", 
                        help="Path to a roughness texture.")
    parser.add_argument("-fs", "--fpath_specular_reflectance_texture", default="", 
                        help="Path to a specular reflectance texture.")
    parser.add_argument("-c", "--cpath", default="", 
                        help="Path to a camera parameter or directory. \
                            With this parameter saves image(s).")
    parser.add_argument("--server", action="store_true", 
                        help="headless=true.")
    parser.add_argument("--scale", default=1.0, type=float, 
                        help="Multiply to base RGB.")
    parser.add_argument("--trans", default=0.0, type=float, 
                        help="Add to base RGB.")
    parser.add_argument("--filter_iters", type=int, default=0, 
                        help="Number of average filter applied.")
    parser.add_argument("--shader", type=str, default="defaultLit", choices=["defaultLit", "defaultUnlit", "normals"], 
                        help="See https://github.com/isl-org/Open3D/tree/master/cpp/open3d/visualization/gui/Materials")
    parser.add_argument("--sun_light_direction", type=float, default=[0.0, 0.0, 1.0], 
                        help="Sun light direction")
    parser.add_argument("--sun_light_color", type=float, default=[1.0, 1.0, 1.0], 
                        help="Sun color direction")
    parser.add_argument("--sun_light_intensity", type=float, default=75000, 
                        help="Sun light intensity")
    parser.add_argument("--indirect_light_intensity", type=float, default=37500, 
                        help="Indirect light intensity")
    parser.add_argument("--background_color", type=float, default=[255.0, 255.0, 255.0, 255.0], 
                        help="Background color")
    parser.add_argument("--envmap", type=str, default="default", 
                        choices=[
                                "brightday", 
                                "crossroads", 
                                "default", 
                                "hall",   
                                "konzerthaus", 
                                "nightlights", 
                                "park", 
                                "park2",      
                                "pillars",     
                                "streetlamp"
                                ],
                        help="Name of environment map given by Open3D.")
    parser.add_argument("--show_skybox", action="store_true", help="Show skybox")
    parser.add_argument("--base_metallic", default=0.0, type=float, help="Base metallic.")
    parser.add_argument("--reflectance_aware_metallic", action="store_true")
    parser.add_argument("--roughness_min", default=0, type=float, help="Roughness min")
    parser.add_argument("--roughness_max", default=255, type=float, help="Roughness max")

    parser.add_argument("--video_name", default="", help="Video name")
    parser.add_argument("--fps", default=30, type=float, help="FPS for video")
    parser.add_argument("--n_frames", default=120, type=int, 
                        help="Num. of frames between views.")

    parser.add_argument("--web", action="store_true", 
                        help="Launch as web server. This is not interactive enough. \
                            Better to launch as local app.")

    args = parser.parse_args()
    main(args)
