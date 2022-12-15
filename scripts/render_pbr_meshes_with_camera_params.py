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

import open3d as o3d
import numpy as np
import argparse
import os, glob, time
import open3d.visualization.rendering as rendering


def smooth(mesh, args):
    if args.filter_iters > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=args.filter_iters)
        mesh.compute_vertex_normals()
    return mesh


def load_mesh(fpath, args):
    mesh = o3d.io.read_triangle_mesh(fpath)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + int(args.trans))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertex_colors) * args.ratio, 0, 1))
    
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


class MeshConf:

    def __init__(self, base_path, shader, 
                 roughness_texture_path="", 
                 specular_reflectance_texture_path="", 
                 triangle_uvs_path="", 
                 args=None):

        mesh = load_mesh(base_path, args)

        material = rendering.MaterialRecord()
        material.shader = shader
        if shader == "defaultLit":
            print("Loading texture maps and triangle_uvs...")
            roughness = o3d.t.io.read_image(roughness_texture_path)
            material.roughness_img = roughness.to_legacy()
            reflectance = o3d.t.io.read_image(specular_reflectance_texture_path)
            material.reflectance_img = reflectance.to_legacy()
            
            triangle_uvs = np.load(triangle_uvs_path)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

        saved_dir_path, _ = os.path.splitext(base_path)
        saved_dir_path = f"{saved_dir_path}_{shader}_{args.envmap}"
        os.makedirs(saved_dir_path, exist_ok=True)

        self.mesh = mesh
        self.material = material
        self.saved_dir_path = saved_dir_path
        
        self.base_path = base_path
        self.shader = shader

        print(self.shader)


def main(args):

    # Renderer and Scene
    render = rendering.OffscreenRenderer(args.width, args.height, headless=True)
    render.scene.set_background(np.asarray(args.background_color))
        
    render.scene.scene.set_sun_light(args.sun_light_direction, args.sun_light_color, args.sun_light_intensity)

    ibl_resource_path = o3d.visualization.gui.Application.instance.resource_path
    render.scene.scene.set_indirect_light(f"{ibl_resource_path}/{args.envmap}")
    render.scene.scene.set_indirect_light_intensity(args.indirect_light_intensity)
    render.scene.show_skybox(args.show_skybox)

    # Mesh configs
    mesh_conf0 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_base_color_mesh00_filtered02.obj", "defaultUnlit", args=args)
    mesh_conf1 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_roughness_mesh00_filtered02.obj", "defaultUnlit", args=args)
    mesh_conf2 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_specular_reflectance_mesh00_filtered02.obj", "defaultUnlit", args=args)
    mesh_conf3 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_implicit_illumination_mesh00_filtered02.obj", "defaultUnlit", args=args)
    mesh_conf4 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_base_color_mesh00_filtered02_ilbaked.obj", "defaultUnlit", args=args)
    mesh_conf5 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_base_color_mesh00_filtered02.obj", "normals", args=args)
    mesh_conf6 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_base_color_mesh00_filtered02.obj", 
                          "defaultLit", 
                          f"{args.dpath}/model_01499_512grid_trimmed_roughness_mesh00_filtered02.png", 
                          f"{args.dpath}/model_01499_512grid_trimmed_specular_reflectance_mesh00_filtered02.png", 
                          f"{args.dpath}/triangle_uvs.npy", 
                          args=args)
    mesh_conf7 = MeshConf(f"{args.dpath}/model_01499_512grid_trimmed_base_color_mesh00_filtered02_ilbaked.obj", 
                          "defaultLit", 
                          f"{args.dpath}/model_01499_512grid_trimmed_roughness_mesh00_filtered02.png", 
                          f"{args.dpath}/model_01499_512grid_trimmed_specular_reflectance_mesh00_filtered02.png", 
                          f"{args.dpath}/triangle_uvs.npy",     
                          args=args)
    mesh_confs = [mesh_conf0, mesh_conf1, mesh_conf2, mesh_conf3, mesh_conf4, mesh_conf5, mesh_conf6, mesh_conf7]

    for mesh_conf in mesh_confs:
        print(f"{mesh_conf.base_path} with {mesh_conf.shader} is being prrocessed...")

        render.scene.clear_geometry()
        render.scene.add_geometry("object", mesh_conf.mesh, mesh_conf.material)

        enable_light = True if mesh_conf.shader == "defaultLit" else False
        render.scene.scene.enable_sun_light(enable_light)
        render.scene.scene.enable_indirect_light(enable_light)

        # Views
        st = time.perf_counter()
        cpaths = sorted(glob.glob(f"{args.cpath}/*.json"))
        for cpath in cpaths:
            print(f"Rendering image with {mesh_conf.shader} at {cpath}...")

            # Camera
            cp = o3d.io.read_pinhole_camera_parameters(cpath)
            intrinsic = cp.intrinsic
            extrinsic_mat = cp.extrinsic
            render.setup_camera(intrinsic, extrinsic_mat)

            # Save image
            idx_view = cpath.split(".")[1]
            img = render.render_to_image()
            img_path = f"{mesh_conf.saved_dir_path}/{idx_view}.png"
            o3d.io.write_image(img_path, img, 9)

        et = time.perf_counter() - st
        print(f"Elapsed time: {et} [s]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dpath", help="Path to result directory", required=True)
    parser.add_argument("-c", "--cpath", help="Path to a camera parameter directory", required=True)

    parser.add_argument("-t", "--trans", default=0, 
                        help="Translation.")
    parser.add_argument("--ratio", default=1.0, type=float, 
                        help="multiply to base RGB.")
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
    parser.add_argument("-W", "--width", default=1600, help="Screen width.")
    parser.add_argument("-H", "--height", default=1200, help="Screen height.")

    args = parser.parse_args()
    main(args)