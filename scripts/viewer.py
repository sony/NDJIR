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


def custom_draw_geometry_with_key_callback(meshes, fpath, cpath, rpath):

    def apply_camera_parameters(vis):
        if cpath != "":
            ctr = vis.get_view_control()
            cp = o3d.io.read_pinhole_camera_parameters(cpath)
            ctr.convert_from_pinhole_camera_parameters(cp, True)
            print("Camera parameter was applied.")

    def apply_render_option(vis):
        if rpath != "":
            ro = vis.get_render_option()
            ro.load_from_json(rpath)
            print("Render option was applied.")

    key_to_callback = {}
    key_to_callback[ord("C")] = apply_camera_parameters
    key_to_callback[ord("R")] = apply_render_option

    W, H = 1600, 1200
    o3d.visualization.draw_geometries_with_key_callbacks(meshes, key_to_callback, 
                                                        width=W, height=H)


def main(args):
    meshes = []
    for fpath in args.fpath:
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
            print(f"median(vertex_colors) = {np.median(np.asarray(mesh.vertex_colors), axis=0)}")
            import matplotlib.pyplot as plt

            
            if "roughness" in fpath:
                dist = np.asarray(mesh.vertex_colors)[:, 1]
            elif "specular_reflectance" in fpath:
                dist = np.mean(np.asarray(mesh.vertex_colors), axis=1)
            elif "implicit_illumination" in fpath:
                dist = np.asarray(mesh.vertex_colors)[:, 2]
            else:
                dist = np.mean(np.asarray(mesh.vertex_colors), axis=1)
            plt.hist(dist, bins=100)
            plt.savefig("hist.png")

        mesh.compute_vertex_normals()
        if args.filter_iters > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=args.filter_iters)
            mesh.compute_vertex_normals()
        if args.drop_color:
            mesh.vertex_colors = o3d.utility.Vector3dVector()
        meshes.append(mesh)

    custom_draw_geometry_with_key_callback(meshes, args.fpath, args.cpath, args.rpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Viewer for mesh")
    parser.add_argument("-f", "--fpath", nargs="+", help="Path to a mesh fpath.")
    parser.add_argument("-t", "--trans", default=0, help="Translation")
    parser.add_argument("-c", "--cpath", help="Path to a camera parameter", default="")
    parser.add_argument("-r", "--rpath", help="Path to a render option", default="")
    parser.add_argument("--ratio", help="multiply to diffuse RGB", default=1.0, type=float)
    parser.add_argument("--drop_color", action="store_true")
    parser.add_argument("--add_indirect", action="store_true")
    parser.add_argument("--filter_iters", type=int, default=0)

    args = parser.parse_args()
    main(args)
