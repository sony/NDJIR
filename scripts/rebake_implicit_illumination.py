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
import os

import numpy as np
import open3d as o3d


def main(args):
    mesh0 = o3d.io.read_triangle_mesh(args.fpath0)
    mesh0.vertices = o3d.utility.Vector3dVector(np.asarray(mesh0.vertices))
    vertex_colors0 = np.asarray(mesh0.vertex_colors)

    mesh1 = o3d.io.read_triangle_mesh(args.fpath1)
    mesh1.vertices = o3d.utility.Vector3dVector(np.asarray(mesh1.vertices))
    vertex_colors1 = np.asarray(mesh1.vertex_colors)
    vertex_colors1[:, :] = vertex_colors1[:, 2:3]

    # # vertex_colors1 = vertex_colors1 > 0.15
    # vertex_colors1 = vertex_colors1 ** 2
    vertex_colors1 = o3d.utility.Vector3dVector(vertex_colors0 * vertex_colors1)
    mesh1.vertex_colors = vertex_colors1

    fpath, _ = os.path.splitext(args.fpath0)
    fpath1 = f"{fpath}_ilbaked.obj"
    o3d.io.write_triangle_mesh(fpath1, mesh1)

    mesh_alpha = o3d.geometry.TriangleMesh(mesh1)
    alphas = np.linspace(0.0, 1.0, args.num_lerps + 2)[1:]
    for alpha in alphas:
        print(f"Interpolating base color and light-baked base color with {alpha}")
        mesh_alpha.vertex_colors = o3d.utility.Vector3dVector((1 - alpha) * vertex_colors0 + alpha * vertex_colors1)
        fpath_alpha = f"{fpath}_ilbaked_{alpha}.obj"
        o3d.io.write_triangle_mesh(fpath_alpha, mesh_alpha)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rebake implicit illumination to base color")
    parser.add_argument("-f0", "--fpath0", help="Path to base color mesh")
    parser.add_argument("-f1", "--fpath1", help="Path to indirect illumination mesh")
    parser.add_argument("-n", "--num_lerps", default=3, type=int, help="number of interpolations")
    
    args = parser.parse_args()
    main(args)
