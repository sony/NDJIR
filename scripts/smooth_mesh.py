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

import open3d as o3d


def main(args):

    fpaths = glob.glob(f"{args.dpath}/*mesh00.obj")
    for fpath in fpaths: 
        print(f"Processing {fpath}...")
        mesh = o3d.io.read_triangle_mesh(fpath)
        if args.filter_iters > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=args.filter_iters)
        
        fpath, ext = os.path.splitext(fpath)
        fpath = f"{fpath}_filtered{args.filter_iters:02d}{ext}"
        o3d.io.write_triangle_mesh(fpath, mesh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smooth meshes")
    parser.add_argument("-d", "--dpath", help="Path to result directory including *mesh00.obj")
    parser.add_argument("--filter_iters", type=int, default=2)

    args = parser.parse_args()
    main(args)
