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
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main(args):

    scan_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    
    n_bins = args.n_bins

    attr_distribution = np.zeros((len(scan_ids), n_bins))
    for i, scan_id in enumerate(scan_ids):
        print(f"Processing scan{scan_id}...")

        fpath = f"{args.base_dir}_scan{scan_id}/model_01499_512grid_trimmed_{args.attribute_name}_mesh00.obj"
        mesh = o3d.io.read_triangle_mesh(fpath)

        if args.attribute_name == "roughness":
            attrs = np.asarray(mesh.vertex_colors)[:, 1]
        else:
            attrs = np.mean(np.asarray(mesh.vertex_colors), axis=1)

        counts, bins = np.histogram(attrs, bins=n_bins, range=(args.range_min, args.range_max))
        dists = counts / counts.sum()
        dists_max_normalized = dists / dists.max()
        dists_max_normalized = np.asarray(dists_max_normalized)

        base_fpath, _ = os.path.splitext(fpath)
        color = "forestgreen" if args.attribute_name == "roughness" else "mediumblue"
        plt.hist(bins[:-1], bins, weights=dists, color=color)
        plt.savefig(f"{base_fpath}_{args.attribute_name}_dist.png")
        plt.clf()

        attr_distribution[i, :] = counts
    
    # Heatmap
    fig, ax = plt.subplots()
    cmap = dict(roughness=plt.cm.Greens, specular_reflectance=plt.cm.Blues)
    attr_distribution = np.log(attr_distribution + 1)
    heatmap = ax.pcolor(attr_distribution, cmap=cmap[args.attribute_name])

    xticks = np.linspace(0, attr_distribution.shape[1], 5)
    ax.set_xticks(xticks, minor=False)
    ax.set_yticks(np.arange(attr_distribution.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    row_labels = scan_ids
    col_labels = np.linspace(args.range_min, args.range_max, 5)
    col_labels = [f"{x:.2f}" for x in col_labels]
    ax.set_yticklabels(row_labels, minor=False)
    ax.set_xticklabels(col_labels, minor=False)

    path = pathlib.Path(fpath)
    dpath = path.parent.parent.absolute()
    prefix = "_".join(path.parent.name.split("_")[:-1])
    opath = f"{dpath}/{prefix}_{args.attribute_name}_dist.png"
    plt.savefig(opath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw distribution (heatmap) of \
        roughness or specular reflectance with DTUMVS dataset")
    parser.add_argument("-b", "--base_dir", required=True, help="Path to base directory.")
    parser.add_argument("-an", "--attribute_name", required=True, help="Attribute name.")
    parser.add_argument("-rm0", "--range_min", type=float, required=True, help="Range min.")
    parser.add_argument("-rm1", "--range_max", type=float, required=True, help="Range max.")
    parser.add_argument("-n", "--n_bins", default=50, help="Bin size.")


    args = parser.parse_args()
    main(args)
