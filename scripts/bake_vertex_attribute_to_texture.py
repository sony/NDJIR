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
import sys
from pathlib import Path

import bpy
import numpy as np


def main(args):
    fpath = args.fpath
    base_path, _ = os.path.splitext(fpath)

    # Delete all objects
    bpy.ops.object.delete(use_global=False, confirm=False)

    print("# Import mesh")
    bpy.ops.wm.obj_import(filepath=fpath)

    print("# Smart UV unwrap")
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.003)

    print("# Settting of texture baking")
    bpy.ops.object.editmode_toggle()
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.margin = 2

    print("# Add Principled BSDF")
    bpy.ops.material.new()
    bpy.data.objects[-1].active_material = bpy.data.materials['Material.001']

    print("# Add vertex color node and link it to Principled BSDF")
    bpy.data.materials["Material.001"].node_tree.nodes.new(type="ShaderNodeVertexColor")
    bpy.data.materials["Material.001"].node_tree.nodes["Color Attribute"].layer_name = "Color"
    bpy.data.materials["Material.001"].node_tree.links.new(
        bpy.data.materials["Material.001"].node_tree.nodes["Color Attribute"].outputs['Color'], 
        bpy.data.materials["Material.001"].node_tree.nodes['Principled BSDF'].inputs['Base Color'])

    print("# Add texutre image node")
    bpy.data.images.new("vertex_attr", width=1024, height=1024)
    bpy.data.images['vertex_attr'].source = 'GENERATED'
    bpy.data.images['vertex_attr'].filepath = f'{base_path}_texture.png'
    bpy.data.materials["Material.001"].node_tree.nodes.new(type="ShaderNodeTexImage")
    bpy.data.materials["Material.001"].node_tree.nodes['Image Texture'].image = bpy.data.images['vertex_attr']

    print("# Bake and save texture")
    bpy.ops.object.bake(type="DIFFUSE", width=1024, height=1024, margin=2, target="IMAGE_TEXTURES", use_clear=True, save_mode="EXTERNAL")
    bpy.data.images['vertex_attr'].save()

    print("# Save triangle_uvs as numpy")
    uv_data = bpy.data.meshes[-1].uv_layers['UVMap'].data
    triangle_uvs = np.zeros((len(uv_data), 2))
    for i in range(len(uv_data)):
        uv = uv_data[i].uv
        triangle_uvs[i, :] = [uv[0], uv[1]]

    path = Path(fpath)
    np.save(f'{path.parent.absolute()}/triangle_uvs.npy', triangle_uvs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bake vertex attribute to texture map using Blender's SmartUV project and Cycles feature.")
    parser.add_argument("-f", "--fpath", required=True, help="Path to a mesh fpath.")
    
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args)
    # Run
    # ./blender.exe --background --python ~/git/finer-inverse-rendering/scripts/bake_vertex_color_to_texture.py --  -f <obj_path> -o <out_name>
