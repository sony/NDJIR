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

#!/bin/bash

result_path=$1
model_iters=$2
prefix=$3
filter_iters=$4

echo "# Smoothing meshes"
python scripts/smooth_mesh.py -d ${result_path} --filter_iters ${filter_iters}

echo "# Rebaking natural light"
python scripts/rebake_implicit_illumination.py \
    -f0 ${result_path}/model_${model_iters}_512grid_${prefix}_base_color_mesh00_filtered0${filter_iters}.obj \
    -f1 ${result_path}/model_${model_iters}_512grid_${prefix}_implicit_illumination_mesh00_filtered0${filter_iters}.obj

echo "# UV unwrapping and texture baking"
/opt/blender-3.3.0-linux-x64/blender --background \
    --python scripts/bake_vertex_attribute_to_texture.py -- \
    -f ${result_path}/model_${model_iters}_512grid_${prefix}_roughness_mesh00_filtered0${filter_iters}.obj

/opt/blender-3.3.0-linux-x64/blender --background \
    --python scripts/bake_vertex_attribute_to_texture.py -- \
    -f ${result_path}/model_${model_iters}_512grid_${prefix}_specular_reflectance_mesh00_filtered0${filter_iters}.obj

