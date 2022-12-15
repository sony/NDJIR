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

data_path=$1
n_images=${2:-100}
single_camera=${3:-1}
colmap_use_mask=${0:-1}

# Create directories
mkdir -p ${data_path}/{image,mask}

# Extract images from video
python scripts/extract_images.py -d ${data_path} -n ${n_images}

# Deblur images
python scripts/deblur_images.py -d ${data_path}

# Create masks
python scripts/create_masks.py -d ${data_path}

# Estimate camera parameters
export QT_QPA_PLATFORM=offscreen

if [ ${colmap_use_mask} == 1 ]; then 
    colmap automatic_reconstructor \
        --workspace_path=${data_path} \
        --image_path=${data_path}/image \
        --mask_path=${data_path}/mask \
        --camera_model=PINHOLE \
        --single_camera=${single_camera} \
        --sparse=1 \
        --use_gpu=0
else
    colmap automatic_reconstructor \
        --workspace_path=${data_path} \
        --image_path=${data_path}/image \
        --camera_model=PINHOLE \
        --single_camera=${single_camera} \
        --sparse=1 \
        --use_gpu=0
fi

colmap model_converter \
    --input_path ${data_path}/sparse/0 \
    --output_path ${data_path}/sparse/0 \
    --output_type TXT

# Normalize camera parameters
python scripts/convert_colmap_to_npz.py \
        -i ${data_path}

python scripts/preprocess_cameras.py \
    --source_dir ${data_path}

mv ${data_path}/cameras.npz ${data_path}/cameras_origin.npz
mv ${data_path}/cameras_new.npz ${data_path}/cameras.npz
