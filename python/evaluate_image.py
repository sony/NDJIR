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

import numpy as np



def psnr(image=None, image_gt=None, mask_obj=None, normalize=lambda x: x * 255):
    mask_obj = np.ones_like(image) if mask_obj is None else mask_obj
    mask_obj = np.broadcast_to(mask_obj, image.shape)
    
    image = normalize(image)
    image_gt = normalize(image_gt)
    sse = np.sum(((image - image_gt) * mask_obj) ** 2)
    mse = sse / np.sum(mask_obj)

    val = 20 * np.log10(255) - 10 * np.log10(mse)
    return val
