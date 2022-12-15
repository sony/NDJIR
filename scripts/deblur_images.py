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

import cv2
import numpy as np
import argparse
import os, glob
from pathlib import Path

import gdown
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def main(args):
    # NAFNet (w/ REDS dataset for debluring task)
    if not os.path.exists('./experiments/pretrained_models/NAFNet-REDS-width64.pth'):
        gdown.download('https://drive.google.com/u/0/uc?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X', 
                        './experiments/pretrained_models/', quiet=False)
    opt_path = '/opt/NAFNet/options/test/REDS/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)

    fpaths = sorted(glob.glob(f"{args.dpath}/image/*"))
    os.makedirs(f"{args.dpath}/image_deblurred", exist_ok=True)
    for fpath in fpaths:
        print(f"Deblurring {fpath}...")

        img = imread(fpath)
        img = img2tensor(img)
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})
        if model.opt['val'].get('grids', False):
            model.grids()
        model.test()
        if model.opt['val'].get('grids', False):
            model.grids_inverse()        
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        
        path = Path(fpath)
        opath = f"{args.dpath}/image_deblurred/{path.name}"
        imwrite(sr_img, opath)

    os.rename(f"{args.dpath}/image", f"{args.dpath}/image_origin")
    os.rename(f"{args.dpath}/image_deblurred", f"{args.dpath}/image")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblur images, then change directory name.")
    parser.add_argument("-d", "--dpath", 
                        help="Path to the parent of `image` directory. E.g., <path_to_parent>/image")

    args = parser.parse_args()
    main(args)
