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
from pathlib import Path

import cv2
from rembg import remove

# Signature:
# remove(
#     data: Union[bytes, PIL.Image.Image, numpy.ndarray],
#     alpha_matting: bool = False,
#     alpha_matting_foreground_threshold: int = 240,
#     alpha_matting_background_threshold: int = 10,
#     alpha_matting_erode_size: int = 10,
#     session: Optional[rembg.session_base.BaseSession] = None,
#     only_mask: bool = False,
#     post_process_mask: bool = False,
# ) -> Union[bytes, PIL.Image.Image, numpy.ndarray]
# Docstring: <no docstring>
# File:      /usr/local/lib/python3.10/dist-packages/rembg/bg.py
# Type:      function


def main(args):
    base_path = f"{args.dpath}/image"
    if not os.path.exists(base_path):
        assert f"args.dpath ({args.dpath}) must contain 'image' directory"
    
    base_opath = f"{args.dpath}/mask"
    os.makedirs(base_opath, exist_ok=True)

    fpaths = glob.glob(f"{base_path}/*")
    for fpath in fpaths: 
        print(f"Background matting {fpath}...")
                
        input = cv2.imread(fpath)
        output = remove(input, alpha_matting=True,
                        only_mask=True, post_process_mask=True)

        # import ipdb
        # ipdb.set_trace()

        path = Path(fpath)
        # convention is different from DTU but must be same as image name for COLMAP: <imagename>.png
        opath = f"{base_opath}/{path.name}.png"
        cv2.imwrite(opath, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Background matting for images.")
    parser.add_argument("-d", "--dpath", 
                        help="Path to the parent of `image` directory. E.g., <path_to_parent>/image")

    args = parser.parse_args()
    main(args)
