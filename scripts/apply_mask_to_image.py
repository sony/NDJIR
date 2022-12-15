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
from pathlib import Path

import cv2


def main(args):
    image = cv2.imread(args.ipath)
    mask = cv2.imread(args.mpath)
    mask = mask / 255.0

    image_ma = image * mask
    if args.white_bg:
        image_ma = image_ma + (1.0 - mask) * 255.0

    path = Path(args.ipath)
    name = path.stem
    suffix = path.suffix
    cv2.imwrite(f"./{name}_ma{suffix}", image_ma)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Write masked image under the currenct directory.")
    parser.add_argument("-i", "--ipath", help="Path to image")
    parser.add_argument("-m", "--mpath", help="Path to mask")
    parser.add_argument("--white-bg", action="store_true")
                        
    args = parser.parse_args()
    main(args)
