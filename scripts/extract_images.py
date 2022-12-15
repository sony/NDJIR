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
from pathlib import Path

import cv2
import numpy as np


def main(args):

    fpaths = sorted(glob.glob(f"{args.dpath}/*.mp4"))

    # Distribute number of images
    num_frames_per_video = []
    for i, fpath in enumerate(fpaths):
        path = Path(fpath)
        cap = cv2.VideoCapture(fpath)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames_per_video.append(frame_count)
    
    ratio_images_per_video = num_frames_per_video / np.sum(num_frames_per_video)
    print(f"Ratio of images used per video: {ratio_images_per_video}")
    num_images_per_video = ratio_images_per_video * args.num_images
    num_images_per_video = list(map(int, num_images_per_video))
    num_images_per_video = list(map(round, num_images_per_video))
    num_images_per_video = list(map(int, num_images_per_video))
        
    # Extract images
    cnt = 0
    for i, fpath in enumerate(fpaths):
        path = Path(fpath)
        cap = cv2.VideoCapture(fpath)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mod = frame_count // num_images_per_video[i]

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        j = 0
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret == True:
                cond = j % mod == 0 \
                    and j > args.n_frames_skip_first \
                    and j < (frame_count - args.n_frames_skip_last)
                if cond:
                    opath = f"{path.parent.absolute()}/image/{cnt:06d}.png"
                    if cv2.__version__ == "4.6.0":
                        image = cv2.flip(image, 0)
                        image = cv2.flip(image, 1)
                    cv2.imwrite(opath, image)
                    cnt += 1
                    print(f"{opath} written")
                j += 1
            else: 
                break

        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from videos.")
    parser.add_argument("-d", "--dpath", 
                        help="Directory path to videos from which images are extracted.")
    parser.add_argument("-n", "--num_images", type=int, default=100, 
                        help="Rough number of images to be extracted.")
    parser.add_argument("--n_frames_skip_first", type=int, default=10, 
                        help="Num. of frames to be skipped first.")
    parser.add_argument("--n_frames_skip_last", type=int, default=10, 
                        help="Num. of frames to be skipped last.")

    args = parser.parse_args()
    main(args)
