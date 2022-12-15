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

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(preds, targets, masks=None, ycbcr=False, args=None):
    dim = (1, 2, 3)
    denorm = 3
    
    if masks is None:
        masks = torch.ones((1, 1, preds.shape[2], preds.shape[3]), device=torch.device(f"cuda:{args.device_id}"))
        
    if ycbcr:
        preds = 0.299 * preds[0, 0, :, :] + 0.587 * preds[0, 1, :, :] + 0.114 * preds[0, 2, :, :]
        preds = preds[None, None, :, :]
        targets = 0.299 * targets[0, 0, :, :] + 0.587 * targets[0, 1, :, :] + 0.114 * targets[0, 2, :, :]
        targets = targets[None, None, :, :]
        dim = (2, 3)
        denorm = 1
        
    masked_mse = torch.sum(((preds - targets) ** 2) * masks, dim=dim) / (torch.sum(masks) * denorm)
    val = 10.0 * torch.log10(255.0 ** 2 / masked_mse)
    val = torch.sum(val)
    return val


def ssim(preds, targets, masks=None, args=None):
    if masks is None:
        masks = torch.ones((1, 1, preds.shape[2], preds.shape[3]), device=torch.device(f"cuda:{args.device_id}"))
    preds = preds.cpu().numpy()[0].transpose((1, 2, 0))
    targets = targets.cpu().numpy()[0].transpose((1, 2, 0))
    masks = masks.cpu().numpy()[0].transpose((1, 2, 0))
    _, val = structural_similarity(preds, targets, sigma=1.5, gaussian_weights=True, full=True, multichannel=True, data_range=255)
    val = np.sum(val * masks) / (3 * np.sum(masks))
    return torch.from_numpy(np.asarray(val))


def main(args):
    
    fpaths_rd = sorted(glob.glob(f"{args.dpath_rd}/*.png"))
    fpaths_gt = sorted(glob.glob(f"{args.dpath_gt}/*.png"))
    fpaths_ma = sorted(glob.glob(f"{args.dpath_ma}/*.png"))

    lpips_vgg = lpips.LPIPS(net='vgg', spatial=True).to(torch.device(f"cuda:{args.device_id}"))

    psnrs, psnrs_ma = [], []
    psnrs_y, psnrs_y_ma = [], []
    ssims, ssims_ma = [], []
    lpipss, lpipss_ma = [], []

    for i, fpaths in enumerate(zip(fpaths_rd, fpaths_gt, fpaths_ma)):
        fpath_rd, fpath_gt, fpath_ma = fpaths
        
        image_rd = cv2.imread(fpath_rd)
        image_rd = image_rd[:, :, ::-1].transpose((2, 0, 1))[None, :, :, :]
        image_gt = cv2.imread(fpath_gt)
        image_gt = image_gt[:, :, ::-1].transpose((2, 0, 1))[None, :, :, :]
        image_ma = cv2.imread(fpath_ma)
        image_ma = (image_ma[:, :, 0:1] > 127.5).transpose((2, 0, 1))[None, :, :, :].astype(np.float32)

        image_rd = torch.from_numpy(image_rd.copy()).to(torch.float32).to(torch.device(f"cuda:{args.device_id}"))
        image_gt = torch.from_numpy(image_gt.copy()).to(torch.float32).to(torch.device(f"cuda:{args.device_id}"))
        image_ma = torch.from_numpy(image_ma.copy()).to(torch.float32).to(torch.device(f"cuda:{args.device_id}"))

        if args.scale:
            dim = (0, 2, 3) if args.channel_wise_mean else (0, 1, 2, 3)
            dernom = 1 if args.channel_wise_mean else 3
            mean_gt = torch.sum(image_gt * image_ma, dim=dim, keepdim=True) / torch.sum(image_ma) / dernom
            mean_rd = torch.sum(image_rd * image_ma, dim=dim, keepdim=True) / torch.sum(image_ma) / dernom
            
            image_rd = image_rd - mean_rd + mean_gt
            image_rd = torch.clip(image_rd, 0.0, 255.0)
            image_rd[image_rd == 255.0] = 0.0 # coerce bg_color to black

        with torch.no_grad():
            # NOTE: when using mask in training, to align the paper's evaluation, 
            # multiply image_ma to image_gt.

            # PSNR
            val = psnr(image_rd, image_gt, args=args).cpu().numpy()
            val_ma = psnr(image_rd, image_gt, image_ma, args=args).cpu().numpy()
            psnrs.append(val)
            psnrs_ma.append(val_ma)

            # PSNR (ycbcr)
            val = psnr(image_rd, image_gt, ycbcr=True, args=args).cpu().numpy()
            val_ma = psnr(image_rd, image_gt, image_ma, ycbcr=True, args=args).cpu().numpy()
            psnrs_y.append(val)
            psnrs_y_ma.append(val_ma)
            
            # SSIM
            val = ssim(image_rd, image_gt, args=args).cpu().numpy()
            val_ma = ssim(image_rd, image_gt, image_ma, args=args).cpu().numpy()
            ssims.append(val)
            ssims_ma.append(val_ma)

            # LPIPS  
            val = lpips_vgg(image_rd / 255.0 * 2.0 - 1.0, 
                            image_gt / 255.0 * 2.0 - 1.0)
            val_ma = torch.sum(val * image_ma) / torch.sum(image_ma)
            val = torch.mean(val).cpu().numpy()
            val_ma = val_ma.cpu().numpy()

            lpipss.append(val)
            lpipss_ma.append(val_ma)

        print(f"Up to {i}: PSNR(RGB),PSNR(Y),SSIM,LPIPS")
        metrics = ",".join(map(str, [np.mean(psnrs), np.mean(psnrs_y), np.mean(ssims), np.mean(lpipss)]))
        metrics_ma = ",".join(map(str, [np.mean(psnrs_ma), np.mean(psnrs_y_ma), np.mean(ssims_ma), np.mean(lpipss_ma)]))
        print(metrics)
        print(metrics_ma)

    # Save
    suffix = "_scaled" if args.scale else ""
    opath = f"{args.dpath_rd}/eval_images{suffix}.txt"
    with open(opath, "w") as fp:
        fp.write(f"PSNR(RGB),PSNR(Y),SSIM,LPIPS\n")
        fp.write(f"{metrics}\n")
        fp.write(f"{metrics_ma}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate neurally-rendered images.")
    parser.add_argument("-d_rd", "--dpath_rd", help="Path to directory of rendered images")
    parser.add_argument("-d_gt", "--dpath_gt", help="Path to directory of GT images")
    parser.add_argument("-d_ma", "--dpath_ma", help="Path to directory of GT masks")
    parser.add_argument("--scale", action="store_true", 
                        help="Scale rendered images such that average is the TG average")
    parser.add_argument("-c", "--channel_wise_mean", action="store_true", 
                        help="Reduce for each channel")
    parser.add_argument("-d", "--device_id", default=0, help="Device ID")
    
    args = parser.parse_args()
    main(args)
