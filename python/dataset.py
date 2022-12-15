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

import glob
import os

import imageio
import nnabla as nn
import numpy as np
from nnabla.utils import image_utils
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource

from helper import load_K_Rt_from_P


class IDRDataSource(DataSource):
    '''
    Load dataset which format is same as in "Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance".
    '''

    def _get_data(self, position):
        img_idx = self._img_indices[position]
        image = self._images[img_idx]
        mask = self._masks[img_idx]
        intrinsic = self._intrinsics[img_idx]
        intrinsic_inv = self._intrinsics_inv[img_idx]
        pose = self._poses[img_idx]
        pixel_idx = self._pixel_idx[img_idx]
        xy = self._xy

        H, W, _ = image.shape
        R = H * W

        image = image.reshape((R, 3))
        mask = mask.reshape((R, 1))
        if self.conf.train.patch_ray_sampling:
            color_p, mask_p, xy = self._generate_patch_rays(image, mask)
        elif self.conf.train.mask_ray_sample_ratio > 0:
            color_p, mask_p, xy = self._generate_mask_rays(image, mask)
        else:
            color_p, mask_p, xy = image[pixel_idx], mask[pixel_idx], xy[pixel_idx]
        
        return color_p, mask_p, intrinsic, pose, xy

    def _generate_patch_rays(self, image, mask):
        H, W = self._H, self._W

        n = int(np.log2(self.conf.train.n_rays))
        is_height_small = self.rng.randint(0, 2)
        if is_height_small:
            nH = n // 2
            nW = n - nH
        else:
            nW = n // 2
            nH = n - nW
        pH = 2 ** nH
        pW = 2 ** nW

        H0 = self.rng.randint(0, H - pH)
        W0 = self.rng.randint(0, W - pW)
        H1 = H0 + pH
        W1 = W0 + pW

        xy = np.asarray(np.meshgrid(np.arange(W0, W1), np.arange(H0, H1))).T
        xy = xy.reshape((pH * pW, 2))

        idx = xy[:, 1] * W + xy[:, 0]
        color_p = image[idx]
        mask_p = mask[idx]

        return color_p, mask_p, xy

    def _generate_mask_rays(self, image, mask):
        n_rays_mask = int(self.conf.train.mask_ray_sample_ratio * self.conf.train.n_rays)
        n_rays_nomask = self.conf.train.n_rays - n_rays_mask
        
        # pixels for mask
        midxs = np.where(mask.flatten() >= 0.5)[0]
        midxs = midxs[self.rng.randint(0, len(midxs), n_rays_mask)]
        y = midxs // self._W
        x = midxs - y * self._W
        xy_mask = np.concatenate([x[:, None], y[:, None]], axis=-1)
        
        # pixels for nomask
        nidxs = np.where(mask.flatten() < 0.5)[0]
        nidxs = nidxs[self.rng.randint(0, len(nidxs), n_rays_nomask)]
        y = nidxs // self._W
        x = nidxs - y * self._W
        xy_nomask = np.concatenate([x[:, None], y[:, None]], axis=-1)
        
        idx = np.concatenate([midxs, nidxs], axis=0)
        color_p = image[idx]
        mask_p = mask[idx]
        xy = np.concatenate([xy_mask, xy_nomask], axis=0)

        return color_p, mask_p, xy

    def _load_data(self, path):
        # Images
        image_files = sorted(glob.glob(os.path.join(path, "image", "*")))
        images = np.asarray([image_utils.imread(f) for f in image_files])
        images = images / 255.0

        # Masks
        mask_files = sorted(glob.glob(os.path.join(path, "mask", "*")))
        masks = np.asarray([imageio.imread(f, as_gray=True)[:, :, np.newaxis] > 127.5
                            for f in mask_files]) * 1.0

        # Camera projection matrix and scale matrix for special correctness
        cameras = np.load(os.path.join(path, "cameras.npz"))
        world_mats = [cameras['world_mat_%d' % idx].astype(
            np.float32) for idx in range(len(images))]
        scale_mats = [cameras['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(len(images))]

        intrinsics, poses = [], []
        for W, S in zip(world_mats, scale_mats):
            P = W @ S
            P = P[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(P)
            intrinsics.append(intrinsic[:3, :3])
            poses.append(pose)

        self.scale = S[0, 0]
        self.trans = S[:3, 3]

        return images.astype(np.float32), masks, np.asarray(intrinsics), np.asarray(poses)

    
    def __init__(self, train=True, shuffle=False, rng=None, conf=None):
        super(IDRDataSource, self).__init__(shuffle=shuffle)
        self.path = conf.data_path
        self._n_rays = conf.train.n_rays
        self._train = train
        self._bounding_sphere_radius = conf.renderer.bounding_sphere_radius
        self.conf = conf

        self.scale = 1
        self.trans = np.zeros((3, ))

        self._images, self._masks, self._intrinsics, self._poses = self._load_data(self.path)

        self._intrinsics_inv = np.linalg.inv(self._intrinsics)
       
        # assume all images have same resolution
        H, W, _ = self._images[0].shape
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        self._xy = np.asarray([xx.flatten(), yy.flatten()]).T

        self._size = len(self._images)
        self._H = H
        self._W = W
        self._pixels = H * W
        self._variables = ('image', 'mask', 'intrinsic', 'pose', 'xy')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

        dname = os.path.split(self.path.rstrip("/"))[-1]
        nn.logger.info(f"--- Finish loading dataset ({dname}). ---")
        nn.logger.info(f"Num. of images = {self._size}")
        nn.logger.info(f"Num. of pixels (H x W) = {self._pixels} ({H} x {W})")
        nn.logger.info(f"Num. of random rays = {self._n_rays}")

    def reset(self):
        if self._shuffle:
            self._img_indices = self.rng.permutation(self._size)
        else:
            self._img_indices = np.arange(self._size)

        B = self._size
        R = self.conf.train.n_rays
        self._pixel_idx = self.rng.randint(0, self._pixels, (B, R))
        super(IDRDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (B, H, W, 3)."""
        return self._images.copy()

    @property
    def poses(self):
        return self._poses.copy()

    @property
    def intrinsics(self):
        return self._intrinsics.copy()

    @property
    def depth(self):
        return self._depth.copy()

    @property
    def masks(self):
        return self._masks.copy()


def data_iterator_idr(data_source,
                         batch_size,
                         rng=None,
                         with_memory_cache=False,
                         with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`IDRDataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`IDRDataSource` is able to store all data into memory.
    '''
    return data_iterator(data_source,
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)
