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


import nnabla.functions as F
import numpy as np

"""
This specular BRDF models assumes the white (monochromatic) light.
"""

def dot(u, v, with_mask=False, eps=1e-8):
    """
    Args: 
        u: direction0 (B, R, M, 3)
        v: direction1 (B, R, M, 3)
    """
    B, R, M, _ = u.shape
    uv = F.sum(u * v, axis=-1, keepdims=True)
    uv = F.reshape(uv, (B, R, M, 1))
    mask = F.greater_scalar(uv, eps).apply(need_grad=False)
    uv = F.maximum_scalar(uv, eps)

    if with_mask:
        return uv, mask
    return uv


def filament_specular_brdf(normal, view_dir, light_dir, roughness, specular_color, conf): 
    """
    Note: all directional vector must be unit-vector normalized.

    Args: 
        norml: Normal (B, R, 3)
        view_dir: Viewing direction (B, R, 1, 3)
        light_dir: Lihgt direction (B, R, M, 3)
        roughness: Roughess (B, R, 1)
        specular_color: Specular color (B, R, 3)
    """

    B, R, _ = normal.shape
    _, _, M, _ = light_dir.shape

    def reshape_bcast(x, feature_size):
        x = F.reshape(x, (B, R, 1, feature_size))
        x = F.broadcast(x, (B, R, M, feature_size))
        return x

    normal = reshape_bcast(normal, 3)
    view_dir = reshape_bcast(view_dir, 3)
    roughness = reshape_bcast(roughness, 1)
    specular_color = reshape_bcast(specular_color, conf.specular_reflectance_network.channels)

    half_dir = light_dir + view_dir
    half_dir = half_dir / F.norm(half_dir, axis=half_dir.ndim - 1, keepdims=True)

    a2 = roughness ** 2

    eps_dot = conf.renderer.eps_dot
    nol, mask_nol = dot(normal, light_dir, True, eps_dot)
    nov, mask_nov = dot(normal, view_dir, True, eps_dot)
    noh, mask_noh = dot(normal, half_dir, True, eps_dot)
    
    eps = 1e-6

    def specular_D():
        denorm = np.pi * (noh ** 2 * (a2 - 1) + 1) ** 2
        denorm = denorm + eps
        D = a2 / denorm
        return D

    def specular_G():
        # not used, just as reference
        def specular_G1(nou):
            denorm = nou + (a2 + (1 - a2) * nou ** 2) ** 0.5
            denorm = denorm + eps
            G1 = 2 * nou / denorm
            return G1
        G = specular_G1(nol) * specular_G1(nov)
        return G

    def specular_V():
        def specular_V1(nou):
            denorm = nou + (a2 + (1 - a2) * nou ** 2) ** 0.5
            denorm = denorm + eps
            V1 = 1 / denorm
            return V1
        V1 = specular_V1(nol) * specular_V1(nov)
        return V1

    def specular_F():
        voh = dot(view_dir, half_dir, False, eps_dot)
        Fs = specular_color + (1 - specular_color) * (1 - voh) ** 5
        return Fs

    if conf.specular_brdf.sampling == "importance":
        V, Fs = specular_V(), specular_F()
        voh = dot(view_dir, half_dir, False, eps_dot)
        noh = dot(normal, half_dir, False, eps_dot)
        sBRDF = V * Fs * (4 * voh / noh)
        sBRDF = sBRDF * (mask_nol * mask_nov * mask_noh)
    elif conf.specular_brdf.sampling == "uniform":
        D, V, Fs = specular_D(), specular_V(), specular_F()
        sBRDF = np.pi * D * V * Fs
        sBRDF = sBRDF * (mask_nol * mask_nov * mask_noh)
        
    return sBRDF, nol


def ue4_specular_brdf(normal, view_dir, light_dir, roughness, specular_color, conf): 
    """
    Args: 
        norml: Normal (B, R, 3)
        view_dir: Viewing direction (B, R, 1, 3)
        light_dir: Lihgt direction (B, R, M, 3)
        roughness: Roughess (B, R, 1)
        specular_color: Specular color (B, R, 3)
    """

    B, R, _ = normal.shape
    _, _, M, _ = light_dir.shape

    def reshape_bcast(x, feature_size):
        x = F.reshape(x, (B, R, 1, feature_size))
        x = F.broadcast(x, (B, R, M, feature_size))
        return x

    normal = reshape_bcast(normal, 3)
    view_dir = reshape_bcast(view_dir, 3)
    roughness = reshape_bcast(roughness, 1)
    specular_color = reshape_bcast(specular_color, conf.specular_reflectance_network.channels)

    half_dir = light_dir + view_dir
    half_dir = half_dir / F.norm(half_dir, axis=half_dir.ndim - 1, keepdims=True)

    a = roughness ** 2
    a2 = a ** 2

    eps_dot = conf.renderer.eps_dot
    nol, mask_nol = dot(normal, light_dir, True, eps_dot)
    nov, mask_nov = dot(normal, view_dir, True, eps_dot)
    noh, mask_noh = dot(normal, half_dir, True, eps_dot)

    eps = 1e-6

    def specular_D():
        noh = dot(normal, half_dir, False, eps_dot)
        denorm = np.pi * (noh ** 2 * (a2 - 1) + 1) ** 2
        denorm = denorm + eps
        D = a2 / denorm
        return D

    def specular_G():
        k = (roughness + 1) ** 2 / 8
        def specular_G1(nou):
            denorm = nou * (1 - k) + k
            denorm = denorm + eps
            G1 = nou / denorm
            return G1
        G = specular_G1(nol) * specular_G1(nov)
        return G

    def specular_F():
        voh = dot(view_dir, half_dir, False, eps_dot)
        power = (-5.55473 * voh - 6.98316) * voh
        Fs = specular_color + (1 - specular_color) * 2 ** power
        return Fs

    if conf.specular_brdf.sampling == "importance":
        G, Fs = specular_G(), specular_F()
        voh = dot(view_dir, half_dir, False, eps_dot)
        noh = dot(normal, half_dir, False, eps_dot)
        sBRDF = G * Fs * (voh / (noh * nov))
        sBRDF = sBRDF * (mask_nol * mask_nov * mask_noh)
    elif conf.specular_brdf.sampling == "uniform":
        D, G, Fs = specular_D(), specular_G(), specular_F()
        sBRDF = np.pi * D * G * Fs / (4 * nov * nol)
        sBRDF = sBRDF * (mask_nol * mask_nov * mask_noh)

    return sBRDF, nol


def specular_brdf_model(normal, view_dir, light_dir, roughness, specular_color, conf): 
    models = dict(filament=filament_specular_brdf, 
                  ue4=ue4_specular_brdf)
    model = models[conf.specular_brdf.model]
    sBRDF, cos = model(normal, view_dir, light_dir, roughness, specular_color, conf)
    return sBRDF, cos
