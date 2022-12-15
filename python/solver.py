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

import nnabla as nn
import nnabla.solvers as S

import numpy as np

class Solvers:

    def __init__(self, conf):
        B, R = conf.train.batch_size, conf.train.n_rays
        self.learning_rate_weight = conf.train.base_learning_rate_weight \
            * (B * R) / (1 * 512)
        self.learning_rate_feat = conf.train.base_learning_rate_feat \
            * (B * R) / (1 * 512)

        self.solver_weight = S.Adam(0)
        self.solver_feat = S.Adam(0)

        self.conf = conf

    def set_parameters(self):
        params = nn.get_parameters()

        param_feats = {}
        param_weights = {}
        for param_name, param in params.items():
            if param_name.endswith("feature/F"):
                param_feats[param_name] = param
            else:
                param_weights[param_name] = param

        self.solver_weight.set_parameters(param_weights)
        self.solver_feat.set_parameters(param_feats)

    def weight_decay(self):
        self.solver_weight.weight_decay(self.conf.train.weight_decay)
        self.solver_feat.weight_decay(self.conf.train.weight_decay)
        

    def clip_grad_by_norm(self):
        if self.conf.train.clip_grad_norm <= 0:
            return
        self.solver_weight.clip_grad_by_norm(self.conf.train.clip_grad_norm)
        self.solver_feat.clip_grad_by_norm(self.conf.train.clip_grad_norm)

    def update(self):
        self.solver_weight.update()
        self.solver_feat.update()

    def zero_grad(self):
        self.solver_weight.zero_grad()
        self.solver_feat.zero_grad()

    def check_inf_or_nan_grad(self):
        return self.solver_weight.check_inf_or_nan_grad() \
            and self.solver_feat.check_inf_or_nan_grad()

    def update_learning_rate(self, i):
        lr = self.compute_learning_rate(i, self.learning_rate_weight)
        self.solver_weight.set_learning_rate(lr)
        
        lr = self.compute_learning_rate(i, self.learning_rate_feat)
        self.solver_feat.set_learning_rate(lr)

        # Addtionally update
        self.update_cos_anneal_ratio(i)
        self.update_light_visibility_gain(i)

    def compute_learning_rate(self, i, lr):
        conf = self.conf
        epoch = conf.train.epoch
        warmup_term_ratio = conf.train.warmup_term_ratio
        warmup_term = int(epoch * warmup_term_ratio)
        warmup_term = 0 if warmup_term < 1 else warmup_term
        lr_end_ratio = conf.train.learning_rate_end_ratio

        if i < warmup_term:
            lr = lr * i / warmup_term
        else:
            x = np.pi * (i - warmup_term) / (epoch - warmup_term)
            a = (1 - lr_end_ratio) * lr / (1 + np.cos(np.pi * warmup_term / epoch))
            b = a + lr_end_ratio * lr
            lr = np.cos(x) * a + b
        
        return lr

    def update_cos_anneal_ratio(self, i):
        conf = self.conf
        end_epoch = conf.train.epoch * conf.train.cos_anneal_term_ratio
        x = i / end_epoch
        ratio = 0.5 * np.cos(np.pi * x) + 0.5 if x < 1.0 else 1.0
        cos_anneal_ratio = nn.parameter.get_parameter_or_create("cos_anneal_ratio", (1, ),
                                        np.asarray([0.0]),
                                        False, False)
        cos_anneal_ratio.d = ratio

    def update_light_visibility_gain(self, i):
        gain = nn.parameter.get_parameter_or_create("/photogrammetric-light-network/gain", (1, ),
                                                    np.asarray([1.0]),
                                                    False, False)
        conf = self.conf
        M = conf.train.sigmoid_gain_lv_end
        b = (M + 1) * 0.5
        a = 1 - b
        g = a * np.cos(np.pi * i / conf.train.epoch) + b
        gain.d = g




    