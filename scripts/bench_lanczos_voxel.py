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

import time

import nnabla as nn
import nnabla.functions as F
import nnabla_ext.cuda
import numpy as np
from lanczos_voxel_feature_composite import \
    lanczos_query_on_voxel as lanczos_query_on_voxel_composite
from nnabla.ext_utils import get_extension_context


def _bench(args):
    rng = np.random.RandomState(412)
    m, M = -1, 1    

    B = args.batch_size
    G = args.grid_size_base
    D = args.feature_size

    query_data = m + rng.rand(B, 3) * (M - m)
    query = nn.Variable.from_numpy_array(query_data).apply(need_grad=True)
    initializer_data = rng.randn(G, G, G, D) * 0.01
    feature = nn.parameter.get_parameter_or_create("F0", (G, G, G, D, ), initializer_data)
    out0 = F.lanczos_query_on_voxel(query, feature, [-1] * 3, [1] * 3)
    out1 = lanczos_query_on_voxel_composite(query, feature, -1, 1)

    def _bench_one(out):
        nnabla_ext.cuda.synchronize(device_id=args.device_id)
        st = time.perf_counter()
        for i in range(args.n_iters):
            out.forward() 
            out.backward(clear_buffer=True) if not args.forward_only else None
        nnabla_ext.cuda.synchronize(device_id=args.device_id)
        et = time.perf_counter() - st
        return et

    et0 = _bench_one(out0)
    et1 = _bench_one(out1)

    return et0, et1

def bench(args):
    ctx = get_extension_context('cudnn', device_id=args.device_id)
    nn.set_default_context(ctx)

    _, _ = _bench(args)
    et0, et1 = _bench(args)
    print(f"Elapsed time (monolithic) [s] = {et0}")
    print(f"Elapsed time (composite) [s] = {et1}")


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark of voxel grid feature with Lanczos filter.')
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--device_id', default="0", type=str)
    parser.add_argument('-B', '--batch_size', default=2**19, type=int)
    parser.add_argument('-G', '--grid_size_base', default=256, type=int)
    parser.add_argument('-D', '--feature_size', default=4, type=int)
    parser.add_argument('--forward_only', action="store_true")

    args = parser.parse_args()    
    bench(args)
        