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
import nnabla.functions as F
import numpy as np
import nnabla_ext.cuda
from nnabla.ext_utils import get_extension_context


from voxel_hash_feature_composite import query_on_voxel_hash as query_on_voxel_hash_composite
from voxel_hash_feature import compute_num_params

import time


def _bench(args):
    rng = np.random.RandomState(412)
    m, M = -1, 1    

    B = args.batch_size
    G0 = args.grid_size_base
    gf = args.growth_factor
    T0 = args.table_size_base
    L = args.n_levels
    D = args.feature_size

    query_data = m + rng.rand(B, 3) * (M - m)
    query = nn.Variable.from_numpy_array(query_data).apply(need_grad=True)
    n_params = compute_num_params(G0, gf, T0, D, L)
    initializer_data = rng.randn(n_params) * 0.01
    feature = nn.parameter.get_parameter_or_create("F0", (n_params, ), initializer_data)
    out0 = F.query_on_voxel_hash(query, feature, G0, gf, T0, L, D)
    out1 = query_on_voxel_hash_composite(query, feature, G0, gf, T0, L, D)

    def _bench_one(out):
        nnabla_ext.cuda.synchronize(device_id=args.device_id)
        st = time.perf_counter()
        for i in range(args.n_iters):
            out.forward() 
            out.backward(clear_buffer=True)# if not args.forward_only else None
        nnabla_ext.cuda.synchronize(device_id=args.device_id)
        et = time.perf_counter() - st
        return et

    et0 = _bench_one(out0)
    et1 = _bench_one(out1)# if args.forward_only else -1

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

    parser = argparse.ArgumentParser(description='Benchmark of voxel hash')
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--device_id', default="0", type=str)
    parser.add_argument('-B', '--batch_size', default=2**19, type=int)
    parser.add_argument('-G0', '--grid_size_base', default=16, type=int)
    parser.add_argument('-gf', '--growth_factor', default=1.5, type=float)
    parser.add_argument('-T0', '--table_size_base', default=2**15, type=int)
    parser.add_argument('-L', '--n_levels', default=16, type=int)
    parser.add_argument('-D', '--feature_size', default=2, type=int)
    parser.add_argument('--forward_only', action="store_true")

    args = parser.parse_args()    
    bench(args)
        