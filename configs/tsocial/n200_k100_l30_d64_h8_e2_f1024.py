# Copyright (c) QIU Tian. All rights reserved.

import os
import sys

sys.path.append('configs')
from _base_ import *

_ = os.path.split(__file__)[0]
_, dataset = os.path.split(_)

batch_size = 2000
n_bin = 32

model_kwargs = dict(
    d_node=10,
    n_path=200, k_path=100, l_path=30, d_path=64,
    n_head=8, n_encoder=2, d_feedforward=1024, dropout=0.1, activation='relu'
)

output_dir = f'{output_root}/{dataset}/{os.path.splitext(os.path.basename(__file__))[0]}'

print_freq = 400
