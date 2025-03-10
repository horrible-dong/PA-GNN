# Copyright (c) QIU Tian. All rights reserved.

import os
import sys

sys.path.append('configs')
from _base_ import *

_ = os.path.split(__file__)[0]
_, dataset = os.path.split(_)

n_bin = 4

model_kwargs = dict(
    d_node=10,
    n_path=200, k_path=150, l_path=5, d_path=128,
    n_head=1, n_encoder=2, d_feedforward=1024, dropout=0.1, activation='relu'
)

optimizer_kwargs = dict(lr=1e-4, weight_decay=5e-4)

output_dir = f'{output_root}/{dataset}/{os.path.splitext(os.path.basename(__file__))[0]}'
