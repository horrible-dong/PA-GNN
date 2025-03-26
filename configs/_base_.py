# Copyright (c) QIU Tian. All rights reserved.

# runtime
device = 'cuda'
seed = 42
batch_size = 256
epochs = 50
clip_max_norm = 1.0
eval_interval = 1
num_workers = None  # auto
pin_memory = True
print_freq = 50
amp = True

# dataset
raw_dir = './data/raw'
processed_dir = './data/processed'
dataset = ...

# data preprocessing
n_bin = ...

# model
model = 'pagnn'

# criterion
criterion = 'default'

# optimizer
optimizer = 'adam'
optimizer_kwargs = dict(lr=1e-3, weight_decay=5e-4)

# lr_scheduler
scheduler = 'plateau'
scheduler_kwargs = dict(mode='min', factor=0.5, patience=5, verbose=False)

# evaluator
evaluator = 'default'

# loading
no_pretrain = True

# saving
save_interval = 10
output_root = './runs'
