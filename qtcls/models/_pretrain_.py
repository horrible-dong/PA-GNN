# Copyright (c) QIU Tian. All rights reserved.

"""
To temporarily use a pre-trained weight path,
you can specify it by command-line argument `--pretrain` / `-p`.

For long-term use of a pre-trained weight path,
it is preferable to write it here in `model_local_paths` or `model_urls`.

**Priority**: `--pretrain` > `model_local_paths` > `model_urls`
"""

# local paths (high priority)
model_local_paths = {
    'your_model_1': '/local/path/to/the/pretrained',
    'your_model_2': ['/local/path_1/to/the/pretrained', '/local/path_2/to/the/pretrained'],
}

# urls (low priority)
model_urls = {
    'your_model': 'url://to/the/pretrained',
}
