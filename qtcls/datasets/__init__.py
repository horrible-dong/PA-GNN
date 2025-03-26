# Copyright (c) QIU Tian. All rights reserved.

from .preprocessing import *

__vars__ = vars()

_num_classes = {  # Required
    # Dataset names must be all in lowercase.
    'amazon_homo': 2,
    'elliptic': 2,
    'tfinance': 2,
    'tsocial': 2,
    'yelpchi_homo': 2,
}


def build_dataset(args):
    import os
    import dgl

    dataset_name = args.dataset.lower()
    raw_dir, processed_dir, processed_file = args.raw_dir, args.processed_dir, f'{dataset_name}.dgldata'
    processed_path = os.path.join(processed_dir, processed_file)

    if not os.path.exists(processed_path):
        print(f'Preprocessing {dataset_name} to {processed_path}')
        try:
            __vars__[f'preprocess_{dataset_name}'](args, raw_dir, processed_dir, processed_file)
        except KeyError:
            print(f"KeyError: Dataset '{dataset_name}' is not found.")
            exit(1)

    print(f'Loading {dataset_name} from {processed_path}')
    graph, _ = dgl.load_graphs(processed_path)
    graph = graph[0]

    return graph
