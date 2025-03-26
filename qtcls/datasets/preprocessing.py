# Copyright (c) QIU Tian. All rights reserved.

__all__ = [
    'preprocess_amazon_homo',
    'preprocess_elliptic',
    'preprocess_tfinance',
    'preprocess_tsocial',
    'preprocess_yelpchi_homo'
]

import os
import pickle

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from toad.transform import Combiner

from ..utils.misc import index_to_mask, mask_to_index
from ..utils.os import makedirs


def preprocess_amazon_homo(args, raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    amazon = loadmat(os.path.join(raw_dir, 'Amazon.mat'))
    net_upu = amazon['net_upu']
    net_usu = amazon['net_usu']
    net_uvu = amazon['net_uvu']
    net_hom = amazon['homo']
    num_nodes = amazon['features'].shape[0]
    y = amazon['label'].reshape(-1)

    X = np.asarray(amazon['features'].todense())
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    train_idx, test_idx, y_train, y_test = train_test_split(range(len(y)), y, stratify=y, train_size=0.4,
                                                            random_state=2, shuffle=True)
    val_idx, test_idx, y_val, y_test = train_test_split(test_idx, y_test, stratify=y_test, test_size=0.67,
                                                        random_state=2, shuffle=True)

    train_idx = pd.Series(train_idx)
    train_idx = train_idx[~train_idx.isin(np.arange(3305))].tolist()
    val_idx = pd.Series(val_idx)
    val_idx = val_idx[~val_idx.isin(np.arange(3305))].tolist()
    test_idx = pd.Series(test_idx)
    test_idx = test_idx[~test_idx.isin(np.arange(3305))].tolist()

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    src_nodes = net_hom.tocoo().col
    dst_nodes = net_hom.tocoo().row

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)

    graph.ndata['feat'] = torch.FloatTensor(X_std)
    graph.ndata['label'] = torch.LongTensor(y)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    graph.ndata['centrality'] = graph.in_degrees()
    pos = bin_encoding(graph, mask_to_index(graph.ndata['train_mask']), n_bin=args.n_bin)
    graph.ndata['pos'] = torch.FloatTensor(pos).contiguous()

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)


def preprocess_elliptic(args, raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    data = pickle.load(open(os.path.join(raw_dir, 'elliptic.dat'), 'rb'))

    X = data.x.numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    num_nodes = data.num_nodes
    src_nodes = data.edge_index[0]
    dst_nodes = data.edge_index[1]

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)

    graph.ndata['feat'] = torch.FloatTensor(X_std)
    graph.ndata['label'] = data.y
    graph.ndata['train_mask'] = data.train_mask
    graph.ndata['val_mask'] = data.val_mask
    graph.ndata['test_mask'] = data.test_mask

    graph.ndata['centrality'] = graph.in_degrees()
    pos = bin_encoding(graph, mask_to_index(graph.ndata['train_mask']), n_bin=args.n_bin)
    graph.ndata['pos'] = torch.FloatTensor(pos).contiguous()

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)


def preprocess_tfinance(args, raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    g, label_dict = load_graphs(os.path.join(raw_dir, 'tfinance'))
    g = g[0]
    num_nodes = g.num_nodes()

    X = g.ndata['feature'].numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    y = g.ndata['label'][:, 1].numpy()

    train_idx, test_idx, y_train, y_test = train_test_split(range(len(y)), y, stratify=y, train_size=0.4,
                                                            random_state=20230415, shuffle=True)
    val_idx, test_idx, y_val, y_test = train_test_split(test_idx, y_test, stratify=y_test, test_size=0.67,
                                                        random_state=20230415, shuffle=True)

    train_idx = torch.LongTensor(np.sort(train_idx))
    val_idx = torch.LongTensor(np.sort(val_idx))
    test_idx = torch.LongTensor(np.sort(test_idx))

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    src_nodes = g.edges()[0]
    dst_nodes = g.edges()[1]

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)

    graph.ndata['feat'] = torch.FloatTensor(X_std)
    graph.ndata['label'] = torch.LongTensor(y)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    graph.ndata['centrality'] = graph.in_degrees()
    pos = bin_encoding(graph, mask_to_index(graph.ndata['train_mask']), n_bin=args.n_bin)
    graph.ndata['pos'] = torch.FloatTensor(pos).contiguous()

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)


def preprocess_tsocial(args, raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    g, label_dict = load_graphs(os.path.join(raw_dir, 'tsocial'))
    g = g[0]
    num_nodes = g.num_nodes()

    X = g.ndata['feature'].numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    y = g.ndata['label'].numpy()

    train_idx, test_idx, y_train, y_test = train_test_split(range(len(y)), y, stratify=y, train_size=0.4,
                                                            random_state=20230415, shuffle=True)
    val_idx, test_idx, y_val, y_test = train_test_split(test_idx, y_test, stratify=y_test, test_size=0.67,
                                                        random_state=20230415, shuffle=True)

    train_idx = torch.LongTensor(np.sort(train_idx))
    val_idx = torch.LongTensor(np.sort(val_idx))
    test_idx = torch.LongTensor(np.sort(test_idx))

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    src_nodes = g.edges()[0]
    dst_nodes = g.edges()[1]

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)

    graph.ndata['feat'] = torch.FloatTensor(X_std)
    graph.ndata['label'] = torch.LongTensor(y)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    graph.ndata['centrality'] = graph.in_degrees()
    pos = bin_encoding(graph, mask_to_index(graph.ndata['train_mask']), n_bin=args.n_bin)
    graph.ndata['pos'] = torch.FloatTensor(pos).contiguous()

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)


def preprocess_yelpchi_homo(args, raw_dir, processed_dir, processed_file):
    makedirs(processed_dir, exist_ok=True)

    yelpchi = loadmat(os.path.join(raw_dir, 'YelpChi.mat'))
    net_rur = yelpchi['net_rur']
    net_rtr = yelpchi['net_rtr']
    net_rsr = yelpchi['net_rsr']
    net_hom = yelpchi['homo']
    y = yelpchi['label'].reshape(-1)
    num_nodes = yelpchi['features'].shape[0]

    X = np.asarray(yelpchi['features'].todense())
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    train_idx, test_idx, y_train, y_test = train_test_split(range(len(y)), y, stratify=y, train_size=0.4,
                                                            random_state=2, shuffle=True)
    val_idx, test_idx, y_val, y_test = train_test_split(test_idx, y_test, stratify=y_test, test_size=0.67,
                                                        random_state=2, shuffle=True)

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    src_nodes = net_hom.tocoo().col
    dst_nodes = net_hom.tocoo().row

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)

    graph.ndata['feat'] = torch.FloatTensor(X_std)
    graph.ndata['label'] = torch.LongTensor(y)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    graph.ndata['centrality'] = graph.in_degrees()
    pos = bin_encoding(graph, mask_to_index(graph.ndata['train_mask']), n_bin=args.n_bin)
    graph.ndata['pos'] = torch.FloatTensor(pos).contiguous()

    processed_path = os.path.join(processed_dir, processed_file)
    save_graphs(processed_path, graph)
    os.chmod(processed_path, 0o777)


def bin_encoding(graph, train_idx, n_bin, col_index=None):
    X = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    train_X = X.iloc[train_idx]
    train_y = pd.DataFrame(y[train_idx])

    combiner = Combiner()
    combiner.fit(train_X, train_y, method='dt', min_samples=0.01, n_bins=n_bin)
    if col_index is None or col_index == 'None':
        col_index = X.columns
    bin_encoded_X = combiner.transform(X[col_index], labels=False)

    bin_encoded_X = bin_encoded_X.values.astype(np.float64)
    bin_encoded_X = bin_encoded_X - bin_encoded_X.mean(axis=0)

    return bin_encoded_X
