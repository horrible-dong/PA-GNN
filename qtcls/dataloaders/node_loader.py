# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['NodeLoader']

import torch


class NodeLoader:
    def __init__(self, node_list, labels, batch_size, shuffle=False, drop_last=False):
        self.node_list = node_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        node_list, labels = self.node_list, self.labels

        if self.shuffle:
            shuffled_indices = torch.randperm(len(self.node_list))
            node_list, labels = node_list[shuffled_indices], labels[shuffled_indices]

        start_index, end_index = 0, 0

        for i in range(len(node_list) // self.batch_size):
            start_index, end_index = i * self.batch_size, (i + 1) * self.batch_size
            yield node_list[start_index: end_index], labels[start_index: end_index]

        if len(node_list) % self.batch_size != 0 and not self.drop_last:
            yield node_list[end_index:], labels[end_index:]

    def __len__(self):
        length = len(self.node_list)
        if self.drop_last:
            return length // self.batch_size
        else:
            from math import ceil
            return ceil(length / self.batch_size)
