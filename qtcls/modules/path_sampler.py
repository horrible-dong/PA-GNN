# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PathSampler']

import dgl
import torch
from torch import nn


class PathSampler(nn.Module):
    def __init__(self, n_path, k_path, l_path):
        super().__init__()

        if l_path < 2:
            raise ValueError('The path length must be greater than 1.')

        self.n_path = n_path
        self.k_path = k_path
        self.l_path = l_path
        self.register_buffer('meta_mask', ~torch.tril(torch.full([l_path, l_path], fill_value=True)))  # <=> U - I

    def random_walk(self, g, start_nodes):
        """
        Args:
            start_nodes: torch.Tensor([n_node * n_path])

        Return:
            paths: torch.Tensor([n_node * n_path, l_path])
        """
        paths = dgl.sampling.random_walk(g, start_nodes, length=self.l_path - 1)[0]
        return paths

    def path_selection(self, g, paths):
        """
        Args:
            paths: torch.Tensor([n_node, n_path, l_path])

        Return:
            paths: torch.Tensor([n_node, k_path, l_path])
        """
        l_path, k_path = self.l_path, self.k_path
        centrality = torch.cat([g.ndata['centrality'], torch.zeros_like(g.ndata['centrality'][[-1]])])
        scores = centrality[paths]
        scores = torch.sum(scores, dim=-1)
        topk_indices = torch.topk(scores, k_path, dim=1)[1]
        paths = torch.gather(paths, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, l_path))

        return paths

    def forward(self, g, nodes):
        """
        Args:
            nodes: torch.Tensor([n_node])

        Return:
            paths: torch.Tensor([n_node, k_path, l_path])
        """
        n_node, n_path, k_path, l_path = len(nodes), self.n_path, self.k_path, self.l_path
        paths = self.random_walk(g, torch.repeat_interleave(nodes, n_path)).reshape(n_node, n_path, l_path)
        mask = self.meta_mask[torch.randint(l_path, (n_node, n_path))]
        paths[mask] = -1
        paths = self.path_selection(g, paths)

        return paths
