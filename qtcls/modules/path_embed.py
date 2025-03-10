# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PathEmbedding']

import torch
from torch import nn


class PathEmbedding(nn.Module):
    def __init__(self, d_node, l_path, d_path):
        super().__init__()
        self.proj = nn.Linear(d_node * l_path, d_path, bias=False)

    def forward(self, g, paths):
        """
        Args:
            paths: torch.Tensor([n_node, k_path, l_path])

        Returns:
            path_embed: torch.Tensor([n_node, k_path, d_path])
        """
        path_feat = torch.cat([g.ndata['feat'], torch.zeros_like(g.ndata['feat'][[-1]])])[paths]
        path_embed = self.proj(path_feat.flatten(-2))

        return path_embed
