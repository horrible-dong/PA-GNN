# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PositionEmbedding']

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, d_node, l_path, d_path):
        super().__init__()
        self.scaler = nn.Parameter(torch.tensor(1.))
        self.proj = nn.Linear(d_node * l_path, d_path, bias=False)

    def forward(self, g, paths):
        """
        Args:
            paths: torch.Tensor([n_node, k_path, l_path])

        Returns:
            pos_embed: torch.Tensor([n_node, k_path, d_path])
        """
        pos_feat = torch.cat([g.ndata['pos'], torch.zeros_like(g.ndata['pos'][[-1]])])[paths]
        pos_embed = self.proj(pos_feat.flatten(-2) * self.scaler)

        return pos_embed
