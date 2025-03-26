# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['PAGNN', 'pagnn']

from torch import nn

from ..modules.path_embed import PathEmbedding
from ..modules.path_sampler import PathSampler
from ..modules.pos_embed import PositionEmbedding
from ..modules.transformer import Transformer


class PAGNN(nn.Module):
    def __init__(self,
                 d_node, n_path, k_path, l_path, d_path,
                 n_head=1, n_encoder=2, d_feedforward=1024, dropout=0.1, activation='relu',
                 num_classes=2):
        super().__init__()
        self.path_sampler = PathSampler(n_path, k_path, l_path)
        self.path_embedding = PathEmbedding(d_node, l_path, d_path)
        self.position_embedding = PositionEmbedding(d_node, l_path, d_path)
        self.transformer = Transformer(d_path, n_head, n_encoder, d_feedforward, dropout, activation)
        self.head = nn.Linear(d_path, num_classes)

    def forward(self, g, nodes):
        paths = self.path_sampler(g, nodes)
        path_embed = self.path_embedding(g, paths)
        pos_embed = self.position_embedding(g, paths)
        path_embed = self.transformer(path_embed, pos_embed=pos_embed)
        path_embed = path_embed.mean(1)
        logits = self.head(path_embed)

        return logits


def pagnn(**kwargs):
    return PAGNN(**kwargs)
