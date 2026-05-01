"""
RGCN model definition — must exactly match the architecture used
to train best_rgcn.pt.
Architecture:
    RGCNConv(788 → 256, num_relations=7)
    → ReLU → Dropout(0.5)
    → RGCNConv(256 → 2, num_relations=7)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .config import NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES, NUM_RELATIONS


class RGCN(torch.nn.Module):
    """
    2-layer Relational Graph Convolutional Network.
    Matches the architecture in '6. Step - Models/rgcn_model.py'.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        num_relations: int = NUM_RELATIONS,
    ):
        super().__init__()
        self.conv1 = RGCNConv(num_features, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x
