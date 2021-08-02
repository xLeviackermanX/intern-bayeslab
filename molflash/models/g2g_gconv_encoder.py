"""
 G2G implementation using a GCN Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv


class GraphConvEncoder(nn.Module):
    """
    Implementation of Graph Conv
    """
    def __init__(self, in_channels=40, hidden_channels=40, out_channels=40):
        super(GraphConvEncoder, self).__init__()
        self.gcn1 = GraphConv(in_channels, hidden_channels)
        self.gcn2 = GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x.view(1600)
