import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SGConv


class SGEncoder(nn.Module):
    """
    Implementation of  Simple Graph Convolutional
    """
    def __init__(self, in_channels=40, hidden_channels=40, out_channels=40):
        super(SGEncoder, self).__init__()
        self.gcn1 = SGConv(in_channels, hidden_channels)
        self.gcn2 = SGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x.view(1600)