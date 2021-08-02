"""
 G2G implementation using a GCN Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import WLConv


class WLEncoder(nn.Module):
    """
    Implementation of Weisfeiler Lehman  Conv
    """
    def __init__(self):  # in_channels=40, hidden_channels=40, out_channels=40):
        super(WLEncoder, self).__init__()
        self.conv1 = WLConv()
        self.conv2 = WLConv()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.view(1600)
