import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SGConv


class SGEncoder(nn.Module):
    """
    Implementation of  Simple Graph Convolutional
    """
    def __init__(self, in_channels=40, hidden_channels=40, out_channels=40, exp: int = 0, layers= [40, 128, 256, 128, 512, 128, 1]):
        super(SGEncoder, self).__init__()
        self.exp = exp
        self.drop =0.5
        if self.exp == 0:
            self.gcn1 = SGConv(in_channels, hidden_channels)
            self.gcn2 = SGConv(hidden_channels, out_channels)
        else:
            self.gcn1 = SGConv(layers[0], layers[1])
            self.gcn2 = SGConv(layers[1], layers[2])
            self.gcn3 = SGConv(layers[2], layers[3])
            self.l1 = nn.Linear(40 * layers[3], layers[4])
            self.l2 = nn.Linear(layers[4], layers[5])
            self.l3 = nn.Linear(layers[5], layers[6])
            self.l = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        if self.exp == 0:
            return x.view(1600)
        else:
            x = F.relu(x)
            x = F.relu(self.gcn3(x,edge_index))
            self.grad_value = x.clone()
            x = x.view(int(len(x) / 40), -1)
            x = self.l(self.l1(x))
            x = F.dropout(x, self.drop, training=self.training)
            x = self.l(self.l2(x))
            x = F.dropout(x, self.drop, training=self.training)
            x = self.l3(x)
            x = torch.sigmoid(x)
            return x
