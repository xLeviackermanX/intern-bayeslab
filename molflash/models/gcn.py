import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv

####################################
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import SGConv


class GCN(torch.nn.Module):

    def __init__(self, hid1: int = 128, hid2: int = 256, hid3: int = 128, lin1: int = 512, lin2 = 128, out: int = 1,
                 drop: int = 0.5, in_channel: int = 40):
        super(GCN, self).__init__()

        self.drop = drop
        self.conv1 = GCNConv(in_channel, hid1)
        self.conv2 = GCNConv(hid1, hid2)
        self.conv3 = GCNConv(hid2, hid3)
        self.l1 = nn.Linear(40 * hid3, lin1)
        self.l2 = nn.Linear(lin1, lin2)
        self.l3 = nn.Linear(lin2, out)
        self.l = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = self.l(self.conv1(x, edge_index))
        x = F.dropout(x, self.drop, training=self.training)
        x = self.l(self.conv2(x, edge_index))
        x = self.l(self.conv3(x, edge_index))
        self.grad_value = x.clone()
        x = x.view(int(len(x) / 40), -1)
        x = self.l(self.l1(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = self.l(self.l2(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = self.l3(x)
        x = torch.sigmoid(x)
        return x
    
    def __repr__(self):
        return {"hid1":hid1 , "hid2": hid2, "hid3": hid3}


    def cam(self):
        return self.l1.weight.data, self.grad_value


if __name__ == "__main__":
    params = {'in_channel': 40,
              'hid1': 128,
              'hid2': 256,
              'hid3': 128,
              'lin1': 512,
              'lin2': 128,
              'out': 1,
              'drop': 0.5,
              }
    model = GCN(**params)
    print(model)

