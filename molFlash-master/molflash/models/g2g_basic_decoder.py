import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn.models import InnerProductDecoder, VGAE, GAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from molflash.utils.preprocess import PreprocessingFunc
from rdkit import Chem
from torch_geometric.data import DataLoader
import torch.nn.functional as F

class BasicDecoder(nn.Module):

    def __init__(self, in_channels=40):
        super(BasicDecoder, self).__init__()
        self.lin1 = nn.Linear(1600, 3200)
        self.lin2 = nn.Linear(3200, 6400)
        self.lin5 = nn.Linear(6400, 9600)
        self.dropout = nn.Dropout(0.4)

    def forward(self, z):
        z = F.relu(self.lin1(z))
        z = F.relu(self.lin2(z))
        z = F.sigmoid(self.lin5(z))

        return z
