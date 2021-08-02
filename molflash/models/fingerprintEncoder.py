import os
import sys
import random
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm, trange
from collections import defaultdict
import traceback


class Highway(nn.Module):
    def __init__(self,size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f
    def forward(self,x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1- gate) * linear
        return x
    
class InScopeFilterNet(nn.Module):
    def __init__(self, product_fp_dim=8192, reaction_fp_dim=2048):
        super().__init__()
        self.prod_fp_dim = product_fp_dim
        self.react_fp_dim = reaction_fp_dim
        #product branch
        self.prod_l1 = nn.Linear(self.prod_fp_dim,599)
        self.dropout = nn.Dropout(0.8)
        self.highway = Highway(599, 6, f=F.elu)
        #reaction branch
        self.react_layer = nn.Linear(2048, 599)
    def forward(self,x):
        # print('in forward',np.array(x).shape,len(x))
        # self.x = torch.from_numpy(np.array(x)).float()
        self.x = x.float()
        
        self.x_prod = self.x[:,:self.prod_fp_dim]
        self.x_prod = torch.log(self.x_prod+1)
        self.x_react = self.x[:,self.prod_fp_dim:]
        prod_x = self.dropout(F.elu(self.prod_l1(self.x_prod)))
        prod_x = self.highway(prod_x)
        react_x = F.elu(self.react_layer(self.x_react))
        prod_norm = F.normalize(prod_x,dim=0,p=2)
        react_norm = F.normalize(react_x,dim=0,p=2)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_sim = cos(prod_norm, react_norm)
        out = torch.sigmoid(cosine_sim)
        return out

class FPSEncoder(nn.Module):
    def __init__(self, fp_dim=1024):
        super().__init__()
        self.fp_dim = fp_dim
        #product branch
        self.l1 = nn.Linear(self.fp_dim,599)
        self.dropout = nn.Dropout(0.8)
        self.highway = Highway(599, 6, f=F.elu)
        self.l2 = nn.Linear(599,128)
        self.l3 = nn.Linear(128,1)
    def forward(self,x):
        self.x = x.float()
        
        self.x = torch.log(self.x+1)
        self.x = self.dropout(F.elu(self.l1(self.x)))
        self.x = self.highway(self.x)
        out = torch.sigmoid(self.l3(F.elu(self.l2(self.x))))
        return out

