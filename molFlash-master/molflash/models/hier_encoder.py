import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from utils import *
from utils import MolGraph

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def get_init_state(self, fmess, init_state=None):
        h = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        return h if init_state is None else torch.cat((h, init_state), dim=0)

    def get_hidden_state(self, h):
        return h

    def GRU(self, x, h_nei):
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x, sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess, bgraph):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0  # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h

    def sparse_forward(self, h, fmess, submess, bgraph):
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_i = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid())
        self.W_o = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid())
        self.W_f = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid())
        self.W = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Tanh())

    def get_init_state(self, fmess, init_state=None):
        h = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        c = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        if init_state is not None:
            h = torch.cat((h, init_state), dim=0)
            c = torch.cat((c, torch.zeros_like(init_state)), dim=0)
        return h, c

    def get_hidden_state(self, h):
        return h[0]

    def LSTM(self, x, h_nei, c_nei):
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i(torch.cat([x, h_sum_nei], dim=-1))
        o = self.W_o(torch.cat([x, h_sum_nei], dim=-1))
        f = self.W_f(torch.cat([x_expand, h_nei], dim=-1))
        u = self.W(torch.cat([x, h_sum_nei], dim=-1))
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, fmess, bgraph):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0  # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            h, c = self.LSTM(fmess, h_nei, c_nei)
            h = h * mask
            c = c * mask
        return h, c

    def sparse_forward(self, h, fmess, submess, bgraph):
        h, c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h, c

#############################
class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential(
                nn.Linear(node_fdim + hidden_size, hidden_size),
                nn.ReLU()
        )

        if rnn_type == 'GRU':
            self.rnn = GRU(input_size, hidden_size, depth)
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth)
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph, mask):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0 #first node is padding

        return node_hiddens * mask, h


class GraphEncoder(nn.Module):
    def __init__(self, avocab, rnn_type, embed_size, hidden_size, depth):
        super(GraphEncoder, self).__init__()
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.atom_size = atom_size = avocab.size() + MolGraph.MAX_POS
        self.bond_size = bond_size = len(MolGraph.BOND_LIST)

        self.E_a = torch.eye( avocab.size() ).cuda()
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cuda()
        self.E_pos = torch.eye( MolGraph.MAX_POS ).cuda()

        self.encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depth)

    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        fnode1 = self.E_a.index_select(index=fnode[:, 0], dim=0)
        fnode2 = self.E_pos.index_select(index=fnode[:, 1], dim=0)
        hnode = torch.cat([fnode1, fnode2], dim=-1)

        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        return hnode, hmess, agraph, bgraph

    def forward(self, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hatom,_ = self.encoder(*tensors, mask=None)
        return hatom
