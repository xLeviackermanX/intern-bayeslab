import torch
import requests
from tqdm.auto import tqdm
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *

class CNN(nn.Module):
    def __init__(self, encoding):
        super(CNN, self).__init__()
        cnn_target_filters = [32, 64, 96],
        # cnn_target_kernels = [4, 8, 12],
        # hidden_dim_protein = 256,

        if encoding == 'protein':
            in_ch = [26] + [32, 64, 96]
            kernels = [4, 8, 12]
            layer_size = len(cnn_target_filters)
            self.conv = nn.ModuleList(
                [nn.Conv1d(in_channels=in_ch[i], out_channels=in_ch[i + 1], kernel_size=kernels[i]) for i in
                 range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))
            self.fc1 = nn.Linear(n_size_p, 256)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        # print('forward feature',x.shape)  the given SIZE should be torch.size([1,26,1000])
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v
