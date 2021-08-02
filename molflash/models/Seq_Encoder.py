import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout,bidirectional=True,batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # print('hidden_dim:',self.hid_dim)
        
        #src = [src len, batch size]
        print(src)
        print(src.shape)
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        # embedded = embedded.view(-1, 2, self.emb_dim)
        # print('enc emb:',embedded.shape)

        
        outputs, (hidden, cell) = self.rnn(embedded)
        # print('enc outputs::',outputs.shape, hidden.shape, cell.shape)
         #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell