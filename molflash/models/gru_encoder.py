import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAEencoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocabulary = vocab
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        self.encoder_rnn = nn.GRU(
            d_emb,
            config.q_d_h,
            num_layers=config.q_n_layers,
            batch_first=True,
            dropout=config.q_dropout if config.q_n_layers > 1 else 0,
            bidirectional=config.q_bidir)


        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)


    def forward(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)
        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss
