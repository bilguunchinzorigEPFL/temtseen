import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Predictor(nn.Module):
    def __init__(self, batchSize, device):
        super(Predictor, self).__init__()
        # self.seqModel = nn.LSTM(16, 16, 5)
        width = 252
        self.linear = nn.Linear(width + 4, 1)
        self.src_mask = None
        # self.h0 = torch.randn(5, batchSize, 16, device=device)
        # self.c0 = torch.randn(5, batchSize, 16, device=device)
        self.ninp = width
        encoder_layers = TransformerEncoderLayer(width + 4, width + 4, width + 4, 0.1)
        self.seqModel = TransformerEncoder(encoder_layers, 4)
        self.encoder = nn.Embedding(1014, width + 4)
        self.pos_encoder = PositionalEncoding(width + 4, 0.1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, X, D, mask):
        X = X.permute(1, 0, 2)

        enc = self.encoder(X[:, :, 0].type(torch.long)) * math.sqrt(self.ninp)
        enc = self.pos_encoder(enc)
        inputMask = (X.sum(dim=2) > 0).permute(1, 0)
        # X = torch.cat((
        #     enc,
        #     (D / 3600).repeat(X.shape[0], 1).reshape(X.shape[0], X.shape[1], 1) / 10,
        #     X[:, :, 1:] / 10
        # ), axis=2)
        out = self.seqModel(enc, src_key_padding_mask=inputMask)
        # out, _ = self.seqModel(X.permute(1, 0, 2), (self.h0, self.c0))
        pred = F.sigmoid(self.linear(out.permute(1, 0, 2))).exp().reshape(mask.shape) * mask
        # pred = X * (D / X.sum(dim=1)).repeat(X.shape[1], 1).permute(1, 0)
        if np.isnan(pred.sum().item()):
            lol = ""
        return pred
