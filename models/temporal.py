# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from IPython import embed

class RNNLayer(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, batch_first=True):
        super(RNNLayer, self).__init__()
        self.rnn =  nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=batch_first, bidirectional=False)
    
    def forward(self, x):
        # x.shape = (B, T, F)
        xlen = x.size(1)
        self.rnn.flatten_parameters() # TODO::Attention
        x, _ = self.rnn(x)
        return x

class LSTMLayer(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, batch_first=True):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=batch_first, bidirectional=False)

    def forward(self, x):
        # x.shape = (B, T, F)
        xlen = x.size(1)
        self.rnn.flatten_parameters() # TODO::Attention
        x, _ = self.rnn(x)
        return x

class GRULayer(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, batch_first=True):
        super(GRULayer, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, dropout=0.1, batch_first=batch_first, bidirectional=False)

    def forward(self, x):
        # x.shape = (B, T, F)
        xlen = x.size(1)
        self.rnn.flatten_parameters() # TODO::Attention
        x, _ = self.rnn(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, drop_ratio=0.1):
        super(AttentionLayer, self).__init__()
        self.multiheads = nn.MultiheadAttention(d_model, nhead, dropout=drop_ratio, batch_first=True)
        self.afflinears = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(p=drop_ratio),
            nn.Linear(d_model * 4, d_model))
        self.normalize1 = nn.LayerNorm(d_model)
        self.normalize2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_ratio)
        self.dropout2 = nn.Dropout(p=drop_ratio)
    
    def forward(self, src, tar):
        _src, _ = self.multiheads(tar, src, src)
        src = self.normalize1(src + self.dropout1(_src))
        _src = self.afflinears(src)
        src = self.normalize2(src + self.dropout2(_src))
        return src

class TemporalModel(nn.Module):
    def __init__(self, tm_base='rnn', input_size=256, hidden_size=128):
        super(TemporalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if tm_base == 'rnn':
            self.temporal_model = RNNLayer(input_size, hidden_size)
        elif tm_base == 'lstm':
            self.temporal_model = LSTMLayer(input_size, hidden_size)
        elif tm_base == 'gru':
            self.temporal_model = GRULayer(input_size, hidden_size)
        elif tm_base == 'attention':
            self.temporal_model()
            raise TypeError('ts_base must be rnn, lstm or gru')
    
    def forward(self, x):
        # x : [n_batch, seqlen, feat_dim]
        x = self.temporal_model(x)
        x = x.reshape(-1, self.hidden_size)
        return x