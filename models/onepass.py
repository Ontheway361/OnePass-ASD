# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.audio import SincDSNet, AudioEncoder 
from models.video import VisualEncoder
from models.sequence import TemporalModel, AttentionLayer
from IPython import embed

class OnePassASD(nn.Module):
    
    def __init__(self, args):
        super(OnePassASD, self).__init__()
        # self.audio_encoder = SincDSNet()
        self.audio_encoder = AudioEncoder(layers=[3, 4, 6, 3],  num_filters=[16, 32, 64, 128])
        self.video_encoder = VisualEncoder()
        self.temporalmodel = TemporalModel(tm_base=args.tm_base, input_size=256, hidden_size=128)
        self.fully_connect = nn.Linear(256, 2)
    
    def forward(self, a, v):
        a = self.audio_encoder(a)
        v = self.video_encoder(v)
        f = torch.cat((a, v), dim=2)
        x = self.temporalmodel(f)
        x = self.fully_connect(x)
        return x

class OnePassASD_MultiHeads(nn.Module):
    
    def __init__(self, args):
        super(OnePassASD_MultiHeads, self).__init__()
        # self.audio_encoder = SincDSNet()
        self.audio_encoder = AudioEncoder(layers=[3, 4, 6, 3],  num_filters=[16, 32, 64, 128])
        self.video_encoder = VisualEncoder()
        self.temporalmodel = TemporalModel(tm_base=args.tm_base, input_size=256, hidden_size=256)
        # self.v2a_attention = AttentionLayer(d_model=128, nhead=8)
        # self.a2v_attention = AttentionLayer(d_model=128, nhead=8)
        # self.avc_attention = AttentionLayer(d_model=256, nhead=8)
        self.fully_connect = nn.Linear(256, 2)
    
    def forward(self, a, v):
        # (a.shape, [B, T, F]), (v.shape, [B, T, H, W])
        a = self.audio_encoder(a)
        v = self.video_encoder(v)
        f = torch.cat((a, v), dim=2)
        f = self.temporalmodel(f)
        f = self.fully_connect(f)
        
        # atta = self.v2a_attention(a, v)
        # attv = self.a2v_attention(v, a)
        # attf = torch.cat((atta, attv), dim=2)
        # attf = self.avc_attention(attf, attf)
        # attf = attf.reshape(-1, 256)
        # logit = self.fully_connect(attf)
        # return (logit, atta, attv)
        return (f, a, v)



