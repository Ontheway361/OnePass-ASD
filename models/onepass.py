# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.audio as alib
import models.video as vlib
import models.sequence as slib
from IPython import embed

class OnePassASD(nn.Module):
    
    def __init__(self, args):
        super(OnePassASD, self).__init__()
        self.audio_ec = alib.AudioNet()
        self.video_en = vlib.VideoNet()
        self.audio_tm = slib.TemporalModel(args.tm_base, 128, 128, 1)
        self.video_tm = slib.TemporalModel(args.tm_base, 128, 128, 1)
        self.avfea_tm = slib.TemporalModel(args.tm_base, 256, 256, 2)
        self.drop_out = nn.Dropout(p=0.2)
        self.fully_connect = nn.Linear(256, 2)
    
    def forward(self, a, v):
        a = self.audio_ec(a)
        a = self.audio_tm(a)
        v = self.video_en(v)
        v = self.video_tm(v)
        f = torch.cat((a, v), dim=2)
        f = self.avfea_tm(f)
        f = f.reshape(-1, 256)
        f = self.fully_connect(f)
        return (f, a, v)

