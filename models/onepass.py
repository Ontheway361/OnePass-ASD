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
        self.video_ec = vlib.VideoNet()
        self.video_tm = slib.VideoTepModel(512, 128, 5)
        self.avfea_tm = slib.TemporalModel(args.tm_base, 256, 128, 2)
        self.drop_out = nn.Dropout(p=0.6)
        self.fully_connect = nn.Linear(128, 2)
    
    def forward(self, a, v):
        a = self.audio_ec(a)
        # a = self.audio_tm(a)
        v = self.video_ec(v)
        v = self.video_tm(v)
        f = torch.cat((a, v), dim=2)
        f = self.avfea_tm(f)
        f = f.reshape(-1, 128)
        f = self.drop_out(f)
        f = self.fully_connect(f)
        return (f, a, v)

