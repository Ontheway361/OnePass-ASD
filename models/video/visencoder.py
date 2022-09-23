# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.video.resnet import ResNet
# from models.video.reslite import ResLite
# from models.video.mobilenet import MobileNetV2
# from models.video.customizednet import CustomizedNet
from IPython import embed

class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class DSConv1d(nn.Module):
    def __init__(self):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1, dilation=1, groups=512, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False))

    def forward(self, x):
        out = self.net(x)
        return out + x

class visualTCN(nn.Module):
    def __init__(self):
        super(visualTCN, self).__init__()
        stacks = []        
        for x in range(5):
            stacks += [DSConv1d()]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(nn.Module):
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1))

    def forward(self, x):
        out = self.net(x)
        return out

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.frame_encoder = None
        # self.frame_encoder = CustomizedNet()
        # self.frame_encoder = ResLite()
        # self.frame_encoder = ResNet()
        # self.frame_encoder = MobileNetV2()
        # self.visualTCN = visualTCN()
        # self.visualConv1D = visualConv1D()
    
    def forward(self, x):
        x = self.frame_encoder(x)
        # x = x.transpose(1, 2)
        # x = self.visualTCN(x)
        # x = self.visualConv1D(x)
        # x = x.transpose(1, 2)
        return x 