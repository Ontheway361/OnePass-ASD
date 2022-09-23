# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from IPython import embed

def conv_bn(inp, oup, k, s, p):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    
    def __init__(self, inp, oup, stride, expand_ratio=4):
        
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AudioNet(nn.Module):

    def __init__(self):
        super(AudioNet, self).__init__()
        interverted_residual_setting = [
            # e, c, n, s
            [2,  16, 2, 1],
            [2,  32, 2, 2],
            [2,  64, 2, 2],
            [1, 128, 2, 1],
        ]
        self.last_channel = 128
        self.layers = []
        self.layers.append(conv_bn(1, 16, k=7, s=(2, 1), p=3))
        # building inverted residual blocks
        inp_c = 16
        for e, c, n, s in interverted_residual_setting:
            out_c = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.append(InvertedResidual(inp_c, out_c, stride, e))
                inp_c = out_c
        self.layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        ## x.shape  = [B, T, F]
        B, _, _ = x.shape
        x = x.unsqueeze(1).transpose(2, 3) 
        x = self.layers(x)
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.view((B, self.last_channel, -1))
        x = x.transpose(1, 2)
        return x