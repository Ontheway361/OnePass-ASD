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

class CustomizedNet(nn.Module):

    def __init__(self):
        super(CustomizedNet, self).__init__()
        interverted_residual_setting = [
            # e, c, n, s
            [2,  64, 2, 1],
            [2, 128, 2, 2],
            [2, 256, 2, 2],
            [1, 512, 2, 2],
        ]
        self.last_channel = 128
        self.layers = []
        self.layers.append(conv_bn(1, 64, k=7, s=2, p=3))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # building inverted residual blocks
        inp_c = 64
        for e, c, n, s in interverted_residual_setting:
            out_c = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.append(InvertedResidual(inp_c, out_c, stride, e))
                inp_c = out_c
        self.layers.append(conv_1x1x1_bn(inp_c, self.last_channel))
        self.layers.append(nn.AvgPool2d(kernel_size=4, stride=1))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # x.shape  = [B, T, W, H]
        x = x.unsqueeze(2)
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.layers(x)
        x = x.reshape(B, T, 128)
        return x

if __name__ == "__main__":
    model = CustomizedNet()
    params = model.parameters()
    totals = sum([p.nelement() for p in params])
    print('params %.4fM' % (totals / 1e6)) 
    inptensor = torch.randn((1, 1, 112, 112))
    outtensor = model(inptensor)
    print(outtensor.shape)