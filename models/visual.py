# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

# class Conv2plus1D(nn.Module):
#     def __init__(self, inplanes=1, outplanes=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)):
#         super(Conv2plus1D, self).__init__()
#         self.spatial_conv2d = nn.Conv2d(inplanes, outplanes, kernel_size[1:], stride[:1], padding[:1]), bias=False)
#         self.num_channels = 64 * 56 * 56
#         self.temporal_conv1d = nn.Conv1d(self.num_channels, self.num_channels)
#         self.spatial_bnrelu = nn.Sequential(
#             nn.BatchNorm2d(outplanes),
#             nn.ReLU())
#         self.temporal_bnrelu = nn.Sequential(
#             nn.BatchNorm1d(outplanes),
#             nn.ReLU())

#     def forward(self, x):
#         # x.shape = [B, T, C, W, H]
#         B, T, C, W, H = x.shape
#         x = x.reshape(-1, C, W, H)
#         x = self.spatial_conv2d(x)
#         x = self.spatial_bnrelu(x)
#         x = x.reshape(B, T, -1)
#         x = x.transpose(2, 1)
#         x = self.temporal_conv1d()

class SpatialTemporal(nn.Module):
    def __init__(self, inplanes=1, outplanes=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)):
        super(SpatialTemporal, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(outplanes, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
    
    def forward(self, x):
        return self.layer0(x)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride):
        super(BasicBlock, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.conv2a(x)
        if self.stride != 1:
            identity = self.downsample(identity)
        x = x + identity
        identity = x
        x = F.relu(self.outbna(x))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.conv2b(x)
        x = x + identity
        x = F.relu(self.outbnb(x))
        return x

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # self.layer0 = SpatialTemporal()
        self.layer1 = BasicBlock(64, 64, stride=1)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, x):
        # x.shape  = [B, T, W, H]
        '''
        ## Conv3D::spatialtemporal
        x = x.unsqueeze(1)
        x = self.layer0(x)
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(-1, C, H, W)
        '''
        ## Conv2D
        x = x.unsqueeze(2)
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(B, T, 512)
        return x

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
        self.frame_encoder = ResNet()
        self.visualTCN = visualTCN()
        self.visualConv1D = visualConv1D()
    
    def forward(self, x):
        x = self.frame_encoder(x)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)
        return x 