# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

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
    layer_channels = {
        'layer0' : (1, 64),
        'layer1' : (64, 64),
        'layer2' : (64, 128),
        'layer3' : (128, 256),
        'layer4' : (256, 512), 
    }
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
        ## Conv3D::spatialtemporal
        # x = x.unsqueeze(1)
        # x = self.layer0(x)
        # B, C, T, H, W = x.shape
        # x = x.transpose(1, 2)

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