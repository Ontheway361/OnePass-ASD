# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, labels=None):
        if labels == None:
            x = F.softmax(x, dim=-1)
            score = x[:,1].view(-1)
            score = score.detach().cpu().numpy()
            return score
        else:
            labels = labels.reshape((-1))
            nloss = self.criterion(x, labels)
            score = F.softmax(x, dim=-1)
            predy = torch.round(score)[:,1]
            nhits = (predy == labels).sum().item()
            return (nloss, score, predy, nhits)

class AuxAudioLoss(nn.Module):

    def __init__(self):
        super(AuxAudioLoss, self).__init__()
        self.fullyconnect = nn.Linear(128, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        labels = labels.reshape((-1))	
        x = self.fullyconnect(x)	
        x = x.reshape(-1, 2)
        aloss = self.criterion(x, labels)

        return aloss

class AuxVisualLoss(nn.Module):

    def __init__(self):
        super(AuxVisualLoss, self).__init__()
        self.fullyconnect = nn.Linear(128, 2)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, labels):
        labels = labels.reshape((-1))
        x = self.fullyconnect(x)
        x = x.reshape(-1, 2)
        vloss = self.criterion(x, labels)
        return vloss