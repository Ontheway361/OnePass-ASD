# -*- coding: utf-8 -*-
import os, time, torch, torchvision

if __name__ == "__main__":

    model = torchvision.models.resnet50(num_classes=10)
    model = model.cuda()
    inptensor = torch.randn((96, 3, 224, 224)).cuda()
    while True:
        outtensor = model(inptensor)
        time.sleep(0.1)
    