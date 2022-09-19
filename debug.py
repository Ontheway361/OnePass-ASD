# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import torchvision
import numpy as np
import pandas as pd

import models as asdlib 
from config import optimize_args
from dataset import AVA_ActiveSpeaker

from IPython import embed 

if __name__ == "__main__":
    
    ## MODEL
    # model = torchvision.models.resnet18()
    # print(model)
    # model = OnePassASD(None)
    # video_data = torch.randn([1, 16, 1, 112, 112])
    # audio_data = torch.randn([1, 9480])
    # print(model)
    # output = model(audio_data, video_data)
    # print(output.shape)
    
    ## DATASET
    args = optimize_args()
    loader = AVA_ActiveSpeaker(args, mode='val')
    audios, videos, labels = loader.__getitem__(1)
    print(audios.shape, videos.shape, labels.shape)
    # model = asdlib.OnePassASD_MultiHeads(args)
    # afeats = model.audio_encoder(audios)
    # vfeats = model.video_encoder(videos)
    # attlayer = asdlib.AttentionLayer(128, 8)
    # afeats = attlayer(afeats, vfeats)
    # print(afeats.shape)
    # output = model(audios, videos)
    # logits, afeats, vfeats = model(audios, videos)
    # print(logits.shape, afeats.shape, vfeats.shape)
    
    model = asdlib.SincDSNet()
    afeat = model(audios)
    print(afeat.shape)
    

    # aloss = asdlib.AuxAudioLoss()
    # vloss = asdlib.AuxVisualLoss()
    # closs = asdlib.ASDLoss()
    # c_loss = closs(logits, labels)
    # a_loss = aloss(afeats, labels)
    # v_loss = vloss(vfeats, labels)
    # print(a_loss, v_loss, c_loss)
    # print(len(c_loss), len(a_loss), len(v_loss))

    # loss = asdloss(output, labels)
    # print(loss[0], loss[-1])

    # tloader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=16)
    # # embed()
    # start_time = time.time()
    # for idx, (audios, videos, labels) in enumerate(tloader, start=1):
    #     print(audios.shape, videos.shape, labels.shape)
    #     if idx % 50 == 0:
    #         end_time = time.time()
    #         cost_time = (end_time - start_time) / 60
    #         print('already iterate %06d, total %06d, cost time %.4f mins' % (idx, eln(loader), cost_time))
     
    #     break

    # face = cv2.imread('1437.07.jpg')
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # face = cv2.resize(face, (112, 112))
    # for i in range(1):
    #     res = loader.augment_image(face)
    #     save_file = 'check/flip_%02d.jpg' % i
    #     cv2.imwrite(save_file, res)
    
    ## TMP
    # device = torch.cuda.current_device()
    # print(device, type(device))
    