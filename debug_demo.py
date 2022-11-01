# -*- coding: utf-8 -*-

# from multiprocessing import Pipe, Process, Manager
from tabnanny import check
import time, torch
from IPython import embed


if __name__=='__main__':

    ckpt_file  = 'checkpoint/one_stage_ASD_baseline_author.pth'
    checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    for key, val in checkpoint.items():
        print(key, val.shape)