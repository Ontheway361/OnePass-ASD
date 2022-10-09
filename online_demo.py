# -*- coding: utf-8 -*-
from importlib.util import find_spec
import os, cv2
from tracemalloc import start
from PIL import Image
import matplotlib.pyplot as plt
import sys, math, tqdm, time, copy, pickle, shutil
import subprocess, torch, torchvision, python_speech_features
import numpy as np
import torch.nn as nn
from scipy import signal
import scenedetect as slib
from scipy.io import wavfile
from scipy.interpolate import interp1d
import utils as ulib
import config as clib
import models as mlib
import dataset as dlib

from IPython import embed

class ASDDemo(object):

    def __init__(self, args):
        self.args = args
        self.data = {}
        self.modelzoos = {}
        self.softmax = nn.Softmax(dim=1)
        self.device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    def report_config_summary(self):
        print('%sEnvironment Versions%s' % ('-' * 26, '-' * 26))
        print("- Python     : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch    : {}".format(torch.__version__))
        print("- TorchVison : {}".format(torchvision.__version__))
        print("- use_gpu    : {}".format(self.args.use_gpu))
        print('- is_debug   : {} [Attention!]'.format(self.args.is_debug))
        print('- pretrain   : {}'.format(self.args.pretrain.split('/')[-1]))
        print('- print_freq : {}'.format(self.args.print_freq))
        print('- test_video : {}'.format(self.args.test_video))
        print('-' * 72)
    
    def load_model(self):
        print('step01. loading model ...')
        try:
            self.modelzoos['asd'] = mlib.OnePassASD(self.args)
            if self.device == 'cuda':
                self.modelzoos['asd'] = self.modelzoos['asd'].cuda()
            if os.path.isfile(self.args.pretrain):
                self.modelzoos['asd'] = ulib.load_weights(
                    model=self.modelzoos['asd'], 
                    pretrain=self.args.pretrain,
                    device=self.device)
            else:
                print('attention !!!, no pretrained model for ASD ...')
        except Exception as e:
            print(e)
            raise TypeError('errors occurs in loading step ...')
        try:
            self.modelzoos['detector'] = ulib.FaceDet(
                longside=self.args.longside, device=self.device)
        except Exception as e:
            print(e)
    
    @staticmethod
    def capture_camera_as_video(self):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        viofps = 25
        screen = (1280, 720) # [w, h]
        vio = cv2.VideoWriter('MacCap.mp4', fourcc, viofps, screen)
        cnt_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            vio.write(frame)
            cv2.waitKey(40)
            cnt_frames += 1
            if cnt_frames == 500:
                break
        cap.release()
        vio.release()
    
    def capture_online_frames(self, fps=25):
        cap = cv2.VideoCapture(0)
        wait = 1000 // fps
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.waitKey(40)
            

    def run_demo(self):
        # self.report_config_summary()
        # self.load_model()
        self.capture_camera()
 

if __name__ == "__main__":
    
    args = clib.demo_args()
    demo = ASDDemo(args)
    test_video = 'testset/videos/shawshank.mp4'
    demo.run_demo()


    # index = 0
    # plt.ion()
    # plt.cla()
    # fig = plt.figure('frame')
    # start_time = time.time()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     plt.clf()
    #     plt.imshow(frame)
    #     plt.axis('off')
        # plt.show()
        # plt.pause(0.001)
        # cv2.imshow('video_%04d' % index, frame)
        # key = cv2.waitKey(40)
        # fig.clf()
    #     index += 1
    #     if index % 50 == 0:
    #         break
    #     print('index %03d ...' % index)
    # finish_time = time.time()
    # cost_time = finish_time - start_time
    # print(cost_time)
    # plt.ioff()
    # cap.release()
    # cv2.destroyAllWindows()
    
    # img = Image.open('PriorBox_grid.png')
    # img.show()
    # img = cv2.imread('PriorBox_grid.png')
    # plt.imshow(img)
    # plt.show()

    