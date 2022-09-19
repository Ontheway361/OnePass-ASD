# -*- coding: utf-8 -*-
import numpy as np 
import os, cv2, glob, torch, random
import python_speech_features
from scipy.io import wavfile
from torch.utils import data
from IPython import embed

class AVA_ActiveSpeaker(data.Dataset):
    
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.load_annofile()
        self.split_lines_into_minibatch()
    
    def load_annofile(self):
        if self.mode == 'train':
            annofile = self.args.train_file
        else:
            annofile = self.args.val_file
        with open(annofile, 'r') as f:
            lines = f.readlines()
        f.close()
        annolines = []
        for line in lines:
            info = line.strip().split('\t')
            gtys = info[3][1:-1].split(', ')
            aline= info[:3] + gtys + [info[-1]]
            annolines.append(aline)
        # self.annolines = sorted(annolines, key=lambda data: (int(data[1]), int(data[-1])), reverse=True) 
        self.annolines = sorted(annolines, key=lambda data:int(data[1]), reverse=True) 
        print('there are %05d lines in %s' % (len(self.annolines), annofile.split('/')[-1]))
        
    def split_lines_into_minibatch(self):
        minibatch_list = []
        if self.mode == 'train':
            start_idx = 0
            while True:
                entity_len = int(self.annolines[start_idx][1])
                end_idx = min(len(self.annolines), start_idx + max(int(self.args.load_seqlen / entity_len), 1))
                minibatch = self.annolines[start_idx:end_idx]
                minibatch_list.append(minibatch)
                if end_idx == len(self.annolines):
                    break
                start_idx = end_idx
        else:
            minibatch_list = [[aline] for aline in self.annolines]
        if self.args.is_debug:
            minibatch_list = minibatch_list[:4]
            print('attention!, debug mode is going ...')
        self.minibatch_list = minibatch_list
        print('group %s annolines into %03d minibatches' % (len(self.annolines), len(self.minibatch_list)))
    
    def __len__(self):
        return len(self.minibatch_list)

    def cache_minibatch_audio(self, minibatch, unitnums):
        audiolist = []
        for entity_info in minibatch:
            audio_file = os.path.join(self.args.audio_dir, entity_info[0] + '.wav')
            _, audio = wavfile.read(audio_file)
            videofps = float(entity_info[2])
            # audio = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025*25/videofps, winstep=0.010*25/videofps)
            maxAudio = int(unitnums * 4)
            # print(audio.shape, unitnums, videofps)
            if audio.shape[0] < maxAudio:
                shortage = maxAudio - audio.shape[0]
                audio = np.pad(audio, (((0, shortage), (0, 0))), mode='wrap')  # maybe 'edge'
            # audio = audio[:maxAudio, :]
            audio = audio[:maxAudio]
            audiolist.append(audio)
        return torch.FloatTensor(np.array(audiolist))  
    
    def augment_image(self, image):
        typelist = ['orig', 'flip', 'crop', 'rotate']
        weights = [0.7, 0.05, 0.2, 0.05]
        aug_type = random.choices(typelist, weights)[0]
        if aug_type == 'flip':
            image = cv2.flip(image, 1)
        elif aug_type == 'crop':
            crop_w = int(self.args.face_size[0] * random.uniform(0.7, 1))
            crop_h = int(self.args.face_size[1] * random.uniform(0.8, 1))
            startx = np.random.randint(0, self.args.face_size[0] - crop_w)
            starty = np.random.randint(0, self.args.face_size[1] - crop_h)
            image = cv2.resize(image[starty:(starty+crop_h), startx:(startx+crop_w)], self.args.face_size)
        elif aug_type == 'rotate':
            center = (self.args.face_size[0] / 2, self.args.face_size[1] / 2)
            alpha = random.uniform(-15, 15)
            enlarge_factor = 1.0
            M = cv2.getRotationMatrix2D(center, alpha, enlarge_factor)
            image = cv2.warpAffine(image, M, self.args.face_size)
        return image

    def cache_minibatch_video(self, minibatch, unitnums):
        videolist = []
        for entity_info in minibatch:
            face_dir = os.path.join(self.args.video_dir, entity_info[0])
            facefiles = glob.glob('%s/*.jpg' % face_dir)
            facefiles = sorted(facefiles, key=lambda ffile:float(ffile.split('/')[-1][:-4])) # reverse=False
            faceslist = []
            for ffile in facefiles[:unitnums]:
                face = cv2.imread(ffile)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # TODO::Attention
                face = cv2.resize(face, self.args.face_size)
                if self.mode == 'train':
                    face = self.augment_image(face)
                face = (face / 255.0 - 0.4161) / 0.1688
                faceslist.append(face)
            if len(faceslist) != unitnums:
                print(entity_info[0], len(facefiles), len(entity_info[3:-1]))
            videolist.append(np.array(faceslist))
        return torch.FloatTensor(np.array(videolist))
    
    def cache_minibatch_label(self, minibatch, unitnums):
        labellist = []
        for entity_info in minibatch:
            label = np.array([int(p) for p in entity_info[3:-1][:unitnums]])
            labellist.append(label)
        return torch.LongTensor(np.array(labellist))

    def __getitem__(self, index):
        minibatch = self.minibatch_list[index].copy()
        unitnums = int(minibatch[-1][1])
        if self.mode == 'train':
            random.shuffle(minibatch)
        audios = self.cache_minibatch_audio(minibatch, unitnums)
        videos = self.cache_minibatch_video(minibatch, unitnums)
        labels = self.cache_minibatch_label(minibatch, unitnums)
        return (audios, videos, labels)


             
        





        
    
    
        
