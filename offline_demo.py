# -*- coding: utf-8 -*-
import os, cv2, sys, math, tqdm, time, copy, pickle, shutil
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
            # self.modelzoos['detector'] = ulib.FaceDetector(device=self.device)
            self.modelzoos['detector'] = ulib.FaceDet(
                longside=self.args.longside, device=self.device)
        except Exception as e:
            print(e)
        
    def parse_video(self, test_video='', is_25_fps=False):
        print('step02. translating test video ...')
        video_file = self.args.test_video
        tmp_vfile, tmp_afile = '', ''
        if len(test_video) and os.path.exists(test_video):
            video_file = test_video
            self.args.test_video = video_file
        try:
            base_name = video_file.split('/')[-1].split('.')[0]
            tmp_dir = os.path.join(self.args.test_dir, 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_vfile = os.path.join(tmp_dir, '%s.mp4' % base_name)
            tmp_afile = os.path.join(tmp_dir, '%s.wav' % base_name)
            if not (os.path.exists(tmp_vfile) and os.path.exists(tmp_afile)):
                if is_25_fps:
                    shutil.copy(video_file, tmp_vfile)
                else:
                    cmd = 'ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic' % ( 
                        video_file, self.args.threads, tmp_vfile)
                    subprocess.call(cmd, shell=True, stdout=None)
                cmd = 'ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic' % (
                    tmp_vfile, self.args.threads, tmp_afile)
                subprocess.call(cmd, shell=True, stdout=None)
        except Exception as e:
            print(e)
        self.data['vfile'] = tmp_vfile
        self.data['afile'] = tmp_afile
    
    def load_video_frames(self):
        print('step03. loading video frames ...')
        framelist = []
        if len(self.data['vfile']) and os.path.exists(self.data['vfile']):
            try:
                cameraCapture = cv2.VideoCapture(self.data['vfile'])
                success, frame = cameraCapture.read()
                while success:
                    framelist.append(frame)
                    success, frame = cameraCapture.read()
                cameraCapture.release()
            except Exception as e:
                print(e)
        self.data['frames'] = framelist

    def load_video_audios(self):
        print('step04. loading video audios ...')
        audios = []
        if len(self.data['afile']) and os.path.exists(self.data['afile']):
            try:
                _, audios = wavfile.read(self.data['afile'])
            except Exception as e:
                print(e)
        self.data['audios'] = audios
        
    def detect_video_scenes(self):
        print('step05. detecting video scenes ...')
        scenelist = []
        if os.path.isfile(self.data['vfile']):
            try:
                # videoManager = slib.VideoManager([self.data['vfile']])
                # statsManager = slib.StatsManager()
                # sceneManager = slib.SceneManager(statsManager)
                # sceneManager.add_detector(slib.ContentDetector())
                # baseTimecode = videoManager.get_base_timecode()
                # videoManager.set_downscale_factor()
                # videoManager.start()
                # sceneManager.detect_scenes(frame_source=videoManager)
                # scenelist = sceneManager.get_scene_list(baseTimecode)
                # if len(scenelist) == 0:
                #     scenelist = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
                scenedet = ulib.SceneDet(self.data['vfile'])
                scenelist = scenedet.detect_scenes()
            except Exception as e:
                print(e)
        self.data['scenelist'] = scenelist
        
    def detect_video_faces(self):
        print('step06. detecting video faces, time consuming, please wait a moment ...')
        facelist = []
        idx = 0
        for frame in tqdm.tqdm(self.data['frames']):
            bboxes = []
            try:
                det_bboxes = self.modelzoos['detector'].facebox_runner(frame)
                for detbox in det_bboxes:
                    rect = [int(p) for p in detbox[:4]]
                    bboxes.append(rect)
            except Exception as e:
                print(e)
            facelist.append((idx, bboxes))
            idx += 1
        self.data['faces'] = facelist
    
    def track_shot_faces(self, shot):
        scene_faces = self.data['faces'][shot[0].frame_num:shot[1].frame_num]
        trackinfos = []
        while True:
            trackinfo = []
            for fidx, frame_faces in scene_faces:
                for face in frame_faces:
                    if len(trackinfo) == 0:
                        trackinfo.append((fidx, face))
                        frame_faces.remove(face)
                    elif fidx - trackinfo[-1][0] <= self.args.min_failed_dets:
                        iou = ulib.calculate_iou(face, trackinfo[-1][1])
                        if iou > self.args.track_iou_thresh:
                            trackinfo.append((fidx, face))
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if len(trackinfo) == 0:
                break
            elif len(trackinfo) >= self.args.min_shot_frames:
                idxlist = [t[0] for t in trackinfo]
                fbboxes = np.array([t[1] for t in trackinfo])
                x_range = np.arange(idxlist[0], idxlist[-1] + 1)
                interpolate_coords = []
                for cidx in range(4):
                    interpfn = interp1d(idxlist, fbboxes[:, cidx])
                    boxcoord = interpfn(x_range)
                    interpolate_coords.append(boxcoord)
                interpolate_bboxes = np.stack(interpolate_coords, axis=1)
                trackinfos.append({'vf_idx':x_range, 'bboxes':interpolate_bboxes})
        return trackinfos

    def track_video_faces(self):
        print('step07. tracking video faces ...')
        alltracks = []
        for shot in self.data['scenelist']:
            num_shot_frames = shot[1].frame_num - shot[0].frame_num
            if num_shot_frames >= self.args.min_shot_frames:
                track_infos = self.track_shot_faces(shot)
                alltracks.extend(track_infos)
        self.data['alltracks'] = alltracks 
    
    def crop_tracked_faces(self):
        print('step08. [core] croping tracked faces ...')
        face_video_segs = []
        for track_info in self.data['alltracks']:
            # smooth face boxes 
            smooth_dets = {'half_side':[], 'center_x':[], 'center_y':[]}
            for bbox in track_info['bboxes']:
                smooth_dets['half_side'].append(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 )
                smooth_dets['center_x'].append((bbox[0] + bbox[2]) / 2)
                smooth_dets['center_y'].append((bbox[3] + bbox[1]) / 2)
            smooth_dets['half_side'] = signal.medfilt(smooth_dets['half_side'], self.args.smooth_win_size)
            smooth_dets['center_x'] = signal.medfilt(smooth_dets['center_x'], self.args.smooth_win_size)
            smooth_dets['center_y'] = signal.medfilt(smooth_dets['center_y'], self.args.smooth_win_size)
            # fetch tracked faces and audio seg
            face_video_seg = {
                'audio' : np.array([]), 
                'timestamp' : (), 
                'vf_idx' : track_info['vf_idx'],
                'smoothdets' : smooth_dets,
                'cropcoords' : [],
                'faces' : []
            }
            timestamp = (track_info['vf_idx'][0] / 25, (track_info['vf_idx'][-1] + 1) / 25)
            face_video_seg['timestamp'] = timestamp
            face_video_seg['audio'] = self.data['audios'][int(timestamp[0] * 16000):int(timestamp[1] * 16000)]
            for idx, vf_idx in enumerate(track_info['vf_idx']):
                vdframe = self.data['frames'][vf_idx]
                halfbox = smooth_dets['half_side'][idx]
                centerx = smooth_dets['center_x'][idx]
                centery = smooth_dets['center_y'][idx]
                imgh, imgw, _ = vdframe.shape

                x1 = max(0, centerx - halfbox * (1 + self.args.crop_face_scale))
                x2 = min(imgw, centerx + halfbox * (1 + self.args.crop_face_scale))
                y1 = max(0, centery - halfbox * (1 + self.args.crop_face_scale))
                y2 = min(imgh, centery + halfbox * (1 + self.args.crop_face_scale))
                x1, y1, x2, y2 = [int(p) for p in [x1, y1, x2, y2]]
                face_video_seg['cropcoords'].append([x1, y1, x2, y2])
                face = cv2.resize(vdframe[y1:y2, x1:x2, :], (112, 112)) # TODO::(224, 224)
                # face = face[56:168, 56:168, :]
                if not self.args.is_debug:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # TODO
                    face = (face / 255.0 - 0.4161) / 0.1688
                face_video_seg['faces'].append(face)
            face_video_segs.append(face_video_seg)
        self.data['tracksegs'] = face_video_segs
        
    def visualize_track_segs(self):
        if not self.args.is_debug:
            print('step09. [closed] visualize middle trackseg ...')
            return 
        print('step09. [opened] visualize middle trackseg ...')
        video_fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        for idx, trackseg in enumerate(self.data['tracksegs']):
            try:
                tmp_vfile = os.path.join(self.args.test_dir, 'tmp', 'seg.avi')
                videoio = cv2.VideoWriter(tmp_vfile, fourcc, video_fps, (112, 112))
                for face in trackseg['faces']:
                    videoio.write(face)
                videoio.release()
                tmp_afile = os.path.join(self.args.test_dir, 'tmp', 'seg.wav') 
                wavfile.write(tmp_afile, rate=16000, data=trackseg['audio'])
                seg_video = os.path.join(self.args.test_dir, 'tmp', 'seg_%02d.avi' % idx)
                cmd = 'ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel quiet' % \
                    (tmp_vfile, tmp_afile, self.args.threads, seg_video)
                output = subprocess.call(cmd, shell=True, stdout=None)
                os.remove(tmp_afile)
                os.remove(tmp_vfile)
            except Exception as e:
                print(e)
    
    def detect_active_speakers(self):
        if self.args.is_debug:
            print('step10. [debug] doing nothing for asd ...')
            return
        print('step10. [core] detecting the active speakers ...')
        self.modelzoos['asd'].eval()
        alltrack_segs_scores = []
        for idx, trackseg in enumerate(self.data['tracksegs']):
            audio = python_speech_features.mfcc(
                signal=trackseg['audio'], samplerate=16000, winlen=0.025, winstep=0.01, numcep=13)
            faces = np.array(trackseg['faces'])
            seconds = min((audio.shape[0] - audio.shape[0] % 4) / 100, len(faces))
            audio = audio[:int(round(seconds * 100)), :]
            faces = faces[:int(round(seconds * 25)), :, :]
            track_seg_scores = []
            for dur in self.args.duration_list:
                cur_dur_scores = []
                num_dur_units = int(math.ceil(seconds / dur))
                with torch.set_grad_enabled(False):
                    for i in range(num_dur_units):
                        inp_audio = audio[i * dur * 100 : (i + 1) * dur * 100, :]
                        inp_faces = faces[i * dur * 25 : (i + 1) * dur * 25, :, :]
                        inp_audio = torch.FloatTensor(inp_audio).unsqueeze(0)
                        inp_faces = torch.FloatTensor(inp_faces).unsqueeze(0)
                        if self.device == 'cuda':
                            inp_audio = inp_audio.cuda()
                            inp_faces = inp_faces.cuda()
                        logit, _, _ = self.modelzoos['asd'](inp_audio, inp_faces)
                        score = self.softmax(logit).detach().cpu().numpy()[:, 1].tolist()
                        cur_dur_scores.extend(score)
                track_seg_scores.append(cur_dur_scores)
            ave_track_seg_score = np.array(track_seg_scores).mean(axis=0).round(decimals=2)
            alltrack_segs_scores.append(ave_track_seg_score)
        self.data['scores'] = alltrack_segs_scores

    def collect_frame_faceinfos(self):
        print('step11. [demo] collecting the faceinfos for each frame ...')
        allframe_face_list = [[] for i in range(len(self.data['frames']))]
        for stidx, shot_track in enumerate(self.data['tracksegs']):
            shot_scores = self.data['scores'][stidx]
            smooth_dets = shot_track['smoothdets']
            crop_coords = shot_track['cropcoords']
            for idx, vf_idx in enumerate(shot_track['vf_idx']):
                idx_win_score = shot_scores[max(idx - 2, 0): min(idx + 3, len(shot_scores) - 1)]
                idx_asd_score = np.mean(idx_win_score)
                cur_vf_face_info = {
                    'track_idx' : stidx,
                    'asd_score' : idx_asd_score,
                    'face_bbox' : [smooth_dets[key][idx] for key in ['half_side', 'center_x', 'center_y']],
                    'cropcoord' : crop_coords[idx]
                }
                allframe_face_list[vf_idx].append(cur_vf_face_info)
        self.data['framefaces'] = allframe_face_list
    
    def visualize_demo(self):
        print('step12. [demo] visualing the demo ...')
        imgh, imgw, _ = self.data['frames'][0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        base_name = self.args.test_video.split('/')[-1].split('.')[0]
        staic_video = os.path.join(self.args.test_dir, 'tmp', '%s.avi' % base_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_fps = 25
        videoio = cv2.VideoWriter(staic_video, fourcc, video_fps, (imgw, imgh))
        colordict = {0:(0, 0, 255), 1:(0, 255, 0)}
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        for vf_idx, frame in enumerate(self.data['frames']):
            for faceinfo in self.data['framefaces'][vf_idx]:
                color = colordict[int(faceinfo['asd_score'] > 0.5)]
                scoretxt = '%.4f' % faceinfo['asd_score']
                '''
                halfside, center_x, center_y = faceinfo['face_bbox']
                x1 = int(max(0, center_x - halfside))
                y1 = int(max(0, center_y - halfside))
                x2 = int(min(imgw, center_x + halfside))
                y2 = int(min(imgh, center_y + halfside))
                '''
                x1, y1, x2, y2 = faceinfo['cropcoord']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=4)
                cv2.putText(frame, scoretxt, (x1, y1), fontFace, 1, color, thickness=2)
            videoio.write(frame)
        videoio.release()
        demo_file = os.path.join(self.args.test_dir, 'demos', '%s_add.avi' % base_name)
        cmd = 'ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel quiet' % (\
            staic_video, self.data['afile'], self.args.threads, demo_file)
        subprocess.call(cmd, shell=True, stdout=None) 
        os.remove(staic_video)

    def run_demo(self, test_video='', is_25_fps=False):
        self.report_config_summary()
        # self.load_model()
        self.parse_video(test_video, is_25_fps)
        self.load_video_frames()
        # self.load_video_audios()
        self.detect_video_scenes()
        print(self.data['scenelist'])
        '''
        self.detect_video_faces()

        # middle_file = 'testset/middle/shawshank.npy'
        # preload_dict = np.load(middle_file, allow_pickle=True).item()
        # self.data.update(preload_dict)

        self.track_video_faces()
        self.crop_tracked_faces()

        # del self.data['frames']
        # np.save(middle_file, self.data)
        
        self.visualize_track_segs()
        self.detect_active_speakers()
        self.collect_frame_faceinfos()
        self.visualize_demo()

        # del self.data['frames']
        # np.save(middle_file, self.data)
        '''

if __name__ == "__main__":
    
    args = clib.demo_args()
    demo = ASDDemo(args)
    # test_video = 'testset/videos/noodles.mp4'
    test_video = 'testset/videos/movie_1_60fps.mp4'
    demo.run_demo(test_video)