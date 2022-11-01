# -*- coding: utf-8 -*-
from tracemalloc import start
import numpy as np
import os, cv2, sys, time, pyaudio, threading
from scipy.interpolate import interp1d
from queue import Queue
import torch, torchvision, python_speech_features
import utils as ulib
import config as clib
import models as mlib
from IPython import embed

class ASDDemo(object):

    def __init__(self, args, capture_time=30):
        self.args = args
        self.modelzoos = {}
        self.capture_time = capture_time
        self.data = {'demo_queue':Queue(maxsize=120), 'infer':True, 'disp':True}
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
        self._report_config_summary()
        # self.avio = ulib.CaptureCameraAV()
        self._load_model()
    
    def _report_config_summary(self):
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
    
    def _load_model(self):
        print('step2. loading model ...')
        try:
            # self.modelzoos['asd'] = mlib.OnePassASD(self.args)
            self.modelzoos['asd'] = mlib.BaseLine(device=self.device)
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
            print('Attention, loading pretrain occurs errors')
        try:
            self.modelzoos['detector'] = ulib.FaceDet(
                longside=self.args.longside, device=self.device)
        except Exception as e:
            print(e)
    
    def fetch_buffer_data(self, verbose=True):
        is_cache, frames, audios = self.avio.cache_buffer()
        self.data['frame_seq'] = frames
        self.data['audio_seq'] = audios
        if verbose:
            print('nframe %03d, naudio %06d ...' % (len(frames), len(audios)))
        return is_cache 
    
    def detect_buffer_faces(self):
        self.data['frames_bboxes'] = []
        for idx, frame in enumerate(self.data['frame_seq']):
            frame_bboxes = []
            try:
                det_bboxes = self.modelzoos['detector'].facebox_runner(frame)
                for detbox in det_bboxes:
                    rect = [int(p) for p in detbox[:4]]
                    frame_bboxes.append(rect)
            except Exception as e:
                print(e)
            self.data['frames_bboxes'].append((idx, frame_bboxes))
    
    def track_buffer_faces(self):
        self.data['faceclips_detinfos'] = []
        while True:
            faceclip_detinfos = []
            for fidx, frame_faces in self.data['frames_bboxes']:
                for face in frame_faces:
                    if len(faceclip_detinfos) == 0:
                        faceclip_detinfos.append((fidx, face))
                        frame_faces.remove(face)
                        break
                    elif fidx - faceclip_detinfos[-1][0] <= self.args.min_failed_dets: # TODO::
                        iou = ulib.calculate_iou(face, faceclip_detinfos[-1][1])
                        if iou > self.args.track_iou_thresh:
                            faceclip_detinfos.append((fidx, face))
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if len(faceclip_detinfos) == 0:
                break
            elif len(faceclip_detinfos) >= self.args.min_shot_frames:  # TODO::
                idxlist = [t[0] for t in faceclip_detinfos]
                fbboxes = np.array([t[1] for t in faceclip_detinfos])
                x_range = np.arange(idxlist[0], idxlist[-1] + 1)
                interpolate_coords = []
                for cidx in range(4):
                    interpfn = interp1d(idxlist, fbboxes[:, cidx])
                    boxcoord = interpfn(x_range)
                    interpolate_coords.append(boxcoord)
                interpolate_bboxes = np.stack(interpolate_coords, axis=1)
                self.data['faceclips_detinfos'].append({'vf_idx':x_range, 'bboxes':interpolate_bboxes})
        
    def crop_buffer_faces(self):
        self.data['tracked_faceclips'] = []
        num_buffer_frames = len(self.data['frame_seq'])
        imgw, imgh = self.args.frame_shape
        for faceclip_detinfos in self.data['faceclips_detinfos']:
            faceclip_info = {
                'vf_idx' : faceclip_detinfos['vf_idx'],
                'face_box' : [], 'faces_seq' : [], 'audio_seq' : np.array([])}
            faceclip_start = int(faceclip_detinfos['vf_idx'][0] / (num_buffer_frames + 1e-3) * 16000)
            faceclip_end = int(faceclip_detinfos['vf_idx'][-1] / (num_buffer_frames + 1e-3) * 16000)
            faceclip_info['audio_seq'] = self.data['audio_seq'][faceclip_start:(faceclip_end + 1)]
            for vf_idx, bbox in zip(faceclip_detinfos['vf_idx'], faceclip_detinfos['bboxes']):
                frame = self.data['frame_seq'][vf_idx]
                bx1, by1, bx2, by2 = bbox
                half_box = max((bx2 - bx1), (by2 - by1)) / 2
                center_x = (bx2 + bx1) / 2
                center_y = (by2 + by1) / 2
                cx1 = int(max(0, center_x - half_box))
                cy1 = int(max(0, center_y - half_box))
                cx2 = int(min(imgw, center_x + half_box))
                cy2 = int(min(imgh, center_y + half_box))
                face = cv2.resize(frame[cy1:cy2, cx1:cx2, :], (112, 112))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = (face / 255.0 - 0.4161) / 0.1688
                faceclip_info['faces_seq'].append(face)
                faceclip_info['face_box'].append([cx1, cy1, cx2, cy2])
            self.data['tracked_faceclips'].append(faceclip_info)
    
    def align_audio_with_frames(self, faces, audio):
        nfaces, naudio = len(faces), len(audio)
        n_faces_audio_len = 4 * nfaces
        if n_faces_audio_len <= naudio:
            sample_index = np.array(np.linspace(0, naudio - 1, n_faces_audio_len), dtype=int).tolist()
            audio = audio[sample_index, :]
        else:
            shortage = n_faces_audio_len - naudio
            audio = np.pad(audio, (((0, shortage), (0, 0))), mode='wrap')
        return faces, audio
            
    def active_speaker_detect(self):
        self.modelzoos['asd'].eval()
        self.data['faceclips_scores'] = []
        for faceclip in self.data['tracked_faceclips']:
            audio = python_speech_features.mfcc(
                signal=faceclip['audio_seq'], samplerate=16000, winlen=0.025, winstep=0.01, numcep=13)
            faces = np.array(faceclip['faces_seq'])
            faces, audio = self.align_audio_with_frames(faces, audio)
            with torch.set_grad_enabled(False):
                audio = torch.FloatTensor(audio).unsqueeze(0)
                faces = torch.FloatTensor(faces).unsqueeze(0)
                if self.device == 'cuda':
                    audio = audio.cuda()
                    faces = faces.cuda()
                logits, _, _ = self.modelzoos['asd'](audio, faces)
                scores = self.softmax(logits).detach().cpu().numpy()[:, 1].round(decimals=2)
            self.data['faceclips_scores'].append(scores)
    
    def collect_buffer_asd_scores(self):
        self.data['score_seq'] = [[] for i in range(len(self.data['frame_seq']))]
        for faceclip, scores in zip(self.data['tracked_faceclips'], self.data['faceclips_scores']):
            score_seq_len = len(scores)
            for idx, vf_idx in enumerate(faceclip['vf_idx']):
                win_score = scores[max(idx - 2, 0): min(idx + 3, score_seq_len - 1)]
                asd_score = np.mean(win_score)
                face_info = {'face_box':faceclip['face_box'][idx], 'asd_score':asd_score}
                self.data['score_seq'][vf_idx].append(face_info)
    
    def text_buffer_asd_scores(self):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        colordict = {0:(0, 0, 255), 1:(0, 255, 0)}
        for frame, faceinfos in zip(self.data['frame_seq'], self.data['score_seq']):
            for faceinfo in faceinfos:
                color = colordict[int(faceinfo['asd_score'] > 0.5)]
                scoretxt = '%.2f' % faceinfo['asd_score']
                x1, y1, x2, y2 = faceinfo['face_box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=4)
                cv2.putText(frame, scoretxt, (x1, y1), fontFace, 1, color, thickness=2)
            if not self.data['demo_queue'].full():
                self.data['demo_queue'].put(frame)
    
    def calculate_decay_factor(self):
        wait_time = 1.0
        num_buffer_frames = len(self.data['frame_seq'])
        if num_buffer_frames >= 40:
            wait_time = 0.001
        elif 20 < num_buffer_frames < 40:
            wait_time = 0.001
        elif 10 < num_buffer_frames <= 20:
            wait_time = 0.001
        else:
            wait_time = 1.0
        return wait_time
    
    def show_log_details(self, time_seq, verbose=True):
        if verbose:
            pip_index = time_seq['pip_index']
            cache_cost = time_seq['cache_done'] - time_seq['pipe_start']
            prep_cost = time_seq['prep_done'] - time_seq['cache_done']
            infer_cost = time_seq['infer_done'] - time_seq['prep_done']
            collect_cost = time_seq['collect_done'] - time_seq['infer_done']
            demo_dur = time_seq['collect_done'] - time_seq['demo_start']
            print('buffer_id : %02d, cache_cost : %.2fs, prep_cost : %.3f, infer_cost : %.2fs, collect_cost : %.2f, demo_dur %.2f...' % (\
                pip_index, cache_cost, prep_cost, infer_cost, collect_cost, demo_dur))

    def online_infer(self):
        print('step3. start the model-infer process ... ')
        time_seq = {'demo_start':None, 'pipe_start':None, 'cache_done':None, \
            'prep_done':None, 'infer_done':None, 'collect_done':None, 'pip_index':0}
        time_seq['demo_start'] = time.time()
        self.avio.start_capture()
        while time.time() - time_seq['demo_start'] < self.capture_time:
            time_seq['pipe_start'] = time.time()
            is_cache = self.fetch_buffer_data(verbose=False)
            if is_cache:
                time_seq['cache_done'] = time.time()
                self.detect_buffer_faces()
                self.track_buffer_faces()
                self.crop_buffer_faces()
                time_seq['prep_done'] = time.time()
                self.active_speaker_detect()
                time_seq['infer_done'] = time.time()
                self.collect_buffer_asd_scores()
                self.text_buffer_asd_scores()
                time_seq['collect_done'] = time.time()
                time_seq['pip_index'] += 1
                self.show_log_details(time_seq, verbose=True)
        self.avio.stop_capture()
    
    def online_imshow(self):
        wait_idx, wait_tolerance_time = 0, 100
        while self.data['disp']:
            if self.data['demo_queue'].empty():
                time.sleep(0.05)
                wait_idx += 1
            else:
                wait_idx = 0
                while not self.data['demo_queue'].empty():
                    print(self.data['demo_queue'].qsize())
                    frame = self.data['demo_queue'].get()
                    cv2.imshow('active speaker detection demo', frame)
                    cv2.waitKey(25)
            if wait_idx > wait_tolerance_time:
                break
        try:
            cv2.distroyAllWindows()
        except:
            pass
    
    def start_demo(self):
        self.data['infer_thread'] = threading.Thread(target=self.online_infer)
        self.data['infer_thread'].start()
        self.data['imshow_thread'] = threading.Thread(target=self.online_imshow)
        self.data['imshow_thread'].start()
    
    def stop_demo(self):
        self.data['infer_thread'].join()
        time.sleep(2)
        self.data['imshow_thread'].join()
        

if __name__ == "__main__":
    
    args = clib.demo_args()
    demo = ASDDemo(args, capture_time=30)
    # demo.start_demo()
    # time.sleep(10)
    # demo.stop_demo()
    