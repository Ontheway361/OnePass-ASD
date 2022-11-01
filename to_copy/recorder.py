# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile
import cv2, os, random, pyaudio, wave, time, threading
from IPython import embed

class CaptureCameraAV(object):
    
    def __init__(self, buffer_len=20):
        self.buffer_len = buffer_len
        self.frame_shape = (640, 360)
        self.sample_rate = 16000
        self.frames_per_buffer = 1024
        self.data = {
            'frame_buffer':[], 'frame_seq':[], 'audio_buffer':[], 'audio_seq': [], 'asr_seq':[]}
        self.avio = {
            'aio':None, 'aio_open':True, 'aio_thread':None, 'audio':None,
            'vio':None, 'vio_open':True, 'vio_thread':None}
        # self.checkout_audio_device()
        self.initialize_io()
    
    def checkout_audio_device(self):
        is_ready = False
        p = pyaudio.PyAudio()
        dev = p.get_device_info_by_index(11) # ubuntu-service
        if 'C922 Pro Stream Webcam' in dev['name']:
            is_ready = True
        if not is_ready:
            assert IOError('Attention, Audio device is not ready !')

    def initialize_io(self):
        print('step1. open the io for video and audio ...')
        try:
            self.avio['audio'] = pyaudio.PyAudio()
            self.avio['aio'] = self.avio['audio'].open(format=pyaudio.paInt16, channels=2, \
                rate=self.sample_rate, input=True, frames_per_buffer=self.frames_per_buffer)
            self.avio['vio'] = cv2.VideoCapture(0)
        except Exception as e:
            print(e)
    
    def fetch_seq_frames(self):
        # ubuntu-30fps
        print('step2. fetch frames in parallel ...')
        while self.avio['vio_open']:
            is_succeed, frame = self.avio['vio'].read()
            frame_ts = time.time()
            if is_succeed:
                frame = cv2.resize(frame, self.frame_shape)
                self.data['frame_seq'].append((frame, frame_ts))
            else:
                break
    
    def fetch_seq_audio(self):
        # ubuntu-15.
        print('step2. fetch audios in parallel ...')
        self.avio['aio'].start_stream()
        while self.avio['aio_open']:
            try:
                audio = self.avio['aio'].read(
                    self.frames_per_buffer, exception_on_overflow=False)
                audio_ts = time.time()
            except:
                print('audio source closed ...')
            else:
                self.data['audio_seq'].append((audio, audio_ts))
                self.data['asr_seq'].append((audio, audio_ts))
            # audio = self.avio['aio'].read(
            #     self.frames_per_buffer, exception_on_overflow=False)
            # self.data['audio_seq'].append(audio)
            if not self.avio['aio_open']:
                break
    
    def translate_audio_bytes(self, buffer, used_as='asd'):
        savefile = 'asd_buffer.wav' if used_as == 'asd' else 'asr_buffer.wav'
        waveFile = wave.open(savefile, 'wb')
        waveFile.setnchannels(2)
        waveFile.setsampwidth(2)
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(buffer))
        waveFile.close()
        _, audio = wavfile.read(savefile)
        return audio
    
    def cache_buffer(self):
        is_cached = False
        self.data['buffer_frame'] = []
        self.data['buffer_audio'] = []
        cache_try_times = 0
        while not is_cached:
            if len(self.data['frame_seq']) >= self.buffer_len:
                buffer_frame = self.data['frame_seq'][:self.buffer_len]
                frame_start_ts = buffer_frame[0][1]
                frame_end_ts = buffer_frame[-1][1]
                num_audio_buffer = len(self.data['audio_seq'])
                audio_start_idx = 0
                while audio_start_idx < num_audio_buffer:
                    if self.data['audio_seq'][audio_start_idx][1] < frame_start_ts:
                        audio_start_idx += 1
                    else:
                        break
                audio_end_idx = audio_start_idx + 1
                while audio_end_idx < num_audio_buffer:
                    if self.data['audio_seq'][audio_end_idx][1] < frame_end_ts:
                        audio_end_idx += 1
                    else:
                        break
                buffer_audio = self.data['audio_seq'][audio_start_idx:audio_end_idx]
                buffer_frame = [b[0] for b in buffer_frame]
                buffer_audio = [b[0] for b in buffer_audio]
                buffer_audio = self.translate_audio_bytes(buffer_audio)
                self.data['buffer_frame'] = buffer_frame
                self.data['buffer_audio'] = buffer_audio
                del self.data['frame_seq'][:self.buffer_len]
                del self.data['audio_seq'][:audio_end_idx]
                is_cached = True
                break 
            else:
                cache_try_times += 1
                time.sleep(0.2)
            if cache_try_times > 10:
                break
        return (is_cached, self.data['buffer_frame'], self.data['buffer_audio'])
    
    def cache_audio(self, audio_duration=5):
        audio_start_ts = self.data['asr_seq'][0][1]
        cur_audio_dur = self.data['asr_seq'][-1][1] - audio_start_ts
        wait_index, max_wait_times = 0, 30
        is_cached = False
        while cur_audio_dur < audio_duration:
            wait_index += 1
            time.sleep(0.5)
            cur_audio_dur = self.data['asr_seq'][-1][1] - audio_start_ts
            if wait_index >= max_wait_times:
                break
        audio_buffer = None
        end_index = None
        if wait_index <= max_wait_times:
            end_index = 0
            while self.data['asr_seq'][end_index][1] - audio_start_ts < audio_duration:
                end_index += 1
        elif cur_audio_dur >= audio_duration * 0.4:
            end_index = -1
        if end_index != -2:
            is_cached = True
            audio_buffer = self.data['asr_seq'][:end_index]
            del self.data['asr_seq'][:end_index]
            audio_buffer = [b[0] for b in audio_buffer]
            audio_buffer = self.translate_audio_bytes(audio_buffer, used_as='asr')
        return (is_cached, audio_buffer)

    def start_capture(self):
        print('step2. fetch avdata in parallel style ...')
        self.avio['aio_thread'] = threading.Thread(target=self.fetch_seq_audio)
        self.avio['aio_thread'].start()
        self.avio['vio_thread'] = threading.Thread(target=self.fetch_seq_frames)
        self.avio['vio_thread'].start()

    def stop_capture(self):
        # stop vio
        if self.avio['vio_open']:
            print('step3. kill the video thread ...')
            self.avio['vio_open'] = False
            self.avio['vio'].release()
            self.avio['vio_thread'].join()
            
        # stop aio
        if self.avio['aio_open']:
            print('step3. kill the audio thread ...')
            self.avio['aio_open'] = False
            self.avio['aio'].stop_stream()
            self.avio['aio'].close()
            self.avio['audio'].terminate()
            self.avio['aio_thread'].join()
    
if __name__ == "__main__":

    avio = CaptureCameraAV(buffer_len=30)
    avio.start_capture()
    for i in range(2):
        # is_cached, frames, audios = avio.cache_buffer()
        # print(is_cached, len(frames), len(audios), audios.shape)
        is_cached, auido = avio.cache_audio(audio_duration=10)
        # print(is_cached, auido.shape)
    # time.sleep(15)
    avio.cache_buffer()
    avio.stop_capture()