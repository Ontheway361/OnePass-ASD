# -*- coding: utf-8 -*-
from importlib.util import find_spec
import numpy as np
from scipy.io import wavfile
import cv2, os, pyaudio, wave, time, threading
from IPython import embed

class CaptureCameraAV(object):
    
    def __init__(self):
        self.frame_shape = (640, 360)
        self.sample_rate = 16000
        self.frames_per_buffer = 1024
        self.data = {
            'frame_buffer':[], 'frame_seq':[],'audio_buffer':[], 'audio_seq': []}
        self.avio = {
            'aio':None, 'aio_open':True, 'aio_thread':None, 'audio':None,
            'vio':None, 'vio_open':True, 'vio_thread':None}
        self.initialize_io()
    
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
        print('step2. fetch frames in parallel ...')
        while self.avio['vio_open']:
            is_succeed, frame = self.avio['vio'].read()
            if is_succeed:
                frame = cv2.resize(frame, self.frame_shape)
                self.data['frame_seq'].append(frame)
            else:
                break
    
    def fetch_seq_audio(self):
        print('step2. fetch audios in parallel ...')
        self.avio['aio'].start_stream()
        while self.avio['aio_open']:
            audio = self.avio['aio'].read(
                self.frames_per_buffer, exception_on_overflow=False)
            self.data['audio_seq'].append(audio)
            if not self.avio['aio_open']:
                break
    
    def translate_audio_bytes(self):
        waveFile = wave.open('buffer.wav', 'wb')
        waveFile.setnchannels(2)
        waveFile.setsampwidth(2)
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(self.data['audio_buffer']))
        waveFile.close()
        _, audio = wavfile.read('buffer.wav')
        return audio
    
    def cache_buffer(self):
        num_fetch_frames = len(self.data['frame_seq']) * 1 // 2
        num_fetch_audios = len(self.data['audio_seq']) * 1 // 2
        self.data['frame_buffer'] = self.data['frame_seq'][:num_fetch_frames]
        self.data['audio_buffer'] = self.data['audio_seq'][:num_fetch_audios]
        del self.data['frame_seq'][:num_fetch_frames]
        del self.data['audio_seq'][:num_fetch_audios]
        audio = self.translate_audio_bytes()
        return (self.data['frame_buffer'], audio)
    
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

    avio = CaptureCameraAV()
    avio.start_capture()
    time.sleep(30)
    frames, audios = avio.cache_buffer()
    print(len(frames), audios.shape)
    time.sleep(3)
    avio.stop_capture()