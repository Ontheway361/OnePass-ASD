# -*- coding: utf-8 -*-
import numpy as np
import cv2, os, ffmpeg, pyaudio, wave, time, threading, subprocess
from IPython import embed

class VideoRecorder():  
    def __init__(self, video="video.avi", view_xy=(1280, 720), fps=25):
        self.open = True
        self.video_filename = video
        self.fps = fps
        self.view_xy = view_xy                  
        self.video_cap = cv2.VideoCapture(0)
        self.video_cap.set(cv2.CAP_PROP_FPS, fps)
        self.video_out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'MJPG'), fps, view_xy)
        self.frame_counts = 1
        self.start_time = time.time()

    def record(self):
        # counter = 1
        timer_start = time.time()
        timer_current = 0
        while self.open:
            ret, frame = self.video_cap.read()
            if ret:
                # frame = cv2.resize(frame, self.view_xy)
                self.video_out.write(frame)
                self.frame_counts += 1
                # counter += 1
                # timer_current = time.time() - timer_start
                # time.sleep(1/self.fps)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('frame', gray)
                cv2.waitKey(40)
            else:
                break

    def stop(self):
        if self.open:
            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()

class AudioRecorder():
    def __init__(self, filename='audio.wav'):
        self.open = True
        self.rate = 16000
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer)
        self.audio_frames = []

    def record(self):
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if not self.open:
                break

    def stop(self):
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

def start_video_recording():
    global video_thread
    video_thread = VideoRecorder()
    video_thread.start()

def start_audio_recording(filename="test"):
    global audio_thread
    audio_thread = AudioRecorder()
    audio_thread.start()
    return filename

def start_AVrecording(filename="test"):
    global video_thread
    global audio_thread
    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()
    audio_thread.start()
    video_thread.start()
    return filename

def stop_AVrecording(filename="test"):
    audio_thread.stop()
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 
    
    # print('flag2 ...')
    # cmd = 'ffmpeg -y -i video.avi -pix_fmt yuv420p -r ' + str(recorded_fps) + 'video.avi'
    # subprocess.call(cmd, shell=True)
    print(threading.active_count())
    cmd = 'ffmpeg -y -ac 2 -channel_layout stereo -i audio.wav -i video.avi -pix_fmt yuv420p fusion.avi -loglevel quiet'
    subprocess.call(cmd, shell=True)
    
    

    # video_stream = ffmpeg.input(video_thread.video_filename)
    # audio_stream = ffmpeg.input(audio_thread.audio_filename)
    # ffmpeg.output(audio_stream, video_stream, 'out.avi').run(overwrite_output=True)

if __name__ == '__main__':
    # start_video_recording()
    
    start_AVrecording()
    time.sleep(5)
    stop_AVrecording()

    # video_thread.stop()
