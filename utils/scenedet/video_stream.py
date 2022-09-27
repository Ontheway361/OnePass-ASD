# -*- coding: utf-8 -*-

import cv2
from scenedetect.frame_timecode import FrameTimecode
from IPython import embed

class VideoOpenFailure(Exception):
    def __init__(self):
        message = 'VideoCapture.isOpened() returned False. Ensure the input file is a valid video, \
            and check that OpenCV is installed correctly.\n'
        super().__init__(message)

class VideoStream(object):

    def __init__(self, video_file, max_decode_attempts=5):
        self._video_file = video_file
        self._is_device = isinstance(self._video_file, int)
        self._cap = None
        self._has_grabbed = False
        self._max_decode_attempts = max_decode_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        self._open_capture()
    
    def _open_capture(self):
        cap = cv2.VideoCapture(self._video_file)
        if not cap.isOpened():
            raise VideoOpenFailure()
        self._cap = cap
        self._frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.base_timecode = FrameTimecode(timecode=0, fps=self._frame_rate)
        self._has_grabbed = False
        
    @property
    def frame_size(self):
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def duration(self):
        return self.base_timecode + int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def position(self):
        if self.frame_number < 1:
            return self.base_timecode
        return self.base_timecode + (self.frame_number - 1)

    @property
    def frame_number(self):
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def reset(self):
        self._cap.release()
        self._open_capture()

    def read(self, decode=True, advance=True):
        if not self._cap.isOpened():
            return False
        if advance:
            has_grabbed = self._cap.grab()
            if not has_grabbed:
                if self.duration > 0 and self.position < (self.duration - 1):
                    for _ in range(self._max_decode_attempts):
                        has_grabbed = self._cap.grab()
                        if has_grabbed:
                            break
                if has_grabbed:
                    self._decode_failures += 1
                    print('Frame failed to decode.')
                    if not self._warning_displayed and self._decode_failures > 1:
                        print('Failed to decode some frames, results may be inaccurate.')
            if not has_grabbed:
                return False
            self._has_grabbed = True
        if decode and self._has_grabbed:
            _, frame = self._cap.retrieve()
            return frame
        return self._has_grabbed
    
    
