# -*- coding: utf-8 -*-
from .video_stream import VideoStream
from .scene_detector import SceneDetector
from .scene_manager import SceneManager

class SceneDet(object):
    def __init__(self, video_file, frame_skip=0, verbose=False):
        self.video = VideoStream(video_file)
        self.frame_skip = frame_skip
        self.verbose = verbose
        self.managers = SceneManager(
            detector=SceneDetector(
                threshold=27.0, 
                min_scene_len=15, 
                luma_only=False
            )
        )
    
    def detect_scenes(self):
        return self.managers.runner(self.video, self.frame_skip, self.verbose)