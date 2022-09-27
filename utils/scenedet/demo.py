# -*- coding: utf-8 -*-
import os
from video_stream import VideoStream
from scene_detector import SceneDetector
from scene_manager import SceneManager
from IPython import embed

root_dir = '/Users/lujie/Documents/audiovideo/dataops/trainvideos/videos'

if __name__ == "__main__":
    
    ## pipeline
    # video_file  = os.path.join(root_dir, 'shawshank.mp4')
    # video_io = VideoStream(video_file)
    # detector = SceneDetector(threshold=27.0, min_scene_len=15, luma_only=False)
    # managers = SceneManager(detector)
    # scenes = managers.runner(video_io, frame_skip=0, verbose=False)
    # for scene in scenes:
    #     print(scene)
    # print(type(scene[0]))
    pass