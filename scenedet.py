# -*- coding: utf-8 -*-
import os
import utils as ulib
from IPython import embed

root_dir = '/Users/lujie/Documents/audiovideo/dataops/trainvideos/videos'

if __name__ == "__main__":
    
    ## pipeline
    video_file = os.path.join(root_dir, 'shawshank.mp4')
    scenedet = ulib.SceneDet(video_file, 0, True)
    scenes = scenedet.detect_scenes()
    for scene in scenes:
        print(scene)
    print(type(scene[0]))