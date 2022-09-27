# -*- coding: utf-8 -*-
import cv2
import numpy as np
from IPython import embed

class SceneDetector(object):

    FRAME_SCORE_KEY = 'content_val'
    DELTA_H_KEY, DELTA_S_KEY, DELTA_V_KEY = ('delta_hue', 'delta_sat', 'delta_lum')
    METRIC_KEYS = [FRAME_SCORE_KEY, DELTA_H_KEY, DELTA_S_KEY, DELTA_V_KEY]

    def __init__(self, threshold=27.0, min_scene_len=15, luma_only=False):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.luma_only = luma_only
        self.last_frame = None
        self.last_scene_cut = None
        self.last_hsv = None

    def _calculate_frame_score(self, curr_hsv, last_hsv):
        curr_hsv = [x.astype(np.int32) for x in curr_hsv]
        last_hsv = [x.astype(np.int32) for x in last_hsv]
        delta_hsv = [0, 0, 0, 0]
        for i in range(3):
            num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
            delta_hsv[i] = np.sum(np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
        delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
        score = delta_hsv[-2] if self.luma_only else delta_hsv[-1]
        return score

    def process_frame(self, frame_num, frame_img):
        cut_list = []
        if self.last_scene_cut is None:
            self.last_scene_cut = frame_num
        if self.last_frame is not None:
            curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
            last_hsv = self.last_hsv
            if not last_hsv:
                last_hsv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))
            frame_score = self._calculate_frame_score(curr_hsv, last_hsv)
            self.last_hsv = curr_hsv
            if frame_score >= self.threshold and (frame_num - self.last_scene_cut) >= self.min_scene_len:
                cut_list.append(frame_num)
                self.last_scene_cut = frame_num
            if self.last_frame is not None:
                self.last_frame = None
        self.last_frame = frame_img.copy()
        return cut_list