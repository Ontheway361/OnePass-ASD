# -*- coding: utf-8 -*-
from re import S
from tqdm import tqdm
import sys, queue, threading
from scenedetect.frame_timecode import FrameTimecode
from IPython import embed

DEFAULT_MIN_WIDTH = 256
MAX_FRAME_QUEUE_LENGTH = 4

class SceneManager(object):

    def __init__(self, detector):
        self._cutting_list = []
        self._event_list = []
        self._detector = detector
        self._start_pos = None
        self._last_pos = None
        self._base_timecode = None
        self._only_cuts = True
        self._exception_info = None

    def _process_frame(self, frame_num, frame_im):
        cuts = self._detector.process_frame(frame_num, frame_im)
        self._cutting_list += cuts

    def _decode_thread(self, video, frame_skip, downscale_factor, out_queue):
        try:
            while True:
                frame_im = video.read()
                if frame_im is False:
                    break
                if downscale_factor > 1:
                    frame_im = frame_im[::downscale_factor, ::downscale_factor, :]
                if self._start_pos is None:
                    self._start_pos = video.position
                out_queue.put((frame_im, video.position))
                if frame_skip > 0:
                    for _ in range(frame_skip):
                        if not video.read(decode=False):
                            break
        except:
            self._exception_info = sys.exc_info()
        finally:
            if self._start_pos is None:
                self._start_pos = video.position
            out_queue.put((None, None))
   
    def detect_scenes(self, video=None, frame_skip=0, verbose=False):
        self._base_timecode = video.base_timecode
        total_frames = (video.duration.get_frames() - video.frame_number)
        if video.frame_size[0] > DEFAULT_MIN_WIDTH:
            downscale_factor = video.frame_size[0] // DEFAULT_MIN_WIDTH
        else:
            downscale_factor = 1

        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=int(total_frames), unit='frames', dynamic_ncols=True)

        frame_queue = queue.Queue(MAX_FRAME_QUEUE_LENGTH)
        decode_thread = threading.Thread(
                            target=SceneManager._decode_thread,
                            args=(self, video, frame_skip, downscale_factor, frame_queue),
                            daemon=True,
                        )
        decode_thread.start()
        frame_im = None
        while True:
            next_frame, position = frame_queue.get()
            if next_frame is None and position is None:
                break
            if not next_frame is None:
                frame_im = next_frame
            self._process_frame(position.frame_num, frame_im)
            if progress_bar is not None:
                progress_bar.update(1 + frame_skip)
        if progress_bar is not None:
            progress_bar.close()
        decode_thread.join()
        if self._exception_info is not None:
            raise self._exception_info[1].with_traceback(self._exception_info[2])
        self._last_pos = video.base_timecode + video.frame_number
    
    def get_cut_list(self, base_timecode=None):
        if base_timecode is None:
            base_timecode = self._base_timecode
        if base_timecode is None:
            return []
        return [FrameTimecode(cut, base_timecode) for cut in sorted(list(set(self._cutting_list)))]
    
    def get_scenes_from_cuts(self, cut_list, base_timecode):
        scene_list = []
        if not cut_list:
            scene_list.append((base_timecode + self._start_pos, base_timecode + self._last_pos))
            return scene_list
        last_cut = base_timecode + self._start_pos
        for cut in cut_list:
            scene_list.append((last_cut, cut))
            last_cut = cut
        # Last scene is from last cut to end of video.
        scene_list.append((last_cut, base_timecode + self._last_pos))
        return scene_list

    def get_scene_list(self, base_timecode=None, start_in_scene=False):
        if base_timecode is None:
            base_timecode = self._base_timecode
        if base_timecode is None:
            return []
        cut_list = self.get_cut_list(base_timecode)
        scene_list = self.get_scenes_from_cuts(cut_list, base_timecode)
        if not cut_list and not start_in_scene:
            scene_list = []
        return sorted(scene_list)
    
    def runner(self, video, frame_skip=0, verbose=False):
        self.detect_scenes(video, frame_skip, verbose)
        scenes = self.get_scene_list()
        return scenes
