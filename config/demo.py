# -*- coding: utf-8 -*-
import os
import argparse

def demo_args():
    
    test_dir = './testdemo'
    ckpt_dir = './checkpoint/one_stage_talknet_drop_conv3d'
    parser = argparse.ArgumentParser('Configure for Train/Val of OnePassASD')
    
    # -- stage
    parser.add_argument('--is_debug', type=bool, default=True)

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)

    # -- i/o
    parser.add_argument('--threads',    type=int, default=4)
    parser.add_argument('--test_dir',   type=str, default=test_dir)
    parser.add_argument('--test_video', type=str, default=os.path.join(test_dir, 'video', 'demo.mp4'))
    parser.add_argument('--demo_dir',   type=str, default=os.path.join(test_dir, 'demos'))
    
    # -- model
    parser.add_argument('--pretrain',    type=str,   default=os.path.join(ckpt_dir, 'epoch_08_loss_0.2540_acc_0.8969_auc_0.9488.pth'))
    parser.add_argument('--tm_base',     type=str,   default='rnn', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--face_size',   type=tuple, default=(112, 112))
    parser.add_argument('--conf_thresh', type=float, default=0.9)
    parser.add_argument('--scalelist',   type=list,  default=[0.25])
    
    # -- track
    parser.add_argument('--min_shot_frames',  type=int,   default=10)
    parser.add_argument('--min_failed_dets',  type=int,   default=10)
    parser.add_argument('--track_iou_thresh', type=float, default=0.5)
    parser.add_argument('--smooth_win_size',  type=int,   default=13)
    parser.add_argument('--crop_face_scale',  type=float, default=0.4)


    # -- verbose
    parser.add_argument('--print_freq', type=int,   default=10)
    args = parser.parse_args()
    return args