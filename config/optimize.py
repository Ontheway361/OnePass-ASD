# -*- coding: utf-8 -*-
import os
import argparse

root_dir = '/workspace/lujie/Benchmark/ava_active_speaker'

def optimize_args():
    parser = argparse.ArgumentParser('Configure for Train/Val of OnePassASD')
    
    # -- stage
    parser.add_argument('--is_debug', type=bool, default=False)

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])

    # -- i/o
    parser.add_argument('--root_dir',   type=str, default=root_dir)
    parser.add_argument('--audio_dir',  type=str, default=os.path.join(root_dir, 'slices'))
    parser.add_argument('--video_dir',  type=str, default=os.path.join(root_dir, 'cropfaces'))
    parser.add_argument('--train_file', type=str, default=os.path.join(root_dir, 'augmentcsv/ava_activespeaker_train_loader_checked.csv'))
    parser.add_argument('--val_file',   type=str, default=os.path.join(root_dir, 'augmentcsv/ava_activespeaker_val_loader_checked.csv'))
    parser.add_argument('--save_dir',   type=str, default='./checkpoint/one_stage_baseline_adam')
    
    # -- model
    parser.add_argument('--face_size', type=tuple, default=(112, 112))
    parser.add_argument('--tm_base',   type=str,   default='rnn', choices=['rnn', 'lstm', 'gru'])

    # -- optimize
    parser.add_argument('--pretrain',    type=str,   default='')
    parser.add_argument('--load_seqlen', type=int,   default=2500)
    parser.add_argument('--num_epochs',  type=int,   default=25)
    parser.add_argument('--start_epoch', type=int,   default=1)
    parser.add_argument('--batch_size',  type=int,   default=1)
    parser.add_argument('--num_workers', type=int,   default=8)
    parser.add_argument('--base_lr',     type=float, default=1e-4)
    parser.add_argument('--lr_steps',    type=list,  default=[10, 16, 22])
    parser.add_argument('--lr_gamma',    type=float, default=0.95)
    parser.add_argument('--weight_decay',type=float, default=5e-4)
    parser.add_argument('--evaluate',    type=bool,  default=True)

    # -- verbose
    parser.add_argument('--print_freq', type=int,   default=10) # TODO::Attention
    args = parser.parse_args()
    return args