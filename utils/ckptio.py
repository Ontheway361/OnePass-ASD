# -*- coding: utf-8 -*-

import torch
from IPython import embed

def check_keys(model, ckpt_dict):
    ckpt_keys = set(ckpt_dict.keys())
    model_keys = set(model.state_dict().keys())
    common_keys = model_keys & ckpt_keys
    unused_keys = ckpt_keys - model_keys
    missed_keys = model_keys - ckpt_keys
    if len(missed_keys):
        print('missed_keys : ', missed_keys)
    

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_weights(model, pretrain='', device='cpu'):
    if device == 'cpu':
        checkpoints = torch.load(pretrain, map_location=lambda storage, loc: storage)
    else:
        checkpoints = torch.load(pretrain, map_location=lambda storage, loc: storage.cuda(0))
    if "state_dict" in checkpoints.keys():
        checkpoints = remove_prefix(checkpoints['state_dict'], 'module.')
    else:
        checkpoints = remove_prefix(checkpoints, 'module.')
    check_keys(model, checkpoints)
    model.load_state_dict(checkpoints, strict=True)
    return model
