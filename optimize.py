# -*- coding: utf-8 -*-
import os, sys, time, copy, shutil, torch, torchvision
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

import config as clib
import models as mlib
import dataset as dlib

from IPython import embed

class Optimizer(object):

    def __init__(self, args):
        self.args = args
        self.optimize = {}
        self.data = {}
        self.device = 'cpu'
        if args.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
    
    def report_config_summary(self):
        print('%sEnvironment Versions%s' % ('-' * 26, '-' * 26))
        print("- Python     : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch    : {}".format(torch.__version__))
        print("- TorchVison : {}".format(torchvision.__version__))
        print("- use_gpu    : {}".format(self.args.use_gpu))
        print('- is_debug   : {} [Attention!]'.format(self.args.is_debug))
        print('- workers    : {}'.format(self.args.num_workers))
        print('- epochs     : {}'.format(self.args.num_epochs))
        print('- print_freq : {}'.format(self.args.print_freq))
        print('- train_file : {}'.format(self.args.train_file.split('/')[-1]))
        print('- val_file   : {}'.format(self.args.val_file.split('/')[-1]))
        print('-' * 72)

    def load_model(self):
        # self.optimize['model'] = mlib.OnePassASD(self.args)
        self.optimize['model'] = mlib.OnePassASD_MultiHeads(self.args)
        self.optimize['criterion'] = {
            'ASDLoss' : mlib.ASDLoss(),
            'AuxAudioLoss' : mlib.AuxAudioLoss(),
            'AuxVisualLoss' : mlib.AuxVisualLoss()}
        # self.optimize['optimizer'] = torch.optim.SGD(
        self.optimize['optimizer'] = torch.optim.Adam(
            [{'params': self.optimize['model'].parameters()},
            {'params': self.optimize['criterion']['AuxAudioLoss'].parameters()},
            {'params': self.optimize['criterion']['AuxVisualLoss'].parameters()}],
            lr=self.args.base_lr) 
            # weight_decay=self.args.weight_decay, momentum=0.9, nesterov=True)
        # self.optimize['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
        #    self.optimize['optimizer'], milestones=self.args.lr_steps, gamma=self.args.lr_gamma)
        self.optimize['scheduler'] = torch.optim.lr_scheduler.StepLR(
            self.optimize['optimizer'], step_size=1, gamma=self.args.lr_gamma)
        if self.device == 'cuda':
            self.optimize['model'] = self.optimize['model'].cuda()
            self.optimize['criterion']['AuxAudioLoss'] = self.optimize['criterion']['AuxAudioLoss'].cuda()
            self.optimize['criterion']['AuxVisualLoss'] = self.optimize['criterion']['AuxVisualLoss'].cuda()
            if len(self.args.gpu_ids) > 1:
                self.optimize['model'] = nn.DataParallel(self.optimize['model'], device_ids=self.args.gpu_ids)
        if len(self.args.pretrain) > 0:
            checkpoint = torch.load(self.args.pretrain, map_location=lambda storage, loc: storage)
            self.optimize['model'].load_state_dict(checkpoint)
            print('loading checkpoints from %s ...' % self.args.pretrain)
        print('step1. model is loaded ...')

    def load_dataset(self):
        self.data['train_loader'] = DataLoader(
            dlib.AVA_ActiveSpeaker(args=self.args, mode='train'),
            batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.data['val_loader'] = DataLoader(
            dlib.AVA_ActiveSpeaker(args=self.args, mode='val'),
            batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        print('step2. dataset is loaded ...')
    
    def save_weights(self, attachinfo):
        if attachinfo[0] == 1 and os.path.exists(self.args.save_dir):
            print('delete the exists %s' % self.args.save_dir)
            shutil.rmtree(self.args.save_dir)
        os.makedirs(self.args.save_dir, exist_ok=True)
        details = 'epoch_%02d_loss_%.4f_acc_%.4f_auc_%.4f.pth' % attachinfo
        model_file = os.path.join(self.args.save_dir, details)
        save_flag = True
        try:
            torch.save(self.optimize['model'].state_dict(), model_file)
        except Exception as e:
            print(e)
            save_flag = False
        return save_flag
        
    def train_step(self, epoch):
        self.optimize['model'].train()
        current_lr =self.optimize['optimizer'].param_groups[0]['lr']
        gt_label_list, pd_score_list = [], []
        running_loss, running_corrects = 0.0, 0.0
        for idx, (audios, videos, labels) in enumerate(self.data['train_loader'], start=1):
            audios.requires_grad = False
            videos.requires_grad = False
            labels.requires_grad = False
            if self.device == 'cuda':
                audios = audios.cuda()
                videos = videos.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(True):
                # logits = self.optimize['model'](audios[0], videos[0]) # TODO::Attention
                logits, afeats, vfeats = self.optimize['model'](audios[0], videos[0])
                aloss = self.optimize['criterion']['AuxAudioLoss'](afeats, labels[0])
                vloss = self.optimize['criterion']['AuxVisualLoss'](vfeats, labels[0])
                closs, score, predy, nhits = self.optimize['criterion']['ASDLoss'](logits, labels[0])
                floss = closs + 0.4 * aloss + 0.4 * vloss
                # self.optimize['optimizer'].zero_grad()
                floss.backward()
                self.optimize['optimizer'].step()
                self.optimize['optimizer'].zero_grad()
            gt_label_list.extend(labels[0].reshape((-1)).detach().cpu().numpy().tolist())
            pd_score_list.extend(score.detach().cpu().numpy()[:, 1].tolist())
            running_loss += floss.item() * len(score)
            running_corrects += nhits
            if idx % self.args.print_freq == 0:
                ave_loss = running_loss / (len(gt_label_list) + 1e-3)
                ave_acc = running_corrects / (len(gt_label_list) + 1e-3)
                iter_progress = float(idx) / len(self.data['train_loader']) * 100
                verbose_info = '%s, [%02d], lr %.08f, iter %.2f%%, loss %.4f, acc %.4f\r' % (time.strftime('%Y-%m-%d %H:%M:%S'), \
                    epoch, current_lr, iter_progress, ave_loss, ave_acc)
                sys.stderr.write(verbose_info)
                sys.stderr.flush()
        sys.stdout.write("\n") 
        ave_epoch_loss = running_loss / (len(gt_label_list) + 1e-3)
        ave_epoch_acc = running_corrects / (len(gt_label_list) + 1e-3)
        ave_epoch_auc = roc_auc_score(gt_label_list, pd_score_list)
        # ave_epoch_ap = average_precision_score(gt_label_list, pd_score_list)
        return (ave_epoch_loss, ave_epoch_acc, ave_epoch_auc)

    def evaluate_step(self, epoch):
        self.optimize['model'].eval()
        gt_label_list, pd_score_list = [], []
        running_loss, running_corrects = 0.0, 0.0
        print_freq = max(len(self.data['val_loader']) // 4, self.args.print_freq)
        for idx, (audios, videos, labels) in enumerate(self.data['val_loader'], start=1):
            if self.device == 'cuda':
                audios = audios.cuda()
                videos = videos.cuda()
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                logits, _, _ = self.optimize['model'](audios[0], videos[0])
                loss, score, predy, nhits = self.optimize['criterion']['ASDLoss'](logits, labels[0])
            gt_label_list.extend(labels[0].reshape((-1)).detach().cpu().numpy().tolist())
            pd_score_list.extend(score.detach().cpu().numpy()[:, 1].tolist())
            running_loss += loss.item() * len(score)
            running_corrects += nhits
            if idx % print_freq == 0:
                ave_loss = running_loss / (len(gt_label_list) + 1e-3)
                ave_acc = running_corrects / (len(gt_label_list) + 1e-3)
                iter_progress = float(idx) / len(self.data['val_loader']) * 100
                verbose_info = '%s, iter %.2f%%, loss %.4f, acc %.4f\r' % (time.strftime('%Y-%m-%d %H:%M:%S'), \
                    iter_progress, ave_loss, ave_acc)
                sys.stderr.write(verbose_info)
                sys.stderr.flush()
        sys.stdout.write("\n")
        ave_epoch_loss = running_loss / (len(gt_label_list) + 1e-3)
        ave_epoch_acc = running_corrects / (len(gt_label_list) + 1e-3)
        ave_epoch_auc = roc_auc_score(gt_label_list, pd_score_list)
        # ave_epoch_ap = average_precision_score(gt_label_list, pd_score_list)
        return (ave_epoch_loss, ave_epoch_acc, ave_epoch_auc)
    
    def optimize_step(self):
        for epoch in range(self.args.start_epoch, self.args.num_epochs + 1):
            print('epoch - %02d||%02d is starting ...' % (epoch, self.args.num_epochs))
            start_timestamp = time.time()
            train_loss, train_acc, train_auc = self.train_step(epoch)
            self.optimize['scheduler'].step()
            train_finish = time.time()
            val_loss, val_acc, val_auc = self.evaluate_step(epoch)
            eval_finish = time.time()
            train_time = float(train_finish - start_timestamp) / 60.0
            eval_time = float(eval_finish - train_finish) / 60.0
            flag = self.save_weights((epoch, val_loss, val_acc, val_auc))
            printinfo_list = [
                '%s epoch-%03d summarys %s' % ('-' * 16, epoch, '-' * 16),
                'train_loss %.4f, train_acc %.4f, train_auc %.4f' % (train_loss, train_acc, train_auc),
                ' eval_loss %.4f,  eval_acc %.4f,  eval_auc %.4f' % (val_loss, val_acc, val_auc),
                'train_time %.4f, eval_time %.4f' % (train_time, eval_time),
                '-' * 52]
            logs = [print(s) for s in printinfo_list]
            if self.args.is_debug:
                break
        
    def run_optimier(self):
        self.report_config_summary()
        self.load_model()
        self.load_dataset()
        self.optimize_step()

if __name__ == "__main__":

    train_config = clib.optimize_args()
    optiminizer = Optimizer(args=train_config)
    optiminizer.run_optimier()







