# -*- coding: utf-8 -*-
import os, sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# テストコメント

# proxy
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from torch.optim.lr_scheduler import MultiStepLR
import tensorboardX as tbx

from utils.dataloader import make_dataloaders
from utils.util import *
from utils.optimizer import *

from models.FNST import FastNeuralStyleTransfer as FNST
from models.FNST_STL import FastNeuralStyleTransfer_STL as FNST_STL
from models.discriminator import Discriminator

# Setup seeds
# torch.manual_seed(1234)
# np.random.seed(1234)
# random.seed(1234)

def get_args():
    parser = argparse.ArgumentParser(description='multi task learning',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--autoencoder', action='store_true')
    parser.add_argument('--segment_semantic', action='store_true')
    parser.add_argument('--edge_texture', action='store_true')
    parser.add_argument('--edge_occlusion', action='store_true')
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--principal_curvature', action='store_true')
    parser.add_argument('--keypoints2d', action='store_true')
    parser.add_argument('--keypoints3d', action='store_true')
    parser.add_argument('--depth_zbuffer', action='store_true')
    
    parser.add_argument('--mode', default='train', choices=['train', 'val'])
    parser.add_argument('--mode_model', default='FNST', choices=['FNST_STL', 'FNST'])
    parser.add_argument('--random_lr', action='store_true') 
    parser.add_argument('--optim', default='adam', choices=['sgd', 'sgd2', 'sgd3', 'sgd3-2', 'sgd3-3', 'sgd3-4', 'sgd3-5', 'adam', 'adam2'])
    
    # if Ours or EncIN
    parser.add_argument('--fc', default=5, type=int, choices=[None, 1, 3, 5])
    parser.add_argument('--fc_nc', default=64, type=int, choices=[None, 64, 128])
    
    parser.add_argument('-e', '--epochs', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--imgsize', type=int, default=128)
#     parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--load', default=False, nargs='*')
    parser.add_argument('--version', type=str, default=None)
    
    parser.add_argument('--save_iter', type=int, default=1)
    parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--val_score', action='store_true')
    
    return parser.parse_args()

#-------------------------------------------------------------------
# train
#-------------------------------------------------------------------

class Manager():
    def __init__(
        self, args,
        netG, netD,
        dataloaders_dict,
        criterion, optimizer, weight,
        device,
        task_vecs, do_task_list,
        save_model_path,
        task_val_func_dic, val_func_dic,
        save_name,
        scheduler
    ):
        
        self.args = args
        self.netG = netG
        self.netD = netD
        self.dataloaders_dict = dataloaders_dict
        self.criterion = criterion
        self.optimizerG, self.optimizerD = optimizer
        self.weight = weight
        self.device = device
        self.task_vecs = task_vecs
        self.do_task_list = do_task_list
        self.save_model_path = save_model_path
        self.task_val_func_dic = task_val_func_dic
        self.val_func_dic = val_func_dic
        self.save_name = save_name
        self.scheduler = scheduler
        
    def epoch(self, epoch):
        train_loss_dic = dict([(task_name, []) for task_name in self.do_task_list])
        val_loss_dic = dict([(task_name, []) for task_name in self.do_task_list])
        
        '''
        ------------------------------------------------------------------------
        学習
        ------------------------------------------------------------------------
        '''  
        if self.args.mode=='train':
            print('-------------')
            print('Epoch {}/{}'.format(epoch, self.args.epochs))
            print('-------------')
            # train
            for task_name in self.netD.keys():
                self.netD[task_name].train()
            self.netG.train() 
#             scheduler = MultiStepLR(self.optimizerG, milestones=[50], gamma=0.5)
            
            print('(train)')
            for train_iter, datas in enumerate(self.dataloaders_dict['train']):
                task = random.choice(self.do_task_list)
                train_batch_loss = self.train(train_iter, datas, task)
                train_loss_dic[task] += [train_batch_loss] # バッチ毎のlossをリストに入れる
                if args.optim=='adam2':
                    self.scheduler.step() 
                
            # save networks
            if (epoch) % self.args.save_iter == 0:
                torch.save(
                    self.netG.state_dict(), 
                    '{}/{:0=5}.pth'.format(self.save_model_path, epoch))
                for task in self.do_task_list:
                    if task in ['autoencoder', 'edge_texture', 'edge_occlusion',
                                'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                        torch.save(self.netD[task].state_dict(), 
                            '{}/D_{}/{:0=5}.pth'.format(self.save_model_path, task, epoch))
        
        '''
        ------------------------------------------------------------------------
        評価
        ------------------------------------------------------------------------
        '''  
        self.netG.eval() 
        print('(val)')
        
        val_score_dic = {}
        for task_name in self.do_task_list:
            val_score_dic[task_name] = {}
            for val_func_name in self.task_val_func_dic[task_name]:
                val_score_dic[task_name][val_func_name] = []
            
        for val_iter, datas in enumerate(self.dataloaders_dict['val']):
            for task in self.do_task_list:
                val_batch_loss, val_batch_scores = self.val(val_iter, datas, task)
                val_loss_dic[task] += [val_batch_loss] # バッチ毎のlossをリストに入れる

                for val_func_name, val_batch_score in val_batch_scores.items():
                    val_score_dic[task][val_func_name] += [val_batch_score]

            # 保存する画像の枚数を指定
            if self.args.save_imgs: 
                break # バッチサイズ分のみ保存

        for task_name in self.do_task_list:
            if self.args.mode=='train':
                train_loss_dic[task_name] = sum(train_loss_dic[task_name]) / len(train_loss_dic[task_name])
            else:
                train_loss_dic[task_name] = 0
            val_loss_dic[task_name] = sum(val_loss_dic[task_name]) / len(val_loss_dic[task_name])
            
            for val_func_name in val_score_dic[task_name]:
                if len(val_score_dic[task_name][val_func_name]) != 0:
                    val_score_dic[task_name][val_func_name] = sum(val_score_dic[task_name][val_func_name]) / len(val_score_dic[task_name][val_func_name])
        return train_loss_dic, val_loss_dic, val_score_dic
        
        
    def train(self, iteration, datas, task):
        # img and annotation img
        imgs, targets_dict = datas
        imgs, targets = imgs.to(self.device), targets_dict[task].to(self.device)
        # batch size
        batch = imgs.shape[0]
        
        task_vec = resize_taskvec(self.task_vecs[task], batch).to(self.device)

        ##### trainG #####
        with torch.set_grad_enabled(True):
            if task in ['autoencoder', 'edge_texture', 'edge_occlusion',
                        'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                outputs = self.netG(imgs, task, task_vec)
                loss1 = self.criterion[task](outputs, targets)
                
                D_fake = self.netD[task](outputs)
                errG_gan = self.criterion['gan'](D_fake, torch.tensor(1.0).expand_as(D_fake).to(self.device)) 
#                 errG_id = self.criterion['identity'](outputs, imgs)
                errG_id = 0
                loss2 = errG_gan + errG_id * 5
            
                loss = (loss1*0.996 + loss2*(1-0.996)) * self.weight[task]
                
            if task=='segment_semantic':
                outputs = self.netG(imgs, task, task_vec)
                loss = self.criterion[task](outputs, targets.long())
                loss = loss * self.weight[task]
                
                
            self.netG.zero_grad()
            for task_name in self.netD.keys():
                self.netD[task_name].zero_grad()
            loss.backward()
            if args.random_lr:
                self.optimizerG.step(task_name=task)
            else:
                self.optimizerG.step()
        
        ##### trainD #####
            if task in ['autoencoder', 'edge_texture', 'edge_occlusion',
                        'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                # train with real
                D_real = self.netD[task](targets)
                errD_real = self.criterion['gan'](D_real, torch.tensor(1.0).expand_as(D_real).to(self.device))
                # train with fake
                D_fake = self.netD[task](outputs.detach())
                errD_fake = self.criterion['gan'](D_fake, torch.tensor(0.0).expand_as(D_fake).to(self.device))
                errD = (errD_real + errD_fake) / 2

                self.netG.zero_grad()
                for task_name in self.netD.keys():
                    self.netD[task_name].zero_grad()
                errD.backward()
                self.optimizerD[task].step()

        return loss

    
    def val(self, iteration, datas, task):
        # img and annotation img
        imgs, targets_dict = datas
        imgs, targets = imgs.to(self.device), targets_dict[task].to(self.device)
        # batch size
        batch = imgs.shape[0]
        
        task_vec = resize_taskvec(self.task_vecs[task], batch).to(self.device)
        
        ##### trainG #####
        with torch.set_grad_enabled(False):
            
            if task in ['autoencoder', 'edge_texture', 'edge_occlusion',
                        'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                outputs = self.netG(imgs, task, task_vec)
                loss = self.criterion[task](outputs, targets)
                loss *= self.weight[task]
            elif task=='segment_semantic':
                outputs = self.netG(imgs, task, task_vec)
                loss = self.criterion[task](outputs, targets.long())
                loss *= self.weight[task]
                
            if self.args.save_imgs:
                save_outputs = outputs.detach().cpu() 
                save_targets = targets.cpu()
                if task in ['autoencoder', 'edge_texture', 'edge_occlusion',
                            'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                    save_outputs = [rescale_pixel_values(save_output, new_scale=[0.,1.], current_scale=[-1.,1.]) for save_output in save_outputs]
                    # save_targets = [rescale_pixel_values(save_target, new_scale=[0.,1.], current_scale=[-1.,1.]) for save_target in save_targets]
                save_imgs(save_outputs, task, self.args.batch_size, iteration, self.save_name, task)
                # save_imgs(save_targets, task, self.args.batch_size, iteration, self.save_name, 'target_'+task)

            score = {}
            for val_func_name in self.task_val_func_dic[task]:
                score[val_func_name] = self.val_func_dic[val_func_name](outputs.detach().cpu(), targets.cpu())
                    
        return loss, score



#-------------------------------------------------------------------
# main
#-------------------------------------------------------------------

def main(args):
    
    ##### Setup #####
    '''
    ------------------------------------------------------------------------
    保存ファイルの設定
    ------------------------------------------------------------------------
    '''
    task_dict = {
        'autoencoder': args.autoencoder,
        'segment_semantic': args.segment_semantic,
        'edge_texture': args.edge_texture,
        'edge_occlusion': args.edge_occlusion,
        'normal': args.normal, 
        'principal_curvature': args.principal_curvature,
        'keypoints2d': args.keypoints2d,
        'keypoints3d': args.keypoints3d,
        'depth_zbuffer': args.depth_zbuffer,
    }
    do_task_list = [t for t,val in task_dict.items() if val==True]
    task_num = len(do_task_list)
    print('DO TASK LIST:', do_task_list)

    save_name = []
    if args.random_lr: save_name.extend(['RandomLR'])
    save_name.extend(['OPTIM'+args.optim])
    save_name.extend(['Model'+args.mode_model])
    if args.fc!=None: save_name.extend(['FC{}'.format(args.fc)])
    # if args.fc_nc!=None: save_name.extend([str(args.fc_nc)])
    if args.version: save_name.extend([args.version])
        
    save_name.extend(do_task_list)
    save_name = '_'.join(save_name)
    save_model_path = './weights/{}'.format(save_name)
    save_log_path = './logs/{}'.format(save_name)
    
    # make folder to save network
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)

    for task_name in do_task_list:
        if task_name in ['autoencoder', 'edge_texture', 'edge_occlusion',
                         'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
            if not os.path.exists(os.path.join(save_model_path, 'D_{}'.format(task_name))):
                os.mkdir(os.path.join(save_model_path, 'D_{}'.format(task_name)))

    # define writer
    if args.mode=='train':
        writer = tbx.SummaryWriter(log_dir=save_log_path)

    '''
    ------------------------------------------------------------------------
    モデルの設定
    ------------------------------------------------------------------------
    '''    
    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    
    # model
    if args.mode_model == 'FNST':
        netG = FNST(3, do_task_list, args.fc, args.fc_nc, n=3)
    elif args.mode_model == 'FNST_STL':
        netG = FNST_STL(3, do_task_list, args.fc, args.fc_nc, n=3)
    else:
        sys.exit('Error:No model selected')
            
    netG.to(device)
    if not args.random_lr:
        netG = torch.nn.DataParallel(netG)
    
    # count parameters
    param = 0
    for p in netG.parameters():
#         print(p.shape)
        param += p.numel()
    print('PARAM:{}'.format(param))
    
    netD = {
        'autoencoder':Discriminator(in_ch=3).to(device),
        'edge_texture':Discriminator(in_ch=1).to(device),
        'edge_occlusion':Discriminator(in_ch=1).to(device),
        'normal':Discriminator(in_ch=3).to(device),
        'principal_curvature':Discriminator(in_ch=3).to(device),
        'keypoints2d':Discriminator(in_ch=1).to(device),
        'keypoints3d':Discriminator(in_ch=1).to(device),
        'depth_zbuffer':Discriminator(in_ch=1).to(device),
    }
    for task_name in netD.keys():
        netD[task_name] = torch.nn.DataParallel(netD[task_name])
        
    # load parameter
    load_epoch = 0
    if args.load:
        load_path, load_epoch = args.load
        load_epoch = int(load_epoch)
        param = torch.load('{}/{:0=5}.pth'.format(load_path, load_epoch))
        netG.load_state_dict(param)
        if args.mode == 'train':
            for task_name in do_task_list:
                if task_name in ['autoencoder', 'edge_texture', 'edge_occlusion',
                            'normal', 'principal_curvature', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
                    param = torch.load('{}/D_{}/{:0=5}.pth'.format(load_path, task_name, load_epoch))
                    netD[task_name].load_state_dict(param)
    
    # torch.backends.cudnn.benchmark = True

    '''
    ------------------------------------------------------------------------
    学習の設定
    ------------------------------------------------------------------------
    '''   
    # train parameter
    # optimizerの参考元：https://github.com/facebookresearch/astmt
    scheduler = None
    if args.random_lr:
        if args.optim=='sgd':
            optimizerG = SGD_c(params=[ # lambda:共有率
                    {"params": netG.film_generator.parameters(), "lambda": 1.0},
                    {"params": netG.encoder.parameters(), "lambda": 1.0},
                    {"params": netG.res.parameters(), "lambda": 0.0},
                    {"params": netG.decoder.parameters(), "lambda": 0.0},
                    {"params": netG.lastconv_dic.parameters(), "lambda": 1.0},
                    # {"params": netG.relu.parameters(), "lambda": 1.0},
                    ], lr=1e-3, momentum=0.9, weight_decay=1e-04, do_task_list=do_task_list)
        elif args.optim=='sgd2':
            optimizerG = SGD_c(params=[ # lambda:共有率
                    {"params": netG.film_generator.parameters(), "lambda": 1.0},
                    {"params": netG.encoder.parameters(), "lambda": 0.0},
                    {"params": netG.res.parameters(), "lambda": 0.0},
                    {"params": netG.decoder.parameters(), "lambda": 0.0},
                    {"params": netG.lastconv_dic.parameters(), "lambda": 1.0},
                    # {"params": netG.relu.parameters(), "lambda": 1.0},
                    ], lr=1e-3, momentum=0.9, weight_decay=1e-04, do_task_list=do_task_list)
        netG = torch.nn.DataParallel(netG)
    else:
        if args.optim=='sgd':
            optimizerG = torch.optim.SGD(netG.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-04)
        elif args.optim=='sgd2':
            optimizerG = torch.optim.SGD(netG.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-04)
        elif args.optim=='adam':
            optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6)
        elif args.optim=='adam2':
            optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.5)

    optimizerD = {
        'autoencoder':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'edge_texture':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'edge_occlusion':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'normal':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'principal_curvature':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'keypoints2d':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'keypoints3d':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
        'depth_zbuffer':torch.optim.Adam(netG.parameters(), lr=1e-4, weight_decay=2e-6),
    }
    
    criterion = { 
        'gan':nn.BCEWithLogitsLoss(),
        'identity':nn.L1Loss(),
        'autoencoder':nn.L1Loss(),
        'segment_semantic':nn.CrossEntropyLoss(),
        'edge_texture':nn.L1Loss(),
        'edge_occlusion':nn.L1Loss(),
        'normal':nn.L1Loss(),
        'principal_curvature':nn.L1Loss(),
        'keypoints2d':nn.L1Loss(),
        'keypoints3d':nn.L1Loss(),
        'depth_zbuffer':nn.L1Loss(),
    }
    
    task_val_func_dic = {
        'autoencoder':['mse'],
        'segment_semantic':['miou'],
        'edge_texture':['mse'],
        'edge_occlusion':['mse'],
        'normal':['mse'],
        'principal_curvature':['mse'],
        'keypoints2d':['mse'],
        'keypoints3d':['mse'],
        'depth_zbuffer':['mse'],
    }
    val_func_dic = {
        'mse': nn.MSELoss(),
        'miou':miou(),
    }
    
    # weight
    weight = {}
    for task_name in do_task_list:
        weight[task_name] = 1.0

    # task vectors
    task_vecs = {}
#     task_vecs_one_hot = torch.Tensor([[0] * (task_num-1)])
#     task_vecs_one_hot = torch.cat([task_vecs_one_hot , torch.eye(task_num-1)], dim=0)
    task_vecs_one_hot = torch.eye(task_num)
    for i,task_name in enumerate(do_task_list):
        task_vecs[task_name] = task_vecs_one_hot[i]
    print('TASK VECS:', task_vecs)

    '''
    ------------------------------------------------------------------------
    データロード
    ------------------------------------------------------------------------
    '''     
    rootpath = '/home/yanai-lab/takeda-m/space/dataset/taskonomy-sample-model-1'
    dataloaders_dict = {}
    train_dataloader, val_dataloader = make_dataloaders(args, rootpath, do_task_list)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # number of images
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    print('train num:{}, val num:{}'.format(num_train_imgs, num_val_imgs))


    '''
    ------------------------------------------------------------------------
    学習
    ------------------------------------------------------------------------
    '''  
    manager = Manager(
        args,
        netG, netD,
        dataloaders_dict,
        criterion, (optimizerG, optimizerD), weight,
        device,
        task_vecs, do_task_list,
        save_model_path,
        task_val_func_dic, val_func_dic,
        save_name,
        scheduler
    )

    max_score = {}
    for task_name in do_task_list:
        max_score[task_name] = {}
        for val_func_name in task_val_func_dic[task_name]:
            if val_func_name=='miou':
                max_score[task_name][val_func_name] = 0
            elif val_func_name=='mse':
                max_score[task_name][val_func_name] = 1

    max_score_epoch = {}
    for task_name in do_task_list:
        max_score_epoch[task_name] = {}
        for val_func_name in task_val_func_dic[task_name]:
            max_score_epoch[task_name][val_func_name] = 0
                
    
    for epoch in range(load_epoch+1, args.epochs+1):
        train_loss_dic, val_loss_dic, val_score_dic = manager.epoch(epoch)
        if args.mode=='val': 
            print(val_score_dic)
            break
        else:
            # write tensorboardX
            writer.add_scalars('train_loss', train_loss_dic, epoch)
            writer.add_scalars('val_loss', val_loss_dic, epoch)
            # 結果出力
            for task_name in do_task_list:
                for val_func_name in task_val_func_dic[task_name]:
                    if val_func_name=='miou': # miouバージョン
                        if max_score[task_name][val_func_name] < val_score_dic[task_name][val_func_name]:
                            max_score[task_name][val_func_name] = val_score_dic[task_name][val_func_name]
                            max_score_epoch[task_name][val_func_name] = epoch
                    else: # mseバージョン
                        if max_score[task_name][val_func_name] > val_score_dic[task_name][val_func_name]:
                            max_score[task_name][val_func_name] = val_score_dic[task_name][val_func_name]
                            max_score_epoch[task_name][val_func_name] = epoch
                print('EPOCH: {:04d} | DATASET: {:s} || TRAIN: {} || TEST: {} {} || MAX: {} ({}ep)'
                    .format(epoch, task_name,
                            train_loss_dic[task_name],
                            val_loss_dic[task_name], val_score_dic[task_name], 
                            max_score[task_name], max_score_epoch[task_name]))
                    
    if args.mode=='train': writer.close()

if __name__ == '__main__':
    args = get_args()
    main(args)