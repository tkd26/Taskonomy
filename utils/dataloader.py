# -*- coding: utf-8 -*-
# パッケージのimport
import os.path as osp
import sys
from PIL import Image
import scipy.io
import numpy as np
import torch
import torch.utils.data as data
import glob

from utils.data_augumentation import *

def make_datapath_list(do_task_list, rootpath):
    """
    学習、検証の画像データとターゲットデータへのファイルパスリストを作成する。
    """
    
    img_list = glob.glob(osp.join(rootpath, 'rgb', '*.png'))
    img_list.sort()
    train_img_list = img_list[:8000]
    val_img_list = img_list[8000:]
    
    target_list_dic = {}
    train_target_list_dic = {}
    val_target_list_dic = {}
    for task in do_task_list:
        if task=='autoencoder':
            target_list_dic[task] = glob.glob(osp.join(rootpath, 'rgb', '*.png'))
        else:
            target_list_dic[task] = glob.glob(osp.join(rootpath, task, '*.png'))
        target_list_dic[task].sort()

        if len(img_list) != len(target_list_dic[task]):
            print('[ERROR]The number of data of input and target is different.')
            print('input:', len(img_list), ',target:', len(target_list_dic[task]))
            sys.exit(1)
    
        train_target_list_dic[task] = target_list_dic[task][:8000]
        val_target_list_dic[task] = target_list_dic[task][8000:]

    return train_img_list, train_target_list_dic, val_img_list, val_target_list_dic


class DataTransform():

    def __init__(self, input_size, task):
            
        self.data_transform = {
            'train': Compose([
                Resize(input_size, do_target=True),  # リサイズ(input_size)
                to_Tensor(task, do_target=True),
            ]),
            'val': Compose([
                Resize(input_size, do_target=True),  # リサイズ(input_size)
                to_Tensor(task, do_target=True)
            ])
        }

    def __call__(self, phase, img, target):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, target)
    
    
class TaskonomyDataset(data.Dataset):
    """
    TaskonomyのDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, target_list_dic, phase, transforms, do_task_list):
        self.img_list = img_list
        self.target_list_dic = target_list_dic
        self.phase = phase
        self.transforms = transforms
        self.do_task_list = do_task_list

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, targets = self.pull_item(index)
        return img, targets

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        targets_dict = {}
        pull_targets_dict = {}
        for task in self.do_task_list:
            # 2. ターゲット画像読み込み
            target_file_path = self.target_list_dic[task][index]
            targets_dict[task] = Image.open(target_file_path)   # [高さ][幅]

            # 3. 前処理を実施
            pull_img, pull_targets_dict[task] = self.transforms[task](self.phase, img, targets_dict[task])

        return pull_img, pull_targets_dict

    
def make_dataloaders(args, rootpath, do_task_list):
    imgsize = args.imgsize
    batch_size = args.batch_size
    
    train_img_list, train_target_list_dic, val_img_list, val_target_list_dic = make_datapath_list(do_task_list, rootpath)
    
    transform_dic = {}
    for task in do_task_list:
        transform_dic[task] = DataTransform(input_size=imgsize, task=task)
    
    train_dataset = TaskonomyDataset(train_img_list, train_target_list_dic, phase="train", 
                                     transforms=transform_dic, do_task_list=do_task_list)

    val_dataset = TaskonomyDataset(val_img_list, val_target_list_dic, phase="val", 
                                   transforms=transform_dic, do_task_list=do_task_list)

    # DataLoader作成
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader