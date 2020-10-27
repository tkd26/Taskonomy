# -*- coding: utf-8 -*-
# 第3章セマンティックセグメンテーションのデータオーギュメンテーション
# 注意　アノテーション画像はカラーパレット形式（インデックスカラー画像）となっている。

# パッケージのimport
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from utils.util import *

class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # アノテーションのリサイズ
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # input_sizeよりも短い辺はpaddingする
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(
                anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """引数input_sizeに大きさを変形するクラス"""

    def __init__(self, input_size, do_target):
        self.input_size = input_size
        self.do_target = do_target

    def __call__(self, img, target):

        # width = img.size[0]  # img.size=[幅][高さ]
        # height = img.size[1]  # img.size=[幅][高さ]

        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        if self.do_target:
            target = target.resize((self.input_size, self.input_size),
                                               Image.NEAREST)

        return img, target


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img=None):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)
        
        if anno_class_img != None:
            # アノテーション画像をNumpyに変換
            anno_class_img = np.array(anno_class_img)  # [高さ][幅]

            # 'ambigious'には255が格納されているので、0の背景にしておく
            index = np.where(anno_class_img == 255)
            anno_class_img[index] = 0

            # アノテーション画像をTensorに
            anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img

class to_Tensor(object):
    def __init__(self, task, do_target):
        self.task = task
        self.do_target = do_target
        
    def __call__(self, img, target, task=None):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)
        # [-1, 1]にリスケール
        img = rescale_pixel_values(img, new_scale=[-1.,1.], current_scale=[0, 1.], no_clip=False)
        
        if self.do_target:
            if self.task=='autoencoder':
                target = transforms.functional.to_tensor(target)
                # [-1, 1]にリスケール
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, 1.], no_clip=False)
            elif self.task=='segment_semantic':
                target = np.array(target)  # [高さ][幅]
                # 'ambigious'には0が格納されているので、backgroundの1にしておく
                index = np.where(target < 1)
                target[index] = 1
                index = np.where(target > 17)
                target[index] = 1
                # アノテーション画像をTensorに
                target = torch.from_numpy(target) - 1
            elif self.task=='edge_texture':
                target = np.array(target)
                target = torch.from_numpy(target).unsqueeze(0)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, target.max()], no_clip=False)
            elif self.task=='edge_occlusion':
                target = np.array(target)
                target = torch.from_numpy(target).unsqueeze(0)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, target.max()], no_clip=False)
            elif self.task=='normal':
                target = transforms.functional.to_tensor(target)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0.,1.], no_clip=False)
            elif self.task=='principal_curvature':
                target = transforms.functional.to_tensor(target)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0.,1.], no_clip=False)
            elif self.task=='keypoints2d':
                target = np.array(target)
                target = torch.from_numpy(target).unsqueeze(0)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, target.max()], no_clip=False)
            elif self.task=='keypoints3d':
                target = np.array(target)
                target = torch.from_numpy(target).unsqueeze(0)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, target.max()], no_clip=False)
            elif self.task=='depth_zbuffer':
                target = np.array(target)
                target = torch.from_numpy(target).unsqueeze(0)
                target = rescale_pixel_values(target, new_scale=[-1.,1.], current_scale=[0, target.max()], no_clip=False)


        return img, target
    