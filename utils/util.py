import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torchvision
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

def resize_taskvec(taskvec, batch):
    return taskvec.unsqueeze(0).repeat(batch,1)

def save_imgs(imgs, task, batch, iteration, dir1, dir2):
    savepath = './outputs/{}/{}'.format(dir1, dir2)
    if not os.path.exists(savepath):
        os.makedirs('./outputs/{}/{}'.format(dir1, dir2))
    
    if task in ['autoencoder', 'normal', 'principal_curvature']:
        for n,img in enumerate(imgs):
            img = img.numpy().transpose((1,2,0))
            img = img.astype(np.float32)
            plt.imsave('{}/{:0=5}.png'.format(savepath, (iteration*batch)+n), img)
    elif task=='segment_semantic':
        for n,img in enumerate(imgs):
            imgs = img.numpy()
            if len(imgs.shape)==3:
                imgs = coloring_mask(imgs.argmax(axis=0))
            elif len(imgs.shape)==2:
                imgs = coloring_mask(imgs)
            plt.imsave('{}/{:0=5}.png'.format(savepath, (iteration*batch)+n), imgs)
    elif task in ['edge_texture', 'edge_occlusion', 'keypoints2d', 'keypoints3d', 'depth_zbuffer']:
        for n,img in enumerate(imgs):
            img = img.repeat(3,1,1)
            img = img.numpy().transpose((1,2,0))
            img = img.astype(np.float32)
            plt.imsave('{}/{:0=5}.png'.format(savepath, (iteration*batch)+n), img)
            
def coloring_mask(mask):
    '''
    0:background, 1:bottle, 2:chair, 3:couch, 4:potted plant, 5:bed, 6:dining table, 7:toilet, 
    8:tv, 9:microwave, 10:oven, 11:toaster, 12:sink, 13:refrigerator, 14:book, 15:clock, 16:vase
    '''
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 100, 100, 100]

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
#     plt.imshow(new_mask)
#     plt.show()
    return new_mask


def rescale_pixel_values(img, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A tensor, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image (type:tensor)
    """
    img = img.numpy().astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            img = np.clip(img, min_val, max_val)
        img = img - min_val
        img /= (max_val - min_val) 
    min_val, max_val = new_scale
    img *= (max_val - min_val)
    img += min_val

    return torch.Tensor(img)


class miou():
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        
    def __call__(self, outputs, labels):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x Class x H x W shape

        # BATCH x Class x H x W => BATCH x H x W
        outputs = [output.argmax(axis=0) for output in outputs.numpy()]
        outputs = np.array(outputs)
        outputs = torch.from_numpy(outputs)
        
        outputs = outputs.long()
        labels = labels.long()

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + self.smooth) / (union + self.smooth)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded.mean()  # thresholded Or thresholded.mean() if you are interested in average across the batch