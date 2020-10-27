import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FastNeuralStyleTransfer(nn.Module):
    def __init__(self, in_ch, do_task_list, fc=1, fc_nc=64, n=1):
        super(FastNeuralStyleTransfer, self).__init__()
        self.nc_list = [32*n, 64*n, 128*n, 
                        128*n, 128*n, 128*n, 128*n, 128*n, 
                        128*n, 128*n, 128*n, 128*n, 128*n, 
                        64*n, 32*n]
        self.do_task_list = do_task_list
        task_num = len(do_task_list)
        
#         self.film_generator = film_generator(sum(self.nc_list), task_num-1, fc, fc_nc)
        self.film_generator = film_generator(sum(self.nc_list), task_num, fc, fc_nc)

        # Initial convolution layers
        self.encoder = nn.ModuleDict({
                'conv1': ConvLayer(in_ch, 32*n, kernel_size=9, stride=1),
                'film1': film(32*n, task_num),
                'conv2': ConvLayer(32*n, 64*n, kernel_size=3, stride=2),
                'film2': film(64*n, task_num),
                'conv3': ConvLayer(64*n, 128*n, kernel_size=3, stride=2),
                'film3': film(128*n, task_num),

        })
        
        # Residual layers
        self.res = nn.ModuleDict({
            'res1': ResidualBlock(128*n, task_num),
            'res2': ResidualBlock(128*n, task_num),
            'res3': ResidualBlock(128*n, task_num),
            'res4': ResidualBlock(128*n, task_num),
            'res5': ResidualBlock(128*n, task_num),
        })
        
        # Upsampling Layers
        self.decoder = nn.ModuleDict({
            'deconv1': UpsampleConvLayer(128*n, 64*n, kernel_size=3, stride=1, upsample=2),
            'film4': film(64*n, task_num),
            'deconv2': UpsampleConvLayer(64*n, 32*n, kernel_size=3, stride=1, upsample=2),
            'film5': film(32*n, task_num)
        })
        
        self.lastconv_dic = nn.ModuleDict({})
        for task in self.do_task_list:
            if task=='autoencoder':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 3, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='segment_semantic':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 17, kernel_size=9, stride=1))
                
            elif task=='edge_texture':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 1, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='edge_occlusion':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 1, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='normal':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 3, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='principal_curvature':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 3, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='keypoints2d':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 1, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='keypoints3d':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 1, kernel_size=9, stride=1),
                                                        nn.Tanh())
            elif task=='depth_zbuffer':
                self.lastconv_dic[task] = nn.Sequential(ConvLayer(32*n, 1, kernel_size=9, stride=1),
                                                        nn.Tanh())
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
        self._initialize_weights()
        

    def forward(self, X, task, task_vec):
        factor, bias = self.film_generator(task_vec)
        factor_list, bias_list = [], []
        s = 0
        for nc in self.nc_list:
            factor_list.append(factor[:,s:s+nc,:,:])
            bias_list.append(bias[:,s:s+nc,:,:])
            s += nc
            
        y = self.relu(self.encoder['film1'](self.encoder['conv1'](X), factor_list[0], bias_list[0]))
        y = self.relu(self.encoder['film2'](self.encoder['conv2'](y), factor_list[1], bias_list[1]))
        y = self.relu(self.encoder['film3'](self.encoder['conv3'](y), factor_list[2], bias_list[2]))
        y = self.res['res1'](y, factor_list[3], bias_list[3], factor_list[4], bias_list[4])
        y = self.res['res2'](y, factor_list[5], bias_list[5], factor_list[6], bias_list[6])
        y = self.res['res3'](y, factor_list[7], bias_list[7], factor_list[8], bias_list[8])
        y = self.res['res4'](y, factor_list[9], bias_list[9], factor_list[10], bias_list[10])
        y = self.res['res5'](y, factor_list[11], bias_list[11], factor_list[12], bias_list[12])
        y = self.relu(self.decoder['film4'](self.decoder['deconv1'](y), factor_list[13], bias_list[13]))
        y = self.relu(self.decoder['film5'](self.decoder['deconv2'](y), factor_list[14], bias_list[14]))
        
        y = self.lastconv_dic[task](y)
        return y
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
#                 init.orthogonal_(m.weight.data, gain=init.calculate_gain('relu'))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, task_num):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.film1 = film(channels, task_num)
        
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.film2 = film(channels, task_num)
        self.relu = torch.nn.ReLU()

    def forward(self, x, factor1, bias1, factor2, bias2):
        residual = x
        out = self.relu(self.film1(self.conv1(x), factor1, bias1))
        out = self.film2(self.conv2(out), factor2, bias2)
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    
class film_generator(nn.Module):
    def __init__(self, norm_nc, cond_nc, fc=5, fc_nc=64):
        super().__init__()
        
        self.fc = fc
        self.relu = torch.nn.ReLU()

        if self.fc==1:
            self.transform = nn.Linear(cond_nc, norm_nc*2)
        if self.fc==3:
            self.transform1 = nn.Linear(cond_nc, fc_nc)
            self.transform2 = nn.Linear(fc_nc, fc_nc)
            self.transform = nn.Linear(fc_nc, norm_nc*2)
        if self.fc==5:
            self.transform1 = nn.Linear(cond_nc, fc_nc)
            self.transform2 = nn.Linear(fc_nc, fc_nc)
            self.transform3 = nn.Linear(fc_nc, fc_nc)
            self.transform4 = nn.Linear(fc_nc, fc_nc)
            self.transform = nn.Linear(fc_nc, norm_nc*2)

        self.transform.bias.data[:norm_nc] = 1
        self.transform.bias.data[norm_nc:] = 0
        
    def forward(self, cond):
        
        if self.fc==1:
            param = self.transform(cond).unsqueeze(2).unsqueeze(3)
        if self.fc==3:
            param = self.relu(self.transform1(cond))
            param = self.relu(self.transform2(param))
            param = self.transform(param).unsqueeze(2).unsqueeze(3)
        if self.fc==5:
            param = self.relu(self.transform1(cond))
            param = self.relu(self.transform2(param))
            param = self.relu(self.transform3(param))
            param = self.relu(self.transform4(param))
            param = self.transform(param).unsqueeze(2).unsqueeze(3)
        factor, bias = param.chunk(2, 1)
        return factor, bias
    
    
    
class film(nn.Module):
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
            
    def forward(self, x, factor, bias):
        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        # apply scale and bias
        out = normalized * factor + bias

        return out