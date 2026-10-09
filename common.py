import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def gmsd(dis_img:Type[Union[torch.Tensor,np.ndarray]],ref_img:Type[Union[torch.Tensor,np.ndarray]],c=170,device='cuda'):
    # 输入类型检查
    if type(dis_img) == np.ndarray:
        assert dis_img.ndim == 2 or dis_img.ndim == 3
        if dis_img.ndim == 2:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0).unsqueeze(0)
        else:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0)

    if type(ref_img) == np.ndarray:
        assert ref_img.ndim == 2 or ref_img.ndim == 3
        if ref_img.ndim == 2:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0)
        else:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0)
    # 算法需要输入为灰度图像，像素值0-255
    if torch.max(dis_img) <= 1:
        dis_img = dis_img * 255
    if torch.max(ref_img) <= 1:
        ref_img = ref_img * 255

    '''算法主体'''
    hx=torch.tensor([[1/3,0,-1/3]]*3,dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)#Prewitt算子
    ave_filter=torch.tensor([[0.25,0.25],[0.25,0.25]],dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)#均值滤波核
    down_step=2#下采样间隔
    hy=hx.transpose(2,3)

    dis_img=dis_img.float().to(device)
    ref_img=ref_img.float().to(device)

    #均值滤波
    ave_dis=F.conv2d(dis_img,ave_filter,stride=1)
    ave_ref=F.conv2d(ref_img,ave_filter,stride=1)
    #下采样
    ave_dis_down=ave_dis[:,:,0::down_step,0::down_step]
    ave_ref_down=ave_ref[:,:,0::down_step,0::down_step]
    #计算mr md等中间变量
    mr_sq=F.conv2d(ave_ref_down,hx)**2+F.conv2d(ave_ref_down,hy)**2
    md_sq=F.conv2d(ave_dis_down,hx)**2+F.conv2d(ave_dis_down,hy)**2
    mr=torch.sqrt(mr_sq)
    md=torch.sqrt(md_sq)
    GMS=(2*mr*md+c)/(mr_sq+md_sq+c)
    GMSD=torch.std(GMS.view(-1))
    return GMSD



