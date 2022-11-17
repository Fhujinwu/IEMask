# -*- coding: utf-8 -*-
# @Author  : HUJINWU
# @Time    : 2021-05-14 17:01
# @Function: Global Enchance Module

import torch
from torch import nn

class GlobalEnchanceModule(nn.Module):
    def __init__(self, in_channels, out_channels):  #2048, 256
        super(GlobalEnchanceModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //2 , 5, stride=2, padding=2),   #2048-----1024
            nn.Conv2d(in_channels // 2, in_channels // 4, 5, stride=2, padding=2), #1024-----512
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv1 = nn.Conv2d(in_channels //4 , out_channels, 1, 1)

    def forward(self, x):
        w = int(x.shape[2])
        h = int(x.shape[3])
        x = self.conv(x)
        x = x.repeat(1, 1, w, h)
        x = self.conv1(x)
        return x