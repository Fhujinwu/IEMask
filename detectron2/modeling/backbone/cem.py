# -*- coding: utf-8 -*-
# @Author  : HUJINWU
# @Time    : 2021-05-22 17:01
# @Function: Channel Enchance Module
import torch
from torch import nn
from torch.nn.parameter import Parameter
class CEMayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CEMayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = x * y
        return y