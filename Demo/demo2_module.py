#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 8:56
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : demo2_module.py
# @Function: 参考：https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py
# @Software: PyCharm


import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, in_feature):
        super(Discriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )

    def forward(self, x):
        for module in self.main:
            x = module(x)
        return x


if __name__ == '__main__':
    discriminator = Discriminator(in_feature=500)
    d_input = torch.zeros(3, 500)
    writer = SummaryWriter('../runs/module_example_0915')

    writer.add_graph(discriminator, d_input, True)

