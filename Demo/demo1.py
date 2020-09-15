#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 21:32
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : demo1.py
# @Function: 
# @Software: PyCharm

from tensorboardX import SummaryWriter
import cv2

if __name__ == '__main__':
    # writer = SummaryWriter('../runs/scalar_example_09')
    writer = SummaryWriter('../runs/scalar_example2_0914')
    for i in range(10):
        writer.add_scalar('my_value_1', i + 1, global_step=i)
        writer.add_scalar('my_value_2', i * 10, global_step=i)
