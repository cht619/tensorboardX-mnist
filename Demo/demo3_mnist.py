#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 9:19
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : demo3_complete.py
# @Function: 
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.optim as optim
from torch.autograd.variable import *

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
feature_dim = 1*28*28
num_classes = 10
img_size = 28


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, out_dim),
            # nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def get_mnist(path, batch_size=100, train=True):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                         [0.5], [0.5])])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=path,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True)

    return mnist_data_loader


def evaluate(classifier, dataloader):
    classifier.eval()
    acc = 0
    for (imgs, labels) in dataloader:
        imgs = Variable(imgs.type(FloatTensor)).reshape(imgs.shape[0], -1)
        labels = Variable(labels.type(LongTensor))
        predict = classifier(imgs).data.max(1)[1]
        acc += predict.eq(labels).sum().item()
    acc_all = int(acc) / len(dataloader.dataset)
    return acc_all


def train(classifier, train_epochs, path, writer):
    classifier.train()
    mnist_dataloader = get_mnist(path)
    mnist_eval_dataloader = get_mnist(path, train=False)
    optimizer = optim.Adam(classifier.parameters())
    loss = nn.CrossEntropyLoss()
    for epoch in range(train_epochs):
        for step, (imgs, labels) in enumerate(mnist_dataloader):
            imgs = Variable(imgs.type(FloatTensor)).reshape(imgs.shape[0], -1)
            labels = Variable(labels.type(LongTensor))

            optimizer.zero_grad()
            loss_ = loss(classifier(imgs), labels)
            loss_.backward()
            optimizer.step()

            if step % 200 == 0:
                acc = evaluate(classifier, mnist_eval_dataloader)
                writer.add_scalar('batch_loss', loss_.item(), global_step=epoch)
                writer.add_scalar('Average', acc, global_step=epoch)


if __name__ == '__main__':
    from torch.backends import cudnn

    torch.backends.cudnn.benchmark = True

    classifier = Classifier(in_dim=feature_dim, out_dim=num_classes)
    c_input = torch.zeros(3, feature_dim)
    with SummaryWriter('../runs/classifier_mnist_0915') as writer:
        writer.add_graph(classifier, c_input, True)
        classifier = Classifier(in_dim=feature_dim, out_dim=num_classes).cuda()
        train(classifier, 10, '../mnist', writer)

