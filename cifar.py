"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddle.vision
import paddle.vision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import datasets
import math

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN
from test import kNN
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--low-dim', default=128, type=int, metavar='D',
                        help='feature dimension')
    parser.add_argument('--nce-k', default=4096, type=int, metavar='K',
                        help='number of negative samples for NCE')
    parser.add_argument('--nce-t', default=0.1, type=float, metavar='T',
                        help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float, metavar='M',
                        help='momentum for non-parametric updates')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    paddle.device.set_device(device)

    best_acc = 0
    start_epoch = 0

    print('==> Preparing data..')

    if np.random.random(1) < 0.2:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    trainset = datasets.CIFAR10Instance(mode='train', download=True, transform=transform_train)
    trainloader = paddle.io.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10Instance(mode='test', download=True, transform=transform_test)
    testloader = paddle.io.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ndata = trainset.__len__()

    print('==> Building model..')
    # net = models.__dict__['ResNet18'](low_dim=args.low_dim)
    net = paddle.vision.models.resnet18()

    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    if device == 'gpu':
        net = paddle.DataParallel(net)

    if args.test_only or len(args.resume) > 0:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        checkpoint = paddle.load('./checkpoint/' + args.resume)
        net.load_state_dict(checkpoint['net'])
        lemniscate = checkpoint['lemniscate']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if hasattr(lemniscate, 'K'):
        criterion = NCECriterion(ndata)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.test_only:
        acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
        sys.exit(0)

    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=args.lr, momentum=0.9,
                                          weight_decay=0.0005)

    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr
        if epoch >= 80:
            lr = args.lr * 0.1 ** ((epoch - 80) // 40)
        print(lr)
        optimizer.set_lr(lr)
        """for param_group in optimizer._param_groups:
            param_group['lr'] = lr"""

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        adjust_learning_rate(optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        net.train()

        end = time.time()

        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            data_time.update(time.time() - end)

            optimizer.clear_grad()

            features = net(inputs)
            outputs = lemniscate(features, indexes)
            loss = criterion(outputs, indexes)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        acc = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            paddle.save(state, './checkpoint/ckpt.pdparams')
            best_acc = acc

        print('best accuracy: {:.2f}'.format(best_acc * 100))

    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    print('last accuracy: {:.2f}'.format(acc * 100))


if __name__ == '__main__':
    main()
