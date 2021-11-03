import paddle
import time
import datasets
from lib.utils import AverageMeter
import paddle.vision.transforms as transforms
import numpy as np


def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.0
    total = 0
    testsize = testloader.dataset.__len__()
    trainFeatures = lemniscate.memory.t()
    """if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()"""
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = paddle.to_tensor([y for (p, y) in trainloader.dataset.imgs], dtype=paddle.int64)
    else:
        trainLabels = paddle.to_tensor(trainloader.dataset.train_labels, dtype=paddle.int64)
    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = paddle.io.DataLoader(trainloader.dataset,
            batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize +
                batchSize] = features.data.t()
        """trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()"""
        trainLabels = paddle.to_tensor(temploader.dataset.train_labels, dtype=paddle.int64)
        trainloader.dataset.transform = transform_bak
    end = time.time()
    with paddle.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()
            """dist = torch.mm(features, trainFeatures)"""
            dist = paddle.mm(features, trainFeatures)
            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            """retrieval = torch.gather(candidates, 1, yi)"""
            retrieval = paddle.gather(candidates, 1, yi)
            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)
            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            cls_time.update(time.time() - end)
            end = time.time()
            print(
                'Test [{}/{}]\tNet Time {net_time.val:.3f} ({net_time.avg:.3f})\tCls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\tTop1: {:.2f}'
                .format(total, testsize, correct * 100.0 / total, net_time=\
                net_time, cls_time=cls_time))
    return correct / total


def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma,
    recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()
    trainFeatures = lemniscate.memory.t()
    """if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()"""
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = paddle.to_tensor([y for (p, y) in trainloader.dataset.imgs], dtype=paddle.int64)
    else:
        trainLabels = paddle.to_tensor(trainloader.dataset.train_labels, dtype=paddle.int64)
    C = trainLabels.max() + 1
    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = paddle.io.DataLoader(trainloader.dataset,
            batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize +
                batchSize] = features.data.t()
        """trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()"""
        trainloader.dataset.transform = transform_bak
    top1 = 0.0
    top5 = 0.0
    end = time.time()
    with paddle.no_grad():
        retrieval_one_hot = paddle.to_tensor(paddle.zeros([K, C]), stop_gradient=True)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()
            """dist = torch.mm(features, trainFeatures)"""
            dist = paddle.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            """retrieval = torch.gather(candidates, 1, yi)"""
            retrieval = paddle.gather(candidates, 1, yi)
            retrieval_one_hot = paddle.reshape(retrieval_one_hot, shape=[batchSize * K, C]).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = paddle.exp(paddle.divide(yd.clone(), sigma))
            probs = paddle.sum(paddle.multiply(retrieval_one_hot.view
                (batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
            print(
                'Test [{}/{}]\tNet Time {net_time.val:.3f} ({net_time.avg:.3f})\tCls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\tTop1: {:.2f}  Top5: {:.2f}'
                .format(total, testsize, top1 * 100.0 / total, top5 * 100.0 /
                total, net_time=net_time, cls_time=cls_time))
    print(top1 * 100.0 / total)
    return top1 / total
