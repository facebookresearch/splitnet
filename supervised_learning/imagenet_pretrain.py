#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

# Taken and slightly adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import random
import shutil
import time
import warnings
from collections import deque

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from networks.networks import ImagenetModel
from utils import python_util
from utils import pytorch_util as pt_util
from utils import tensorboard_logger

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
)
parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr"
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument(
    "--dist-url", default="tcp://224.66.41.62:23456", type=str, help="url used to set up distributed training"
)
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument("--pytorch-gpu-ids", type=str, required=True, help="ID(s) of the gpu(s) to use for pytorch")
parser.add_argument(
    "--clear-weights", action="store_true", default=False, help="do not load previous model weights if they exist"
)
parser.add_argument("--no-tensorboard", action="store_true", default=False, help="disable tensorboard logging")
parser.add_argument("--log-prefix", type=str, default="", required=True, help="path to logs, checkpoints, etc")
parser.add_argument("--no-save-checkpoints", action="store_true", default=False, help="disable saving checkpoints")
parser.add_argument(
    "--tensorboard-dirname", type=str, default="tensorboard", help="path under log-prefix for tensorboard logs."
)

parser.add_argument(
    "--checkpoint-dirname", type=str, default="checkpoints", help="path under log-prefix for checkpoints."
)

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.tensorboard = not args.no_tensorboard
    args.load_model = not args.clear_weights
    args.save_checkpoints = not args.no_save_checkpoints

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    log_prefix = args.log_prefix
    time_str = python_util.get_time_str()
    checkpoint_dir = os.path.join(log_prefix, args.checkpoint_dirname, time_str)

    torch_devices = [int(gpu_id.strip()) for gpu_id in args.pytorch_gpu_ids.split(",")]
    args.gpu = torch_devices[0]
    device = "cuda:" + str(torch_devices[0])

    model = ImagenetModel()
    model = pt_util.get_data_parallel(model, torch_devices)
    model.to(device)

    start_iter = 0
    if args.load_model:
        start_iter = pt_util.restore_from_folder(model, os.path.join(log_prefix, args.checkpoint_dirname, "*"))
    args.start_epoch = start_iter

    train_logger = None
    test_logger = None
    if args.tensorboard:
        train_logger = tensorboard_logger.Logger(
            os.path.join(log_prefix, args.tensorboard_dirname, time_str + "_train")
        )
        test_logger = tensorboard_logger.Logger(os.path.join(log_prefix, args.tensorboard_dirname, time_str + "_test"))

    main_worker(model, args.gpu, args, train_logger, test_logger, checkpoint_dir)


def main_worker(model, gpu, args, train_logger, test_logger, checkpoint_dir):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    pt_util.save(model, checkpoint_dir, num_to_keep=5, iteration=0)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255),
            ]
        ),
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * 255),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, train_logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, test_logger, (epoch + 1) * len(train_loader.dataset))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best and args.save_checkpoints:
            pt_util.save(model, checkpoint_dir, num_to_keep=5, iteration=(epoch + 1) * len(train_loader.dataset))


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = RollingAverageMeter()
    data_time = RollingAverageMeter()
    losses = RollingAverageMeter()
    top1 = RollingAverageMeter()
    top5 = RollingAverageMeter()

    # switch to train mode
    model.train()

    logger.network_conv_summary(model, epoch * len(train_loader.dataset))
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0])
        top5.update(acc5[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
            if logger is not None:
                logger.dict_log(
                    dict(
                        batch_time=batch_time.avg,
                        data_time=data_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ),
                    epoch * len(train_loader.dataset) + i * len(target),
                )


def validate(val_loader, model, criterion, args, logger, iteration):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5
                    )
                )

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    if logger is not None:
        logger.dict_log(dict(batch_time=batch_time.avg, loss=losses.avg, top1=top1.avg, top5=top5.avg), iteration)
    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RollingAverageMeter(object):
    def __init__(self, window_size=10):
        super(RollingAverageMeter, self).__init__()
        self.deque = deque(maxlen=window_size)

    def reset(self):
        self.deque = deque(maxlen=self.deque.maxlen)

    def update(self, val):
        self.deque.append(val)
        self.val = val
        self.avg = sum(self.deque) / len(self.deque)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
