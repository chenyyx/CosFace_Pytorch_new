#!usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import argparse
import os
import time

# torch 包
import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms
# 使用 gpu cuda 编程（mac 上不能使用 gpu，所以这个引入就无用了）
# import torch.backends.cudnn as cudnn
# 这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 分为 2 种情况：1.如果网络的输入数据维度或类型变化不大，可以增加运行效率；
# 2。如果网络的输入数据在每次 iteration 都变化的话，会导致 cuDNN 每次都去寻找一遍最优配置，这样反而降低运行效率。
# cudnn.benchmark = True

# 引用 sephere 网络结构
from net import sphere20
from dataset import ImageList
# 引用 lfw_eval
import n_eval
import layer

'''
Desc:
    此文件 完成 train 过程。使用数据是 casia-webface 数据
Date：
    2018-08-22 13:36
'''



# 训练 参数 设置
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# 数据 root 目录
parser.add_argument('--root_path', default='/Users/chenyao/Documents/dataset/CASIA-WebFace/CASIA-WebFace-112X96-pre/', type=str, help='path to root path of images')
# 训练数据集 list
parser.add_argument('--train_list', default='/Users/chenyao/Documents/dataset/CASIA-WebFace/CASIA-WebFace-112X96-pre/CASIA-WebFace-112X96-pre.txt', type=str, help='path to training list')
# batch-size default=512
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
# 模型 class 的数目-10572
parser.add_argument('--num_class', default=3, type=int, help='number of people(class) (default: 10572)')
# LR policy
# 训练的 epoch
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 20)')
# learning rate - default=0.1
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
# lr 的衰变步长
# parser.add_argument('--step_size', type=list, default=[16000, 24000], metavar='SS', help='lr decay step (default: [10,15,18])')  # [15000, 22000, 26000]
parser.add_argument('--step_size', type=list, default=[16, 24], metavar='SS', help='lr decay step (default: [10,15,18])')  # [15000, 22000, 26000]
# SGD 的动量
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
# weight 的衰变
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W', help='weight decay (default: 0.0005)')
# 在记录状态之前要等待多少 batches
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
# 保存 checkpoint 的路径
parser.add_argument('--save_path', default='/Users/chenyao/Documents/checkpoint/pre/', type=str, metavar='PATH', help='path to save checkpoint')
# 是否不使用 cuda（不使用 cuda 编程，那我这里就设置为 default=True）
# parser.add_argument('--no-cuda', type=bool, default=False, help='disables CUDA training')
parser.add_argument('--no-cuda', type=bool, default=True, help='disables CUDA training')
# worker 的数目代表 用于数据加载的子进程数。0 表示数据将加载到主进程中。（默认值：0）
parser.add_argument('--workers', type=int, default=4, metavar='N', help='how many workers to load data')

args = parser.parse_args()
# 是否使用 cuda 编程，其中 cuda.is_available() 判断是否支持 gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    # --------------------------------------model----------------------------------------
    # 调用 net.py 文件中的 sphere20() 网络
    model = sphere20()
    # DataParallel 的作用是让数据在多个 gpu 上运行
    # 改为 cpu 版本（因 mac 不支持gpu运行）
    # model = torch.nn.DataParallel(model).cuda()
    print(model)
    # 检测保存路径是否已存在，保存 checkpoint
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model.save(args.save_path + 'CosFace_0_checkpoint.pth')
    print('save checkpoint finished！')

    # ------------------------------------load image---------------------------------------
    # 加载训练数据集
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list,
                  # 进行图像预处理
                  transform=transforms.Compose([
                      # 以 0.5 的概率水平翻转给定的 PIL 图像
                      transforms.RandomHorizontalFlip(),
                      # 将一个 PIL 图像（H*W*C）在 【0，255】范围内转化为 torch.Tensor(C*H*W) 在 【0.0，1.0】范围内
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      # 使用 均值 mean 和标准差 standard deviation 来标准化数据 
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                  ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # 打印 train dataset 的 length
    print('length of train Dataset: ' + str(len(train_loader.dataset)))
    # 打印 train dataset 的类别数目
    print('Number of Classses: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    # 实现 cos face 的核心部分，但是使用了 cuda
    # MCP = layer.MarginCosineProduct(512, args.num_class).cuda()
    MCP = layer.MarginCosineProduct(512, args.num_class)
    # MCP = layer.AngleLinear(512, args.num_class).cuda()
    # MCP = torch.nn.Linear(512, args.num_class, bias=False).cuda()
    
    # 修改为不用 cuda
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # 使用（被优化的） SGD 优化器
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': MCP.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ----------------------------------------train----------------------------------------
    # lfw_eval.eval(args.save_path + 'CosFace_0_checkpoint.pth')
    # 开始训练
    # 训练 epoch 次，每完整训练一次，存储一次 checkpoint
    for epoch in range(1, args.epochs + 1):
        # scheduler.step()
        train(train_loader, model, MCP, criterion, optimizer, epoch)
        # model.module.save(args.save_path + 'CosFace_' + str(epoch) + '_checkpoint.pth')
        model.save(args.save_path + 'CosFace_' + str(epoch) + '_checkpoint.pth')
        n_eval.eval(args.save_path + 'CosFace_' + str(epoch) + '_checkpoint.pth')
    print('Finished Training')

# 训练函数
def train(train_loader, model, MCP, criterion, optimizer, epoch):
    # 训练
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    # 获取当前时间
    time_curr = time.time()
    # 展示出来的 loss
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        # 迭代次数
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        # 调整 lr
        adjust_learning_rate(optimizer, iteration, args.step_size)
        # 判断是否支持 gpu 运行，是——转为cuda版本，否——不处理
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # compute output
        output = model(data)
        output = MCP(output, target)
        loss = criterion(output, target)
        loss_display += loss.data[0]
        # compute gradient and do SGD step
        # 计算 梯度 并做 SGD
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新网络参数
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(MCP.m, MCP.s)
            # INFO = ' lambda: {:.4f}'.format(MCP.lamb)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0

# 带有打印时间的 print
def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)

# 调整 lr 的形式
def adjust_learning_rate(optimizer, iteration, step_size):
    """
    设置 lr 衰减，每一个 step size 就衰减为原来的 10%，也就是之前的 1/10
    Sets the learning rate to the initial LR decayed by 10 each step size
    """
    if iteration in step_size:
        # 调整 lr 的 具体形式
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        # optimizer 通过 param_group 来管理参数组。param_group 中保存了参数组及其对应的学习率，动量等
        # 因此，我们可以通过更改 param_group['lr'] 的值来更改对应参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


if __name__ == '__main__':
    main()
