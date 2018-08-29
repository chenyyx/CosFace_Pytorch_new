#!usr/bin/python
# -*- coding: utf-8 -*-

import torch.utils.data as data
from PIL import Image, ImageFile
import os
# 添加本行，否则出现 IOError: image file is truncated
ImageFile.LOAD_TRUNCATED_IAMGES = True

'''
Desc:
    此文件完成 加载数据 过程。
Date：
    2018-08-23 16:21
Author：
    yao.chen@nttdata.com
'''

# 加载指定 path 下的数据
def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img

# 加载 fileList 并返回 imgList
def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


class ImageList(data.Dataset):
    '''
    Desc：
        重写 ImageList 中的 init ,getitem, len 方法
    Args:
        root (string): Root directory path.
        fileList (string): Image list file path 
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    '''

    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
