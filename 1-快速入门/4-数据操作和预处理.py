# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:19:28 2021

@author: I'am the best
"""

'''1-高维数组'''
#回归数据
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.datasets import load_boston,load_iris

boston_X,boston_Y = load_boston(return_X_y=True)
print("boston_X.dtype:",boston_X.dtype)
print("boston_Y.dtype:",boston_Y.dtype)

## 训练集X转化为张量,训练集y转化为张量
train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_Y.astype(np.float32))
print("train_xt.dtype:",train_xt.dtype)
print("train_xt.dtype:",train_yt.dtype)

## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt,train_yt)
## 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 0, # 使用两个进程 
)

##  检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break
## 输出训练图像的尺寸和标签的尺寸，和数据类型
print("b_x.shape:",b_x.shape)
print("b_y.shape:",b_y.shape)
print("b_x.dtype:",b_x.dtype)
print("b_y.dtype:",b_y.dtype)
#################分割线######################
#分类数据
iris_X,iris_Y = load_iris(return_X_y=True)
print("iris_x.dtype:",iris_X.dtype)
print("irisy:",iris_Y.dtype)

## 训练集X转化为张量,训练集y转化为张量
train_xt = torch.from_numpy(iris_X.astype(np.float32))
train_yt = torch.from_numpy(iris_Y.astype(np.int64))
print("train_xt.dtype:",train_xt.dtype)
print("train_xt.dtype:",train_yt.dtype)

## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt,train_yt)
## 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=10, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 1, # 使用两个进程 
)

##  检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break
## 输出训练图像的尺寸和标签的尺寸，和数据类型
print("b_x.shape:",b_x.shape)
print("b_y.shape:",b_y.shape)
print("b_x.dtype:",b_x.dtype)
print("b_y.dtype:",b_y.dtype)

'''2-图像数据'''
import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

## 使用FashionMNIST数据，准备训练数据集
train_data  = FashionMNIST(
    root = "./data", # 数据的路径
    train = True, # 只使用训练数据集
    transform  = transforms.ToTensor(),
    download= True  #数据已经下载过的话，就不会再下载了
)
## 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 2, # 使用两个进程 
)
## 计算train_loader有多少个batch
print("train_loader的batch数量为:",len(train_loader))

## 对测试集进行处理
test_data  = FashionMNIST(
    root = "./data", # 数据的路径
    train = False, # 不使用训练数据集
    download= True # 因为数据已经下载过，所以这里不再下载
)
## 为数据添加一个通道纬度,并且取值范围缩放到0～1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x,dim = 1)
test_data_y = test_data.targets  ## 测试集的标签
print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)
