# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:50:40 2021

@author: I'am the best
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import seaborn as sns

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
# 设置随机数种子
setup_seed(2021)

## 使用FashionMNIST数据
## 准备训练数据集
train_data  = FashionMNIST(
    root = "./data", # 数据的路径
    train = True, # 只使用训练数据集
    transform  = transforms.ToTensor(),
    download= True
)
## 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = False, # 每次迭代前不乱数据 
)
## 计算train_loader有多少个batch
print("train_loader的batch数量为:",len(train_loader))
##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break
    


## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)

## 准备需要使用的测试数据集
test_data  = FashionMNIST(
    root = "./data", # 数据的路径
    train = False, # 不使用训练数据集
    transform  = transforms.ToTensor(),
    download= False # 因为数据已经下载过，所以这里不再下载
)
## 定义一个数据加载器
test_loader = Data.DataLoader(
    dataset = test_data, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    # num_workers = 2, # 使用两个进程 
)
##  可视化训练数据集的一个batch的样本来查看图像内容
for step, (b_x, b_y) in enumerate(test_loader):  
    if step > 0:
        break

## 输出训练图像的尺寸和标签的尺寸，都是torch格式的数据
print(b_x.shape)
print(b_y.shape)

class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim:输入数据的维度(图片每行的数据像素点)
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(RNNimc, self).__init__()
        self.hidden_dim = hidden_dim ## RNN神经元个数
        self.layer_dim = layer_dim ## RNN的层数
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')
        
        # 连接全连阶层
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x:[batch, time_step, input_dim]
        # 本例中time_step＝图像所有像素数量／input_dim
        # out:[batch, time_step, output_size]
        # h_n:[layer_dim, batch, hidden_dim]
        out, h_n = self.rnn(x, None) # None表示h0会使用全0进行初始化
        # 选取最后一个时间点的out输出
        out = self.fc1(out[:, -1, :]) 
        return out
    
## 模型的调用
input_dim=28   # 图片每行的像素数量
hidden_dim=128  # RNN神经元个数
layer_dim = 1  # RNN的层数
output_dim=10  # 隐藏层输出的维度(10类图像)
MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
print(MyRNNimc)

## 对模型进行训练
optimizer = torch.optim.RMSprop(MyRNNimc.parameters(), lr=0.0003)  
criterion = nn.CrossEntropyLoss()   # 损失函数
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epochs = 30
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    MyRNNimc.train() ## 设置模型为训练模式
    corrects = 0
    train_num  = 0
    for step,(b_x, b_y) in enumerate(train_loader):
        # input :[batch, time_step, input_dim]
        xdata = b_x.view(-1, 28, 28)
        output = MyRNNimc(xdata)     
        pre_lab = torch.argmax(output,1)
        loss = criterion(output, b_y) 
        optimizer.zero_grad()        
        loss.backward()       
        optimizer.step()  
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        train_num += b_x.size(0)
    ## 计算经过一个epoch的训练后在训练集上的损失和精度
    train_loss_all.append(loss.item() / train_num)
    train_acc_all.append(corrects.double().item()/train_num)
    print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
        epoch, train_loss_all[-1], train_acc_all[-1]))
    ## 设置模型为验证模式
    MyRNNimc.eval()
    corrects = 0
    test_num  = 0
    for step,(b_x, b_y) in enumerate(test_loader):
        # input :[batch, time_step, input_dim]
        xdata = b_x.view(-1, 28, 28)
        output = MyRNNimc(xdata)     
        pre_lab = torch.argmax(output,1)
        loss = criterion(output, b_y) 
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        test_num += b_x.size(0)
    ## 计算经过一个epoch的训练后在测试集上的损失和精度
    test_loss_all.append(loss.item() / test_num)
    test_acc_all.append(corrects.double().item()/test_num)
    print('{} Test Loss: {:.4f}  Test Acc: {:.4f}'.format(
        epoch, test_loss_all[-1], test_acc_all[-1]))

## 可视化模型训练过程中
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(train_loss_all,"ro-",label = "Train loss")
plt.plot(test_loss_all,"bs-",label = "Test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(train_acc_all,"ro-",label = "Train acc")
plt.plot(test_acc_all,"bs-",label = "Test acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()