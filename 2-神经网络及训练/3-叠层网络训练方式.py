# -*- coding: utf-8 -*-
"""
Created on Sat May  8 20:47:34 2021

@author: I'am the best
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  8 20:47:00 2021

@author: I'am the best
"""

import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
# 设置随机数种子
setup_seed(2021)

## 读取数据
boston_X,boston_y = load_boston(return_X_y=True)
print("boston_X.shape:",boston_X.shape)

plt.figure()
plt.hist(boston_y,bins=20)
plt.show()

## 数据标准化处理
ss = StandardScaler(with_mean=True,with_std=True)
boston_Xs = ss.fit_transform(boston_X)
# boston_ys = ss.fit_transform(boston_y)
np.mean(boston_Xs,axis=0)
np.std(boston_Xs,axis=0)

## 将数据预处理为可以使用pytorch进行批量训练的形式
## 训练集X转化为张量
train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
## 训练集y转化为张量
train_yt = torch.from_numpy(boston_y.astype(np.float32))
## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt,train_yt)
## 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=128, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 0, # 使用两个进程 
)

##  检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break
## 输出训练图像的尺寸和标签的尺寸，都是torch格式的数据
print(b_x.shape)
print(b_y.shape)

## 使用定义网络时使用nn.Sequential的形式
class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2,self).__init__()
        ## 定义隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
        )
        ## 预测回归层
        self.regression = nn.Linear(10,1)

    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
    
## 输出我们的网络结构
mlp1 = MLPmodel2()
print(mlp1)

## 对回归模型mlp1进行训练并输出损失函数的变化情况
# 定义优化器和损失函数
optimizer = SGD(mlp1.parameters(),lr=0.001)  
loss_func = nn.MSELoss()  # 最小平方根误差
train_loss_all = [] ## 输出每个批次训练的损失函数
## 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        output = mlp1(b_x).flatten()      # MLP在训练batch上的输出
        train_loss = loss_func(output,b_y) # 平方根误差
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all,"r-")
plt.title("Train loss per iteration")
plt.show()