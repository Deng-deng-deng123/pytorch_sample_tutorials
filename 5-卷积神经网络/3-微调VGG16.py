# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:28:38 2021

@author: I'am the best
"""

## 导入本章所需要的模块
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
# 设置随机数种子
setup_seed(2021)

## 导入预训练好的VGG16网络
vgg16 = models.vgg16(pretrained=True)
print(vgg16)

## 获取vgg16的特征提取层
vgg = vgg16.features
# 将vgg16的特征提取层参数冻结，不对其进行更新
for param in vgg.parameters():
    param.requires_grad_(False)
    
## 使用VGG16的特征提取层＋新的全连接层组成新的网络
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel,self).__init__()
        ## 预训练的vgg16的特征提取层
        self.vgg = vgg
        ## 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,10),
            nn.Softmax(dim=1)
        )
        

    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
    
## 输出我们的网络结构
Myvggc = MyVggModel()
device = torch.device('cuda')
Myvggc.to(device)
print(Myvggc)

## 使用10类猴子的数据集
## 对训练集的预处理
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),# 随机长宽比裁剪为224*224
    transforms.RandomHorizontalFlip(),# 依概率p=0.5水平翻转
    transforms.ToTensor(), # 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
## 对验证集的预处理  
val_data_transforms = transforms.Compose([
    transforms.Resize(256), # 重置图像分辨率
    transforms.CenterCrop(224),#依据给定的size从中心裁剪 
    transforms.ToTensor(),# 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## 读取图像
train_data_dir = "./data/10kindofmonkey/training"
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
train_data_loader = Data.DataLoader(train_data,batch_size=32,
                                    shuffle=True)
## 读取验证集
val_data_dir = "./data/10kindofmonkey/validation"
val_data = ImageFolder(val_data_dir, transform=val_data_transforms)
val_data_loader = Data.DataLoader(val_data,batch_size=32,
                                  shuffle=True)


print("训练集样本数:",len(train_data.targets))
print("验证集样本数:",len(val_data.targets))


##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_data_loader):  
    if step > 0:
        break

## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)

## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)

print(len(val_data.targets))
print(len(train_data.targets))
print(len(val_data_loader))
print(len(train_data_loader))

## 可视化训练集其中一个batch的图像
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
plt.figure(figsize=(12,6))
for ii in np.arange(len(b_y)):
    plt.subplot(4,8,ii+1)
    image = b_x[ii,:,:,:].numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(b_y[ii].data.numpy())
    plt.axis("off")
plt.subplots_adjust(hspace = 0.3)

##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(val_data_loader):  
    if step > 0:
        break
    
## 可视化验证集其中一个batch的图像
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
plt.figure(figsize=(12,6))
for ii in np.arange(len(b_y)):
    plt.subplot(4,8,ii+1)
    image = b_x[ii,:,:,:].numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(b_y[ii].data.numpy())
    plt.axis("off")
plt.subplots_adjust(hspace = 0.3)

# 定义优化器
optimizer = torch.optim.Adam(Myvggc.parameters(), lr=0.0003)  
loss_func = nn.CrossEntropyLoss()   # 损失函数
train_loss_all = []
train_acc_all = []
val_loss_all = []
val_acc_all = []
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
since = time.time()
for epoch in range(10):
    print('start!')
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects =0
    val_corrects = 0
    ## 对训练数据的迭代器进行迭代计算
    Myvggc.train()
    for step, (b_x, b_y) in enumerate(train_data_loader): 
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        ## 计算每个batch的
        output = Myvggc(b_x)            # CNN在训练batch上的输出
        loss = loss_func(output, b_y)   # 交叉熵损失函数
        pre_lab = torch.argmax(output,1)
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        loss.backward()                 # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss_epoch += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab == b_y.data)
    ## 计算一个epoch的损失和精度
    train_loss = train_loss_epoch / len(train_data.targets)
    train_acc = train_corrects.double() / len(train_data.targets)
    
    ## 计算在验证集上的表现
    Myvggc.eval()
    for step, (val_x, val_y) in enumerate(val_data_loader):  
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        output = Myvggc(val_x)
        loss = loss_func(output, val_y)
        pre_lab = torch.argmax(output,1)
        val_loss_epoch += loss.item() * val_x.size(0)
        val_corrects += torch.sum(pre_lab == val_y.data)
    ## 计算一个epoch的损失和精度
    val_loss = val_loss_epoch / len(val_data.targets)
    val_acc = val_corrects.double() / len(val_data.targets)
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc)
    val_loss_all.append(val_loss)
    val_acc_all.append(val_acc)
    print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
        epoch, train_loss_all[-1], train_acc_all[-1]))
    print('{} Val Loss: {:.4f}  val Acc: {:.4f}'.format(
        epoch, val_loss_all[-1], val_acc_all[-1]))
    time_use = time.time() - since
    print("Train and val complete in {:.0f}m {:.0f}s".format(
        time_use // 60, time_use % 60))
    
torch.save(Myvggc,'./model/Myvggc')