# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:23:28 2021

@author: I'am the best
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

## 使用手写字体数据
## 准备训练数据集
train_data  = torchvision.datasets.FashionMNIST(
    root = "./data", # 数据的路径
    train = True, # 只使用训练数据集
    transform  = torchvision.transforms.ToTensor(),
    download= True # 下载过后就不用下载了
)
## 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=128, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
)

##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break

## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)


## 准备需要使用的测试数据集
test_data  = torchvision.datasets.FashionMNIST(
    root = "./data", # 数据的路径
    train = False, # 不使用训练数据集
    download= True # 下载过后就不用下载了
)
## 为数据添加一个通道纬度,并且取值范围缩放到0～1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x,dim = 1)
test_data_y = test_data.targets  ## 测试集的标签

print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)

## 搭建一个卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        ## 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,## 输入的feature map
                out_channels = 16, ## 输出的feature map
                kernel_size = 3, ##卷积核尺寸
                stride=1,   ##卷积核步长
                padding=1, # 进行填充
            ), 
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(
                kernel_size = 2,## 平均值池化层,使用 2*2
                stride=2,   ## 池化步长为2 
            ), 
        )
        ## 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1), 
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2,2) ## 最大值池化
        )
        ## 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(
                in_features = 32*7*7, ## 输入特征
                out_features = 128, ## 输出特证数
            ),
            nn.ReLU(),  # 激活函数
            nn.Linear(128,64),
            nn.ReLU()  # 激活函数
        )
        self.out = nn.Linear(64,10) ## 最后的分类层


    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        x = self.fc(x)
        output = self.out(x)
        return output
    
## 输出我们的网络结构
MyConvnet = ConvNet()
print(MyConvnet)

'''1-HiddenLayer库可视化网络'''
import hiddenlayer as hl
## 可视化卷积神经网络
hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
hl_graph.theme = hl.graph.THEMES["blue"].copy()  
## 将可视化的网路保存为图片,默认格式为pdf
hl_graph.save("pic/MyConvnet_hl.pdf")

'''2-torchviz可视化网络'''
from torchviz import make_dot
## 使用make_dot可视化网络
x = torch.randn(1, 1, 28, 28).requires_grad_(True)
y = MyConvnet(x)
MyConvnetvis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
## 将mlpvis保存为图片
## 指定文件保存位置
MyConvnetvis.directory = "pic/MyConvnet_vis"
MyConvnetvis.view() ## 会自动在当前文件夹生成文件