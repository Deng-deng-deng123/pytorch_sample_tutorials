# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:01:40 2021

@author: I'am the best
"""

'''1-tensorboardx可视化'''
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

## 从tensorboardX库中导入需要的API
from tensorboardX import SummaryWriter
SumWriter = SummaryWriter(log_dir="data/log")

# 定义优化器
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)  
loss_func = nn.CrossEntropyLoss()   # 损失函数
train_loss = 0
print_step = 100 ## 每经过100次迭代后,输出损失
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(10):
    ## 对训练数据的迭代器进行迭代计算
    print('epoch:',epoch)
    for step, (b_x, b_y) in enumerate(train_loader):  
        ## 计算每个batch的
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        output = MyConvnet(b_x)            # CNN在训练batch上的输出
        loss = loss_func(output, b_y)   # 交叉熵损失函数
        loss.backward()                 # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss = train_loss+loss    # 计算损失的累加损失
        ## 计算迭代次数
        niter = epoch * len(train_loader) + step+1
        ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            ## 为日志添加训练集损失函数
            SumWriter.add_scalar("train loss",
                                 train_loss.item() / niter,
                                 global_step=niter)
            ## 计算在测试集上的精度
            output = MyConvnet(test_data_x)
            _,pre_lab = torch.max(output,1)
            acc = accuracy_score(test_data_y,pre_lab)
            ## 为日志添加在测试集上的预测精度
            SumWriter.add_scalar("test acc",acc.item(),niter)
            ## 为日志中添加训练数据的可视化图像，使用当前batch的图像
            ## 将一个batch的数据进行预处理
            b_x_im = vutils.make_grid(b_x,nrow=12)
            SumWriter.add_image('train image sample', b_x_im,niter)
            ## 使用直方图可视化网络中参数的分布情况
            for name, param in MyConvnet.named_parameters():
                SumWriter.add_histogram(name, param.data.numpy(),niter)

## 为日志中添加训练数据的可视化图像，使用最后一个batch的图像
## 将一个batch的数据进行预处理
b_x_im = vutils.make_grid(b_x,nrow=12)
SumWriter.add_image('train image sample', b_x_im)
## 使用直方图可视化网络中参数的分布情况
for name, param in MyConvnet.named_parameters():
    SumWriter.add_histogram(name, param.data.numpy())
    
'''
在窗口设置更是时间就可以实时查看数据了
'''