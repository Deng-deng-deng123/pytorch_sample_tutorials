# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:07:02 2021

@author: I'am the best
"""

import torch
import torch.nn as nn

'''1-卷积层'''
#官网例子：https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)
################################分割线#################################

#对图像进行一次卷积
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#读取图像->转换为灰度图->转换为numpy数组
image = Image.open('pic/leaf.jpg')
image_gray = np.array(image.convert('L'),dtype = np.float32)
#可视化图片
plt.figure(figsize=(10,10))
plt.imshow(image_gray,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
#将数组转换为合适的张量
imh,imw = image_gray.shape
timage = torch.from_numpy(image_gray.reshape(1,1,imh,imw))
print(timage.shape)
#定义边缘检测卷积核
kersize = 5
ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
ker[2,2] = 24
ker = ker.reshape((1,1,kersize,kersize))
#定义卷积操作
conv2d = nn.Conv2d(1,2,(kersize,kersize),bias = False)
conv2d.weight.data[0] = ker
#卷积运算
output = conv2d(timage)
print(output.shape)
output = output.squeeze()
plt.figure(figsize=(18,24))
plt.subplot(3,1,1)
plt.imshow(image_gray,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(3,1,2)
plt.imshow(output[0].detach().numpy(),cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(3,1,3)
plt.imshow(output[1].detach().numpy(),cmap=plt.cm.gray)
plt.axis('off')
plt.show()

'''2-池化层'''
#官网例子：https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html?highlight=maxpool2d
maxpool = nn.MaxPool2d(2,stride=2)
pool_out = maxpool(output) #可以通过在这换不同的池化层
plt.figure(figsize=(18,16))
plt.subplot(2,2,1)
plt.imshow(output[0].data,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(output[1].data,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(pool_out[0].data,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(pool_out[1].data,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(output.shape)
print(pool_out.shape)

'''3-激活函数'''
x = torch.linspace(-6,6,100)

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()
softplus = nn.Softplus()

ysigmoid = sigmoid(x)
ytanh = tanh(x)
yrelu = relu(x)
ysoftplus = softplus(x)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(x.numpy(),ysigmoid.numpy())
plt.title('Sigmoid')
plt.grid()
plt.subplot(2,2,2)
plt.plot(x.numpy(),ytanh.numpy())
plt.title('Tanh')
plt.grid()
plt.subplot(2,2,3)
plt.plot(x.numpy(),yrelu.numpy())
plt.title('Relu')
plt.grid()
plt.subplot(2,2,4)
plt.plot(x.numpy(),ysoftplus.numpy())
plt.title('Softplus')
plt.grid()

'''4-循环层'''
#官网例子：https://pytorch.org/docs/master/generated/torch.nn.RNN.html#rnn
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)

'''5-全连接'''
#官网例子：https://pytorch.org/docs/master/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())

