# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:57:32 2021

@author: I'am the best
"""

import torch
#定义张量为2x2的矩阵
x = torch.tensor(torch.arange(4.0).reshape(2,2),requires_grad=True)
#定义前向运算
y = torch.sum(x**2 + x * 2 + 1)
#反向传播
y.backward()
#查看梯度
print(x.grad)