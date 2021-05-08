# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:10:06 2021

@author: I'am the best
"""

'''

for input,target in dataset :
    optimizer.zero_grad() #梯度清零
    output = model(input) #模型预测预测
    loss = loss_fn(output,target) #计算损失
    loss.backward() #反向传播计算参数
    optimizer.step() #更新网络参数

'''