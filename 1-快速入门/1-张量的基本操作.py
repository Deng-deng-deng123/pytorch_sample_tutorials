# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:47:01 2021

@author: I'am the best
"""

import torch 

'''1-更改默认张量数据类型'''
print(torch.tensor([1.2,3.4]).dtype)
torch.set_default_tensor_type(torch.HalfTensor) #将浮点类型自动变成64位
print(torch.tensor([1.2,3.4]).dtype)
torch.set_default_tensor_type(torch.DoubleTensor) #将浮点类型自动变成64位
print(torch.tensor([1.2,3.4]).dtype)
#获取默认的数据类型
print(torch.get_default_dtype())


'''2-更改数据类型'''
a = torch.tensor([1.2,3.4])
print('a.dtype:',a.dtype)
print('a.long():',a.long().dtype)
print('a.int():',a.int().dtype)
print('a.float():',a.float().dtype)

'''3-张量基本信息'''
a = torch.tensor([[1.0,1.0],[2,2]])
print(a.shape) #查看形状
print(a.size()) #计算形状
print(a.numel()) #计算数量

'''4-梯度计算'''
b = torch.tensor([1,2,3],dtype=torch.float32,requires_grad=True)  
y = b.pow(2).sum()
y.backward()
print(b.grad)

#默认不能计算梯度
# b = torch.tensor([1,2,3],dtype=torch.float32)  
# y = b.pow(2).sum()
# y.backward()
# print(b.grad)
# #只有浮点类型可以计算梯度
# b = torch.tensor([1,2,3],dtype=torch.int32,requires_grad=True)  
# y = b.pow(2).sum()
# y.backward()
# print(b.grad)

'''5-生成简单张量'''
c = torch.Tensor(3,3,requires_grad=True)
#torch.xx_like() 形状相同，内容不同
torch.ones_like(c)
torch.zeros_like(c)
torch.rand_like(c)

#.new_xx() 形状不同，数据类型相同
d = [1,2,3]
d = c.new_tensor(d)
print(d.dtype)

#特定规律
 
'''6-numpy与Tensor之间相互转换'''
#从numpy生成张量
import numpy as np
e = np.ones([3,3])
print(e)
f = torch.as_tensor(e)
f = torch.from_numpy(e)
f = torch.tensor(e)
print(f)
#Tensor转换回numpy
g = f.numpy()
print(g)
'''7-生成随机张量'''
torch.manual_seed(2021)
#[0,1]均匀分布
h = torch.rand(3,3)
print(h)
h = torch.rand_like(i)
print(h)
#正太分布
i = torch.normal(mean = 0.0,std = torch.arange(1,5.0)) 
i = torch.normal(mean = torch.arange(1,5.0),std = torch.arange(1,5.0))
print(i)
#正态分布
j = torch.randn(2,2)
print(j)
j = torch.randn_like(j)
print(j)

'''8-改变形状'''
#改变形状
a = torch.arange(12).reshape(3,4) #直接赋值
a.reshape(2,-1) #更改视图
a.resize_(2,-1) #直接更改
torch.reshape(a,(2,-1)) #更改视图
b = torch.arange(10,19).reshape(3,3)
a.resize_as_(b) #直接更改

#拓展缩减1维度
a = torch.ones(3,3)
a = torch.unsqueeze(a,dim = 0)
print(a.shape)
a = a.unsqueeze(3)
print(a.shape)
a = a.squeeze(0)
a = a.squeeze()
print(a.shape)

#张量广播
a = torch.arange(3)
print(a.shape)
b = a.expand(3,-1)
print(b.shape)
c = b.repeat(2,2)
print(c.shape)

'''9-获取元素'''
#切片获取
a = torch.arange(12).reshape(1,3,4)
print(a[0])
print(a[0,:,0:3])
print(a[0,-4:,0])

#where
b = -a
print(torch.where(a>3,a,b))
print(a[a>10])

#矩阵获取
#上三角
a = torch.arange(9).reshape(3,3)
print(a.tril(diagonal=0)) #通过diagonal更改对角线,正数往上，负数往下
print(a.tril(diagonal=-1))
#下三角
a = torch.arange(9).reshape(3,3)
print(a.triu(diagonal=0)) #通过diagonal更改对角线
print(a.triu(diagonal=-1))
#对角线
print(a.diag())
print(a.diag(1))

'''10-拼接与拆分'''
a = torch.arange(6.0).reshape(2,3)
b = torch.linspace(0,10,6).reshape(2,3)

#cat()同一纬度连接
c = torch.cat((a,b),dim=0)
print(c)
print(c.shape)
c = torch.cat((a,b),dim=1)
print(c)
print(c.shape)
c = torch.cat((a[:,0:2],a,b),dim=1)
print(c)
print(c.shape)

#stack()同一纬度堆叠
c = torch.stack((a,b),dim = 0)
print(c)
print(c.shape)
c = torch.stack((a,b),dim = 1)
print(c)
print(c.shape)
c = torch.stack((a,b),dim = 2)
print(c)
print(c.shape)

#chunk拆分成同形状的块
d = torch.arange(12).reshape(2,-1)
e1,e2 = torch.chunk(d,2,dim=0)
#split拆分成指定大小的块
e1,e2,e3 = torch.split(d,[1,2,3],dim=1)

'''11-比较大小'''
#torch.allclose()函数比较两元素是否接近，|A-B|<=atol+rtolx|B|
a = torch.tensor([10.0])
b = torch.tensor([10.1])
print(torch.allclose(a,b,rtol=0.1,atol=0.01,equal_nan=False))
print(torch.allclose(a,b,rtol=1e-5,atol=1e-5,equal_nan=False))
#equal_nan=True时判断nan为接近
c = torch.tensor(float('nan'))
print(torch.allclose(c,c,rtol=1e-5,atol=1e-5,equal_nan=True))

#torch.eq()判断是否相等，torch.equal()判断是否有相同的形状和大小
a = torch.arange(1,7)
b = torch.tensor([1,2,3,4,5,6])
c = a.unsqueeze(0)
print(torch.eq(a,b))
print(torch.eq(a,c))
print(torch.equal(a,b))
print(torch.equal(a,c))

#torch.ge()是否大于等于，gt()是否小于
#torch.le()是否小于等于，lt()是否小于
print(torch.ge(a,b))
print(torch.gt(a,b))
print(torch.le(a,b))
print(torch.lt(a,b))
#torch.ne()是否不等于
print(torch.ne(a,b))
#torch.isnan()是否为缺失值
c = torch.tensor([1,3,float('nan'),4])
print(torch.isnan(c))

'''12-基本运算'''
#加 减 乘 除 整除
a = torch.eye(3)
b = torch.ones(3,3)
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a//b)
#幂
a[2,2] = 2
print(a.pow(2))
print(a**3)
#指数、对数、平方根
print(torch.exp(a))
print(torch.log(a))
print(torch.sqrt(a))

#最大值、最小值、范围裁剪
a = torch.arange(6).reshape(2,3)
print(torch.clamp_max(a,4))
print(torch.clamp_min(a,2))
print(torch.clamp(a,2,3))

#转置、矩阵乘法
b = torch.t(a)
print(b)
c = torch.matmul(a,b)
print(c)

#矩阵的逆
a = torch.arange(4.0).reshape(2,2)
c = torch.inverse(a)
print(c)

#矩阵的迹（对角线元素之和）
print(torch.trace(a))

'''13-统计运算'''
#获得最大值，最小值
a = torch.arange(2,14).reshape(3,4)
print(torch.max(a))
print(torch.min(a))
#获得最大值最小值的索引
print(a.argmax())
print(a.argmin())

#计算平均值
b = torch.arange(9.0).reshape(3,3)
print(torch.mean(b,dim = 1,keepdim=True))
print(torch.mean(b,dim = 0,keepdim=True))
#计算和
print(torch.sum(b,dim = 1,keepdim = True))
print(torch.sum(b,dim = 0,keepdim = True))
#计算累加
print(torch.cumsum(b,dim = 1))
print(torch.cumsum(b,dim = 0))
#计算中位数
print(torch.median(b,dim = 1,keepdim=True))
print(torch.median(b,dim = 0,keepdim=True))
#计算乘机
print(torch.prod(b,dim = 1))
print(torch.prod(b,dim = 0))
#计算累乘
print(torch.cumprod(b,dim = 1))
print(torch.cumprod(b,dim = 0))
#计算标准差
print(torch.std(b))