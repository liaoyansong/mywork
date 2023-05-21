#!/usr/bin/python3
# @Time    : 2023/5/20 21:54
# @Author  : luke
# @FileName: 手动实现线性回归
# @Software: PyCharm
import torch
from matplotlib import pyplot as plt
# 1.准备数据
x = torch.rand([500, 1])
y_true = x * 3 + 0.8

# 2.通过模型计算y_pre
w = torch.rand([1, 1], requires_grad=True)
b = torch.zeros([1], requires_grad=True)

# 4.通过循环，反向传播，更新参数
learning_rate = 0.01
for i in range(500):
    # 3.计算loss
    y_pre = torch.matmul(x, w) + b
    loss = (y_true - y_pre).pow(2).mean()

    if w.grad is not None:  # 设置的梯度默认是None值
        w.grad.data.zero_()
    if b.grad is not None:  # 设置的梯度默认是None值
        b.grad.data.zero_()
    # .data是对数据的浅拷贝(同一个内存地址),.detach()是对数据的深拷贝(复制为另一份)
    loss.backward()
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if i % 100== 0:
        print(f'w={w.item()},b={b.item()},loss={loss.item()}')


#画图
plt.figure(figsize=(20,8))
y_pre=torch.mm(x,w)+b
plt.plot(x.numpy().reshape(-1),y_pre.detach().numpy().reshape(-1))
plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1),c='r')
plt.show()

