#!/usr/bin/python3
# @Time    : 2023/5/21 9:51
# @Author  : luke
# @FileName: 线性回归简易版
# @Software: PyCharm
import torch
from torch import nn

# 1.准备数据
x = torch.rand([500, 1])
y_true = x * 5 + 10


# 2.定义模型
class Model(nn.Module):
    def __init__(self):
        # 继承父类的init
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# 3.实例化模型，优化器，损失函数
net = Model()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss = nn.MSELoss()

# 4.模型训练，参数更新
for i in range(200):
    y_pre = net(x)  # 得到预测值
    LS = loss(y_pre, y_true)  # 计算损失
    optimizer.zero_grad()  # 梯度置零
    LS.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if i % 50 == 0:
        print(f'loss={LS}')
for param in net.parameters():
    print(param.data)