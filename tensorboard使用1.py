#!/usr/bin/python3
# @Time    : 2023/5/21 15:03
# @Author  : luke
# @FileName: tensorboard
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 定义一个简单的PyTorch模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建TensorBoard的SummaryWriter对象
writer = SummaryWriter()

# 实例化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 示例训练循环
for epoch in range(10):
    # 假设每个epoch有100个训练样本
    for i in range(100):
        # 假设输入x和目标y为随机数据
        x = torch.randn(10)
        y = torch.randn(1)

        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(x)

        # 计算损失
        loss = criterion(output, y)

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

        # 每个batch打印一次损失，并将损失写入TensorBoard
        print(f'Epoch [{epoch+1}/10], Step [{i+1}/100], Loss: {loss.item()}')
        writer.add_scalar('Loss/train', loss.item(), epoch * 100 + i)

# 将模型的图形结构写入TensorBoard
dummy_input = torch.randn(1, 10)  # 假设输入大小为1x10
writer.add_graph(model, dummy_input)

# 关闭TensorBoard的SummaryWriter
writer.close()
