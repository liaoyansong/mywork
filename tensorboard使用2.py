#!/usr/bin/python3
# @Time    : 2023/5/21 15:07
# @Author  : luke
# @FileName: tensorboard使用2
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
writer = SummaryWriter()
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for images, labels in train_loader:
    # 假设images是输入图像，labels是对应的标签
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 在每个epoch或特定的步骤中，将特征图写入TensorBoard
    if (epoch + 1) % 10 == 0:
        features = model.conv1(images)
        writer.add_images('Conv1 Features', features, global_step=epoch)
writer.close()
