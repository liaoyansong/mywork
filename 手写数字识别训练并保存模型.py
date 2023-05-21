#!/usr/bin/python3
# @Time    : 2023/5/21 15:20
# @Author  : luke
# @FileName: 手写数字识别
# @Software: PyCharm
import torch.nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch import nn
# 创建TensorBoard的SummaryWriter对象
# writer = SummaryWriter()
batch_size = 128


# 1准备数据集
def get_dataloader(train=True):
    transform = Compose([ToTensor(),
                         Normalize(mean=0.1307, std=0.3081)])
    dataset = MNIST(root='./mnist_data', train=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader

# 2构建模型

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(1*28*28,28),
            nn.ReLU(),
            nn.Linear(28,10)
        )
    def forward(self,x):
        x=x.view(-1,1*28*28)
        x=self.layer(x)
        return x

net=Model()
Loss=torch.nn.CrossEntropyLoss()
optim=torch.optim.SGD(net.parameters(),lr=0.01)

# 3模型训练
def train(epoch):
    net.train()
    train_data=get_dataloader()
    lost=0
    for j in range(epoch):
        for i,(input,label) in enumerate(train_data):
            output=net(input)
            loss=Loss(output,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lost=loss
        print(f'loss={lost}')



if __name__ == '__main__':
    train(3)
    torch.save(net.state_dict(),'./model/mnist.pth')#模型保存