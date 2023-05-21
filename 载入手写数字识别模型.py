#!/usr/bin/python3
# @Time    : 2023/5/21 16:25
# @Author  : luke
# @FileName: 载入手写数字识别模型
# @Software: PyCharm
import os.path
import numpy as np
import torch.nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch import nn

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

def test():
    net.eval()
    data_test = get_dataloader(False)
    correct=0
    acc_list=[]
    loss_list=[]
    with torch.no_grad():
        for i,(input,label) in enumerate(data_test):
            out=net(input)
            loss=Loss(out,label)
            loss_list.append(loss)
            correct=sum(torch.argmax(out,dim=1)==label)
            acc=correct/len(label)
            acc_list.append(acc)
        print(f'acc={sum(acc_list)/len(acc_list)}')
        print(f'acc={np.mean(acc_list)}')
        print(f'loss={np.mean(loss_list)}')


if __name__ == '__main__':
    net=Model()
    Loss = torch.nn.CrossEntropyLoss()
    if os.path.exists('./model/mnist.pth'):
        net.load_state_dict(torch.load('./model/mnist.pth'))
    test()