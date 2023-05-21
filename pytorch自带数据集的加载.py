#!/usr/bin/python3
# @Time    : 2023/5/21 11:59
# @Author  : luke
# @FileName: pytorch自带数据集的使用
# @Software: PyCharm
import torchvision.transforms
from torchvision.datasets import MNIST
from torchvision import transforms
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(10,1)
                                          ])
mnist=MNIST(root='./mnist_data',train=True,download=True,transform=transform)
print(mnist[0][0])
# ret=transforms.ToTensor()(mnist[0][0])
# print(ret)
# print('*'*100)
# norm_img=transforms.Normalize(10,1)(ret)
# print(norm_img)