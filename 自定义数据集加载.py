#!/usr/bin/python3
# @Time    : 2023/5/21 11:31
# @Author  : luke
# @FileName: 自定义数据集加载
# @Software:
from torch.utils.data import Dataset,DataLoader
path='dataset'
# file=open(path)
class Mydata(Dataset):
    def __init__(self,path):
        self.lines=open(path,encoding = 'utf-8').readlines()

    def __getitem__(self, index):
        #获取索引对应位置的一条数据
        cur_line= self.lines[index].strip()
        label=cur_line[:4].strip()
        content=cur_line[4:].strip()
        return label,content

    def __len__(self):
        #返回数据的总数量
        return len(self.lines)

if __name__ == '__main__':
    my_data=Mydata(path)
    data_loader=DataLoader(dataset=my_data,batch_size=128,shuffle=True,drop_last=True)
    # for i,(label,data) in enumerate(data_loader):
    #     print(i,label,data)
    print(len(data_loader))

