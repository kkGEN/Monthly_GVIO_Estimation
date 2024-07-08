from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import torchvision
import numpy as np
import pandas as pd
import torch
import os

def Check_FolderPath_Exist(outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
        print(f'{outFolderPath} is created successflly!')
    return

def Read_YTensor(inYFilepath, outYTensor):
    y_df = pd.read_excel(inYFilepath)
    y = y_df.iloc[:, 1:2]
    ytensor = torch.from_numpy(y.values).squeeze()
    torch.save(ytensor, outYTensor)
    print('YTensor存储成功！')
    return

def Read_tensors():
    # #读取tensor
    xtensor = torch.load(Xtensorpath)
    ytensor = torch.load(Ytensorpath)
    # print('Tensor读取成功！')

    # 转换tensor字符类型为float，匹配卷积操作的字符类型
    xtensor = xtensor.float()
    ytensor = ytensor.float()
    print('原始xtensor和ytensor大小分别为：', xtensor.shape, ytensor.shape)
    # print(xtensor)

    # 归一化Xtensor和Ytensor
    x_mean, x_std = torch.mean(xtensor, dim=0), torch.std(xtensor, dim=0)
    y_mean, y_std = torch.mean(ytensor, dim=0), torch.std(ytensor, dim=0)
    # x_norm = (xtensor - x_mean) / x_std
    y_norm = (ytensor - y_mean) / y_std
    print('归一化完成！')
    x_norm = xtensor
    # print(x_norm)

    # 数据读进dataloader，方便后续训练
    torch_dataset = TensorDataset(x_norm, y_norm)  # 组成torch专门的数据库
    # 划分训练集测试集与验证集
    torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
    # 先将数据集拆分为训练集+验证集（共96组），测试集（10组）
    train_validaion, test = random_split(torch_dataset, [90, 6])
    # 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
    train_data_dl = DataLoader(train_validaion, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    print('Dataloader完成！')

if __name__ == "__main__":
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    Tensor_Stored_Folder = os.path.join(outPath, 'Step04_Store_All_Tensors/')
    Check_FolderPath_Exist(Tensor_Stored_Folder)
    # 自定义Read_Tensors函数的输入参数：
