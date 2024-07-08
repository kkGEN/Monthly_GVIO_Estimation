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

def Read_Xtensors():
    # 读取X
    files = os.listdir(XtensorFolderpath)
    # 定义一个空tensor储存所有月的数据
    xtensor = torch.empty([0, 1, 256, 180])
    # print(xtensor)
    for file in files:
        # 1.读取每月excel中的灯光统计信息为dataframe
        df_file = pd.read_excel(XtensorFolderpath + file, header=None)
        df = df_file.iloc[:, 0:181]  # 读取维度(288,179)
        # 2.将dataframe转换为nparray
        df_arr = df.values
        #     print(df_arr)
        # 3.使用transforms.ToTensor可以将tensor增加一维，本来主要是用来处理图片
        trans = transforms.ToTensor()  # 会增加一个维度([1, 289,180])
        tensor = trans(df_arr)
        #     print(tensor)
        #     print(tensor.size())
        # 4.继续给tensor增加维度，为了使其符合卷积的输入要求
        tensor = tensor.unsqueeze(0)  # 继续增加一维([1, 1, 289,180])
        #     print(tensor.size())
        # 将每一个月的数据作为一张灰度图像，即只有一个通道的二维图像，然后拼接为一个
        xtensor = torch.vstack((xtensor, tensor))
    print(xtensor.size())
    # 将创建的全零向量删除
    # xtensor = xtensor[0:-1, ...]
    xtensor = xtensor.squeeze(0)
    print(xtensor.shape)
    # 存储tensor
    torch.save(xtensor, Xtensorpath)
    print('XTensor存储成功！')
    print(xtensor)

if __name__ == "__main__":
    outPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    Tensor_Stored_Folder = os.path.join(outPath, 'Step04_Store_All_Tensors/')
    Check_FolderPath_Exist(Tensor_Stored_Folder)
    # 自定义Read_YTensor函数的输入参数：
    YtensorFilePath = os.path.join(outPath, '工业总产值建模14-22.xlsx')
    YtensorPath = os.path.join(Tensor_Stored_Folder, 'YTensor.pt')
    if not os.path.exists(YtensorPath):
        Process_9 = Read_YTensor(YtensorFilePath, YtensorPath)
    else:
        print('Ytensor is already exist!')

    # 自定义Read_Xtensors函数的输入参数：
    XtensorFolderpath = os.path.join(outPath, )

    Process_10 = Read_Xtensors(XtensorFolderpath, XtensorPath)
