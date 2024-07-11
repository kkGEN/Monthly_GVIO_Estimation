from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import pandas as pd
import torch
import os

def Get_File_Path(path, string):
    # path: 要匹配的文件夹
    # string：待匹配的文件名称含有的字符串

    filepath = [file for file in os.listdir(path) if string in file]
    outpath = os.path.join(path, filepath[0])
    return outpath

def Read_tensors(Xtensorpath, Ytensorpath, Batchsize):
    # 读取xytensor，并完成归一化等预处理，然后使用dataloader操作数据
    # Xtensorpath: xtensor的路径
    # Ytensorpath: ytensor的路径

    #读取tensor
    xtensor = torch.load(Xtensorpath)
    ytensor = torch.load(Ytensorpath)
    print('Tensor读取成功！')

    # 转换tensor字符类型为float，匹配卷积操作的字符类型
    xtensor = xtensor.float()
    ytensor = ytensor.float()
    # 归一化Xtensor和Ytensor
    x_mean, x_std = torch.mean(xtensor, dim=0), torch.std(xtensor, dim=0)
    y_mean, y_std = torch.mean(ytensor, dim=0), torch.std(ytensor, dim=0)
    # x_norm = 1E-06 + ((xtensor - x_mean) / x_std)
    x_norm = xtensor
    y_norm = (ytensor - y_mean) / y_std

    # 数据读进dataloader，方便后续训练
    torch_dataset = TensorDataset(x_norm, y_norm)  # 组成torch专门的数据库
    # 划分训练集测试集与验证集
    torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
    # 先将数据集拆分为训练集+验证集（共108组），测试集（10组）
    train_validaion, test = random_split(torch_dataset, [90, 18])
    # 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
    train_data_dl = DataLoader(train_validaion, batch_size=Batchsize, shuffle=True)
    test_dl = DataLoader(test, batch_size=Batchsize, shuffle=True)
    print('Dataloader完成！')
    return train_data_dl, test_dl, y_mean, y_std

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  #1.第一层卷积输入的灯光统计数据1通道，相当于灰度图，大小为256行X180列，即(1, 288, 180)
            nn.Conv2d(
                in_channels=1,  #数据输入的通道数，对于彩色图片是3，对于灰度图是1
                out_channels=16,  #卷积核的个数，每个卷积核都会生成一层新的卷积特征
                kernel_size=3,  #卷积核的大小
                stride=1,  #卷积核每次移动的距离
                padding='same',  #如果想要卷积出来的图片长宽没有变化, padding=(kernel_size-1)/2
            ),  #2.输出时的形状为(16, 288, 180),形状保持不变
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=5, stride=2)  #3.在2x2的空间里使用最大值向下采样,输出的形状为(16,144,90)
        )
        self.conv2 = nn.Sequential(  #4.第二层卷积输入的为上一层的输出，即(16,144,90)
            nn.Conv2d(16, 32, 3, 1, 'same'),  #5.输出的形状为(32,144,90)
            nn.ReLU(),
            nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(32,72,45)
        )
        self.conv3 = nn.Sequential(  #4.第三层卷积输入的为上一层的输出，即(16,144,90)
            nn.Conv2d(32, 64, 3, 1, 'same'),  #5.输出的形状为(64,144,90)
            nn.ReLU(),
            nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(64,33,19)
        )
#         self.conv4 = nn.Sequential(  #4.第三层卷积输入的为上一层的输出，即(16,144,90)
#             nn.Conv2d(64, 128, 3, 1, 'same'),  #5.输出的形状为(64,144,90)
#             nn.ReLU(),
#             nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(64,33,19)
#         )
        #通过改变全连接层的输出，决定时分类还是回归，此外，还要注意损失函数的定义
        self.out = nn.Linear(36864, 1)  # 全连接层输出：36864(3层卷积)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
#         x = self.conv4(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        output = self.out(x)
        return output

def Trian_Modle(train_data_dl, epoch, device):
    # train_data_dl：用于训练的dataloader

    total_train_step = 0
    for ep in range(epoch):
        if ep % 1000 == 0:
            print(f'第{ep}次epoch')
        cnn.train()
        for batch_idx, (inputs, targets) in enumerate(train_data_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = cnn(inputs)
            loss = loss_func(output, targets.unsqueeze(1))
            # 以下是固定写法
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 误差传播
            optimizer.step()  # 应用梯度

            # 打印训练过程
            total_train_step += 1
            if total_train_step % 1000 == 0:
                print(f'第{total_train_step}次训练，loss = {loss.item()}')
    return

def Test_Model(test_dl, y_mean, y_std, outexcel, device):
    # test_dl: 用于测试的dataloader
    # y_mean，y_std：ytensor的均值和标准差

    df = pd.DataFrame(columns=['True', 'Pred'])
    for batch_idx, (testinputs, targets) in enumerate(test_dl):
        testinputs = testinputs.to(device)
        targets = targets.to(device)
        testoutput = cnn(testinputs)
        pred = testoutput.data.cpu().numpy().squeeze() * y_std.numpy() + y_mean.numpy()
        real = targets.cpu().numpy() * y_std.numpy() + y_mean.numpy()
        # 将每一个batch的数据装进df中
        df_temp = pd.DataFrame({'True': real, 'Pred': pred})
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
    print('Predicted number:', df['Pred'])
    print('Real number:', df['True'])
    df.to_excel(outexcel)
    # 展示测试数据结果
    x_true, y_pred = df['True'], df['Pred']
    sns.regplot(x=x_true, y=y_pred)
    plt.show()
    return


if __name__ == "__main__":
    # 若cuda存在，则将网络放到gpu上运算
    rootPath = r'E:/ShanghaiFactory/Shanghai_Final/'
    EPOCH = 10000
    BATCH_SIZE = 20
    LR = 0.0001
    BufferSize = '1500 METERS'  # <<Caution!!!>> 缓冲区的距离，这是一个可变参数，可选500m,1000m,1500m,2000m

    # 自定义Read_Tensors函数的输入参数：
    Tensor_Stored_Folder = os.path.join(rootPath, 'Step04_Store_All_Tensors/')
    XTensorpath = Get_File_Path(Tensor_Stored_Folder, f'{BufferSize}')
    YTensorpath = Get_File_Path(Tensor_Stored_Folder, 'Y')
    TrainDL, TestDL, YMean, YStd = Read_tensors(XTensorpath, YTensorpath, BATCH_SIZE)

    # 初始化网络
    cnn = CNN()
    # 若CUDA存在，则将网络放到GPU上运算
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(Device)
    print('网络初始化完成！')
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    print('损失函数定义完成！')
    # 定义结果输出路径
    outCNNResult = os.path.join(rootPath, f'{BufferSize}_Result.xlsx')
    # 开始训练网络
    Trian_Modle(TrainDL, EPOCH, Device)
    # 测试网络结果
    OutResultExcel = os.path.join(rootPath, f'{BufferSize}_result.xlsx')
    Test_Model(TestDL, YMean, YStd, OutResultExcel, Device)
