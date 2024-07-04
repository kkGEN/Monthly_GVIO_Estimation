import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms

# 数据预处理函数
def preprocess_data(xtensor_folder, ytensor_file):
    # 读取y
    y_df = pd.read_excel(YtensprFilepath)
    y = y_df.iloc[:, 1:2]
    print(y)
    ytensor = torch.from_numpy(y.values).squeeze()
    print(ytensor.shape)
    torch.save(ytensor, Ytensorpath)
    print('YTensor存储成功！')

    #读取X
    files = os.listdir(XtensorFolderpath)
    #定义一个空tensor储存所有月的数据
    xtensor = torch.empty([0, 1, 256, 180])
    # print(xtensor)
    for file in files:
        #1.读取每月excel中的灯光统计信息为dataframe
        df_file = pd.read_excel(XtensorFolderpath + file, header=None)
        df = df_file.iloc[:, 0:181]  # 读取维度(288,179)
        #2.将dataframe转换为nparray
        df_arr = df.values
        #     print(df_arr)
        #3.使用transforms.ToTensor可以将tensor增加一维，本来主要是用来处理图片
        trans = transforms.ToTensor()  # 会增加一个维度([1, 289,180])
        tensor = trans(df_arr)
        #     print(tensor)
        #     print(tensor.size())
        #4.继续给tensor增加维度，为了使其符合卷积的输入要求
        tensor = tensor.unsqueeze(0)  #继续增加一维([1, 1, 289,180])
        #     print(tensor.size())
        #将每一个月的数据作为一张灰度图像，即只有一个通道的二维图像，然后拼接为一个
        xtensor = torch.vstack((xtensor, tensor))
    print(xtensor.size())
    #将创建的全零向量删除
    # xtensor = xtensor[0:-1, ...]
    xtensor = xtensor.squeeze(0)
    print(xtensor.shape)
    # 存储tensor
    torch.save(xtensor, Xtensorpath)
    print('XTensor存储成功！')
    print(xtensor)
    #转换tensor字符类型为float，匹配卷积操作的字符类型
    xtensor = xtensor.float()
    ytensor = ytensor.float()
    print('原始xtensor和ytensor大小分别为：', xtensor.shape, ytensor.shape)
    # #读取tensor
    # xtensor = torch.load(Xtensorpath)
    # ytensor = torch.load(Ytensorpath)
    # print('Tensor读取成功！')
    return xtensor, ytensor

# 模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # ... 模型定义的代码 ...
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
        self.conv4 = nn.Sequential(  #4.第三层卷积输入的为上一层的输出，即(16,144,90)
            nn.Conv2d(64, 128, 3, 1, 'same'),  #5.输出的形状为(64,144,90)
            nn.ReLU(),
            nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(64,33,19)
        )
        #通过改变全连接层的输出，决定是分类还是回归，此外，还要注意损失函数的定义
        self.out = nn.Linear(13312, 1)  #全连接层64*33*19  35264

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #         print(x.shape)
        x = x.view(x.size(0), -1)
        #         print(x.shape)
        output = self.out(x)
        return output

# 模型初始化函数
def initialize_model():
    cnn = CNN()
    cnn = cnn.to(device)
    return cnn

# 训练函数
def train_model(model, train_data_loader, optimizer, loss_func, epochs=10000):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_func(output, targets.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 1000 == 0:
                print(f'Epoch {epoch}, Step {total_train_step}, Loss: {loss.item()}')

# 测试函数
def test_model(model, data_loader):
    model.eval()
    for batch_idx, (testinputs, targets) in enumerate(data_loader):
        testinputs, targets = testinputs.to(device), targets.to(device)
        testoutput = model(testinputs)
        # ... 其他测试部分的代码 ...
        pred = testoutput.data.cpu().numpy().squeeze() * y_std.numpy() + y_mean.numpy()
        real = targets.cpu().numpy() * y_std.numpy() + y_mean.numpy()
        print('prediction number:', pred)
        print('real number:', real)
        # 预测测试数据效果
        df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame(real)], axis=1, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(pred)], axis=1, ignore_index=True)
        df.columns = ['True', 'Pred']
        df.to_excel(CNNresult)
        print(df)
        x_t = df['True']
        y_p = df['Pred']
        sns.regplot(x=x_t, y=y_p, data=df)

if __name__ == "__main__":
    rootpath = r'C:\Users\KJ\Documents\ShanghaiFactory\Shanghai\\'
    YtensprFilepath = rootpath + r'工业总产值建模.xlsx'
    XtensorFolderpath = rootpath + r'3-500m缓冲区卷积网络使用数据\\'
    Ytensorpath = rootpath + r'Tensor存储\YTensor.pt'
    Xtensorpath = rootpath + r'Tensor存储\500m缓冲区Xtensor\XTensor.pt'
    CNNresult = rootpath + r'卷积结果_500m缓冲区.xlsx'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCH = 10000
    BATCH_SIZE = 50
    LR = 0.0001

    xtensor, ytensor = preprocess_data(XtensorFolderpath, YtensprFilepath)

    # 存储tensor
    torch.save(ytensor, Ytensorpath)
    torch.save(xtensor, Xtensorpath)

    cnn = initialize_model()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    total_train_step = 0
    train_model(cnn, train_data_dl, optimizer, loss_func, epochs=EPOCH)

    # 将所有数据装进新的 DataLoader
    all_dl = DataLoader(torch_dataset, batch_size=96, shuffle=False)
    test_model(cnn, all_dl)
