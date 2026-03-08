#深度主从复合网络结构
import torch
from torch import nn
import scipy.io as scio
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt
from PreprocessBlock import PreprocessBlock
from PasBlock import PasBlock
from MANetLoss import MANetLoss
from SelfAttention import SelfAttention
import numpy as np
import pandas as pd
from View import View
from Anti_Embad import Anti_Embad

##########
#1.读取和划分数据集
n = 5#用几条数据去预测
length = 199
lr, num_epochs, batch_size = 0.02, 500, 32
data = scio.loadmat('F:/vb_model/damage_simulation_true_speed80_deguo3_fftimf_normal.mat')
#加速度
AX = torch.tensor(data['Xn'], dtype=torch.float32)
#加速度FFT
AY = torch.tensor(data['FFT'], dtype=torch.float32)
#EMD的第一个IMF分量
AZ = torch.tensor(data['IMF'], dtype=torch.float32)
#label
AAX = torch.tensor(data['AX'], dtype=torch.float32)
#划分训练集、验证集和测试集的输入
train_AX = AX[:, :, 0:80, [0, 1, 2, 4, 5, 6], :].reshape(15360, 1, (n+1), length)
test_AX = AX[:, :, 80:90, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
add_AX = AX[:, :, 90:100, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
train_AY = AY[:, :, 0:80, [0, 1, 2, 4, 5, 6], :].reshape(15360, 1, (n+1), length)
test_AY = AY[:, :, 80:90, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
add_AY = AY[:, :, 90:100, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
train_AZ = AZ[:, :, 0:80, [0, 1, 2, 4, 5, 6], :].reshape(15360, 1, (n+1), length)
test_AZ = AZ[:, :, 80:90, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
add_AZ = AZ[:, :, 90:100, [0, 1, 2, 4, 5, 6], :].reshape(1920,  1, (n+1), length)
#划分训练集、验证集和测试集的输出
train_BX = AAX[:, :, 0:80, 3:4, :].reshape(15360, length)
test_BX = AAX[:, :, 80:90, 3:4, :].reshape(1920, length)
add_BX = AAX[:, :, 90:100, 3:4, :].reshape(1920, length)
#数据拼接，整理输入(批次、通道、高度、宽度)、输出形状（批次、宽度）
train_x = torch.cat((train_AX, train_AY, train_AZ), dim=1)#输入：（批量、通道3、高度、宽度）
test_x = torch.cat((test_AX, test_AY, test_AZ), dim=1)
train_y = train_BX#输出：（批量、宽度）
test_y = test_BX
add_x = torch.cat((add_AX, add_AY, add_AZ), dim=1)
add_y = add_BX
#构建训练集、验证集、测试集
train_data = (train_x, train_y)
test_data = (test_x, test_y)
add_data = (add_x, add_y)
train_iter = d2l.load_array(train_data, batch_size)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
add_iter = d2l.load_array(add_data, batch_size, is_train=False)

#实验数据(一个和一批)
one_A = AZ[0:1, 0:1, 90:91, [0, 1, 2, 4, 5, 6], :].reshape(1,  1, (n+1), length)
one_B = AAX[0:1, 0:1, 90:91, 3:4, :].reshape(1, length)
one_data = (one_A, one_B)
one_iter = d2l.load_array(one_data, 1, is_train=False)

batch_A = AZ[0:1, 0:1, 90:100, [0, 1, 2, 4, 5, 6], :].reshape(10,  1, (n+1), length)
batch_B = AAX[0:1, 0:1, 90:100, 3:4, :].reshape(10, length)
batch_data = (batch_A, batch_B)
batch_iter = d2l.load_array(batch_data, 10, is_train=False)

#2.初始化模型参数train_ch6里的函数
##########
#3.定义网络######
#卷积块，卷积-批量规范化-激活
def Conv_block (input_channels, num_channels, kernel_size, padding):
    return nn.Sequential(nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding),
                         nn.BatchNorm2d(num_channels), nn.ReLU())

#连接总网络
net = nn.Sequential(PreprocessBlock(3, 32, 32),#主从5*32：1*32
                    PasBlock((192, 128), (32, 32), (5, 1)),
                    nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),#5*99
                    PasBlock((160, 64), (32, 16), (4, 1)),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),#4*49
                    PasBlock((80, 64), (16, 16), (4, 1)),
                    nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),#4*24
                    PasBlock((80, 64), (16, 16), (4, 1)),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),#3*12
                    View(),#[1, 141, 80](批次、长度、 嵌入维度)
                    SelfAttention(8, 80, 80, 0.2),#(num_attention_heads, input_size, hidden_size, hidden_dropout_prob)#[1, 141, 80]
                    Anti_Embad(80),
                    nn.Linear(141, length)
                    )
#检查模型输出大小，确保与我们期望一致
X = torch.rand(size=(1, 3, 6, length), dtype=torch.float32)
print(X.size())
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
##########
#4.训练模型
#自己的数据集
#训练函数（损失函数）
#@save
def evaluate(net, data_iter, device=None): #@save
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:#循环15次，测试
            X = X.to(device)
            y = y.to(device)
            loss = MANetLoss()
            metric.add(loss(net(X), y, net)*y.numel(), y.numel())
    return metric[0] / metric[1]

def criterion(net, data_iter, device=None): #@save
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:#循环15次，测试
            X = X.to(device)
            y = y.to(device)
            loss = nn.L1Loss()# 均方差损失，小批量梯度下降
            metric.add(loss(net(X), y)*y.numel(), y.numel())
    return metric[0] / metric[1]

#@save
def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)#权重初始化ok
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = MANetLoss()#均方差损失，小批量梯度下降
    plt.rcParams["font.family"] = "STSong"
    animator = d2l.Animator(xlabel='Epoch', xlim=[1, num_epochs],
                            legend=['训练损失', '测试损失'], figsize=(6, 4))
    timer, num_batches = d2l.Timer(), len(train_iter)
    losslist = []
    testlosslist = []
    crilist = []
    for epoch in range(num_epochs):#循环20次，一共20轮
        # 训练损失之和，样本数
        metric = d2l.Accumulator(2)
        #epochloss = torch.zeros(num_epochs)
        net.train()
        for i, (X, y) in enumerate(train_iter):#一次一个batch,循环60次，一共60个batch
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y, net)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])#一个批次的
            timer.stop()
            train_l = metric[0] / metric[1]#一个批次的损失
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, None))
        print(f'epoch {epoch+1:.0f} loss {train_l:.11f}')#一轮的损失
        test_loss = evaluate(net, test_iter)
        cri = criterion(net, test_iter)
        animator.add(epoch + 1, (None, test_loss))
        losslist.append(train_l)
        testlosslist.append(test_loss)
        crilist.append(cri)
    mid = pd.DataFrame(losslist)
    testmid = pd.DataFrame(testlosslist)
    crimid = pd.DataFrame(crilist)
    mid.to_excel(r'loss_self.xlsx', header='loss', index=True)
    testmid.to_excel(r'test_self.xlsx', header='test_loss', index=True)
    crimid.to_excel(r'cri_self.xlsx', header='cri', index=True)
    print(f'loss {train_l:.11f}, '
          f'test loss {test_loss:.11f}')
    print(f'{metric[1] * num_epochs / timer.sum():.7f} examples/sec '
          f'on {str(device)}')
    plt.savefig("lossdeguonew.svg", dpi=600, format="svg")
    plt.show()


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')



train(net, train_iter, test_iter, num_epochs, lr, try_gpu())
torch.save(net, os.path.join('.', 'fz_manet_mae_l_self.pth'))# 保存模型结构和参数
#net = torch.load(os.path.join('.', 'fz_manet_mae_l_1000.pth')).to(device='cpu')# 加载模型结构和参数
#5.预测模型
def predict(net, test_iter,device):  #@save
    for i, (X, y) in enumerate(test_iter):
        net.to(device='cpu').eval()
        trues = y
        preds = net(X)
        loss = MANetLoss()  # 交叉熵损失，小批量梯度下降
        l = loss(preds, trues, net)
        print(f'loss {l:.11f}')
    #绘图
    x=torch.arange(1,length+1)#.to(device)
    print(x.device)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(num=1, figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time(0.01s)", fontsize=30)
    ax.set_xlim([1, length+1])
    ax.set_ylabel("Acceleration(m/s$^{2}$)", fontsize=30)
    ax.plot(x.numpy(), trues[15].detach().numpy(), "b-", label="VBI analysis")
    ax.plot(x.numpy(), preds[15].detach().numpy(), "r-.", label="MANET prediction")
    true = list(trues[15].detach().numpy())
    pred = list(preds[15].detach().numpy())
    datatruemid = pd.DataFrame(true)
    datapredmid = pd.DataFrame(pred)
    datatruemid.to_excel(r'true_data_self.xlsx', header='data', index=True)
    datapredmid.to_excel(r'pred_data_self.xlsx', header='data', index=True)
    plt.tick_params(labelsize=30)
    ax.legend(loc=1, labelspacing=1, handlelength=3, fontsize=30)#, shadow=True)
    plt.savefig("test_new.svg", dpi=600, format="svg")
    plt.show()

def predict_batch(net, test_iter,device):  #@save
    for i, (X, y) in enumerate(test_iter):
        net.to(device='cpu').eval()
        trues = y
        preds = net(X)
        loss = nn.L1Loss()
        l = loss(preds, trues)
        print(f'loss {l:.11f}')
    #绘图
    x=torch.arange(1,length+1)#.to(device)
    print(x.device)
    #图格式
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(num=1, figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time(0.01s)", fontsize=30)
    ax.set_xlim([1, length+1])
    ax.set_ylabel("Acceleration(m/s$^{2}$)", fontsize=30)
    ax.plot(x.numpy(), trues[3].detach().numpy(), "b-", label="VBI analysis")
    ax.plot(x.numpy(), preds[3].detach().numpy(), "r-.", label="MANET prediction")
    #打印数据
    true = list(trues.detach().numpy())
    pred = list(preds.detach().numpy())
    datatruemid = pd.DataFrame(true)
    datapredmid = pd.DataFrame(pred)
    datatruemid.to_excel(r'true_data_100.xlsx', header='data', index=True)
    datapredmid.to_excel(r'pred_data_100.xlsx', header='data', index=True)
    #画图
    plt.tick_params(labelsize=30)
    ax.legend(loc=1, labelspacing=1, handlelength=3, fontsize=30)#, shadow=True)
    plt.savefig("test_new.svg", dpi=600, format="svg")
    plt.show()

predict(net, test_iter,try_gpu())


