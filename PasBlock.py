import torch
from torch import nn
from ResidualBlock import ResidualBlock

#主从卷积块
class PasBlock (nn.Module): #先普通卷积 后面可以用残差网络
    def __init__(self, c1, c2, ratio):#(5,1)
        super(PasBlock, self).__init__()
        # 送入的比例
        self.split = int(ratio[0] * c1[0] / (ratio[0] + ratio[1]))
        # 线路1，卷积层
        self.p1 = ResidualBlock(c1[0], c1[1])
        # 线路2，输出通道为1的1/8
        self.p2 = ResidualBlock(c2[0], c2[1])
    def forward(self, x):
        x1 = x
        x2 = x[:, self.split:x.size(1), :, :]#用的是后半部分，就是车辆的
        q1 = self.p1(x1)
        q2 = self.p2(x2)
        return torch.cat((q1, q2), dim=1)#车辆在后