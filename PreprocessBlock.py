import torch
from torch import nn
n = 5
def Conv_block (input_channels, num_channels, kernel_size, padding):
    return nn.Sequential(nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, padding=padding),
                         nn.BatchNorm2d(num_channels, 4), nn.ReLU())

#预处理块
class PreprocessBlock (nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super(PreprocessBlock, self).__init__()
        # 每层的卷积核大小和填充都自己设一下 为了输出矩阵大小不变
        # 线路1，单1x5卷积核
        self.p1 = Conv_block(in_channels, out_channels1, (1, 5), (0, 2))#（行数、列数）
        # 线路2，单2x5卷积核
        self.p2 = Conv_block(in_channels, out_channels1, (2, 5), (0, 2))
        # 线路3，单3x5卷积核
        self.p3 = Conv_block(in_channels, out_channels1, (3, 5), (1, 2))
        # 线路4，单4x5卷积核
        self.p4 = Conv_block(in_channels, out_channels1, (4, 5), (0, 2))
        # 线路5，单5x5卷积核
        self.p5 = Conv_block(in_channels, out_channels1, (5, 5), (2, 2))
        # 线路6，单1x5卷积核
        self.p6_1 = Conv_block(in_channels, out_channels2, (1, 5), (0, 2))
        self.p6_2 = Conv_block(out_channels2, out_channels2, (1, 5), (0, 2))
        # 线路7，单1x11卷积核
        self.p7_1 = Conv_block(in_channels, out_channels2, (1, 11), (0, 5))
        self.p7_2 = Conv_block(out_channels2, out_channels2, (1, 11), (0, 5))
        self.p8 = Conv_block(in_channels, out_channels2, (1, 1), (0, 0))
        self.pad2 = nn.ZeroPad2d((0, 0, 1, 0))#左右上下
        self.pad4 = nn.ZeroPad2d((0, 0, 1, 2))
        self.out_channels2 = out_channels2
    def forward(self, x):#x1和x2可以变成一个矩阵
        x1 = x[:, :, 0:n, :].resize_(x.size(0), x.size(1), 5, x.size(3))
        x2 = x[:, :, n:(n+1), :]#x2应该是哪一个，第6个才是车
        #x2 = x[:, :, 0:1, :]
        q1 = self.p1(x1)
        x1_2 = self.pad2(x1)
        q2 = self.p2(x1_2)
        q3 = self.p3(x1)
        x1_4 = self.pad4(x1)
        q4 = self.p4(x1_4)
        q5 = self.p5(x1)
        q6_1 = self.p6_1(x2)
        q6_2 = self.p6_2(q6_1)
        q7_1 = self.p6_1(x2)
        q7_2 = self.p7_2(q7_1)
        x2_0 = self.p8(x2)
        channels1 = torch.cat((q1, q2, q3, q4, q5), dim=1)
        channels2 = torch.cat((x2_0, q6_1, q6_2, q7_1, q7_2), dim=2)  # x2维度不一致 按行拼接
        return torch.cat((channels1, channels2), dim=1)# 按通道拼接，车辆在后