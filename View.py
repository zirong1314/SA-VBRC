import torch
import torch.nn as nn
import numpy as np
# 定义残差模块
class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()


    def forward(self, x):
        data = x.view(x.size(0), x.size(1), -1)
        reshaped_data = data.permute(0, 2, 1)
        return reshaped_data


