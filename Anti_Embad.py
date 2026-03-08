import torch
import torch.nn as nn
import numpy as np
# 定义残差模块
class Anti_Embad(nn.Module):
    def __init__(self, input_size):
        super(Anti_Embad, self).__init__()
        self.anti = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.zeros((x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        for i in range(x.size(1)):
            y = x[:, i, :].view(x.size(0),  -1)
            out[:, i] = self.anti(y).squeeze(-1)
        return out


