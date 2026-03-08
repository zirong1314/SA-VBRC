import torch
from torch import nn
class MANetLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.0001, gamma=0.0001):
        super(MANetLoss, self).__init__()
        self.alpha = alpha  # 权衡MAE和MSE的系数
        self.beta = beta    # L1正则化的系数
        self.gamma = gamma  # L2正则化的系数

    def forward(self, predicted, actual, model):
        # 计算MSE损失
        mse_loss = torch.mean((predicted - actual) ** 2)
        # 计算MAE损失
        mae_loss = torch.mean(torch.abs(predicted - actual))
        #mae_loss = torch.sqrt(torch.sum(torch.abs(predicted)) ** 2 - torch.abs(torch.sum(actual)) ** 2)
        # 计算L1正则化项 l1_reg 和 l2_reg应被定义为不需要梯度的张量，而且在损失计算中不需要对其进行梯度传播requires_grad=False
        l1_reg = torch.tensor(0.0, device=predicted.device)
        for param in model.parameters():
            l1_reg += torch.sum(torch.abs(param))
        # 计算L2正则化项
        l2_reg = torch.tensor(0.0, device=predicted.device)
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        # 组合损失项
        total_loss = self.alpha * mae_loss + (1-self.alpha) * mse_loss + self.beta * l1_reg + self.gamma * l2_reg
        return total_loss
