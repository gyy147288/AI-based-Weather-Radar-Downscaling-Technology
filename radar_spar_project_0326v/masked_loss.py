# masked_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

#自定义损失函数，两个都是为了含掩码的模型建立的
# class MaskedMSELoss(nn.Module):
#     def __init__(self, fill_value=-500):
#         super().__init__()
#         self.fill_value = fill_value
#
#     def forward(self, pred, target):
#         # 生成掩码（1表示有效值，0表示无效）
#         mask = (target != self.fill_value).float()
#
#         # 计算加权损失
#         loss = (pred - target) ** 2
#         return (loss * mask).sum() / (mask.sum() + 1e-6)  # 避免除以零
#
# class HybridMaskedLoss(nn.Module):
#     def __init__(self, fill_value=-500, alpha=0.3):
#         super().__init__()
#         self.fill_value = fill_value
#         self.alpha = alpha  # L1损失权重
#
#     def forward(self, pred, target):
#         # 生成掩码（1表示有效值，0表示无效）
#         mask = (target != self.fill_value).float()
#
#         # 计算加权损失
#         l2_loss = (pred - target) ** 2 * mask
#         l1_loss = torch.abs(pred - target) * mask
#
#         # 组合损失
#         valid_count = mask.sum() + 1e-6
#         return ((1 - self.alpha) * l2_loss.sum() + self.alpha * l1_loss.sum()) / valid_count
#
# class FillValueAwareLoss(nn.Module):
#     def __init__(self, fill_value=-3000):
#         super().__init__()
#         self.fill_value = fill_value
#
#     def forward(self, pred, target):
#         # 创建有效值掩码
#         valid_mask = (target != self.fill_value).float()
#         # 计算均方误差
#         loss = torch.mean((pred - target)**2 * valid_mask)
#         return loss

#实际使用的模型的损失函数，标记了-10dbz的
# class WeightedMSELoss(nn.Module):
#     def __init__(self, fill_value=-1000, null_weight=1.5, high_value_threshold=40, high_value_weight=3.0):
#         """
#         加权均方误差损失函数
#         参数:
#             fill_value (float): 表示空值的特殊标记值，默认-1000
#             null_weight (float): 空值区域的损失权重，默认2.0
#             high_value_threshold (float): 强回波阈值，默认4000
#             high_value_weight (float): 强回波区域的损失权重，默认2.0
#         """
#         super().__init__()
#         self.fill_value = fill_value
#         self.null_weight = null_weight
#         self.high_value_threshold = high_value_threshold
#         self.high_value_weight = high_value_weight
#
#     def forward(self, predictions, targets):
#         """
#         前向计算
#         参数:
#             predictions (Tensor): 模型预测值，形状任意
#             targets (Tensor): 真实标签值，形状需与predictions一致
#         返回:
#             Tensor: 加权后的均方误差损失
#         """
#         # 创建有效值掩码（排除空值区域）
#         valid_mask = targets.ne(self.fill_value)
#
#         # 初始化权重张量
#         weights = torch.ones_like(targets)
#
#         # 设置空值区域权重（虽然这些区域会被掩码排除，但为了代码完整性保留）
#         # weights.masked_fill_(targets.eq(self.fill_value), self.null_weight)
#
#         # 设置强回波区域权重
#         weights.masked_fill_(targets.gt(self.high_value_threshold), self.high_value_weight)
#
#         # 计算加权MSE（仅计算有效区域）
#         loss = (weights[valid_mask] * (predictions[valid_mask] - targets[valid_mask]) ** 2).mean()
#         return loss

class WeightedMSELoss(nn.Module):
    def __init__(self, high_value_threshold=40, high_value_weight=3.0):
        """
        加权均方误差损失函数（不再考虑 fill_value）。
        """
        super().__init__()
        self.high_value_threshold = high_value_threshold
        self.high_value_weight = high_value_weight

    def forward(self, predictions, targets):
        """
        计算加权均方误差。
        """
        # 直接计算 MSE
        weights = torch.ones_like(targets)
        weights.masked_fill_(targets.gt(self.high_value_threshold), self.high_value_weight)

        loss = (weights * (predictions - targets) ** 2).mean()
        return loss
