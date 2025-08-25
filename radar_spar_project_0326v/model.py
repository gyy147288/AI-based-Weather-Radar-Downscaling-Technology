import torch
import torch.nn as nn
import torch.nn.functional as F

#一代目版本模型，特别简单
# class RadarCNN(nn.Module):
#     def __init__(self):
#         super(RadarCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 输入通道数为3，输出通道数为16
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 输出通道数为32
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # 最大池化
#         self.fc1 = nn.Linear(32 * 1 * 1, 64)                    # 全连接层
#         self.fc2 = nn.Linear(64, 1)                             # 输出层，回归任务输出单个值
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))  # 卷积1 + 激活
#         x = self.pool(self.relu(self.conv2(x)))  # 卷积2 + 激活 + 池化
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.relu(self.fc1(x))  # 全连接层1 + 激活
#         x = self.fc2(x)             # 输出层
#         return x

#二代目版本模型，增加了一点复杂度，主要是池化的位置改变了并且把最大池化改为了平均值池化
# class RadarCNN(nn.Module):
#     def __init__(self, in_channels=3, base_channels=64): #根据不同层数训练使用in_channels = 1,2,3
#         super().__init__()
#         self.features = nn.Sequential(
#             # 第一卷积组
#             nn.Conv2d(in_channels, base_channels, 3, padding=1),
#             nn.ReLU(),
#             # 第二卷积组
#             nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
#             nn.ReLU(),
#             # 自适应池化
#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(base_channels*2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         x = self.features(x)      # (B,3,3,3) → (B,128,3,3) → (B,128,1,1)
#         x = x.view(x.size(0), -1) # → (B,128)
#         x = self.classifier(x)    # → (B,1)
#         return x

#三代目版本模型，存储的模型文件为.new版本 第一次使用注意力机制的模型
# class RadarCNN(nn.Module):
#     def __init__(self, num_layers=3, base_channels=32):  # 减少基础通道数
#         super().__init__()
#
#         # 输入处理
#         self.input_conv = nn.Sequential(
#             nn.Conv2d(num_layers * 2, base_channels, 3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU()
#         )
#
#         # 残差块
#         self.res_block = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, 3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(),
#             nn.Conv2d(base_channels, base_channels, 3, padding=1),
#             nn.BatchNorm2d(base_channels)
#         )
#
#         # 轻量级注意力
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(base_channels, base_channels // 4, 1),
#             nn.ReLU(),
#             nn.Conv2d(base_channels // 4, base_channels, 1),
#             nn.Sigmoid()
#         )
#
#         # 回归头
#         self.regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(base_channels, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         # 输入形状: [B, L, 3, 3, 2]
#         B = x.size(0)
#         x = x.view(B, -1, 3, 3)  # 合并层和通道
#
#         # 特征提取
#         x = self.input_conv(x)  # [B, 32, 3, 3]
#         residual = x
#         x = self.res_block(x) + residual  # 残差连接
#         x = F.relu(x)
#
#         # 注意力
#         attn = self.attention(x)  # [B, 32, 1, 1]
#         x = x * attn
#
#         # 回归预测
#         return self.regressor(x).squeeze(1)

#四代目版本模型，加入了层注意力机制，但是还是没有解决8°，9°仰角数据面积过大的情况
class RadarCNN(nn.Module):
    def __init__(self, num_layers=3, base_channels=32):
        super().__init__()
        self.num_layers = num_layers

        # 每层共享的卷积处理（处理2通道数据）
        self.conv_shared = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        # 残差块
        self.res_block = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        # ! NEW 注意力输入维度从1->3（差异方向+绝对差异+有效数据比例）
        self.layer_attn_fc = nn.Sequential(
            nn.Linear(3, base_channels // 2),  # 输入特征维度增加
            nn.ReLU(),
            nn.Linear(base_channels // 2, 1),
            nn.Sigmoid()
        )
        # 回归头保持原样
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, elevations, target_elevation):
        B, L, H, W, C = x.size()
        features = []
        attn_weights = []

        # ! NEW 提取掩码并计算有效数据比例
        mask = x[..., 1]  # [B, L, H, W]
        valid_ratio = mask.mean(dim=(-1, -2))  # [B, L]

        for i in range(L):
            # ! NEW 预处理回波数据（将无效区域置零）
            layer_echo = x[:, i, ..., 0] * mask[:, i]  # [B, H, W]
            layer_mask = x[:, i, ..., 1]  # [B, H, W]
            layer_data = torch.stack([layer_echo, layer_mask], dim=1)  # [B, 2, H, W]

            # 特征提取部分保持不变
            feat = self.conv_shared(layer_data)
            residual = feat
            feat = self.res_block(feat) + residual
            feat = F.relu(feat)
            features.append(feat)

            # ! NEW 计算三重注意力特征 ------------------------
            # 带符号的仰角差
            signed_diff = elevations[:, i] - target_elevation.squeeze(1)  # [B]
            # 绝对差异
            abs_diff = torch.abs(signed_diff)  # [B]
            # 有效数据比例
            valid = valid_ratio[:, i]  # [B]
            # 组合特征
            attn_feats = torch.stack([
                signed_diff,
                abs_diff,
                valid
            ], dim=1)  # [B, 3]

            attn = self.layer_attn_fc(attn_feats)  # [B, 1]
            attn_weights.append(attn)

        # 特征融合部分保持结构不变
        features = torch.stack(features, dim=1)  # [B, L, C, H, W]
        attn_weights = torch.stack(attn_weights, dim=1)  # [B, L, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # 维度扩展保持原方式
        attn_weights = attn_weights.view(B, L, 1, 1, 1)  # [B, L, 1, 1, 1]
        weighted_features = (features * attn_weights).sum(dim=1)  # [B, C, H, W]

        out = self.regressor(weighted_features).squeeze(1)
        return out

#还可以的新模型：
class LayerAttention(nn.Module):
    """层注意力机制"""

    def __init__(self, num_layers, reduction=1):  # 将 reduction 调整为1
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_layers, num_layers // reduction),
            nn.ReLU(),
            nn.Linear(num_layers // reduction, num_layers),
            nn.Sigmoid()
            # nn.Softmax(dim=1)  # 让注意力更聚焦
        )

    def forward(self, x_list):
        # x_list: 各层的特征图列表 [ (B,C,H,W) ]
        batch_size = x_list[0].shape[0]

        # 计算各层权重
        weights = []
        for x in x_list:
            w = self.avgpool(x).view(batch_size, -1)  # (B, C)
            weights.append(w.mean(1, keepdim=True))  # (B, 1)
        weights = torch.cat(weights, dim=1)  # (B, L)

        # 通过全连接生成注意力权重
        attn = self.fc(weights)  # (B, L)
        return attn

class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.bn(x + residual)
        return F.relu(x)

class RadarResNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # 残差块
        # self.res_blocks = nn.Sequential(
        #     ResidualBlock(base_channels),
        #     ResidualBlock(base_channels),
        #     ResidualBlock(base_channels)
        # )
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels),
        )

        # 层注意力机制
        self.layer_attn = LayerAttention(num_layers=3)  # 假设使用3个输入层

        # 输出层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 1)
        )

    def forward(self, x):
        # x输入形状: (B, L, H, W) → 分割各层
        layer_features = []
        for i in range(x.shape[1]):
            layer = x[:, i]  # (B, H, W)
            layer = self.init_conv(layer.unsqueeze(1))  # (B, C, H, W)
            layer = self.res_blocks(layer)
            layer_features.append(layer)

        # 计算层注意力权重
        attn_weights = self.layer_attn(layer_features)  # (B, L)

        # 加权融合
        combined = torch.zeros_like(layer_features[0])
        for i, (feat, w) in enumerate(zip(layer_features, attn_weights.unbind(1))):
            combined += feat * w.view(-1, 1, 1, 1)

        # 最终预测
        return self.fc(combined).squeeze(1)

# class ResidualBlock(nn.Module):
#     """残差块"""
#
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 1)
#         self.bn = nn.BatchNorm2d(in_channels)
#
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.conv1(x))
#         x = self.conv2(x)
#         x = self.bn(x + residual)
#         return F.relu(x)

# class RadarResNet(nn.Module):
#     def __init__(self, in_channels=3, base_channels=64):
#         super().__init__()
#         # 初始卷积
#         self.init_conv = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, 3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU()
#         )
#
#         # 残差块
#         self.res_blocks = nn.Sequential(
#             ResidualBlock(base_channels),
#             ResidualBlock(base_channels),
#             ResidualBlock(base_channels)
#         )
#
#         # 输出层
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(base_channels, 1)
#         )
#
#     def forward(self, x):
#         # x输入形状: (B, L, H, W) → 分割各层
#         layer_features = []
#         for i in range(x.shape[1]):
#             layer = x[:, i]  # (B, H, W)
#             layer = self.init_conv(layer.unsqueeze(1))  # (B, C, H, W)
#             layer = self.res_blocks(layer)
#             layer_features.append(layer)
#
#         # 直接平均融合所有层（去掉注意力机制）
#         combined = sum(layer_features) / len(layer_features)
#
#         # 最终预测
#         return self.fc(combined).squeeze(1)

# 残差块，以下是残差网络，估计后续可能不是很能用得上了，我决定先删除，后续如果需要再说，gitee上0302的备份有网络代码
