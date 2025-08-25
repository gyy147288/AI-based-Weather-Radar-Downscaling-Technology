import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from model import RadarCNN, RadarResNet  # 导入模型定义

def plot_predictions_vs_targets(model_path, dataloader):
    """
    加载保存的模型权重，绘制模型预测值与真实值的散点图，并叠加高斯核密度估计分布。

    参数:
    - model_path: 已保存模型权重的路径
    - dataloader: 数据加载器
    """
    # 初始化模型并加载权重
    model = RadarCNN()  # 替换为你的模型定义
    # model = RadarResNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    all_targets = []
    all_predictions = []

    # 关闭梯度计算，避免不必要的开销
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs).squeeze()  # 获取预测值
            all_targets.append(targets)
            all_predictions.append(outputs)

    # 拼接所有批次的预测值和真实值
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_predictions = torch.cat(all_predictions).cpu().numpy()

    all_targets /= 100
    all_predictions /= 100

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_targets, y=all_predictions, alpha=0.5, edgecolor=None, label='Scatter Points')

    # 添加高斯核密度估计
    sns.kdeplot(x=all_targets, y=all_predictions, cmap="Blues", fill=True, alpha=0.5, levels=20, label='KDE Contours')

    # 添加辅助线
    max_val = max(all_targets.max(), all_predictions.max())
    min_val = min(all_targets.min(), all_predictions.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')

    # 设置坐标轴范围
    plt.xlim(-15, max_val)  # 设置横轴从-15开始
    plt.ylim(-15, max_val)  # 设置纵轴从-15开始

    # 设置图例和标签
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    # plt.title('Resnet-18', fontsize=16)
    plt.title('CNN', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用时调用该函数
if __name__ == "__main__":
    from calculate import load_and_process_data, combine_data_from_files # 导入数据处理函数

    # 定义数据路径和文件名
    root_path = 'E:/福建数据/spar/04/test1/'
    filename_ZH = 'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv'
    filename_CC = 'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_Z9600_20240430060124_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_Z9600_20240430060124_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),#有电磁干扰
        ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv')#针对高仰角数据不足添加
        # ... 更多文件对
    ]

    # 加载和处理数据
    # features, labels = load_and_process_data(root_path, filename_ZH, filename_CC)
    features, labels = combine_data_from_files(
        root_path=root_path,
        file_list=file_pairs,
        angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
        selected_layers=[0, 1, 2, 3, 4],  # 第一层
        # selected_layers=[4, 5, 6, 7, 8],#第二层
        # selected_layers=[7, 8, 9, 10, 11, 12, 13, 14],#第三层
        # selected_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 第四层
        selected_range_indices=[0, 2],  # 处理选择的区间,低层用了两个，高层用了3个
        elevations=[0.5, 0.89, 1.45, 2.4],  # 仰角
        # elevations=[2.4, 3.0, 3.5, 4.3],
        # elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        # elevations=[10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5],
        target_layer_idx=1,
        max_bins=1520,
        grid_size=3
    )

    # 调整形状为 (batch_size, channels, height, width)
    features = features.reshape(-1, 3, 3, 3)

    # 准备数据
    from torch.utils.data import DataLoader, TensorDataset
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 调用绘图函数
    model_path = "best_models_cnn/best_model0.89.pth"  # 替换为你的模型路径
    # model_path = "best_models_resnet/best_model0.89.pth"  # 替换为你的模型路径
    plot_predictions_vs_targets(model_path, dataloader)

