import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from sklearn.metrics import explained_variance_score  # 导入解释方差计算函数
# from calculate_for_nomask import load_and_process_data, combine_data_from_files  # 根据实际情况调整导入
from calculate import load_and_process_data, combine_data_from_files  # 根据实际情况调整导入
from torch.utils.data import DataLoader, TensorDataset

# 从train.py中导入必要的函数和模型定义
from train import (split_dataset, prepare_dataloader,
                   RadarResNet, normalize, device)  # 请确保这些导入与您的实际代码匹配

def load_trained_model(model_path, input_channels=1):
    """加载预训练模型"""
    model = RadarResNet(in_channels=input_channels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# def filter_invalid_values(predictions, targets, invalid_value=-10.0):
#     """过滤预测值和实际值中的无效值"""
#     valid_mask = (targets != invalid_value) & (predictions != invalid_value)
#     return predictions[valid_mask], targets[valid_mask]
def filter_invalid_values(predictions, targets, invalid_threshold=-5.0):
    """过滤预测值和实际值中小于指定阈值的无效值"""
    valid_mask = (targets >= invalid_threshold) & (predictions >= invalid_threshold)
    return predictions[valid_mask], targets[valid_mask]


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    # 计算解释方差
    explained_variance = explained_variance_score(targets, predictions)
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    covariance = torch.cov(torch.stack([predictions, targets], dim=0))
    correlation = covariance[0, 1] / (torch.std(predictions) * torch.std(targets))
    return explained_variance, rmse.item(), correlation.item()


if __name__ == "__main__":
    # 配置参数
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # MODEL_PATH = "best_models_resnet/best_model7.0.pth"
    MODEL_PATH = "best_models_forplot/best_model_0.89dbz.pth"
    INVALID_VALUE = -5.0  # 根据实际情况调整无效值

    # 加载数据 (使用和训练时相同的处理流程)
    # root_path = "E:/福建数据/spar/04/test1/"
    root_path = "E:/福建数据/spar/04/output_v/"

    # 准备文件对列表
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),#针对高仰角数据不足添加
        # ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',#11°以上建模使用
        # 'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv')
        # ... 更多文件对
    ]
    # file_pairs = [
    #     ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
    #      'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
    #     ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
    #      'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
    #     ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
    #      'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
    #     ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
    #      'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),  # 针对高仰角数据不足添加
    #     ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',  # 11°以上建模使用
    #      'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
    #     ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
    #      'Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv')
    #     # ... 更多文件对
    # ]

    try:
        features, labels = combine_data_from_files(
            root_path=root_path,
            file_list=file_pairs,
            angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
            selected_layers=[0, 1, 2, 3, 4],  # 第一层
            # selected_layers=[4, 5, 6, 7, 8],#第二层
            # selected_layers=[7, 8, 9, 10, 11, 12, 13, 14],#第三层
            # selected_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 第四层
            selected_range_indices=[0, 1, 2],  # 处理选择的区间,低层用了两个，高层用了3个
            elevations=[0.5, 0.89, 1.45, 2.4],  # 仰角
            # elevations=[2.4, 3.0, 3.5, 4.3],
            # elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            # elevations=[10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5],
            target_layer_idx=1,
            max_bins=1520,
            grid_size=3
        )

        # 数据预处理
        features = features.reshape(-1, 3, 3, 3)
        features = features / 100.0
        labels = labels / 100.0

        # X_min, X_max = -30, 30  # 你当前数据的范围
        # features = (features - X_min) / (X_max - X_min) * 2 - 1  # 归一化到 [-1, 1]
        # labels = (labels - X_min) / (X_max - X_min) * 2 - 1  # 归一化到 [-1, 1]

        # 划分数据集
        # _, test_dataset = split_dataset(features, labels)
        # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # test_loader = DataLoader(TensorDataset(torch.tensor(features), torch.tensor(labels)), batch_size=64,
        #                          shuffle=False)
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(features, dtype=torch.float32),  # 确保数据类型匹配
                torch.tensor(labels, dtype=torch.float32)
            ),
            batch_size=64,
            shuffle=False
        )

        # 加载模型
        model = load_trained_model(MODEL_PATH)

        # 进行预测
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                preds = model(inputs).squeeze()

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        # 合并结果并转换为numpy
        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        # #反归一化
        # predictions = (predictions + 1) / 2 * (X_max - X_min) + X_min
        # targets = (targets + 1) / 2 * (X_max - X_min) + X_min

        # 反归一化（如果训练时做了归一化）
        # predictions = predictions * 100.0  # 根据训练时的缩放调整
        # targets = targets * 100.0

        # 过滤无效值
        # valid_preds, valid_targets = filter_invalid_values(predictions, targets, INVALID_VALUE)
        valid_preds = predictions
        valid_targets = targets

        # 计算指标
        explained_variance, rmse, corr = calculate_metrics(torch.tensor(valid_preds),
                                            torch.tensor(valid_targets))
        metrics = {
            "explained_variance": explained_variance,
            "rmse": rmse,
            "correlation": corr
        }
        print(f"评估指标: explained_variance={explained_variance:.4f}, RMSE={rmse:.4f}, R={corr:.4f}")

        # 绘制图形
        # plot_density_scatter(valid_preds, valid_targets, metrics)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

    # 保存示例 (在原始代码最后添加)
    # np.savez(
    #     f"model_data/model_vr/9.0.npz",
    #     predictions=valid_preds,
    #     targets=valid_targets,
    #     metrics=np.array([explained_variance, rmse, corr])
    # )