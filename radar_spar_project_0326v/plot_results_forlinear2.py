from plot_results_forlinear import combine_data_from_files
import numpy as np
import pandas as pd
from plot_results import calculate_metrics
import torch

def filter_invalid_values(predictions, targets, invalid_threshold=-5.0):
    """过滤预测值和实际值中小于指定阈值的无效值"""
    valid_mask = (targets >= invalid_threshold) & (predictions >= invalid_threshold)
    return predictions[valid_mask], targets[valid_mask]

def linear_interpolation(feature_data, elevations, target_elv):
    """
    对选定的两层数据进行线性插值，计算目标层的值。

    参数:
        feature_data (np.ndarray): 形状为 (n, 2)，包含选出的两层数据
        elevations (list): 选中两层的仰角，如 [6.0, 7.0]
        target_elv (float): 目标层的仰角，如 6.5

    返回:
        np.ndarray: 目标层的插值结果，形状为 (n,)
    """
    elv1, elv2 = elevations  # 选定的两层仰角
    feature1, feature2 = feature_data[:, 0], feature_data[:, 1]  # 取出两层数据

    # 线性插值计算目标层
    target_values = ((elv2 - target_elv) * feature1 + (target_elv - elv1) * feature2) / (elv2 - elv1)

    return target_values  # 返回插值后的目标层数据

if __name__ == "__main__":
    # 定义数据路径和文件名
    INVALID_VALUE = -5.0  # 根据实际情况调整无效值
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
        ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),  # 针对高仰角数据不足添加
        ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',#11°以上建模使用
        'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv')
        # ... 更多文件对
    ]

    # 加载和处理数据
    # features, labels = combine_data_from_files(root_path, filename_ZH, filename_CC)
    try:
        # 处理所有文件并合并数据
        features, labels = combine_data_from_files(
            root_path=root_path,
            file_list=file_pairs,
            angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
            # selected_layers=[0, 1, 2, 3, 4],#第一层
            # selected_layers=[4, 5, 6, 7, 8],#第二层
            selected_layers=[7, 8, 9, 10, 11, 12, 13, 14],  # 第三层
            # selected_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 第四层
            selected_range_indices=[0, 1, 2],  # 处理选择的区间,低层用了两个，高层用了3个
            # elevations=[0.5, 0.89, 1.45, 2.4],  # 仰角
            # elevations=[2.4, 3.0, 3.5, 4.3],
            elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            # elevations=[10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5],
            target_layer_idx=5,
            max_bins=1520,
        )

        print("\n数据处理完成！")
        print(f"最终数据集大小: {len(labels)} 样本")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

    features = features / 100.0
    targets = labels / 100.0

    # 选择索引 0 和 2 作为插值层（你需要手动指定）
    selected_indices = [1, 2]  # 例如，你希望用 features 第 1 和第 3 层
    selected_elevations = [6.0, 10.0]  # 这两层的仰角
    target_elevation = 9.0  # 目标插值层

    # 手动索引选定的两层
    selected_features = features[:, selected_indices]  # 变成 (n, 2)

    # 进行线性插值
    predictions = linear_interpolation(selected_features, selected_elevations, target_elevation)

    # valid_preds, valid_targets = filter_invalid_values(predictions, targets, INVALID_VALUE)#dbz用
    valid_preds = predictions
    valid_targets = targets

    explained_variance, rmse, corr = calculate_metrics(torch.tensor(valid_preds), torch.tensor(valid_targets))

    metrics = {
        "explained_variance": explained_variance,
        "rmse": rmse,
        "correlation": corr
    }
    print(f"评估指标: explained_variance={explained_variance:.4f}, RMSE={rmse:.4f}, R={corr:.4f}")

    # np.savez(
    #     f"model_data/linear_vr/9.0.npz",
    #     predictions=valid_preds,
    #     targets=valid_targets,
    #     metrics=np.array([explained_variance, rmse, corr],dtype=np.float32)
    # )