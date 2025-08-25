import numpy as np
import pandas as pd
import torch
from calculate import align_radar_data, extract_to_ndarray, interpolate_layer, calculate_relative_shifts
from sklearn.metrics import explained_variance_score
import os

# === 常量定义 ===
MIN_DBZ = -3000
TRUE_FILL = -1000  # 统一定义空值
FILL_VALUE = -32768


def create_dataset(layer_arrays, distance_shifts, layer_indices, elevations, target_layer_idx, max_bins=3200):
    """
    构建简化版数据集 (仅使用对应点值)

    Args:
        layer_arrays: 雷达数据数组
        distance_shifts: 距离库偏移字典
        layer_indices: 要使用的层索引列表，如 [0, 2, 3]
        elevations: 对应的仰角列表，如 [0.5, 1.45, 2.4]
        target_layer_idx: 目标层索引
        max_bins: 最大距离库数

    Returns:
        tuple: (特征数据, 标签数据)
    """
    feature_data = []
    label_data = []

    # 统计计数器
    total_points = 0
    valid_samples = 0

    target_layer = layer_arrays[target_layer_idx]

    # 检查输入参数
    if len(layer_indices) != len(elevations):
        raise ValueError("层索引列表和仰角列表长度必须相同")

    # 遍历数据点
    for i in range(target_layer.shape[0]):  # 方位角维度
        for j in range(min(target_layer.shape[1], max_bins)):  # 距离库维度
            total_points += 1

            # 获取标签值
            label = target_layer[i, j]

            # 跳过无效标签
            if label == FILL_VALUE or label < MIN_DBZ:
                continue

            # 初始化特征向量
            feature_vector = []
            valid_feature = True

            # 提取各层特征
            for layer_idx, elv in zip(layer_indices, elevations):
                # 计算偏移后的距离库索引
                shifted_j = j + distance_shifts[elv][j]

                # 检查索引有效性
                if (shifted_j < 0 or
                        shifted_j >= layer_arrays[layer_idx].shape[1] or
                        i >= layer_arrays[layer_idx].shape[0]):
                    valid_feature = False
                    break

                # 获取特征值
                feature_value = layer_arrays[layer_idx][i, shifted_j]

                # 检查特征有效性
                if feature_value == FILL_VALUE or feature_value < MIN_DBZ:
                    valid_feature = False
                    break

                feature_vector.append(feature_value)

            if valid_feature:
                feature_data.append(feature_vector)
                label_data.append(label)
                valid_samples += 1

    # 转换数据格式
    feature_array = np.array(feature_data)
    label_array = np.array(label_data)

    # 数据统计
    print("\n=== 数据统计 ===")
    print(f"总数据点: {total_points}")
    print(f"有效样本数: {valid_samples} ({valid_samples / total_points:.1%})")
    print(f"特征形状: {feature_array.shape}")
    print(f"标签形状: {label_array.shape}")

    return feature_array, label_array

def load_and_process_data(
    root_path,
    filename_ZH,
    filename_CC,
    angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],  # 方位角范围
    selected_layers=[0, 1, 2, 3, 4],  # 选择的层索引
    selected_range_indices=[0],  # 选择的区间索引
    elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 9.9],  # 仰角列表
    target_layer_idx=1,  # 目标层索引
    max_bins=1520,  # 最大距离库数
):
    """
    加载并处理雷达数据，支持灵活使用多个区间。

    参数:
        root_path (str): 数据文件根目录。
        filename_ZH (str): ZH 数据文件名。
        filename_CC (str): CC 数据文件名。
        angle_ranges (list of tuple): 方位角范围列表，每个范围是一个元组 (start, end)。
        selected_layers (list of int): 选择的层索引列表。
        selected_range_indices (list of int): 选择的区间索引列表。
        elevations (list of float): 仰角列表。
        target_layer_idx (int): 目标层索引。
        max_bins (int): 最大距离库数。
        grid_size (int): 特征矩阵大小。

    返回:
        feature_data (np.ndarray): 特征数据。
        label_data (np.ndarray): 标签数据。
    """
    # 构建文件路径
    file_path_ZH = root_path + filename_ZH
    file_path_CC = root_path + filename_CC

    # 加载数据
    data_ZH = pd.read_csv(file_path_ZH, header=None)
    data_CC = pd.read_csv(file_path_CC, header=None)

    # data_ZH = data_ZH / 100  # 将 ZH 数据缩小100倍

    # 利用 CC 数据质控
    filtered_zh = data_ZH.where(data_CC > 0.8, -32768)

    # 对齐数据
    aligned_data = align_radar_data(filtered_zh, angle_ranges)

    # 处理多个区间
    all_arrays = []
    for range_idx in selected_range_indices:
        # 提取指定区间的数据
        array = extract_to_ndarray(aligned_data, selected_layers, range_idx)
        array = array[:, :, 1:]  # 去掉第一列（方位角）
        all_arrays.append(array)

    # 合并多个区间的数据
    combined_array = np.concatenate(all_arrays, axis=1)
    print(f"合并后的数据形状: {combined_array.shape}")

    # 插值处理
    combined_array = np.round(interpolate_layer(combined_array, 0, 1, 0, 4.3, 4.0, 4.5))#第三次
    # combined_array = np.round(interpolate_layer(combined_array, 2, 3, 2, 1.45, 1.29, 1.79))  # 第一次
    # combined_array = np.round(interpolate_layer(combined_array, 3, 4, 3, 4.3, 4.0, 4.5))#第二次
    # combined_array = np.round(interpolate_layer(combined_array, 4, 5, 4, 14.6, 14.0, 15.0))#第四次两次插值
    # combined_array = np.round(interpolate_layer(combined_array, 8, 9, 8, 19.5, 19.0, 20.0))
    print(f"插值后的数据形状: {combined_array.shape}")

    # 计算距离库偏移
    all_distances = np.linspace(0, 200, 3200)  # 全部距离范围
    selected_distances = all_distances[80:1600]  # 选择第 80 到第 1600 个库
    distance_shifts = calculate_relative_shifts(selected_distances, elevations)

    # 提取特征和标签数据
    feature_data, label_data = create_dataset(
        combined_array,
        distance_shifts,
        # layer_indices=[0, 2, 3],  # 选择的特征数据层索引1
        # layer_indices=[0, 2, 3],  # 选择的特征数据层索引2
        layer_indices=[0, 2, 6],  # 选择的特征数据层索引3
        # layer_indices=[0, 4, 8],  # 选择的特征数据层索引4
        # elevations=[0.5, 1.45, 2.4],  # 特征数据仰角列表1
        # elevations=[2.4, 3.5, 4.3],  # 特征数据仰角列表2
        elevations=[4.3, 6.0, 10],  # 特征数据仰角列表3
        # elevations=[10, 14.6, 19.5],  # 特征数据仰角列表4
        target_layer_idx=target_layer_idx,
        max_bins=max_bins,
    )

    # feature_data = feature_data / 100
    # label_data = label_data / 100

    return feature_data, label_data

def combine_data_from_files(
    root_path,
    file_list,
    angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],  # 方位角范围
    selected_layers=[0, 1, 2, 3, 4],  # 选择的层索引
    selected_range_indices=[0],  # 选择的区间索引
    elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 9.9],  # 仰角列表
    target_layer_idx=1,  # 目标层索引
    max_bins=1520,  # 最大距离库数
):
    """
    从多个文件读取并合并数据。

    Args:
        root_path (str): 数据文件根目录。
        file_list (list of tuple): 文件对列表，每对是 (ZH文件名, CC文件名) 的元组。
        angle_ranges (list of tuple): 方位角范围列表，每个范围是一个元组 (start, end)。
        selected_layers (list of int): 选择的层索引列表。
        selected_range_indices (list of int): 选择的区间索引列表。
        elevations (list of float): 仰角列表。
        target_layer_idx (int): 目标层索引。
        max_bins (int): 最大距离库数。
        grid_size (int): 特征矩阵大小。

    Returns:
        combined_features (np.ndarray): 合并后的特征数据。
        combined_labels (np.ndarray): 合并后的标签数据。
    """
    all_features = []
    all_labels = []

    total_files = len(file_list)

    for idx, (zh_file, cc_file) in enumerate(file_list):
        try:
            print(f"\n处理文件 {idx + 1}/{total_files}")
            print(f"正在处理: {zh_file} 和 {cc_file}")

            # 使用更新后的 load_and_process_data 函数处理单个文件对
            features, labels = load_and_process_data(
                root_path=root_path,
                filename_ZH=zh_file,
                filename_CC=cc_file,
                angle_ranges=angle_ranges,
                selected_layers=selected_layers,
                selected_range_indices=selected_range_indices,
                elevations=elevations,
                target_layer_idx=target_layer_idx,
                max_bins=max_bins,
            )

            # 将结果添加到列表
            all_features.append(features)
            all_labels.append(labels)

            print(f"成功处理文件对 {idx + 1}")
            print(f"当前文件特征形状: {features.shape}")
            print(f"当前文件标签形状: {labels.shape}")

        except Exception as e:
            print(f"处理文件 {zh_file} 时出错: {str(e)}")
            continue

    # 合并所有数据
    if all_features and all_labels:
        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        print("\n=== 合并后的数据集信息 ===")
        print(f"总样本数: {len(combined_labels)}")
        print(f"特征形状: {combined_features.shape}")
        print(f"标签形状: {combined_labels.shape}")

        return combined_features, combined_labels
    else:
        raise ValueError("没有成功处理任何文件")

if __name__ == "__main__":
    # 配置参数
    config = {
        'angle_ranges': [(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
        # 'selected_layers': [0, 1, 2, 3],  # 对应 4.0° - 10.0°
        'selected_layers': [7, 8, 9, 10, 11, 12, 13, 14],
        'selected_range_indices': [0, 1, 2]
    }

    # 文件列表
    root_path = "E:/福建数据/spar/04/test1/"
    file_list = [
        "Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv",
        "Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv",
        "Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv",
        "Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv",
    ]

    # 选择导出格式 ("npz" 或 "csv")
    export_format = "npz"  # 或 "csv"

    # 执行处理 & 导出
    # process_and_merge_files(root_path, file_list, config, export_format)



