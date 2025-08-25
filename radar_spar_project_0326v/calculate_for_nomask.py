import numpy as np
import pandas as pd
import math
from calculate_for_nan import calculate_relative_shifts, align_radar_data, extract_to_ndarray, interpolate_layer

MIN_DBZ = -500
TRUE_FILL = -1000  # 统一定义空值
FILL_VALUE = -32768
# 修改后的extract_3x3_matrix函数
def extract_3x3_matrix(layer_data, i, j, grid_size=3):
    half_size = grid_size // 2
    if (i - half_size < 0 or i + half_size >= layer_data.shape[0] or
            j - half_size < 0 or j + half_size >= layer_data.shape[1]):
        return None

    matrix = layer_data[i - half_size:i + half_size + 1,
             j - half_size:j + half_size + 1].copy()

    # 统一空值处理逻辑
    matrix = np.where(np.isnan(matrix), TRUE_FILL, matrix)  # 处理NaN
    matrix[matrix < MIN_DBZ] = TRUE_FILL  # 替换无效值

    return matrix


# 修改后的create_dataset函数（单通道版本）
def create_dataset(layer_arrays, distance_shifts, layer_indices, elevations,
                   target_layer_idx, max_bins=3200, grid_size=3):
    feature_data = []
    label_data = []

    target_layer = layer_arrays[target_layer_idx]

    for i in range(grid_size // 2, target_layer.shape[0] - grid_size // 2):
        for j in range(grid_size // 2, min(target_layer.shape[1] - grid_size // 2, max_bins)):
            # 处理标签
            label = target_layer[i, j]
            label = TRUE_FILL if label <= MIN_DBZ else label

            # 适量保留空值数据（限制最大比例为 20%）
            if label == TRUE_FILL and np.random.rand() > 0.3:
                continue  # 丢弃大部分空值数据，只保留 20%

            # 适量生成纯空数据（只占 5%）
            if np.random.rand() < 0.05:
                feature = np.full((len(layer_indices), grid_size, grid_size), TRUE_FILL, dtype=np.float32)
                feature_data.append(feature)
                label_data.append(TRUE_FILL)
                continue

            # 处理特征
            feature = []
            valid = True

            for layer_idx, elv in zip(layer_indices, elevations):
                shifted_j = j + distance_shifts[elv][j]

                # 越界检查
                if shifted_j < 0 or shifted_j >= layer_arrays[layer_idx].shape[1]:
                    valid = False
                    break

                matrix = extract_3x3_matrix(layer_arrays[layer_idx], i, shifted_j, grid_size)
                if matrix is None:
                    valid = False
                    break

                # 数据增强（随机遮挡）
                if np.random.rand() < 0.1:
                    mask = np.random.choice([0, 1], (grid_size, grid_size), p=[0.3, 0.7])
                    matrix = np.where(mask, matrix, TRUE_FILL)

                feature.append(matrix)

            if valid:
                feature_data.append(np.array(feature))
                label_data.append(label)

    return np.array(feature_data), np.array(label_data)

def load_and_process_data(
    root_path,
    filename_ZH,
    filename_CC,
    angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],  # 方位角范围
    selected_layers=None, #[0, 1, 2, 3, 4],  # 选择的层索引
    selected_range_indices=None, #[0],  # 选择的区间索引
    elevations= None, #[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 9.9],  # 仰角列表
    target_layer_idx=1,  # 目标层索引
    max_bins=1520,  # 最大距离库数
    grid_size=3  # 特征矩阵大小
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

    # # 利用 CC 数据质控
    # 改进的数据质控
    # filtered_zh = data_ZH.where(data_CC > 0.8, FILL_VALUE)
    # # 空值替换为最小有效值
    # filtered_zh = filtered_zh.replace(FILL_VALUE, MIN_DBZ)
    # Step1: 标记无效值
    mask = (data_CC <= 0.8) | (data_ZH <= FILL_VALUE)
    # Step2: 替换无效值为合理值
    filtered_zh = data_ZH.where(~mask, TRUE_FILL)  # 仅替换无效点

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
    # combined_array = np.round(interpolate_layer(combined_array, 2, 3, 2, 1.45, 1.29, 1.79))#第一次
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
        grid_size=grid_size
    )

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
    grid_size=3  # 特征矩阵大小
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
                grid_size=grid_size
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