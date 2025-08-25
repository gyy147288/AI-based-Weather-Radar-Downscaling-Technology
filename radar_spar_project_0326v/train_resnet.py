import numpy as np
import pandas as pd
import math

#基本可能用不到了，某一个版本的整理数据备份
# 新增常量定义
MIN_DBZ = -1500.0 #-1678.0  # 根据实际情况设置的最小有效值
FILL_VALUE = -32768
MASK_VALUE = -9999  # 新增掩码标记值
COEFFICIENT_SCALE = 1000  # 用于恢复相关系数的缩放因子
DBZ_SCALE = 100  # 用于恢复dbz的缩放因子

#计算对应库
def cal_radar_gauge(d, beta):
    beta = math.radians(beta)      #仰角
    R_earth = 6371                  #地球半径
    eq_R_earth = (4.0 / 3.0) * R_earth  #km 等效地球半径
    h_0 = 1.079                         #km雷达海拔高度
    a = d / eq_R_earth                  #地心角
    # alpha = (4.0 / 3.0) * a             #等效地心角
    alpha = a                          #好像这样是对的
    # r = 1
    r = 0.0625                    # 库长 62.5m
    G = ((eq_R_earth + h_0) * np.sin(alpha)) / (r * np.cos(alpha + beta))

    return G

def calculate_distance_shifts(distances, elvs):
    """
    计算不同仰角层的距离库偏移。
    参数:
        distances (list): 距离范围（单位：km）。
        elvs (list): 仰角列表（单位：度）。
    返回:
        dict: 每个仰角对应的距离库偏移。
    """
    shifts = {}
    for elv in elvs:
        shifts[elv] = [int(cal_radar_gauge(d, elv)) for d in distances]
    return shifts

def calculate_relative_shifts(distances, elvs, reference_elv=0.5):
    """
    计算相对库偏移。
    参数:
        distances (list): 距离范围（单位：km）。
        elvs (list): 仰角列表（单位：度）。
        reference_elv (float): 参考仰角（默认最低仰角）。
    返回:
        dict: 每个仰角相对参考仰角的库偏移。
    """
    reference_shifts = [int(cal_radar_gauge(d, reference_elv)) for d in distances]
    shifts = {}
    for elv in elvs:
        current_shifts = [int(cal_radar_gauge(d, elv)) for d in distances]
        shifts[elv] = [current - ref for current, ref in zip(current_shifts, reference_shifts)]
    return shifts

#分仰角层输出一个时次的雷达数据
def process_radar_data(dataarray, layer_n):  # 输出第几层的数据，从第一层开始 第一层:0.5°仰角
    layers = []
    current_layer = []
    prev_angle = -np.inf

    for idx, row in dataarray.iterrows():
        angle = row[0]
        # 提取每行的第一个值，即为方位角度数
        if angle < prev_angle:  # 若切换则存储该层到layers列表中
            layers.append(pd.DataFrame(current_layer))  # 判断是否切换层
            current_layer = []  # 若切换则存储该层到layers列表中

        current_layer.append(row)  # 没切换层则添加到当前层列表中
        prev_angle = angle
        # 存储到过去方位角数据，循环后与下一个方位角比较用来判断是否切换层

    if current_layer:
        layers.append(pd.DataFrame(current_layer))

    return layers, layers[layer_n - 1]

def align_radar_data(dataarray, angle_ranges):
    """
    对齐不同仰角的雷达数据，使得它们在每个范围内的径向数相同。

    Args:
        dataarray (pd.DataFrame): 包含雷达数据的 DataFrame，每行是一条径向数据，第一列为仰角。
        angle_ranges (list of tuple): 方位角范围列表，每个范围是一个元组 (start, end)。

    Returns:
        list of list: 对齐后的所有仰角数据，每个仰角是一个 DataFrame 列表。
    """

    # 提取所有仰角数据
    layers, _ = process_radar_data(dataarray, 1)  # 提取所有层
    all_layers_data = [process_radar_data(dataarray, i + 1)[1] for i in range(len(layers))]

    # 计算每个范围的最小径向数
    selected_rows_shapes = []
    for angle_range in angle_ranges:
        min_rows = np.inf
        for layer_data in all_layers_data:
            # 统计该范围内的径向数
            rows_in_range = layer_data[
                (layer_data.iloc[:, 0] >= angle_range[0]) & (layer_data.iloc[:, 0] <= angle_range[1])
            ]
            min_rows = min(min_rows, len(rows_in_range))
        selected_rows_shapes.append(min_rows)

    # 对齐数据
    aligned_data = []
    for layer_data in all_layers_data:
        aligned_layer = []
        for range_idx, angle_range in enumerate(angle_ranges):
            rows_in_range = layer_data[
                (layer_data.iloc[:, 0] >= angle_range[0]) & (layer_data.iloc[:, 0] <= angle_range[1])
            ]
            if len(rows_in_range) > selected_rows_shapes[range_idx]:
                # 根据第一列的方位角值，选择最接近的数据
                closest_rows = rows_in_range.iloc[
                    np.argsort(
                        np.abs(
                            rows_in_range.iloc[:, 0].values
                            - np.median(rows_in_range.iloc[:, 0].values)
                        )
                    )[:selected_rows_shapes[range_idx]]
                ]
                aligned_layer.append(closest_rows.sort_index())
            else:
                aligned_layer.append(rows_in_range)
        aligned_data.append(aligned_layer)

    return aligned_data

def extract_to_ndarray(aligned_data, selected_layers, selected_range_idx):
    """
    从对齐后的数据中提取指定层和方位范围的数据，转换为三维 ndarray。

    Args:
        aligned_data (list of list): 对齐后的数据，每个仰角包含多个方位范围的数据。
        selected_layers (list of int): 要提取的层索引列表（从 0 开始）。
        selected_range_idx (int): 要提取的方位范围索引（从 0 开始）。

    Returns:
        np.ndarray: 提取后的三维数组，形状为 (len(selected_layers), rows, columns)。
    """
    extracted_data = []

    for layer_idx in selected_layers:
        # 获取指定层的数据
        layer_data = aligned_data[layer_idx][selected_range_idx]
        # 将 DataFrame 转换为 ndarray 并添加到结果中
        extracted_data.append(layer_data.values)

    # 将列表转换为三维 ndarray
    return np.array(extracted_data)


def interpolate_layer(data, lower_idx, upper_idx, new_idx, new_angle, lower_angle, upper_angle):
    """
    改进的插值函数，处理部分有效值的情况
    """
    lower_layer = data[lower_idx].astype(np.float32)
    upper_layer = data[upper_idx].astype(np.float32)

    # 创建有效值掩码
    lower_valid = (lower_layer > FILL_VALUE) & (lower_layer >= MIN_DBZ)
    upper_valid = (upper_layer > FILL_VALUE) & (upper_layer >= MIN_DBZ)

    # 计算权重
    weight = (new_angle - lower_angle) / (upper_angle - lower_angle)

    # 初始化插值层
    interpolated_layer = np.full_like(lower_layer, FILL_VALUE, dtype=np.float32)

    # 情况1: 两层都有效
    both_valid = lower_valid & upper_valid
    interpolated_layer[both_valid] = (
            lower_layer[both_valid] * (1 - weight) +
            upper_layer[both_valid] * weight
    )

    # 情况2: 仅下层有效
    only_lower = lower_valid & ~upper_valid
    interpolated_layer[only_lower] = lower_layer[only_lower]

    # 情况3: 仅上层有效
    only_upper = ~lower_valid & upper_valid
    interpolated_layer[only_upper] = upper_layer[only_upper]

    # 保留原始无效值
    interpolated_layer[interpolated_layer <= FILL_VALUE] = FILL_VALUE

    data[new_idx] = interpolated_layer
    return np.delete(data, upper_idx, axis=0)


def extract_3x3_matrix(layer_data, i, j, grid_size=3):
    """
    提取指定点 (i, j) 周围的 3x3 矩阵。
    如果三层的空值超过一半，放弃数据，否则填充空值为均值。
    """
    # half_size = grid_size // 2
    # if (i - half_size < 0 or i + half_size >= layer_data.shape[0] or
    #         j - half_size < 0 or j + half_size >= layer_data.shape[1]):
    #     return None  # 跳过边界点
    #
    # matrix = layer_data[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
    #
    # # 计算每层的有效值数量
    # valid_count = np.sum(matrix != FILL_VALUE)
    # if valid_count < (grid_size * grid_size) / 2:
    #     return None  # 空值超过一半时，跳过数据
    #
    # # 填充FILL_VALUE的空值为均值
    # nan_mask = matrix == FILL_VALUE
    # matrix[nan_mask] = np.nanmean(matrix[~nan_mask])  # 用有效值的均值填充空值
    #
    # return matrix

    """严格过滤含空值的样本"""
    half_size = grid_size // 2
    if (i - half_size < 0 or i + half_size >= layer_data.shape[0] or
            j - half_size < 0 or j + half_size >= layer_data.shape[1]):
        return None

    matrix = layer_data[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]

    # 完全不允许空值存在
    if np.any(matrix == FILL_VALUE):
        return None

    return matrix


def create_dataset(layer_arrays, distance_shifts, layer_indices, elevations, target_layer_idx, max_bins=3200,
                   grid_size=3):
    """
    构建数据集。
    """
    feature_data = []
    label_data = []
    target_layer = layer_arrays[target_layer_idx]

    # 遍历数据点
    for i in range(grid_size // 2, target_layer.shape[0] - grid_size // 2):
        for j in range(grid_size // 2, min(target_layer.shape[1] - grid_size // 2, max_bins)):
            feature_vector = []
            label = target_layer[i, j]

            # 检查标签的有效性
            target_matrix = extract_3x3_matrix(target_layer, i, j, grid_size)
            # if target_matrix is None or np.sum(target_matrix == FILL_VALUE) > (grid_size * grid_size) / 3:
            if target_matrix is None or np.sum(target_matrix == FILL_VALUE) > 0:
                continue  # 无效标签，跳过

            # 提取特征
            valid_feature = True
            for layer_idx, elv in zip(layer_indices, elevations):
                shifted_j = j + distance_shifts[elv][j]
                matrix = extract_3x3_matrix(layer_arrays[layer_idx], i, shifted_j, grid_size)
                if matrix is None or np.any(matrix == FILL_VALUE):
                    valid_feature = False
                    break
                feature_vector.append(matrix)

            if valid_feature:
                feature_data.append(np.array(feature_vector))
                label_data.append(label)

    return np.array(feature_data), np.array(label_data)

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
    filtered_zh = data_ZH.where(data_CC > 800, FILL_VALUE)
    # # 空值替换为最小有效值
    # filtered_zh = filtered_zh.replace(FILL_VALUE, MIN_DBZ)

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
    # combined_array = np.round(interpolate_layer(combined_array, 0, 1, 0, 4.3, 4.0, 4.5))#第三次
    combined_array = np.round(interpolate_layer(combined_array, 2, 3, 2, 1.45, 1.29, 1.79))#第一次
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
        layer_indices=[0, 2, 3],  # 选择的特征数据层索引1
        # layer_indices=[0, 2, 3],  # 选择的特征数据层索引2
        # layer_indices=[0, 2, 6],  # 选择的特征数据层索引3
        # layer_indices=[0, 4, 8],  # 选择的特征数据层索引4
        elevations=[0.5, 1.45, 2.4],  # 特征数据仰角列表1
        # elevations=[2.4, 3.5, 4.3],  # 特征数据仰角列表2
        # elevations=[4.3, 6.0, 10],  # 特征数据仰角列表3
        # elevations=[10, 14.6, 19.5],  # 特征数据仰角列表4
        target_layer_idx=target_layer_idx,
        max_bins=max_bins,
        grid_size=grid_size
    )

    feature_data = feature_data / 100
    label_data = label_data / 100

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

def process_interpolation_data(root_path, filename_ZH, filename_CC):    #插值
    file_path_ZH = root_path + filename_ZH
    file_path_CC = root_path + filename_CC

    # 加载数据
    data_ZH = pd.read_csv(file_path_ZH)
    data_CC = pd.read_csv(file_path_CC)

    # 利用 cc 数据质控
    filtered_zh = data_ZH.where(data_CC > 0.8, -32768)

    # 分成按层排列的数据
    # layers, layer_1 = process_radar_data(filtered_zh, 1)

    # 归整成三维数组并且计算中间层的线性插值
    angle_ranges = [
        (12000.0, 19000.0),  # 第一段
        (22500.0, 23500.0),  # 第二段
        (25500.0, 26500.0)  # 第三段
    ]

    aligned_data = align_radar_data(filtered_zh, angle_ranges)
    selected_layers = [0, 1, 2, 3, 4]  # 选择第 1、6、11 层（索引从 0 开始）
    selected_range_idx = 0  # 选择第 2 个方位范围（索引从 0 开始）

    array = extract_to_ndarray(aligned_data, selected_layers, selected_range_idx)
    array = array[:, :, 1:]
    array = np.round(interpolate_layer(array, 2, 3, 2, 1.45, 1.29, 1.79))

    return array