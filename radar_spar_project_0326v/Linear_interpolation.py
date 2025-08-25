# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from calculate import process_interpolation_data
from scipy.interpolate import interp1d

import numpy as np
import math
import pandas as pd
from run import calculate_relative_shifts, load_radar_layers, align_radar_layers

# BINS = 4000
BINS = 1000

#修改读取数据文件和仰角即可
def align_layers_by_shifts(layers, shifts, max_bins=None):
    """
    根据计算出的偏移量对两层数据进行对齐，低层数据超出上层范围时填充为空值。

    参数：
        layers (np.ndarray): 包含两层数据的数组，形状为 (2, M, N)，M为纬度，N为距离库
        shifts (dict): 计算出的偏移量，形如 {elevation: [shift1, shift2, ..., shiftM]}
        max_bins (int, optional): 最大距离库数量，如果提供，则用于限制对齐的列数

    返回：
        aligned_layers (np.ndarray): 对齐后的两层数据，形状为 (2, M, N)
    """
    aligned_layers = np.full_like(layers, np.nan)  # 初始化为 NaN

    # 获取数据的维度
    num_layers, num_latitudes, num_bins = layers.shape

    # 确定 max_bins
    max_bins = max_bins if max_bins is not None else num_bins

    for elv_idx, elv in enumerate(shifts):
        # 获取当前层的偏移量
        current_shifts = shifts[elv]
        layer = layers[elv_idx]

        for i in range(num_latitudes):  # 遍历纬度
            for j in range(num_bins):  # 遍历距离库
                shifted_j = j + current_shifts[i]  # 计算偏移后的位置

                # 如果偏移后的距离库在有效范围内，则赋值
                if 0 <= shifted_j < max_bins:
                    aligned_layers[elv_idx, i, shifted_j] = layer[i, j]

    return aligned_layers


import numpy as np
from scipy.interpolate import interp1d

def interpolate_layers(layers, target_elevation, shifts, max_bins=None):
    """
    对已经对齐的两层数据进行插值，插值的层是指定的 target_elevation。

    参数：
        layers (np.ndarray): 对齐后的两层数据，形状为 (2, M, N)
        target_elevation (float): 需要插值的目标仰角
        shifts (dict): 计算出的偏移量，形如 {elevation: [shift1, shift2, ..., shiftM]}
        max_bins (int, optional): 最大距离库数量，默认为 None，表示使用原数据的列数

    返回：
        interpolated_layer (np.ndarray): 插值后的数据，形状为 (M, N)
    """
    # 获取低层和上层的仰角数据
    elvs = list(shifts.keys())

    if target_elevation < min(elvs) or target_elevation > max(elvs):
        raise ValueError("目标仰角超出已知仰角范围")

    lower_elv = max([elv for elv in elvs if elv < target_elevation])  # 上一层
    upper_elv = min([elv for elv in elvs if elv > target_elevation])  # 下一层

    # 获取对应的两层数据
    lower_layer = layers[elvs.index(lower_elv)]
    upper_layer = layers[elvs.index(upper_elv)]

    # 获取这两层的偏移量
    lower_shifts = shifts[lower_elv]
    upper_shifts = shifts[upper_elv]

    # 获取数据的维度
    num_latitudes, num_bins = lower_layer.shape

    # 初始化插值后的结果
    interpolated_layer = np.full((num_latitudes, num_bins), np.nan)

    for i in range(num_latitudes):  # 遍历纬度
        for j in range(num_bins):  # 遍历距离库
            # 计算低层和上层的偏移位置
            lower_shifted_j = j + lower_shifts[i]
            upper_shifted_j = j + upper_shifts[i]

            # 判断低层和上层是否都有有效数据
            lower_value = lower_layer[i, lower_shifted_j] if 0 <= lower_shifted_j < num_bins else np.nan
            upper_value = upper_layer[i, upper_shifted_j] if 0 <= upper_shifted_j < num_bins else np.nan

            # 如果低层和上层都有数据，则进行插值
            if not np.isnan(lower_value) and not np.isnan(upper_value):
                f = interp1d([lower_elv, upper_elv], [lower_value, upper_value], kind='linear')
                interpolated_layer[i, j] = f(target_elevation)
            # 如果只有低层有数据，则直接用低层的数据
            elif not np.isnan(lower_value):
                interpolated_layer[i, j] = lower_value
            # 如果只有上层有数据，则直接用上层的数据
            elif not np.isnan(upper_value):
                interpolated_layer[i, j] = upper_value

    return interpolated_layer


if __name__ == "__main__":
# 示例调用
#     file_zh_1 = "run_data/V_data/0429135859data/layer_1_reflectivity.csv"
#     file_cc_1 = "run_data/0430022017data/layer_5_CC.csv"
#
#     file_zh_2 = "run_data/V_data/0429135859data/layer_1.45_reflectivity.csv"
#     # file_zh_2 = "run_data/0430022017data/layer_7_interpolated.csv"
#     file_cc_2 = "run_data/0430022017data/layer_7_CC.csv"

    # file_zh_3 = "run_data/0430022017data/layer_3_reflectivity.csv"
    # file_cc_3 = "run_data/0430022017data/layer_10_CC.csv"

    file_zh_1 = "run_data/sa_data/0430022017dbz/6.0.csv"
    file_zh_2 = "run_data/sa_data/0430022017dbz/10.02.csv"

    file_paths = [file_zh_1, file_zh_2]
    # file_paths = [file_zh_2, file_zh_3]
    layers = load_radar_layers(file_paths)
    aligned_data, azimuth = align_radar_layers(layers, BINS)

    # elvs = [0.5, 1.45]
    # elvs = [2.4, 3.35]
    # elvs = [4.3, 6.0]
    elvs = [6.0, 9.9]
    select_distances = np.linspace(0, 250, BINS)
    shifts = calculate_relative_shifts(select_distances, elvs=elvs, reference_elv=6.0)

    interpolated_data = align_layers_by_shifts(np.array(aligned_data), shifts)
    interpolated_data = interpolate_layers(interpolated_data, 9.0, shifts, BINS)

    print(interpolated_data.shape)
    pd.DataFrame(interpolated_data).to_csv("run_data/linear_result_dbz/sa/linear_interpolation_9.0.csv", index=False)
    # pd.DataFrame(interpolated_data).to_csv("run_data/linear_result_v/0429135859/linear_interpolation_0.89.csv", index=False)

