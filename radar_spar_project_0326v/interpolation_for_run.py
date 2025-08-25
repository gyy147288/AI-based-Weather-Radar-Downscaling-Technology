# 0307老胡说应该将模型应用于相控阵雷达数据来比较效果好坏，因此有些层需要先插值
import numpy as np
import pandas as pd

def interpolate_between_two_layers(file_low, file_high, target_elv, elv_low, elv_high, output_file):
    """
    读取两个 CSV 文件，进行仰角插值，并保存结果。

    参数：
        file_low (str): 低仰角层 CSV 文件路径
        file_high (str): 高仰角层 CSV 文件路径
        target_elv (float): 目标插值仰角
        elv_low (float): 低仰角值
        elv_high (float): 高仰角值
        output_file (str): 保存插值后的 CSV 文件路径
    """

    # 读取数据
    data_low = pd.read_csv(file_low, header=0)
    data_high = pd.read_csv(file_high, header=0)

    # 检查数据大小是否一致
    if data_low.shape != data_high.shape:
        raise ValueError("输入的两个 CSV 文件尺寸不匹配！")

    # 提取方位角列，并直接使用较低仰角的方位角
    azimuth_values = data_low.iloc[:, 0].values  # 直接使用低仰角的方位角
    data_low_values = data_low.iloc[:, 1:].values  # 去掉第一列（方位角）
    data_high_values = data_high.iloc[:, 1:].values

    # 计算插值因子
    weight = (target_elv - elv_low) / (elv_high - elv_low)

    # 初始化插值结果
    interpolated_data = np.full_like(data_low_values, np.nan)

    # 插值逻辑：
    # 1. 如果两层都有数据，进行线性插值
    mask_valid = ~np.isnan(data_low_values) & ~np.isnan(data_high_values)
    interpolated_data[mask_valid] = (
            data_low_values[mask_valid] * (1 - weight) + data_high_values[mask_valid] * weight
    )

    # 2. 如果某一层 NaN，则直接取另一层数据
    mask_only_low = ~np.isnan(data_low_values) & np.isnan(data_high_values)
    mask_only_high = np.isnan(data_low_values) & ~np.isnan(data_high_values)

    interpolated_data[mask_only_low] = data_low_values[mask_only_low]
    interpolated_data[mask_only_high] = data_high_values[mask_only_high]

    # 3. 如果两层都是 NaN，保持 NaN

    # 组合方位角列，确保格式不变
    result_df = pd.DataFrame(np.column_stack([azimuth_values, interpolated_data]))

    # 保存为 CSV
    result_df.to_csv(output_file, index=False, header=data_low.columns)
    print(f"插值完成，结果已保存至: {output_file}")

# ======== 运行示例 ========
if __name__ == "__main__":
    file_low = "run_data/0429135859data/layer_3_reflectivity.csv"
    file_high = "run_data/0429135859data/layer_4_reflectivity.csv"

    target_elv = 1.29  # 目标插值仰角
    elv_low = 1.79  # 低层仰角
    elv_high = 1.45  # 高层仰角

    output_file = "run_data/0429135859data/layer_1.45_reflectivity.csv"

    interpolate_between_two_layers(file_low, file_high, target_elv, elv_low, elv_high, output_file)



