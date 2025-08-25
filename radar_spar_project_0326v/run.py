import torch
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import math
from model import RadarCNN, RadarResNet  # 导入训练好的模型
from torch.utils.data import DataLoader, TensorDataset

'''注：1，更换层数记得修改仰角索引，输入文件
      2,更换模型记得加载和后面都要换
      3，库偏移匹配记得修改，库偏移函数进行了修改，不知道训练的时候要不要修改一下, 参考仰角要修改
      4，文件名记得更换
'''

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 填充空值和删除低于-1500的值
def preprocess_layer_data(layer_data):
    """
    填充NaN值为-1500，并删除小于-1500的值，填补为-1500。
    """
    layer_data[layer_data < -500] = -500  # 删除低于-1500的值并替换为空值
    layer_data[layer_data == np.nan] = -500
    return layer_data

# 计算雷达库偏移
def cal_radar_gauge(d, beta):
    beta = math.radians(beta)  # 仰角
    R_earth = 6371  # 地球半径
    eq_R_earth = (4.0 / 3.0) * R_earth  # 等效地球半径
    h_0 = 1.079  # 雷达海拔高度
    a = d / eq_R_earth  # 地心角
    alpha = a
    r = 0.25  # 0.0625  # 库长 62.5m
    # r = 0.0625  # 库长 62.5m
    G = ((eq_R_earth + h_0) * np.sin(alpha)) / (r * np.cos(alpha + beta))
    return G

# def calculate_relative_shifts(distances, elvs, reference_elv=0.5):
#     reference_shifts = [int(cal_radar_gauge(d, reference_elv)) for d in distances]
#     shifts = {}
#     for elv in elvs:
#         current_shifts = [int(cal_radar_gauge(d, elv)) for d in distances]
#         shifts[elv] = [current - ref for current, ref in zip(current_shifts, reference_shifts)]
#     return shifts


def calculate_relative_shifts(distances, elvs, reference_elv=0.5):
    # 计算参考仰角的偏移量
    reference_shifts = [cal_radar_gauge(d, reference_elv) for d in distances]

    shifts = {}
    for elv in elvs:
        # 计算当前仰角的偏移量
        current_shifts = [cal_radar_gauge(d, elv) for d in distances]

        # 计算相对偏移量，先相减后取整
        shifts[elv] = [int(current - ref) for current, ref in zip(current_shifts, reference_shifts)]

    return shifts


#新加载数据集代码
def load_radar_layers(file_paths):
    """
    读取多层雷达数据，返回方位角和数据矩阵列表
    参数：
        file_paths (list): 各层CSV文件路径列表
    返回：
        list: 每个元素为元组 (方位角数组, 数据矩阵)
    """
    layers = []
    for path in file_paths:
        df = pd.read_csv(path)
        azimuth = df.iloc[:, 0].values.astype(float)  # 第一列为方位角
        data = df.iloc[:, 1:].values.astype(np.float32)  # 后续列为距离库数据
        layers.append((azimuth, data))
    return layers


def align_radar_layers(layers, target_bins=920):
    """
    对齐多层雷达数据到指定的距离库数，并确保方位角对齐。

    参数：
        layers (list): 每一层数据为一个元组 (azimuth, 0430022017data)，数据为 (方位角, 数据)。
        target_bins (int): 目标距离库数量，用于对齐。

    返回：
        aligned_data (list): 对齐后的数据列表，每个元素是对齐后的层数据。
        base_azimuth (np.ndarray): 对齐后的方位角。
    """

    # 选择径向最少的层作为目标层
    base_idx = np.argmin([len(az) for az, _ in layers])  # 找到最少方位角数量的层作为目标层
    base_azimuth = layers[base_idx][0]  # 目标层的方位角
    # min_bins = min(0430022017data.shape[1] for _, 0430022017data in layers)  # 找到最少的距离库数量

    aligned_data = []

    for az, data in layers:
        # === 方位角对齐 ===
        # 创建插值函数（自动处理360度循环）
        az_norm = az % 360  # 确保方位角是0到360度之间
        sort_idx = np.argsort(az_norm)  # 按方位角排序
        interpolator = RegularGridInterpolator(
            (az_norm[sort_idx], np.arange(data.shape[1])),
            data[sort_idx, :],
            method='nearest',
            bounds_error=False,
            fill_value=np.nan
        )

        # 生成插值网格
        grid_az, grid_bin = np.meshgrid(base_azimuth, np.arange(target_bins), indexing='ij')

        # 执行插值
        data_aligned = interpolator((grid_az, grid_bin))

        # === 距离库裁剪 ===
        if data_aligned.shape[1] < target_bins:
            # 如果当前层的距离库少于目标值，则填充 NaN， -500
            data_aligned = np.pad(data_aligned, ((0, 0), (0, target_bins - data_aligned.shape[1])), mode='constant',
                                  constant_values=-500)
        elif data_aligned.shape[1] > target_bins:
            # 如果当前层的距离库多于目标值，则裁剪掉多余的部分
            data_aligned = data_aligned[:, :target_bins]

        aligned_data.append(data_aligned)

    return aligned_data, base_azimuth


# 填充边界
def fill_edges(layers_data):
    """
    对边缘进行填充，使得数据连续：
    - 在每一层的第一行之前插入一行，内容为该层的最后一行的值
    - 在每一层的最后一行之后插入一行，内容为该层的第一行的值

    参数：
        layers_data (list): 包含多层数据的列表，每一层是一个形状为 (方位角, 距离库) 的数组

    返回：
        filled_layers (list): 填充边缘后的多层数据列表
    """
    filled_layers = []

    for layer in layers_data:
        # 获取该层的第一行和最后一行
        first_row = layer[0, :].copy()  # 第一行数据
        last_row = layer[-1, :].copy()  # 最后一行数据

        # 在该层的方位角数据前面加一行，内容是该层的最后一行
        layer = np.insert(layer, 0, last_row, axis=0)

        # 在该层的方位角数据后面加一行，内容是该层的第一行
        layer = np.insert(layer, layer.shape[0], first_row, axis=0)

        # 将填充后的数据加入列表
        filled_layers.append(layer)

    return np.array(filled_layers)

# 提取3x3矩阵
def extract_3x3_matrix(layer_data, i, j, grid_size=3):
    """
    提取指定点 (i, j) 周围的 3x3 矩阵。
    """
    half_size = grid_size // 2
    if (i - half_size < 0 or i + half_size >= layer_data.shape[0] or
        j - half_size < 0 or j + half_size >= layer_data.shape[1]):
        return None  # 跳过边界点
    return layer_data[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]

# 处理三层数据并生成特征数据集
def create_dataset(layer_arrays, distance_shifts, elevations, grid_size=3, max_bins=920):
    """
    应用场景数据集构建（适配训练模型输入格式）

    参数：
        layer_arrays: 雷达数据数组，形状 [层数, 方位数, 距离库数]
        distance_shifts: 距离库偏移字典 {仰角: 偏移数组}
        elevations: 仰角列表（需与layer_arrays顺序一致）
        grid_size: 特征矩阵大小
        max_bins: 最大距离库数

    返回：
        feature_data: 形状 [样本数, 层数, 3, 3, 2]（数据+掩码双通道）
    """
    layer_height = layer_arrays[0].shape[0]
    feature_data = []

    for i in range(grid_size//2, layer_height - grid_size//2):
        for j in range(grid_size//2, min(layer_arrays[0].shape[1], max_bins)):
            feature_vector = []

            for elv, layer in zip(elevations, layer_arrays):
                # 计算物理对齐偏移
                shifted_j = j + distance_shifts[elv][j]

                # 处理越界情况
                if shifted_j < 0 or shifted_j >= layer.shape[1]:
                    matrix = np.full((grid_size, grid_size), -500)
                else:
                    matrix = extract_3x3_matrix(layer, i, shifted_j, grid_size)
                    if matrix is None:
                        matrix = np.full((grid_size, grid_size), -500)

                # 替换原始无效值并生成掩码
                matrix = np.where(np.isnan(matrix), -500, matrix)  # ✅ 正确替换 NaN
                matrix[matrix == -32768] = -500  # 确保原始无效值也被替换
                mask = (matrix != -500).astype(np.float32)

                # 合并为双通道
                combined = np.stack([matrix, mask], axis=-1)  # (3,3,2)
                feature_vector.append(combined)

            # 重组维度适配模型输入
            stacked = np.array(feature_vector)  # (层数, 3, 3, 2)
            feature_data.append(stacked)

    return np.array(feature_data)

# 创建数据加载器
def create_dataloader(features, batch_size=64):
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


# 加载已经训练好的模型
def load_trained_model(model_path):
    model = RadarCNN(num_layers=3).to(device)
    # model = RadarResNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

# 进行预测
# def predict(model, dataloader):
#     all_predictions = []
#
#     with torch.no_grad():  # 禁用梯度计算
#         for inputs in dataloader:
#             inputs = inputs[0].to(device)
#             outputs = model(inputs).squeeze()
#             all_predictions.append(outputs.cpu().numpy())
#
#     return np.concatenate(all_predictions, axis=0)

def predict(model, dataloader, features):
    """
    修改后的预测函数，自动处理全空输入
    """
    # 计算掩码总和（关键新增代码）
    mask_sum = features[..., 1].sum(axis=(1, 2, 3))  # 对每个样本的所有掩码求和
    base_elevations = torch.tensor([4.3, 6.0, 10.0], dtype=torch.float32).to(device)

    # 原始预测逻辑
    all_predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            # outputs = model(inputs).squeeze()
            batch_size = inputs.shape[0]
            # 构造形状为 [B,3] 的仰角张量
            elevations = base_elevations.unsqueeze(0).repeat(batch_size, 1)
            # 构造目标仰角张量，形状为 [B,1]，例如目标仰角为5.0
            target_elevation = torch.full((batch_size, 1), 9.0, dtype=torch.float32).to(device)
            # 注意：这里调用模型时传入仰角信息
            outputs = model(inputs, elevations, target_elevation).squeeze()

            all_predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)

    # 新增：全空输入置为nan
    predictions[mask_sum == 0] = np.nan

    return predictions

# 将预测结果恢复为原始维度
def reshape_predictions_to_original_shape(predictions, original_shape):
    reshaped_predictions = predictions.reshape(original_shape)
    return reshaped_predictions


# 主函数
if __name__ == "__main__":
    # 定义文件路径
    file_zh_1 = "run_data/0430022017data/layer_5_reflectivity.csv"
    file_cc_1 = "run_data/0430022017data/layer_5_CC.csv"

    file_zh_2 = "run_data/0430022017data/layer_7_reflectivity.csv"
    file_cc_2 = "run_data/0430022017data/layer_7_CC.csv"

    file_zh_3 = "run_data/0430022017data/layer_10_reflectivity.csv"
    file_cc_3 = "run_data/0430022017data/layer_10_CC.csv"

    file_paths = [file_zh_1, file_zh_2, file_zh_3]
    # elvs = [0.5, 1.45, 2.4]
    # elvs = [2.4, 3.35, 4.3]
    elvs = [4.3, 6.0, 9.9]
    # elvs = [10, 14.6, 19.5]    # eles = [10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5]

    # select_distances = np.linspace(0, 230, 920)
    select_distances = np.linspace(0, 250, 1000)
    shifts = calculate_relative_shifts(select_distances, elvs=elvs, reference_elv=4.3)

    layers = load_radar_layers(file_paths)
    # aligned, base_az = align_radar_layers(layers, 920)
    aligned, base_az = align_radar_layers(layers, 1000)
    data = fill_edges(aligned)

    # 空值填充为-500
    data = preprocess_layer_data(data) #用我的新模型的时候更换为这个

    # features = create_dataset(0430022017data, select_distances, elvs, max_bins=1000)#注意select_distances还是shifts
    features = create_dataset(data, shifts, elvs, max_bins=1000)
    # features = features[:, 1:3, :, :]
    # features = features[:, 1:3, :, :]
    # 处理数据为适合模型的输入格式
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # 创建数据加载器
    dataloader = create_dataloader(features_tensor)

    # 加载训练好的模型
    # model = load_trained_model("best_models_forplot/best_model5.0.pth")
    # model = load_trained_model("best_models_cnn/best_model9.0.new.pth")
    model = load_trained_model("best_models_cnn/best_model9.0_concentrate_angle.pth")
    # model = load_trained_model("best_models_resnet/best_model0.89.pth")

    # 进行预测
    # predictions = predict(model, dataloader)
    predictions = predict(model, dataloader, features)  # 修改这里

    # 恢复预测结果为原始的维度
    # 计算目标形状
    first_dim = base_az.shape[0]  # 第一维（固定的行数）
    second_dim = features_tensor.shape[0] // first_dim  # 根据总样本数除以第一维大小来计算第二维大小

    # 目标形状
    original_shape = (first_dim, second_dim)
    # original_shape = (362,999)

    reshaped_predictions = reshape_predictions_to_original_shape(predictions, original_shape)

    # 填充空值部分为NaN
    reshaped_predictions = np.where(reshaped_predictions <= -500, np.nan, reshaped_predictions)

    # 保存结果到CSV文件
    pd.DataFrame(reshaped_predictions).to_csv("run_data/result/predictions.csv", index=False)
    print("预测结果已保存为 predictions.csv")

