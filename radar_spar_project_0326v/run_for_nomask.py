import torch
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import math
from model import RadarCNN, RadarResNet  # 导入训练好的模型
from torch.utils.data import DataLoader, TensorDataset
from run import calculate_relative_shifts, load_radar_layers, align_radar_layers, fill_edges, extract_3x3_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

BINS = 4000
TRUE_FILL = -1000
def preprocess_layer_data(layer_data):
    """
    填充NaN值为-1500，并删除小于-1500的值，填补为-1500。
    """
    layer_data[layer_data < -500] = TRUE_FILL  # 删除低于-1500的值并替换为空值
    layer_data[layer_data == np.nan] = TRUE_FILL
    return layer_data

def extract_3x3_matrix(layer_data, i, j, grid_size=3):
    """
    提取指定点 (i, j) 周围的 3x3 矩阵
    """
    half_size = grid_size // 2
    if (i - half_size < 0 or i + half_size >= layer_data.shape[0] or
        j - half_size < 0 or j + half_size >= layer_data.shape[1]):
        return None
    return layer_data[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]

def create_dataset(layer_arrays, distance_shifts, elevations, grid_size=3, max_bins=920):
    """
    构建数据集，返回的特征数据形状为 (样本数, 层数, 3, 3)
    这里每个样本只包含单通道数据（不含掩码），因为你采用的是新的单通道版本。
    """
    layer_height = layer_arrays[0].shape[0]
    feature_data = []

    for i in range(grid_size // 2, layer_height - grid_size // 2):
        for j in range(grid_size // 2, min(layer_arrays[0].shape[1], max_bins)):
            feature_vector = []
            # 这里不再生成掩码，只提取数值矩阵
            for elv, layer in zip(elevations, layer_arrays):
                shifted_j = j + distance_shifts[elv][j]
                if shifted_j < 0 or shifted_j >= layer.shape[1]:
                    matrix = np.full((grid_size, grid_size), TRUE_FILL, dtype=np.float32)#修改成了0
                else:
                    matrix = extract_3x3_matrix(layer, i, shifted_j, grid_size)
                    if matrix is None:
                        matrix = np.full((grid_size, grid_size), TRUE_FILL, dtype=np.float32)#也修改了0
                matrix = np.where(np.isnan(matrix), TRUE_FILL, matrix)
                matrix[matrix == -32768] = TRUE_FILL
                feature_vector.append(matrix)
            stacked = np.array(feature_vector)  # 形状 (层数, 3, 3)
            feature_data.append(stacked)
    return np.array(feature_data)

def create_dataloader(features, batch_size=64):
    dataset = TensorDataset(features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def reshape_predictions_to_original_shape(predictions, original_shape):
    return predictions.reshape(original_shape)

# ========== 模型加载与预测函数 ==========
def load_trained_model(model_path):
    """
    加载训练好的 RadarResNet 模型。注意：这里我们使用单通道输入，
    所以实例化时将 in_channels 设置为 1。
    """
    model = RadarResNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# def denormalize(data_norm, fill_value=-1000, min_val=-500, max_val=7000):
#     """
#     将归一化后的数据还原
#     """
#     0430022017data = (data_norm + 1) * (max_val - min_val) / 2 + min_val
#     0430022017data[data_norm == -1] = fill_value  # 恢复空值
#     return 0430022017data

def predict(model, dataloader):
    """
    对数据进行预测，模型的输入形状应为 (B, L, H, W)。
    RadarResNet 模型的 forward 不需要额外参数。
    """
    all_predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            # inputs 形状: (B, L, 3, 3)
            inputs = inputs[0].to(device)
            outputs = model(inputs).squeeze()  # 输出形状 (B,)
            all_predictions.append(outputs.cpu().numpy())
    return np.concatenate(all_predictions, axis=0)

# ========== 主流程 ==========
if __name__ == "__main__":
    # 定义文件路径（请根据实际情况修改）
    file_zh_1 = "run_data/0429135859data/layer_4.3_reflectivity.csv"
    # file_zh_1 = "run_data/0429data/layer_4.3_reflectivity.csv"
    file_cc_1 = "run_data/0430022017data/layer_5_CC.csv"
    file_zh_2 = "run_data/0429135859data/layer_11_reflectivity.csv"
    # file_zh_2 = "run_data/0430022017data/layer_1.45_interpolated.csv"
    file_cc_2 = "run_data/0430022017data/layer_7_CC.csv"
    file_zh_3 = "run_data/0429135859data/layer_15_reflectivity.csv"
    file_cc_3 = "run_data/0430022017data/layer_10_CC.csv"
    file_paths = [file_zh_1, file_zh_2, file_zh_3]

    # 定义仰角（与文件顺序对应）
    elvs = [4.3, 6.0, 9.9]
    # elvs = [2.4, 3.5, 4.3]
    # elvs = [0.5, 1.45, 2.4]
    # select_distances = np.linspace(0, 250, 1000) #记得换个数还有run.py里的库长
    select_distances = np.linspace(0, 250, BINS)  # 记得换个数还有run.py里的库长
    shifts = calculate_relative_shifts(select_distances, elvs=elvs, reference_elv=4.3)

    layers = load_radar_layers(file_paths)
    aligned, base_az = align_radar_layers(layers, target_bins=BINS)
    data = fill_edges(aligned)
    data = preprocess_layer_data(data)

    # 使用新的 create_dataset 函数，返回形状 (样本数, 层数, 3, 3)
    features = create_dataset(data, shifts, elvs, grid_size=3, max_bins=BINS)
    print(f"数据集形状: {features.shape}")
    features = features /100.0

    features_tensor = torch.tensor(features, dtype=torch.float32)
    dataloader = create_dataloader(features_tensor, batch_size=64)

    # 加载模型（请修改模型路径）
    model = load_trained_model("best_models_resnet/best_model9.0.pth")

    predictions = predict(model, dataloader)
    # predictions = denormalize(
    #     predictions,
    #     fill_value=TRUE_FILL,  # 根据实际空值定义（如-1000）
    #     min_val=-500,
    #     max_val=7000
    # )
    print(f"预测结果形状: {predictions.shape}")

    # 恢复预测结果原始形状
    first_dim = base_az.shape[0]
    second_dim = features_tensor.shape[0] // first_dim
    original_shape = (first_dim, second_dim)
    reshaped_predictions = reshape_predictions_to_original_shape(predictions, original_shape)
    # 将预测值低于 -500 的置为 NaN
    reshaped_predictions = np.where(reshaped_predictions <= -2, np.nan, reshaped_predictions)

    # pd.DataFrame(reshaped_predictions).to_csv("run_data/result/predictions_new.csv", index=False)
    pd.DataFrame(reshaped_predictions).to_csv("run_data/result_dbz/0429135859/predictions_9.0.csv", index=False)
    print("预测结果已保存为 predictions_new.csv")