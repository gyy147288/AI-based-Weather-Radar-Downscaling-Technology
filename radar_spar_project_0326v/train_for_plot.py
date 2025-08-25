from model import RadarCNN, RadarResNet  # 导入模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from masked_loss import WeightedMSELoss  # 使用加权MSE损失
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据加载和处理
def prepare_dataloader(dataset, batch_size=64, shuffle=True):
    """
    创建 DataLoader。
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_dataset(features, labels, train_ratio=0.8):
    """
    将数据集划分为训练集和测试集。
    """
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)  # 回归任务使用 float 类型

    dataset = TensorDataset(features, labels)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def train_model(model, dataloader, criterion, optimizer, num_epochs=1000, patience=10):
    """
    训练模型。
    """
    best_loss = float('inf')
    best_correlation = -float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            # 确保保留维度
            outputs = outputs.reshape(-1)  # 新增的维度修正

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_targets.append(targets.detach())
            all_predictions.append(outputs.detach())

        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)

        # 计算相关系数
        correlation = torch.corrcoef(torch.stack([all_targets, all_predictions]))[0, 1]
        rmse = torch.sqrt(((all_targets - all_predictions) ** 2).mean())

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, Correlation: {correlation:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_models_forplot/best_model_9.0vr.pth")
            epochs_without_improvement = 0
        elif correlation > best_correlation:
            best_correlation = correlation
            torch.save(model.state_dict(), "best_models_forplot/best_model_9.0vr.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in the last {patience} epochs.")
            break


def evaluate_model(model, dataloader, criterion):
    """
    评估模型。
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()

            # 确保保留维度
            outputs = outputs.reshape(-1)  # 新增的维度修正

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_targets.append(targets.detach())
            all_predictions.append(outputs.detach())

    avg_loss = total_loss / len(dataloader)
    print(f"测试集平均损失: {avg_loss:.4f}")

    all_targets = torch.cat(all_targets)
    all_predictions = torch.cat(all_predictions)
    correlation = torch.corrcoef(torch.stack([all_targets, all_predictions]))[0, 1]
    rmse = torch.sqrt(((all_targets - all_predictions) ** 2).mean())

    print(f"测试集 RMSE: {rmse:.4f}, 测试集 Correlation: {correlation:.4f}")

if __name__ == "__main__":
    # from calculate import load_and_process_data,combine_data_from_files  # 导入数据处理函数
    # from calculate_for_nan import load_and_process_data, combine_data_from_files
    from calculate import load_and_process_data, combine_data_from_files

    # 定义数据路径和文件名
    # root_path = "E:/福建数据/spar/04/test1/"
    root_path = "E:/福建数据/spar/04/output_v/"

    # 准备文件对列表
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
         'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
         'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
         'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
         'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),#针对高仰角数据不足添加
        ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',#11°以上建模使用
        'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',
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
            selected_layers=[7, 8, 9, 10, 11, 12, 13, 14],#第三层
            # selected_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 第四层
            selected_range_indices=[0, 1, 2],  # 处理选择的区间,低层用了两个，高层用了3个
            # elevations=[0.5, 0.89, 1.45, 2.4],  # 仰角
            # elevations=[2.4, 3.0, 3.5, 4.3],
            elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            # elevations=[10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5],
            target_layer_idx=5,
            max_bins=1520,
            grid_size=3
        )

        print("\n数据处理完成！")
        print(f"最终数据集大小: {len(labels)} 样本")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


    features = features / 100.0
    labels = labels / 100.0
    # 数据归一化
    # X_min, X_max = -30, 30  # 你当前数据的范围
    # features = (features - X_min) / (X_max - X_min) * 2 - 1  # 归一化到 [-1, 1]
    # labels = (labels - X_min) / (X_max - X_min) * 2 - 1  # 归一化到 [-1, 1]

    features = features.reshape(-1, 3, 3, 3)  # 新形状 (样本数, 层数, H, W)

    # 划分训练集和测试集
    train_dataset, test_dataset = split_dataset(features, labels, train_ratio=0.8)
    train_dataloader = prepare_dataloader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = prepare_dataloader(test_dataset, batch_size=64, shuffle=False)

    model = RadarResNet(in_channels=1).to(device)
    criterion = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_model(model, train_dataloader, criterion, optimizer)
    evaluate_model(model, test_dataloader, criterion)