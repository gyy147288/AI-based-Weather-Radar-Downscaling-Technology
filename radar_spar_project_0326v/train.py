from calculate import load_and_process_data  # 导入数据处理函数
from model import RadarCNN, RadarResNet  # 导入模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
# from masked_loss import MaskedMSELoss
from masked_loss import WeightedMSELoss
import numpy as np
from torch.utils.data import WeightedRandomSampler

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载和处理
def prepare_dataloader(dataset, batch_size=64, shuffle=True):
    """
    创建 DataLoader。
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# def prepare_dataloader(dataset, batch_size=64, shuffle=True, high_threshold=0.6):
#     """
#     对高值样本过采样
#     """
#     # 获取所有标签
#     labels = dataset.tensors[1].numpy()
#
#     # 计算样本权重
#     sample_weights = np.ones(len(labels))
#     high_value_indices = np.where(labels > high_threshold)[0]
#     sample_weights[high_value_indices] = 5.0  # 高值样本权重提高5倍
#
#     # 创建采样器
#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(labels),  # 过采样后总样本数不变
#         replacement=True
#     )
#
#     dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
#     return dataloader

def filter_valid_samples(features, labels, fill_value=-3000):#换模型可能要改
    """ 过滤掉标签为无效值的样本 """
    valid_mask = (labels != fill_value).flatten()
    return features[valid_mask], labels[valid_mask]

# 划分训练集和测试集
def split_dataset(features, labels, train_ratio=0.8):
    """
    先将 features 和 labels 转换为 Tensor，然后划分训练集和测试集。
    """
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)  # 回归任务使用 float 类型

    dataset = TensorDataset(features, labels)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def train_model(model, dataloader, criterion, optimizer, num_epochs=1000, patience = 10):
    """
    训练模型并打印每个 epoch 的损失、相关系数和均方根误差。

    参数:
    - model: 要训练的模型
    - dataloader: 数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - patience: 早停的耐心轮次（即连续多少轮未改善就停止训练）
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
            inputs, targets = inputs.to(device), targets.to(device)  # 数据移动到 GPU

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # 输出维度调整为 (batch_size,)
            # outputs = model(inputs, elevations, target_elevation).squeeze()  # 传入额外参数

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 累计损失
            running_loss += loss.item()

            # 保存所有的预测值和真实值用于后续计算
            all_targets.append(targets.detach())
            all_predictions.append(outputs.detach())

        # 将所有批次的目标值和预测值拼接成一个张量
        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)

        # 计算相关系数
        target_mean = all_targets.mean()
        pred_mean = all_predictions.mean()
        covariance = ((all_targets - target_mean) * (all_predictions - pred_mean)).mean()
        target_std = all_targets.std()
        pred_std = all_predictions.std()
        correlation = covariance / (target_std * pred_std)

        # 计算均方根误差（RMSE）
        rmse = torch.sqrt(((all_targets - all_predictions) ** 2).mean())

        # 打印结果
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, Correlation: {correlation:.4f}")

        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     torch.save(model.state_dict(), "best_model5.0.pth") #最初的保存方法
        # 如果模型有更好的表现（损失降低或相关系数提高），保存模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_models_resnet/best_model8.0.pth")
            # torch.save(model.state_dict(), "best_models_cnn/best_model8.0_concentrate_angle.pth")
            # torch.save(model.state_dict(), "best_models_cnn/best_model3.0.pth")
            epochs_without_improvement = 0  # 重置连续无改善的轮数
        elif correlation > best_correlation:
            best_correlation = correlation
            torch.save(model.state_dict(), "best_models_resnet/best_model8.0.pth")
            # torch.save(model.state_dict(), "best_models_cnn/best_model8.0_concentrate_angle.pth")
            # torch.save(model.state_dict(), "best_models_cnn/best_model3.0.pth")
            epochs_without_improvement = 0  # 重置连续无改善的轮数
        else:
            epochs_without_improvement += 1

        # 如果连续10轮没有改善，停止训练
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in the last {patience} epochs.")
            break

# 测试函数
def evaluate_model(model, dataloader, criterion):
    """
    评估模型在测试集上的性能。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            # outputs = model(inputs, elevations, target_elevation).squeeze()#含掩码训练增强时使用
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 保存所有的预测值和真实值
            all_targets.append(targets.detach())
            all_predictions.append(outputs.detach())

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"测试集平均损失: {avg_loss:.4f}")

    # 计算相关系数和 RMSE
    all_targets = torch.cat(all_targets)
    all_predictions = torch.cat(all_predictions)
    target_mean = all_targets.mean()
    pred_mean = all_predictions.mean()
    covariance = ((all_targets - target_mean) * (all_predictions - pred_mean)).mean()
    target_std = all_targets.std()
    pred_std = all_predictions.std()
    correlation = covariance / (target_std * pred_std)
    rmse = torch.sqrt(((all_targets - all_predictions) ** 2).mean())

    print(f"测试集 RMSE: {rmse:.4f}, 测试集 Correlation: {correlation:.4f}")

#标准化，看情况使用
def normalize(data, fill_value=-1000):
    """
    分区间归一化：
    - 空值保持为-1（与原有效值区分）
    - 有效值缩放到[-1,1]
    """
    # 创建掩码标识空值
    mask = (data == fill_value)

    # 对有效值进行归一化
    valid_data = data[~mask]
    if len(valid_data) > 0:
        min_val = -500  # 根据您设定的有效值下限（过滤后有效数据）
        max_val = 7000  # 根据您的数据统计设定
        valid_data_norm = 2 * (valid_data - min_val) / (max_val - min_val) - 1  # 缩放到[-1,1]
    else:
        valid_data_norm = valid_data

    # 合并数据
    data_norm = np.full_like(data, -1.0)  # 空值设为-1
    data_norm[~mask] = valid_data_norm
    return data_norm

if __name__ == "__main__":
    # from calculate import load_and_process_data,combine_data_from_files  # 导入数据处理函数
    # from calculate_for_nan import load_and_process_data, combine_data_from_files
    from calculate_for_nomask import load_and_process_data, combine_data_from_files

    # 定义数据路径和文件名
    root_path = "E:/福建数据/spar/04/test1/"

    # 准备文件对列表
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_ZBAAB_20240430014207_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),#针对高仰角数据不足添加
        # ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',#11°以上建模使用
        # 'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv')
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
            target_layer_idx=4,
            max_bins=1520,
            grid_size=3
        )

        print("\n数据处理完成！")
        print(f"最终数据集大小: {len(labels)} 样本")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

    # print(features.shape,labels.shape)

    # features = features[:, :2, :, :]  # 选择前两层
    # 调整形状为 (batch_size, channels, height, width)
    # features, labels = filter_valid_samples(features, labels)

    # features = normalize(features, fill_value=-1000)  # 输入归一化
    # labels = normalize(labels, fill_value=-1000)  # 输出归一化

    # features = features.reshape(-1, 3, 3, 3, 2)
    # features = features[..., 0]  # 仅保留数据通道（假设原数据形状为 [样本数, 层数, H, W, 2]
    features = features / 100.0
    labels = labels / 100.0
    features = features.reshape(-1, 3, 3, 3)  # 新形状 (样本数, 层数, H, W)

    # 划分训练集和测试集
    train_dataset, test_dataset = split_dataset(features, labels, train_ratio=0.8)

    # 准备训练和测试 DataLoader
    train_dataloader = prepare_dataloader(train_dataset, batch_size=64, shuffle=True)
    # train_dataloader = prepare_dataloader(
    #     train_dataset,
    #     batch_size=64,
    #     shuffle=False,  # 必须关闭shuffle，因为使用了Sampler
    #     high_threshold=0.6  # 归一化后的高值阈值
    # )
    test_dataloader = prepare_dataloader(test_dataset, batch_size=64, shuffle=False)

    # 准备数据
    # dataloader = prepare_dataloader(features, labels)

    # 初始化模型、损失函数和优化器
    # model = RadarCNN().to(device)
    # model = RadarCNN(num_layers=3).to(device)
    model = RadarResNet(in_channels=1).to(device)
    # criterion = nn.MSELoss()  # 均方误差损失
    # criterion = WeightedMSELoss(fill_value=-1, high_value_threshold=0.6, high_weight=3.0).to(device)
    # criterion = FillValueAwareLoss(fill_value=-2000)
    criterion = WeightedMSELoss(fill_value=-100)
    # criterion = HybridMaskedLoss(fill_value=-500, alpha=0.5)  # 可调整alpha值
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.AdamW(model.parameters(),
    #                               lr=0.005,  # 降低学习率
    #                               weight_decay=1e-4)  # 添加权重衰减

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer)

    # 测试模型
    evaluate_model(model, test_dataloader, criterion)


