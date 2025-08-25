import numpy as np
import pandas as pd
from calculate import extract_3x3_matrix, process_radar_data, align_radar_data, extract_to_ndarray,interpolate_layer,calculate_relative_shifts, create_dataset,load_and_process_data
def combine_data_from_files(root_path, file_list):
    """
    从多个文件读取并合并数据

    Args:
        root_path: 数据文件根目录
        file_list: 文件对列表，每对是(ZH文件名, CC文件名)的元组

    Returns:
        combined_features: 合并后的特征数据
        combined_labels: 合并后的标签数据
    """
    all_features = []
    all_labels = []

    total_files = len(file_list)

    for idx, (zh_file, cc_file) in enumerate(file_list):
        try:
            print(f"\n处理文件 {idx + 1}/{total_files}")
            print(f"正在处理: {zh_file} 和 {cc_file}")

            # 使用你原有的函数处理单个文件对
            features, labels = load_and_process_data(root_path, zh_file, cc_file)

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



# 使用示例
if __name__ == "__main__":
    root_path = "F:/福建数据/spar/04/test1/"

    # 准备文件对列表
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv', 'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430060124_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv', 'Z_RADR_I_Z9600_20240430060124_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('file3_ZH.csv', 'file3_CC.csv'),
        # ... 更多文件对
    ]

    try:
        # 处理所有文件并合并数据
        features, labels = combine_data_from_files(root_path, file_pairs)

        print("\n数据处理完成！")
        print(f"最终数据集大小: {len(labels)} 样本")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")