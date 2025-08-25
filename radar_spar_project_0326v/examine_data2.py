from calculate import combine_data_from_files, interpolate_layer, extract_3x3_matrix
import matplotlib as plt
import numpy as np

def test_data_processing():
    # 定义数据路径和文件对
    root_path = "E:/福建数据/spar/04/test1/"
    file_pairs = [
        ('Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430024720_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430031733_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        ('Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
         'Z_RADR_I_Z9600_20240430030625_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv')
        # 其他文件对...
    ]

    try:
        # 调用数据处理函数并打印特征和标签的大小
        features, labels = combine_data_from_files(
            root_path=root_path,
            file_list=file_pairs,
            angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
            selected_layers=[0, 1, 2, 3, 4],
            selected_range_indices=[0, 2],
            elevations=[0.5, 0.89, 1.45, 2.4],
            target_layer_idx=1,
            max_bins=1520,
            grid_size=3
        )

        # 打印数据的大小
        print("\n数据处理完成！")
        print(f"最终特征数据集大小: {features.shape}")
        print(f"最终标签数据集大小: {labels.shape}")
        print(np.min(features))

        # 检查特定位置的特征和标签，查看掩码是否正常
        # 假设掩码是根据标签值为0或1来判断
        print("\n检查特定位置的特征和标签是否正常:")
        for i in range(5):  # 打印前5个数据点
            print(f"第{i+1}个样本:")
            print(f"特征数据: {features[i]}")
            print(f"标签: {labels[i]}")
            print(f"特征数据对应标签是否匹配: {'匹配' if labels[i] == 1 else '不匹配'}")
            print()

        # 可以根据需要添加其他的检查，例如是否有缺失值等
        # print(f"特征数据的一部分:\n{features[:5]}")  # 打印前5条特征数据
        # print(features)

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

    return features, labels

if __name__ == "__main__":
    features, labels = test_data_processing()

