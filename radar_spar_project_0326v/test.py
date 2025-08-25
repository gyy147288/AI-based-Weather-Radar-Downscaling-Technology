from calculate import combine_data_from_files, interpolate_layer, extract_3x3_matrix
from calculate import combine_data_from_files
import matplotlib.pyplot as plt
import numpy as np
from run import load_radar_layers, align_radar_layers, fill_edges, calculate_relative_shifts
import pandas as pd
# 测试插值函数
root_path = "E:/福建数据/spar/04/output_v/"

# 准备文件对列表
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
        # ('Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_Vr.csv',#11°以上建模使用
        # 'Z_RADR_I_ZBAAB_20240430013058_O_DOR_PAR-SD_CAP_025.bin.gz_CC.csv'),
        # ('Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv',
        #  'Z_RADR_I_ZBAAB_20240430014653_O_DOR_PAR-SD_CAP_025.bin.gz_ZH.csv')
        # ... 更多文件对
    ]

features, labels = combine_data_from_files(
            root_path=root_path,
            file_list=file_pairs,
            angle_ranges=[(12000.0, 19000.0), (22500.0, 23500.0), (25500.0, 26500.0)],
            # selected_layers=[0, 1, 2, 3, 4],#第一层
            selected_layers=[4, 5, 6, 7, 8],#第二层
            # selected_layers=[7, 8, 9, 10, 11, 12, 13, 14],#第三层
            # selected_layers=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 第四层
            selected_range_indices=[0, 1, 2],  # 处理选择的区间,低层用了两个，高层用了3个
            # elevations=[0.5, 0.89, 1.45, 2.4],  # 仰角
            elevations=[2.4, 3.0, 3.5, 4.3],
            # elevations=[4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            # elevations=[10.0, 11.0, 12.0, 13.0, 14.6, 16.0, 17.0, 18.0, 19.5],
            target_layer_idx=1,
            max_bins=1520,
            grid_size=3
        )

# 绘制标签分布直方图
# plt.hist(labels, bins=np.arange(-15, 70, 0.5), density=True)
# plt.xlabel('Reflectivity (dBZ)')
# plt.ylabel('Density')
# plt.title('Label Distribution')
# plt.axvline(0, color='r', linestyle='--')
# plt.show()
# 统计不同区间的数据数量
# counts = {
#     "-1000": np.sum(labels == -1000),
#     "-1000 to 0": np.sum((labels > -1000) & (labels <= 0)),
#     "0 to 1000": np.sum((labels > 0) & (labels <= 1000)),
#     "1000 to 2000": np.sum((labels > 1000) & (labels <= 2000)),
#     "2000 to 3000": np.sum((labels > 2000) & (labels <= 3000)),
#     "3000 to 4000": np.sum((labels > 3000) & (labels <= 4000)),
#     "4000 to 5000": np.sum((labels > 4000) & (labels <= 5000)),
#     "5000+": np.sum(labels > 5000)
# }

counts = {
    "0": np.sum(labels == 0),
    "-3000 to -2000": np.sum((labels > -3000) & (labels <= -2000)),
    "-2000 to -1000": np.sum((labels > -2000) & (labels <= -1000)),
    "-1000 to 0": np.sum((labels > -1000) & (labels <= 0)),
    "0 to 1000": np.sum((labels > 0) & (labels <= 1000)),
    "1000 to 2000": np.sum((labels > 1000) & (labels <= 2000)),
    "2000 to 3000": np.sum((labels > 2000) & (labels <= 3000)),
    "3000+": np.sum(labels > 3000)
}

# 打印统计结果
for key, value in counts.items():
    print(f"{key}: {value}")

plt.hist(labels.flatten(), bins=50)
plt.xlabel("dBZ Value")
plt.ylabel("Frequency")
plt.title("Histogram of Radar Echoes in Training Data")
plt.show()



