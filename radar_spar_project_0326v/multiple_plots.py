import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# 设置字体
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['mathtext.default'] = 'regular'

# ================== 配置参数 ==================
models = [
    {"name": "0.89°", "file": "model_data/model_vr/0.89.npz"},
    {"name": "3.0°", "file": "model_data/model_vr/3.0.npz"},
    {"name": "5.0°", "file": "model_data/model_vr/5.0.npz"},
    {"name": "7.0°", "file": "model_data/model_vr/7.0.npz"},
    {"name": "8.0°", "file": "model_data/model_vr/8.0.npz"},
    {"name": "9.0°", "file": "model_data/model_vr/9.0.npz"}
]
# models = [
#     {"name": "0.89°", "file": "model_data/linear_vr/0.89.npz"},
#     {"name": "3.0°", "file": "model_data/linear_vr/3.0.npz"},
#     {"name": "5.0°", "file": "model_data/linear_vr/5.0.npz"},
#     {"name": "7.0°", "file": "model_data/linear_vr/7.0.npz"},
#     {"name": "8.0°", "file": "model_data/linear_vr/8.0.npz"},
#     {"name": "9.0°", "file": "model_data/linear_vr/9.0.npz"}
# ]

# ================== 自定义颜色映射 ==================
colors = [
    (0, 0, 1),  # 蓝色
    (0, 1, 0),  # 绿色
    (1, 1, 0),  # 黄色
    (1, 0.5, 0),  # 橙色
    (1, 0, 0)  # 红色
]
custom_cmap = LinearSegmentedColormap.from_list("BlueGreenYellowRed", colors, N=256)

# ================== 创建画布 ==================
fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=100, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.5, hspace=0.4)  # 调整子图间距

# ================== 统一坐标设置 ==================
#dbz
# xmin, xmax = -10, 70
# ymin, ymax = -10, 70
#Vr
xmin, xmax = -20, 20
ymin, ymax = -20, 20

# ================== 主绘图循环 ==================
for idx, model in enumerate(tqdm(models, desc="绘图进度", unit="model")):
    ax = axs[idx // 3, idx % 3]

    # 加载数据
    data = np.load(model["file"])
    pred, true = data["predictions"], data["targets"]

    # 计算核密度
    kde = gaussian_kde(np.vstack([pred, true]))
    density = kde(np.vstack([pred, true]))

    # 绘制散点图
    sc = ax.scatter(pred, true, c=density, s=10, alpha=0.8, cmap=custom_cmap, edgecolors='none')

    # 添加参考线
    ax.plot([xmin, xmax], [xmin, xmax], 'k--', lw=2)

    # 添加子图标签 (a), (b), ...
    ax.text(0.03, 0.92, f'({chr(97 + idx)}) {model["name"]}',
            transform=ax.transAxes, fontsize=16, fontweight='bold')

    ax.set_xticks(np.arange(xmin, xmax + 1, 10))
    ax.set_yticks(np.arange(ymin, ymax + 1, 10))

    # ================== 添加独立色标 ==================
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # 创建色标
    cbar = plt.colorbar(sc, cax=cax)

    # 设置色标刻度字体大小
    cbar.ax.tick_params(labelsize=14)

    # 获取并设置色标刻度
    ticks = np.linspace(density.min(), density.max(), num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick * 1e3:.0f}' for tick in ticks])

    # 添加科学计数法标签
    cbar.ax.set_title('×10⁻³', fontsize=10, pad=5)

    # ================== 格式设置 ==================
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.tick_params(axis='both', which='major', labelsize=14)

    if idx >= 3:
        ax.set_xlabel('Predicted (m/s)', fontsize=16, fontweight='bold')
    if idx % 3 == 0:
        ax.set_ylabel('Actual (m/s)', fontsize=16, fontweight='bold')

print("🎉 所有模型绘图完成！")

# ================== 最终调整 ==================
plt.tight_layout()
plt.savefig('D:/读研的司马事儿/雷达降尺度/小论文/radar_comparison_linear_vr.png', dpi=300, bbox_inches='tight')
plt.show()
