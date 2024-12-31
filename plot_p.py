import matplotlib.pyplot as plt
import numpy as np

f1_values_run1 = {
    "GCN": [80.10, 78.85, 84.60, 70.25, 65.30],
    "NIEFTY": [81.00, 69.30, 83.40, 68.50, 60.30],
    "FairGNN": [81.50, 77.55, 81.90, 68.80, 63.20],
    "FairVGNN": [81.55, 79.60, 87.50, 68.30, 62.40],
    "FuGNN": [82.40, 87.80, 85.30, 68.20, 62.10],
    "FairSAD": [82.35, 78.20, 87.00, 69.00, 62.60],
    "Ours": [82.65, 79.35, 87.70, 70.70, 62.40]
}


eo_values_run1 = {
    "GCN": [28.80, 4.90, 9.60, 4.92, 11.20],
    "NIEFTY": [2.40, 2.70, 6.50, 2.80, 7.20],
    "FairGNN": [3.30, 4.60, 4.00, 2.10, 3.10],
    "FairVGNN": [3.00, 5.00, 1.30, 2.10, 4.70],
    "FuGNN": [0.30, 2.00, 1.00, 1.35, 1.15],
    "FairSAD": [0.05, 1.80, 1.00, 1.38, 2.90],
    "Ours": [0.01, 0.95, 0.50, 1.05, 0.65]
}
f1_values_run2 = {
    "GCN": [80.05, 78.80, 84.50, 70.15, 65.20],
    "NIEFTY": [81.10, 69.25, 83.55, 68.55, 60.25],
    "FairGNN": [81.42, 77.40, 81.85, 68.78, 63.18],
    "FairVGNN": [81.48, 79.65, 87.48, 68.32, 62.38],
    "FuGNN": [82.45, 87.85, 85.35, 68.25, 62.12],
    "FairSAD": [82.25, 78.18, 86.98, 69.02, 62.58],
    "Ours": [82.60, 79.30, 87.66, 70.65, 62.37]
}
eo_values_run2 = {
    "GCN": [28.75, 4.85, 9.62, 4.90, 11.25],
    "NIEFTY": [2.45, 2.68, 6.48, 2.85, 7.25],
    "FairGNN": [3.35, 4.68, 3.95, 2.14, 3.12],
    "FairVGNN": [3.08, 4.93, 1.33, 2.11, 4.75],
    "FuGNN": [0.32, 1.98, 1.03, 1.36, 1.16],
    "FairSAD": [0.04, 1.78, 1.01, 1.39, 2.94],
    "Ours": [0.01, 0.97, 0.51, 1.04, 0.66]
}
f1_values_run3 = {
    "GCN": [80.00, 78.90, 84.55, 70.18, 65.25],
    "NIEFTY": [81.15, 69.20, 83.50, 68.52, 60.22],
    "FairGNN": [81.48, 77.45, 81.88, 68.81, 63.19],
    "FairVGNN": [81.50, 79.67, 87.52, 68.33, 62.36],
    "FuGNN": [82.48, 87.88, 85.40, 68.27, 62.13],
    "FairSAD": [82.28, 78.21, 87.01, 69.01, 62.61],
    "Ours": [82.63, 79.32, 87.68, 70.68, 62.39]
}
eo_values_run3 = {
    "GCN": [28.77, 4.88, 9.59, 4.94, 11.22],
    "NIEFTY": [2.42, 2.66, 6.46, 2.84, 7.23],
    "FairGNN": [3.38, 4.63, 3.98, 2.13, 3.13],
    "FairVGNN": [3.02, 4.92, 1.32, 2.13, 4.72],
    "FuGNN": [0.33, 1.97, 1.01, 1.34, 1.14],
    "FairSAD": [0.03, 1.77, 1.03, 1.37, 2.93],
    "Ours": [0.01, 0.96, 0.53, 1.03, 0.64]
}


# 帕累托前沿计算函数
def pareto_frontier(x, y, maximize_x=True, minimize_y=True):
    sorted_indices = np.lexsort((y, -x))
    pareto_indices = []
    best_y = float('inf') if minimize_y else float('-inf')
    for idx in sorted_indices:
        if (y[idx] < best_y and minimize_y) or (y[idx] > best_y and not minimize_y):
            pareto_indices.append(idx)
            best_y = y[idx]
    return pareto_indices


# 绘图准备
methods = list(f1_values_run1.keys())
datasets = ["German", "Bail", "Credit", "Pokec-z", "Pokec-n"]
methods = ["GCN", "NIEFTY", "FairGNN", "FairVGNN", "FuGNN", "FairSAD", "Ours"]
markers = ['o', 'v', 's', 'p', '*', 'X', 'D']
colors = ['b', 'g', 'k', 'c', 'm', 'y', 'r']
sizes = [140, 140, 140, 140, 140, 140, 150]
fig, axes = plt.subplots(1, 5, figsize=(24.5, 4.5), sharey=False)

for i, dataset in enumerate(datasets):
    ax = axes[i]
    for j, method in enumerate(methods):
        # 三次实验结果
        f1_list = [f1_values_run1[method][i], f1_values_run2[method][i], f1_values_run3[method][i]]
        eo_list = [eo_values_run1[method][i], eo_values_run2[method][i], eo_values_run3[method][i]]

        # 绘制所有结果点（不同实验结果）
        ax.scatter(f1_list, eo_list, 
                   color=colors[j], 
                   marker=markers[j], 
                   s=sizes[j], 
                   label=method if i == 0 else "")

        # 帕累托前沿
        pareto_indices = pareto_frontier(np.array(f1_list), np.array(eo_list))
        pareto_x = [f1_list[idx] for idx in pareto_indices]
        pareto_y = [eo_list[idx] for idx in pareto_indices]
        pareto_x, pareto_y = zip(*sorted(zip(pareto_x, pareto_y)))
        ax.plot(pareto_x, pareto_y, '--', color=colors[j], linewidth=2)

    # 设置图像标题和坐标轴
    ax.set_title(dataset, fontsize=16)
    ax.set_xlabel("F1-score (%)", fontsize=14)
    if i == 0:
        ax.set_ylabel("ΔEO (↓)", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True)

# 图例
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(methods), fontsize=12)

# 布局和保存
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/yzhen/code/fair/Pareto_F1_vs_EO_all_runs.png', dpi=400)
plt.show()