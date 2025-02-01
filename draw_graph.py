import numpy as np
import matplotlib.pyplot as plt

# 数据集和模型名称
datasets = ["Dataset A", "Dataset B"]
models = ["GPT-4o", "DeepSeek-r1"]
metrics = ["F1 Score", "Accuracy"]

# 假设的实验数据
data = {
    "Dataset A": {"GPT-4o": {"F1 Score": 0.85, "Accuracy": 0.87},
                  "DeepSeek-r1": {"F1 Score": 0.88, "Accuracy": 0.89}},
    "Dataset B": {"GPT-4o": {"F1 Score": 0.80, "Accuracy": 0.83},
                  "DeepSeek-r1": {"F1 Score": 0.84, "Accuracy": 0.86}},
}

# 设置柱状图参数
bar_width = 0.2  # 每个柱子的宽度
x = np.arange(len(datasets))  # 数据集的位置
offsets = [-0.3, -0.1, 0.1, 0.3]  # 4根柱子的偏移

# 提取数据
f1_gpt4o = [data[d]["GPT-4o"]["F1 Score"] for d in datasets]
acc_gpt4o = [data[d]["GPT-4o"]["Accuracy"] for d in datasets]
f1_deepseek = [data[d]["DeepSeek-r1"]["F1 Score"] for d in datasets]
acc_deepseek = [data[d]["DeepSeek-r1"]["Accuracy"] for d in datasets]

# 颜色方案（同一模型的 F1 和 Accuracy 颜色相同）
colors = ["blue", "blue", "orange", "orange"]

# 画图
plt.figure(figsize=(12, 7))
bars = []
bars.append(plt.bar(x + offsets[0], f1_gpt4o, bar_width, color=colors[0], label="GPT-4o"))
bars.append(plt.bar(x + offsets[1], acc_gpt4o, bar_width, color=colors[0], alpha=0.6))  # 透明度区分
bars.append(plt.bar(x + offsets[2], f1_deepseek, bar_width, color=colors[2], label="DeepSeek-r1"))
bars.append(plt.bar(x + offsets[3], acc_deepseek, bar_width, color=colors[2], alpha=0.6))  # 透明度区分

# 设置标签
plt.title("Model Performance on Two Datasets", fontsize=16)
plt.xlabel("Datasets", fontsize=14)
plt.ylabel("Scores", fontsize=14)
plt.xticks(x, datasets)
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)

# 标注数值
for i, bars_set in enumerate(bars):
    for bar in bars_set:
        height = bar.get_height()
        label_text = "F1 Score"
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
                 ha="center", va="bottom", fontsize=12)
        if i % 2 == 0:  # 在 F1 Score 和 Accuracy 之间留空，并标注
            plt.text(bar.get_x() + bar.get_width()/2, 0.05, label_text,
                     ha="center", va="bottom", fontsize=12, fontweight="bold")

# 重新调整图例位置，避免遮挡柱子
plt.legend(loc="upper right", fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()
