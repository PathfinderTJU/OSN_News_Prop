import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from dataloader import deephawkes

# 生成增长数量 / 时间曲线

cascade_network, user_network = deephawkes()
cascades = []

cas_sum = 0
# 提取每个级联的起始时间和大小
for cas in cascade_network:
    start_time = int(cas["start_time"])
    cascade_events = cas["Ec"]
    cascade_size = len(cascade_events)
    cascade_times = sorted([int(e[2]) for e in cascade_events])
    cascades.append((start_time, cascade_times, cascade_size))

    cas_sum += cascade_size

print(cas_sum / len(cascade_network))

# 计算每个级联的时间差和大小比例
time_diffs_per_cascade = []
size_ratios_per_cascade = []

for start_time, cascade_times, cascade_size in cascades:
    time_diffs = [(t - start_time) / 60 for t in cascade_times]  # 转换为小时
    size_ratios = np.arange(1, cascade_size + 1) / cascade_size
    time_diffs_per_cascade.append(time_diffs)
    size_ratios_per_cascade.append(size_ratios)

# 确定所有级联中的最长时间
longest_time = max([td[-1] for td in time_diffs_per_cascade])

# 创建一个时间范围（x轴）
time_range = np.linspace(0, longest_time, 2160)  # 可以调整为需要的精度

# 初始化一个数组来存储所有级联在不同时间点的平均大小比例
average_size_ratio_at_time = np.zeros_like(time_range)

# 对每个时间点，计算所有级联的平均大小比例
for i, time_point in enumerate(time_range):
    size_ratios = []
    for cascade_index, time_diffs in enumerate(time_diffs_per_cascade):
        # 查找当前时间点之前的最近事件
        indices = np.where(np.array(time_diffs) <= time_point)[0]
        if indices.size > 0:
            last_index = indices[-1]
            # 获取对应的大小比例
            size_ratio = size_ratios_per_cascade[cascade_index][last_index]
            size_ratios.append(size_ratio)
    # 计算当前时间点的平均大小比例
    if size_ratios:
        average_size_ratio_at_time[i] = np.mean(size_ratios)

# 定义幂律函数模型
def power_law(x, a, b):
    return a * np.power(x, b)

# 使用 curve_fit 函数来拟合模型参数
# 注意：忽略 time_range 中的第一个点，因为它是零点，可能会导致数学问题
valid_time_range = time_range[1:]
valid_average_size_ratio = average_size_ratio_at_time[1:]
params, covariance = curve_fit(power_law, valid_time_range, valid_average_size_ratio)

# 使用拟合参数来计算拟合值
fitted_vals = power_law(valid_time_range, *params)

print(params)

# 绘制曲线

plt.rcParams.update({'font.size': 20})
a, b = params
plt.text(0.5, 0.2, f'Fitted Curve: y = {a:.2f} * x^{b:.2f}', transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
plt.figure(figsize=(10, 5))
plt.plot(time_range, average_size_ratio_at_time, label='Average Cascade Size Ratio')
plt.plot(valid_time_range, fitted_vals, label='Power Law Fit')
plt.xlabel('Time (Minutes)')
plt.ylabel('Cascade Size')
plt.title('Average Cascade Size Over Time')
plt.legend()
plt.grid(True)
plt.savefig("time.png")
plt.show()