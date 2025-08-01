import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from dataloader import deephawkes

## 生成增长概率 / 时间曲线

def constant_fit(x, a):
    return a * np.ones_like(x)

def power_law(x, A, k):
    return A * np.power(x, k)

cascade_network, user_network = deephawkes()
cascades = []

for cas in cascade_network:
    start_time = int(cas["start_time"])
    cascade_time = [int(e[2]) for e in cas["Ec"]]
    cascades.append((start_time, cascade_time))

# 解析数据并计算时间差（以分钟为单位）
time_diffs_per_cascade = []
probabilities_per_cascade = []


for start_time, propagation_times in cascades:
    time_diffs = [(t - start_time) / 60 for t in propagation_times if t > start_time]  # 确保时间差为正
    time_diffs_per_cascade.append(time_diffs)

    # 计算每个时间差的增量
    incremental_counts = np.bincount(np.array(time_diffs).astype(int))
    total_incremental_count = np.sum(incremental_counts)

    # 计算每个时间差的概率
    probabilities = incremental_counts / total_incremental_count if total_incremental_count > 0 else np.zeros_like(
        incremental_counts)
    probabilities_per_cascade.append(probabilities)


# 计算所有级联的最大时间差
max_time_diff = max([len(p) for p in probabilities_per_cascade])

# 初始化一个数组来存储所有时间差的平均概率
average_probabilities = np.zeros(max_time_diff)

# 对于每个时间差，计算所有级联的该时间差的概率平均值
for p in probabilities_per_cascade:
    average_probabilities[:len(p)] += p

average_probabilities /= len(cascades)

times = np.arange(len(average_probabilities))

# 分段拟合，分为x<10和x>=10两部分
x_break = 10
x_constant = times[times < x_break]
y_constant = average_probabilities[times < x_break]
x_power_law = times[times >= x_break]
y_power_law = average_probabilities[times >= x_break]

# 使用 curve_fit 来进行常数拟合
params_constant, _ = curve_fit(constant_fit, x_constant, y_constant)

# 使用 curve_fit 来进行幂律分布拟合
params_power_law, _ = curve_fit(power_law, x_power_law, y_power_law, p0=[1, -1])

# 绘制增量级联增长的概率分布和拟合的函数
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12, 6))
plt.scatter(times, average_probabilities, color='blue', alpha=0.6, label='Empirical Data')

# 绘制常数部分拟合
plt.plot(x_constant, constant_fit(x_constant, *params_constant), 'g--',
         label=f'Constant Fit: y={params_constant[0]:.4f} (x<{x_break})')

# 绘制幂律部分拟合
plt.plot(x_power_law, power_law(x_power_law, *params_power_law), 'r--',
         label=f'Power Law Fit: y={params_power_law[0]:.4f}*x^{params_power_law[1]:.4f} (x>={x_break})')

plt.xlabel('Time (minutes)')
plt.ylabel('Average Incremental Probability')
plt.title('Average Incremental Distribution of Cascades Size Over Time')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.savefig("time_prob.png")
plt.show()