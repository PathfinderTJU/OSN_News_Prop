import matplotlib.pyplot as plt
from dataloader import deephawkes
from utils import deephawkes_pop

cascades_network, user_network = deephawkes()

cascade = cascades_network[-1]
print(cascade["ID"])


start_time = int(cascade["start_time"])
te = start_time + 3600 * 12

origin_pop = {}
# 遍历每个三元组 (A, B, T)
for A, B, T in cascade["Ec"]:
    # 计算相对于起始时间的时间差（秒）
    time_diff = int(T) - start_time

    # 找到时间差落在哪个300秒的时间间隔内
    interval = (time_diff // 300) * 300

    # 更新流行度字典，记录该时间间隔的路径数量
    if interval in origin_pop:
        origin_pop[interval] += 1
    else:
        origin_pop[interval] = 1

dp_pop = deephawkes_pop(origin_pop)

sorted_pop = sorted(dp_pop.items())

new_dict = []

for i in sorted_pop:
    new_dict.append({"time": int(i[0]), "value": int(i[1])})

print(new_dict)

def calculate_y(t):
    return 9.6562 * (t ** 0.3417)

def generate_dataset():
    dataset = []
    time_slice = 300  # 时间片长度，单位为秒
    for t in range(0, 4 * 3600, time_slice):  # 0到12小时的时间范围
        y_end = calculate_y(t + time_slice)  # 时间片结束时的y值
        y_start = calculate_y(t)  # 时间片开始时的y值
        increment = y_end - y_start  # 增量
        dataset.append({'time': int(t), 'value': int(increment)})  # 存储为字典，并转换为int类型
    return dataset

# 生成数据集
data_list = generate_dataset()

# print(data_list)
#
# durations = []
# for cascade in cascades_list:
#     start_time = int(cascade['start_time'])
#     latest_time = int(start_time)
#     for path in cascade['Ec']:
#         _, _, T = path
#         if int(T) > latest_time:
#             latest_time = int(T)
#     duration = latest_time - start_time
#     durations.append(duration)
#
# # 将持续时间转换为小时（如果它们是以秒为单位的时间戳）
# durations_hours = [duration / 3600 for duration in durations]
#
# # 绘制持续时间的分布图
# plt.figure(figsize=(10, 6))  # 设置图像大小
# plt.hist(durations_hours, bins=50, alpha=0.7, color='blue')  # bins参数根据数据的特性调整
# plt.title('Distribution of Cascade Durations')
# plt.xlabel('Duration (hours)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# 假设你有一个整数列表，例如：
integer_list = []

for cas in  cascades_network:
    integer_list.append(len(cas["Ec"]))

# 使用matplotlib绘制直方图
plt.rcParams.update({'font.size': 20})
plt.hist(integer_list, bins='auto', alpha=0.7, color='blue', edgecolor='black')

# 计算小于100的整数数量和比例
count_less_than_100 = sum(1 for i in integer_list if i < 100)
proportion_less_than_100 = count_less_than_100 / len(integer_list)
percentage_less_than_100 = proportion_less_than_100 * 100

# 在直方图上添加文本显示小于100的整数数量比例
plt.text(max(integer_list)*0.6, max(plt.ylim())*0.9, f'Proportion < 100: {percentage_less_than_100:.2f}%', fontsize=20, color='red')

# 命名坐标轴
plt.xlabel('Cascade Size')
plt.ylabel('Number of Cascades')

# 添加标题
plt.title('Histogram of Cascade Size Distribution')

# 显示网格（可选）
plt.grid(True)

# 显示图表
plt.show()
plt.savefig("Size.png")
plt.close()