import os
from collections import defaultdict

filepath = "./simulate_result.txt"
with open(filepath, "r") as f:
    dataset = [line.strip() for line in f]

results = []
i = 0
while i < len(dataset):
    res = dataset[i + 1]
    results.append(eval(res))
    i += 3

data = []
for r in results:
    data.extend(list(r))

# print(data)

def count_nodes(transmissions, target_nodes):
    # 创建字典来存储时间戳以及相关的统计数据
    stats = defaultdict(lambda: {"time": 0})

    # 为每个target_node初始化计数器
    node_counts = {node: 0 for node in target_nodes}

    # 初始化传输总数计数器
    transmission_count = 0

    # 找到最早的时间戳
    min_time = min(transmissions, key=lambda x: x[2])[2]

    # 先按时间戳T排序传播列表
    transmissions.sort(key=lambda x: x[2])

    for A, B, T in transmissions:
        # 计算与最早时间戳的差值
        time_diff = T - min_time

        # 累积传输总数
        transmission_count += 1

        # 如果这是第一次见到这个时间戳，初始化它
        if T not in stats:
            stats[T] = {"time": time_diff, "sum": transmission_count}
            # 继承之前所有时间点的节点计数
            for node in target_nodes:
                stats[T][node] = node_counts[node]
        else:
            # 更新当前时间戳的传输总数
            stats[T]["sum"] = transmission_count

        # 如果A是目标节点之一，累积计数
        if A in target_nodes:
            stats[T][A] += 1
            node_counts[A] += 1

    # 转换stats为列表，并按照time排序
    stats_list = sorted([value for key, value in stats.items()], key=lambda x: x["time"])

    return stats_list



# 目标节点
target_nodes = ['925941', '1373670', '1217099', '1030233', '1003900', '1217106']

# 调用函数
stats = count_nodes(data, target_nodes)

# 输出结果
for s in stats:
    print(s)