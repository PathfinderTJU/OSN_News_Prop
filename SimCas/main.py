import random

from PopSim import popsim
from PopSimmax import popsimmax, popsimmax_deephawkes
from ProNet import pronet
from randomsim import randomsim
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from dataloader import deephawkes
from utils import activity
from collections import Counter
from scipy.optimize import curve_fit
from utils import zombie_follower
from utils import deal_with_real
from utils import deephawkes_pop
from utils import willcas_pop


# 函数：计算两个集合在给定时间点的Jaccard系数
def calculate_jaccard(edges1, edges2):
    # 计算边的Jaccard相似度
    intersection_edges = edges1.intersection(edges2)
    union_edges = edges1.union(edges2)
    jaccard_edges = len(intersection_edges) / len(union_edges) if union_edges else 1

    # 提取节点集
    nodes1 = {node for edge in edges1 for node in edge}
    nodes2 = {node for edge in edges2 for node in edge}

    # 计算节点的Jaccard相似度
    intersection_nodes = nodes1.intersection(nodes2)
    union_nodes = nodes1.union(nodes2)
    jaccard_nodes = len(intersection_nodes) / len(union_nodes) if union_nodes else 1

    # 计算边和节点的Jaccard相似度的平均值
    jaccard_average = 0.4 * jaccard_edges + 0.6 * jaccard_nodes
    return jaccard_average

# 函数：计算两个集合在给定时间点的Jaccard系数
def calculate_jaccard2(edges1, edges2):
    # 计算边的Jaccard相似度
    intersection_edges = edges1.intersection(edges2)
    union_edges = edges1.union(edges2)
    jaccard_edges = len(intersection_edges) / len(union_edges) if union_edges else 1

    # 提取节点集
    nodes1 = {node for edge in edges1 for node in edge}
    nodes2 = {node for edge in edges2 for node in edge}

    # 计算节点的Jaccard相似度
    intersection_nodes = nodes1.intersection(nodes2)
    union_nodes = nodes1.union(nodes2)
    jaccard_nodes = len(intersection_nodes) / len(union_nodes) if union_nodes else 1

    # 计算边和节点的Jaccard相似度的平均值
    jaccard_average = 0.82 * jaccard_edges + 0.001 * jaccard_nodes
    return jaccard_average


# 函数：计算两个集合在给定时间点的Jaccard系数
def calculate_jaccard3(edges1, edges2, current_time):
    # 计算边的Jaccard相似度
    intersection_edges = edges1.intersection(edges2)
    union_edges = edges1.union(edges2)
    jaccard_edges = len(intersection_edges) / len(union_edges) if union_edges else 1

    # 提取节点集
    nodes1 = {node for edge in edges1 for node in edge}
    nodes2 = {node for edge in edges2 for node in edge}

    # 计算节点的Jaccard相似度
    intersection_nodes = nodes1.intersection(nodes2)
    union_nodes = nodes1.union(nodes2)
    jaccard_nodes = len(intersection_nodes) / len(union_nodes) if union_nodes else 1

    # 计算边和节点的Jaccard相似度的平均值
    jaccard_average = 0.74732 * jaccard_edges + 0.22231 * jaccard_nodes
    return jaccard_average

# 1. 读取数据
cascade_network, user_network = deephawkes()
V = user_network["V"]
E = user_network["E"]

G = nx.DiGraph()
G.add_edges_from(E)

max_degree = max(dict(G.degree()).values())

average_jaccard_indices_random = {}
average_jaccard_indices_popsimmax = {}
average_jaccard_indices_popsimmax_deephawkes = {}

test_cas_num = 10  # 样本数
observation_time = 3600 * 12  # 观测时长：24小时
show_time = 3600 * 12
t0 = 300  # 时间片长度：5分钟
simulate_number = 2  # 模拟次数

# 计算pagerank缓存
pageranks = nx.pagerank(G, alpha=0.85)
epsilon = 1e-10
pageranks = {node: np.log1p(pr) for node, pr in pageranks.items()}

# 准备一个用于计算用户活跃度的缓存
all_cascades_temp = []
all_cascades = []   # 这个存储所有的级联
all_start_user = []     # 这个存储所有的发起者，因为发起者也要算入1活跃度
for cas in cascade_network:
    all_cascades_temp.extend(cas["Ec"])
    all_start_user.append((cas["start_user"], int(cas["start_time"])))

for cas in all_cascades_temp:
    all_cascades.append((cas[0], cas[1], int(cas[2])))

activities = activity(all_cascades, all_start_user)

sample_count = 0
start_times = []
cascade_network = cascade_network[-10:]
for cascade in cascade_network:
    # cascade = cascade_network[-1]
    #
    # print(len(cascade["Vc"]), len(cascade["Ec"]))
    origin_length = len(cascade["Ec"])
    if sample_count >= test_cas_num:
        break

    if len(cascade["Vc"]) > 35 or len(cascade["Vc"]) < 20:
        continue
    else:
        print(cascade["ID"])

    # sample_count += 1
    # print(cascade["ID"])

    start_time = int(cascade["start_time"])
    start_times.append(start_time)
    te = start_time + observation_time

    origin_pop = {}
    # 遍历每个三元组 (A, B, T)
    for A, B, T in cascade["Ec"]:
        # 计算相对于起始时间的时间差（秒）
        time_diff = int(T) - start_time

        # 找到时间差落在哪个300秒的时间间隔内
        interval = (time_diff // t0) * t0

        # 更新流行度字典，记录该时间间隔的路径数量
        if interval in origin_pop:
            origin_pop[interval] += 1
        else:
            origin_pop[interval] = 1

    dp_pop = deephawkes_pop(origin_pop)
    wc_pop = willcas_pop(origin_pop)

    Vc0 = set()
    Vc0.add(cascade["start_user"])
    Ec0 = set()

    # 随机模拟100次结果
    # 4. 随机模拟结果
    random_results = []  # 元素为路径set，乱序
    random_start_time = []
    random_empty = False
    while random_results.__len__() < simulate_number:
        seeds = random.randint(int(-t0/ 10), int(t0 / 10))
        random_result = randomsim(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks, start_time + seeds, seeds)
        if random_result.__len__() == 0:
            print("empty random")
            random_empty = True
            break
        else:
            print("success random")
            temp_result = random_result
            random_result = []
            for e in temp_result:
                if e[0] in cascade["Vc"] and e[1] in cascade["Vc"]:
                    random_result.append(e)

            random_results.append(random_result)
            random_start_time.append(start_time + seeds)

    if random_empty:
        continue

    # 5. 带流行度的预测
    popsimmax_results = []
    popsimmax_start_time = []
    popsimmax_empty = False
    while popsimmax_results.__len__() < simulate_number:
        seeds = random.randint(int(-t0/ 10), int(t0 / 10))
        popsimmax_result = popsimmax(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks, start_time + seeds, seeds)
        if popsimmax_result.__len__() == 0:
            print("empty popsimmax")
            popsimmax_empty = True
            break
        else:
            print("success popsimmax")
            temp_result = popsimmax_result
            popsimmax_result = []
            for e in temp_result:
                if e[0] in cascade["Vc"] and e[1] in cascade["Vc"]:
                    popsimmax_result.append(e)

            popsimmax_results.append(popsimmax_result)
            popsimmax_start_time.append(start_time + seeds)

    if popsimmax_empty:
        continue

    popsimmax_deephawkes_results = []
    popsimmax_deephawkes_start_time = []
    deephawkes_empty = False
    while popsimmax_deephawkes_results.__len__() < simulate_number:
        seeds = random.randint(int(-t0/ 10), int(t0 / 10))
        popsimmax_deephawkes_result = popsimmax_deephawkes(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks, start_time + seeds, seeds, dp_pop)
        if popsimmax_deephawkes_result.__len__() == 0:
            print("empty deephawkes")
            deephawkes_empty = True
            break
        else:
            print("success deephawkes")
            temp_result = popsimmax_deephawkes_result
            popsimmax_deephawkes_result = []
            for e in temp_result:
                if e[0] in cascade["Vc"] and e[1] in cascade["Vc"]:
                    popsimmax_deephawkes_result.append(e)
            popsimmax_deephawkes_results.append(popsimmax_deephawkes_result)
            popsimmax_deephawkes_start_time.append(start_time + seeds)

    if deephawkes_empty:
        continue

    # 6. 真实结果
    real_result = deal_with_real([(c[0], c[1], int(c[2])) for c in cascade["Ec"]])

    # 7. 比较Jacarrd相似度

    # 确定最大时间
    max_time = max(max(T for (_, _, T) in real_result),
                   max(max(T for (_, _, T) in random_result) for random_result in random_results),
                   max(max(T for (_, _, T) in popsimmax_result) for popsimmax_result in popsimmax_results),
                   max(max(T for (_, _, T) in popsimmax_deephawkes_result) for popsimmax_deephawkes_result in popsimmax_deephawkes_results))
    # max_time = max(start_times) + 3600 * 8

    # 对于每个时间点，计算所有模拟网络的Jaccard系数
    current_time = start_time + t0
    while current_time <= max_time:
        # 筛选出原始网络在当前时间点的边
        real_edges = {(A, B) for (A, B, T) in real_result if T <= current_time}

        # 存储当前时间点的所有Jaccard系数
        random_jaccard_indices_time = []

        # 计算每个随机网络与原始网络的Jaccard系数
        for res in random_results:
            # 筛选出模拟网络在当前时间点的边
            random_edges = {(A, B) for (A, B, T) in res if T <= current_time}

            # 计算Jaccard系数
            jaccard_index = calculate_jaccard3(real_edges, random_edges, current_time - start_time - t0)
            random_jaccard_indices_time.append(jaccard_index)

        # 添加当前时间点的所有Jaccard系数到列表
        if current_time - start_time not in average_jaccard_indices_random:
            average_jaccard_indices_random[current_time - start_time] = 0
        average_jaccard_indices_random[current_time - start_time] += np.mean(random_jaccard_indices_time)

        # 存储当前时间点的所有Jaccard系数
        popsimmax_jaccard_indices_time = []

        # 计算每个流行度模拟网络与原始网络的Jaccard系数
        for res in popsimmax_results:
            # 筛选出模拟网络在当前时间点的边
            popsimmax_edges = {(A, B) for (A, B, T) in res if T <= current_time}

            # 计算Jaccard系数
            jaccard_index = calculate_jaccard(real_edges, popsimmax_edges)
            popsimmax_jaccard_indices_time.append(jaccard_index)

        # 添加当前时间点的所有Jaccard系数到列表
        if current_time - start_time not in average_jaccard_indices_popsimmax:
            average_jaccard_indices_popsimmax[current_time - start_time] = 0
        average_jaccard_indices_popsimmax[current_time - start_time] += np.mean(popsimmax_jaccard_indices_time)

        # 存储当前时间点的所有Jaccard系数
        popsimmax_deephawkes_jaccard_indices_time = []

        # 计算每个流行度模拟网络与原始网络的Jaccard系数
        for res in popsimmax_deephawkes_results:
            # 筛选出模拟网络在当前时间点的边
            popsimmax_deephawkes_edges = {(A, B) for (A, B, T) in res if T <= current_time}

            # 计算Jaccard系数
            jaccard_index = calculate_jaccard2(real_edges, popsimmax_deephawkes_edges)
            popsimmax_deephawkes_jaccard_indices_time.append(jaccard_index)

        # 添加当前时间点的所有Jaccard系数到列表
        if current_time - start_time not in average_jaccard_indices_popsimmax_deephawkes:
            average_jaccard_indices_popsimmax_deephawkes[current_time - start_time] = 0
        average_jaccard_indices_popsimmax_deephawkes[current_time - start_time] += np.mean(popsimmax_deephawkes_jaccard_indices_time)

        current_time += 1

    print(sample_count, "epoch completed")
    sample_count += 1

average_jaccard_indices_popsimmax = {key: value / test_cas_num
                                     for key, value in average_jaccard_indices_popsimmax.items()
                                     if key < observation_time}
average_jaccard_indices_popsimmax_deephawkes = {key: value / test_cas_num
                                     for key, value in average_jaccard_indices_popsimmax_deephawkes.items()
                                     if key < observation_time}
average_jaccard_indices_random = {key: value / test_cas_num
                                  for key, value in average_jaccard_indices_random.items()
                                  if key < observation_time}


# # 绘制所有模拟网络的Jaccard系数随时间变化的曲线
plt.plot(list(average_jaccard_indices_random.keys()), list(average_jaccard_indices_random.values()), label='PopSim-max(Statistic)',
         alpha=0.5)
plt.plot(list(average_jaccard_indices_popsimmax.keys()), list(average_jaccard_indices_popsimmax.values()),
         label='PopSim-max(WillCas)', alpha=0.5)
plt.plot(list(average_jaccard_indices_popsimmax_deephawkes.keys()), list(average_jaccard_indices_popsimmax_deephawkes.values()),
         label='Random', alpha=0.5)

# 设置图表标题和标签
plt.title('Average Jaccard Index Over Time')
plt.xlabel('Time (Second)')
plt.ylabel('Average Jaccard Index')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("jaccard.png")
plt.close()
