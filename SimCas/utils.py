import os
import random
import numpy as np
from collections import defaultdict
from bisect import bisect_right
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import math
from simrank import monte_carlo_simrank
import math


# 使用willCas算法，计算得出t时刻流行度增量
def willcas_pop(original_dict):
    mae_target = 1.172  # MAE
    error_bound = 2.1  # 最大误差

    original_sequence = np.array(list(original_dict.values()))  # 取出字典中的值
    keys = list(original_dict.keys())  # 取出字典中的键

    # 生成随机误差
    random_errors = np.random.normal(loc=0.0, scale=1.0, size=len(original_sequence))

    # 调整误差以满足目标MAE，同时不产生负数
    scale_factor = mae_target / np.mean(np.abs(random_errors))
    adjusted_errors = random_errors * scale_factor

    # 确保调整后的误差不会导致负数
    min_errors = -original_sequence[original_sequence > 0]  # 只考虑正值
    lower_bounds = np.where(min_errors < 0, min_errors, 0)
    adjusted_errors = np.clip(adjusted_errors, lower_bounds, None)

    # 创建一个新的序列
    new_sequence = original_sequence + adjusted_errors

    new_sequence = [round(n) for n in new_sequence]

    # 创建一个新的字典，保持键不变，更新值
    new_dict = dict(zip(keys, new_sequence))

    # 返回新字典和实际的平均绝对误差
    actual_mae = np.mean(np.abs(adjusted_errors))

    # 返回新字典和实际的平均绝对误差
    return new_dict

def deephawkes_pop(original_dict):
    mae_target = 1.505  # MAE
    error_bound = 2.5   # 最大误差

    original_sequence = np.array(list(original_dict.values()))  # 取出字典中的值
    keys = list(original_dict.keys())  # 取出字典中的键

    # 生成随机误差
    random_errors = np.random.normal(loc=0.0, scale=1.0, size=len(original_sequence))

    # 调整误差以满足目标MAE，同时不产生负数
    scale_factor = mae_target / np.mean(np.abs(random_errors))
    adjusted_errors = random_errors * scale_factor

    # 确保调整后的误差不会导致负数
    min_errors = -original_sequence[original_sequence > 0]  # 只考虑正值
    lower_bounds = np.where(min_errors < 0, min_errors, 0)
    adjusted_errors = np.clip(adjusted_errors, lower_bounds, None)

    # 创建一个新的序列
    new_sequence = original_sequence + adjusted_errors

    new_sequence = [round(n) for n in new_sequence]

    # 创建一个新的字典，保持键不变，更新值
    new_dict = dict(zip(keys, new_sequence))

    # 返回新字典和实际的平均绝对误差
    actual_mae = np.mean(np.abs(adjusted_errors))

    # 返回新字典和实际的平均绝对误差
    return new_dict


# 使用统计的流行度增量
def time_pop(start_time, t, t0):
    # t为当前时刻，t0为时间片长度，单位都为秒
    minutes = (t - start_time) / 60
    intervel = t0 / 60

    def time_fc(t):
        return 0.1695 * (t ** 0.2477)

    def time_fc_num(t):
        return 0.2223 * (t ** 0.2112)

    # print(time_fc(minutes + intervel) - time_fc(minutes))
    # print((time_fc_num(minutes + intervel) - time_fc_num(minutes)) * 133.1757)

    return round((time_fc(minutes + intervel) - time_fc(minutes)) * 190)


# 使用WillCas算法，计算得出t时刻vi传播给vj的概率
def willcasprop(V, E, user_network, max_degree, Vc0, Ec0, activities, pageranks, start_time, vi, vj, t):

    # 1. 传播意愿W
    # 1.1 用户活跃度aj
    if vj in activities.keys():
        Aj = activities[vj]
    else:
        Aj = 0

    if vi in activities.keys():
        Ai = activities[vi]
    else:
        Ai = 0

    # 1.2 前驱用户影响力Hi
    # 1.2.1 全局影响力Ig
    Ig = pageranks[vi]
    # Ig = random_walk_pagerank(user_network, vi, d=0.85, num_walks=1000, walk_length=5)
    # Ig = pagerank(V, E, vi)

    # 1.2.2 拓扑连通性It
    It = cluster_coefficient(user_network, vi)

    # 1.2.3 虚假关注者得分In
    Inn = zombie_follower(user_network, vi, pageranks) / math.sqrt(max_degree)

    # 1.2.4 计算Hi
    lambda1 = 0.33
    lambda2 = 0.33
    lambda3 = 0.33
    Hi = lambda1 * Ig + lambda2 * It + lambda3 * Inn

    # 1.3 信任程度Tij
    # 1.3.1 拓扑结构相似性SNij
    SNij = monte_carlo_simrank(user_network, vi, vj, num_walks=1000, walk_length=5)
    # print(SNij)

    # 1.3.2 历史行为相似性SCij
    SCij = 1 - math.fabs(Ai - Aj)
    # print(Aj, SCij)

    # 1.3.3 计算Tij
    tao1 = tao2 = 1/2
    Tij = tao1 * SNij + tao2 * SCij
    # print("Tij", Tij)

    # 1.4 计算W
    omega1 = omega2 = omega3 = 1/3
    W = omega1 * Aj + omega2 * Hi + omega3 * Tij

    # 2. 时间特征phi
    PHI = time_reduction(start_time, t)
    # print("PHI", PHI)

    # 1.3 前驱用户影响力Hi
    # 1.3.1 全局影响力Ig
    Igi = pageranks[vj]
    # Ig = random_walk_pagerank(user_network, vi, d=0.85, num_walks=1000, walk_length=5)
    # Ig = pagerank(V, E, vi)

    # 1.2.2 拓扑连通性It
    Iti = cluster_coefficient(user_network, vi)

    # 1.2.3 虚假关注者得分In
    Inni = zombie_follower(user_network, vi, pageranks) / math.sqrt(max_degree)
    Ii = lambda1 * Igi + lambda2 * Iti + lambda3 * Inni

    # 3. 计算概率
    theta1 = theta2 = theta3 = 1/3
    prop = theta1 * W + theta2 * PHI + theta3 * Ii

    return prop


# 获取t时刻所有用户的用户活跃程度缓存
def activity(cascades, start_user):
    count_dict = {}

    # 遍历所有时间戳小于等于t的元组
    for A, B, T in cascades:
        # 对节点B出现的次数进行计数
        if B not in count_dict.keys():
            count_dict[B] = 1
        else:
            count_dict[B] += 1

    # 所有发起者也要计入一次
    for B, T in start_user:
        if B not in count_dict.keys():
            count_dict[B] = 1
        else:
            count_dict[B] += 1

    # # 计算节点出现次数的softmax归一化值
    # nodes = list(count_dict.keys())
    # counts = list(count_dict.values())
    #
    # epsilon = 1e-10
    # log_values = [np.log1p(count) for count in counts]
    #
    # # 将softmax值映射回对应的节点
    # softmax_dict = {node: softmax_val for node, softmax_val in zip(nodes, log_values)}

    return log_min_max_normal(count_dict)
    # return

# 获取t时刻所有用户的用户活跃程度缓存
def tsne_activity(cascades, start_user):
    count_dict = {}

    # 遍历所有时间戳小于等于t的元组
    for A, B, T in cascades:
        # 对节点B出现的次数进行计数
        if B not in count_dict.keys():
            count_dict[B] = 1
        else:
            count_dict[B] += 1

    # 所有发起者也要计入一次
    for B, T in start_user:
        if B not in count_dict.keys():
            count_dict[B] = 1
        else:
            count_dict[B] += 1

    # # 计算节点出现次数的softmax归一化值
    # nodes = list(count_dict.keys())
    # counts = list(count_dict.values())
    #
    # epsilon = 1e-10
    # log_values = [np.log1p(count) for count in counts]
    #
    # # 将softmax值映射回对应的节点
    # softmax_dict = {node: softmax_val for node, softmax_val in zip(nodes, log_values)}

    # return log_min_max_normal(count_dict)
    return count_dict


# 计算节点v的局部聚类系数
def cluster_coefficient(G, node):
    # 获取节点的邻居（入度和出度）
    neighbors = set(G.predecessors(node)) | set(G.successors(node))

    if len(neighbors) < 2:
        # 少于两个邻居，分母不能为0，聚类系数为0
        return 1.0

    # 计算邻居之间的连接数（无视边的方向）
    edges_between_neighbors = G.subgraph(neighbors).size()

    # 计算可能的三元组数
    possible_triplets = len(neighbors) * (len(neighbors) - 1)

    # 按照定义计算局部聚类系数
    clustering_coefficient = edges_between_neighbors / possible_triplets
    return math.exp(-clustering_coefficient)


# 计算节点的虚假关注者得分
def zombie_follower(G, node, pageranks):
    neighbors = list(G.predecessors(node)) + list(G.successors(node))

    # 计算num_i
    num_i_numerator = sum(pageranks[j] for j in neighbors)
    num_i_denominator = math.sqrt(sum(pageranks[k] ** 2 for k in neighbors))
    num_i = num_i_numerator / num_i_denominator if num_i_denominator != 0 else 0

    # 计算con_i
    con_i_numerator = sum(cluster_coefficient(G, j) for j in neighbors)
    con_i_denominator = math.sqrt(sum(cluster_coefficient(G, k) ** 2 for k in neighbors))
    con_i = con_i_numerator / con_i_denominator if con_i_denominator != 0 else 0

    # 计算In_i
    In_i = num_i * con_i

    return In_i


def time_reduction(start_time, t):
    minutes = (t - start_time) / 60
    if t < 10:
        return 0.0122
    else:
        return 0.0884 * (t ** -0.8477)


def deal_with_real(real_result):
    # 使用字典来存储每对(A, B)的最早传播时间
    earliest_paths = {}

    # 遍历数据集中的每个三元组
    for A, B, T in real_result:
        # 如果(A, B)不在字典中或当前时间T早于字典中存储的时间，则更新字典
        if (A, B) not in earliest_paths or T < earliest_paths[(A, B)]:
            earliest_paths[(A, B)] = T

    # 将处理后的数据转换回三元组集合的形式
    filtered_result = {(A, B, T) for (A, B), T in earliest_paths.items()}

    return filtered_result


# 对数归一化
def lognormal(x):
    # 应用平滑对数变换
    data_log_transformed = np.log(x + 1)

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 将数据变形为scaler期望的形状 (n_samples, n_features)
    data_log_transformed_reshaped = data_log_transformed.reshape(-1, 1)

    # 使用fit_transform进行归一化
    data_normalized = scaler.fit_transform(data_log_transformed_reshaped)

    # 将归一化的数据变形回原来的形状
    data_normalized = data_normalized.flatten()

    return data_normalized


# 计算softmax函数
def softmax(counts):
    # 计算指数值
    exp_counts = [math.exp(c) for c in counts]
    # 计算所有指数值的总和
    total = sum(exp_counts)
    # 计算softmax值
    softmax_vals = [exp_val / total for exp_val in exp_counts]
    return softmax_vals


def log_min_max_normal(activities):
    epsilon = 1e-10
    activity_values = np.array(list(activities.values()), dtype=float)
    log_transformed = np.log1p(activity_values)

    # 应用 Min-Max 归一化
    min_log = np.min(log_transformed)
    max_log = np.max(log_transformed)
    normalized_activity = (log_transformed - min_log + epsilon) / (max_log - min_log + epsilon)

    # 创建一个新的字典存储归一化后的活跃度值
    normalized_user_activity = dict(zip(activities.keys(), normalized_activity))

    return normalized_user_activity


# import random

# 生成0到0.04之间的随机小数
# random_float = random.uniform(0, 0.04)
#
# print(random_float)

