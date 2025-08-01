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

# 获取t时刻所有用户的用户活跃程度缓存
def activity(cascades, start_user):
    # 初始化一个字典来存储节点出现的次数
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
    cp = [cluster_coefficient(G, j) for j in neighbors]
    con_i_numerator = sum(cluster_coefficient(G, j) for j in neighbors)
    con_i_denominator = math.sqrt(sum(cluster_coefficient(G, k) ** 2 for k in neighbors))
    con_i = con_i_numerator / con_i_denominator if con_i_denominator != 0 else 0

    # 计算In_i
    In_i = num_i * con_i

    return In_i


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