import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataloader import deephawkes

ob_time = 3600
cascade_network, user_network = deephawkes(ob_time)

user_network_G = nx.DiGraph()
user_network_G.add_edges_from(user_network["E"])
n = len(user_network["V"])

print("平均聚类系数C", nx.average_clustering(user_network_G))

lengths = []
counts = []
for component in nx.weakly_connected_components(user_network_G):
    subgraph = user_network_G.subgraph(component)
    avg_length_comp = nx.average_shortest_path_length(subgraph)
    lengths.append(avg_length_comp)
    counts.append(len(component))
# 计算加权平均值
total_count = sum(counts)
avg_length_weighted = sum(l * c for l, c in zip(lengths, counts)) / total_count
print("平均路径长度L", avg_length_weighted)

degrees = dict(user_network_G.degree())

kbar = np.mean(list(degrees.values()))

print("等规模平均路径长度Lbar",math.log(n) / math.log(kbar))

print("等规模平均聚类系数Cbar",kbar / n)