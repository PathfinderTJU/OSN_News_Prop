import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
from dataloader import deephawkes
from utils import activity, cluster_coefficient, zombie_follower
import networkx as nx
import time

cascade_network, user_network = deephawkes()

cascade = cascade_network[25588]
print(cascade["ID"])

V = user_network["V"]
E = user_network["E"]

G = nx.DiGraph()
G.add_edges_from(E)


def get_in_edge_neighbors(G, node_set, depth):
    in_edge_neighbors = set()
    edges_list = []

    # 初始节点集合的入边邻居
    for node in node_set:
        in_edges = G.in_edges(node)
        for edge in in_edges:
            in_edge_neighbors.add(edge[0])
            edges_list.append(edge)

    # 如果深度大于1，则递归调用函数，直到达到所需深度
    if depth > 1:
        more_in_edge_neighbors, more_edges_list = get_in_edge_neighbors(G, in_edge_neighbors, depth - 1)
        in_edge_neighbors.update(more_in_edge_neighbors)
        edges_list.extend(more_edges_list)

    return in_edge_neighbors, edges_list


# 调用函数获取入边邻居及其边，重复三次
final_neighbors, final_edges_list = get_in_edge_neighbors(G, cascade["Vc"], 3)
final_node_list = []
# 最终结果：final_edges_list是包含所有找到的边的列表
# print(final_edges_list)
for e in final_edges_list:
    if e[0] not in final_node_list:
        final_node_list.append(e[0])
    if e[1] not in final_node_list:
        final_node_list.append(e[1])
# print(final_node_list.__len__())
# print(final_edges_list.__len__())


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

final_results = []
for n in final_node_list:
    random.seed(time.time_ns())
    r1 = random.uniform(0.999, 1.001)
    random.seed(time.time_ns())
    r2 = random.uniform(1.0, 1.01)
    # a = activities[n] * r1 ** r2
    if n not in activities:
        a = 1e-5
    else:
        a = activities[n]

    Ig = pageranks[n]
    # Ig = random_walk_pagerank(user_network, vi, d=0.85, num_walks=1000, walk_length=5)
    # Ig = pagerank(V, E, vi)

    # 1.2.2 拓扑连通性It
    It = cluster_coefficient(G, n)

    # 1.2.3 虚假关注者得分In
    Inn = zombie_follower(G, n, pageranks)

    lambda1 = 0.33
    lambda2 = 0.33
    lambda3 = 0.33
    random.seed(time.time_ns())
    Hi = (lambda1 * Ig + lambda2 * It + lambda3 * Inn)
    # Hi = (lambda1 * Ig + lambda2 * It + lambda3 * Inn) + random.uniform(-0.0001, 0.0001)
    final_results.append([a, Hi])


X = np.array(final_results)
noise = np.random.normal(0, 1e-4, X.shape)
X = X + noise

# 初始化t-SNE模型
tsne = TSNE(n_components=2, random_state=40, perplexity=38, learning_rate=80)

# 使用t-SNE进行降维
X_tsne = tsne.fit_transform(X)

# 使用K-means聚类算法自动分类
kmeans = KMeans(n_clusters=4, random_state=10)  # 假设我们想要将数据分类成4个类别
kmeans.fit(X_tsne)
y_kmeans = kmeans.predict(X_tsne)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.title('t-SNE Visualization with K-means Clustering')
plt.show()
plt.savefig("tsne.png")

output_data = []
for x in list(X_tsne):
    output_data.append(list(x))
print(output_data)
print(list(y_kmeans))