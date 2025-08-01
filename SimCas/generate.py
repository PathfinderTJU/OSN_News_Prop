from PopSim import popsim
from PopSimmax import popsimmax
from ProNet import pronet
from randomsim import randomsim
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time
import random
from dataloader import deephawkes
from utils import activity
from collections import Counter
from scipy.optimize import curve_fit
from ProNet import pronet
from utils import deephawkes_pop,activity,zombie_follower,cluster_coefficient, tsne_activity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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


# 1. 读取数据
cascade_network, user_network = deephawkes()
V = user_network["V"]
E = user_network["E"]
# print(cascade_network)
cascade = cascade_network[-1]
# print(cascade)
print(cascade["ID"], len(cascade["Vc"]), len(cascade["Ec"]))

G = nx.DiGraph()
G.add_edges_from(E)

max_degree = max(dict(G.degree()).values())

start_time = int(cascade["start_time"])
te = start_time + 3600 * 12      # 观测时长：24小时
t0 = 300     # 时间片长度：1分钟

Vc0 = set()
Vc0.add(cascade["start_user"])
Ec0 = set()

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

# 按时间从小到大排序
all_cascades = sorted(all_cascades, key=lambda x:x[2])

# 3. 计算pagerank缓存
pageranks = nx.pagerank(G, alpha=0.85)

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

# 随机模拟100次结果
# 4. popsim模拟结果
popsim_results = []     # 存储模拟结果
popsim_pe = []          # 存储概率
simualte_number = 100     # 模拟次数

with open("./simulate_result.txt", "w") as f:

    for i in range(simualte_number):
        Res, Pe = popsim(V, E, G, max_degree, Vc0, Ec0, te, t0, activities, pageranks, start_time, dp_pop)
        popsim_results.append(Res)
        popsim_pe.append(Pe)

        f.write(str(i + 1) + "popsim completed\n")
        f.write(str(Res) + "\n")
        f.write(str(Pe) + "\n")
        print(i + 1, "popsim completed")
        # print(Res, Pe)

    # 5. 生成概率加权传播网络
    EP, VP = pronet(popsim_results, popsim_pe)
    f.write("已激活加权网络\n")
    f.write(str(EP) + "\n")
    f.write(str(VP) + "\n")
    f.write(str(VP.__len__()) + "\n")

    print("已激活加权网络")
    print(EP, VP)
    print(VP.__len__())



    # 调用函数获取入边邻居及其边，重复三次
    final_neighbors, final_edges_list = get_in_edge_neighbors(G, cascade["Vc"], 3)
    final_node_list = []
    # 最终结果：final_edges_list是包含所有找到的边的列表
    print("显示的网络", final_edges_list)
    for e in final_edges_list:
        if e[0] not in final_node_list:
            final_node_list.append(e[0])
        if e[1] not in final_node_list:
            final_node_list.append(e[1])
    print(final_node_list.__len__())
    print(final_edges_list.__len__())

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

    activities = tsne_activity(all_cascades, all_start_user)

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
    tsne = TSNE(n_components=2, random_state=40, perplexity=41, learning_rate=100)

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

    final_results = output_data
    final_labels = list(y_kmeans)

    user_labels = {}
    for i in range(0, len(final_labels)):
        user_labels[final_node_list[i]] = final_labels[i]

    print("展示的用户标签", user_labels)

    output = []
    for index, r in enumerate(final_results):
        new_dict = {}
        new_dict["id"] = str(index)
        new_dict['x'] = r[0]
        new_dict['y'] = r[1]
        new_dict['category'] = str(final_labels[index])
        new_dict['special'] = 0
        output.append(new_dict)


    print("用于D3可视化的", output)