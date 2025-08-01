import numpy as np
import random

# 测试数据
V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}


def random_walk_pagerank(G, v, d=0.85, num_walks=1000, walk_length=5):
    """
    使用随机游走来近似计算图G中指定节点v的PageRank值。

    参数:
    G : networkx graph
        表示网络结构的有向图。
    v : node
        需要计算PageRank值的节点。
    d : float
        随机游走继续的概率（阻尼因子）。
    num_walks : int
        进行的随机游走总次数。
    walk_length : int
        每次随机游走的最大长度。

    返回:
    pagerank_v : float
        节点v的PageRank值。
    """
    # 初始化所有节点的访问次数为0
    visits = {node: 0 for node in G.nodes()}

    # 执行随机游走
    for _ in range(num_walks):
        # 从随机节点开始
        current_node = random.choice(list(G.nodes()))

        for _ in range(walk_length):
            # 如果当前节点是我们感兴趣的节点v，则增加其访问次数
            if current_node == v:
                visits[v] += 1

            # 以概率d继续在链接上游走，或者以概率(1-d)跳到任意节点
            if random.random() < d and list(G[current_node]):  # 检查是否有出边
                current_node = random.choice(list(G[current_node]))  # 沿链接游走
            else:
                current_node = random.choice(list(G.nodes()))  # 随机跳到某个节点

    # 根据节点v的访问次数计算其PageRank值并归一化
    pagerank_v = visits[v] / (num_walks * walk_length)
    # print(pagerank_v)
    return pagerank_v


def pagerank(V, E, v0):
    nodes = list(V)
    edges = []
    N = len(nodes)

    for e in E:
        edges.append([e[0], e[1]])

    node_to_num = {}
    for i, node in enumerate(nodes):
        node_to_num[node] = i

    for edge in edges:
        edge[0] = node_to_num[edge[0]]
        edge[1] = node_to_num[edge[1]]

    # 生成初步的S矩阵
    S = np.zeros([N, N])
    for edge in edges:
        S[edge[1], edge[0]] = 1

    # 计算比例：即一个网页对其他网页的PageRank值的贡献，即进行列的归一化处理
    for j in range(N):
        sum_of_col = sum(S[:, j])
        for i in range(N):
            S[i, j] /= sum_of_col

    # 计算矩阵A
    alpha = 0.85
    A = alpha * S + (1 - alpha) / N * np.ones([N, N])

    # 生成初始的PageRank值，记录在P_n中，P_n和P_n1均用于迭代
    P_n = np.ones(N) / N
    P_n1 = np.zeros(N)

    e = 100000  # 误差初始化
    k = 0  # 记录迭代次数
    # print('loop...')

    while e > 0.000000001:  # 开始迭代
        P_n1 = np.dot(A, P_n)  # 迭代公式
        e = P_n1 - P_n
        e = max(map(abs, e))  # 计算误差
        P_n = P_n1
        k += 1

    # print(nodes)
    # print(P_n)

    index = nodes.index(v0)
    return P_n[index]


# pagerank(V, E, 1)
