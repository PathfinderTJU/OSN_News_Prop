import numpy as np

# 测试数据
V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}


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
    print(P_n[index])
    return P_n[index]


pagerank(V, E, 1)
