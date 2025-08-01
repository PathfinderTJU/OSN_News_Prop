import networkx as nx
from karateclub import Graph2Vec
from numpy import random
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


# 比较两个图在结构上的相似性，输入为networkx有向图对象
def similar(G1, G2):
    graphs = [G1, G2]

    # 使用Graph2Vec进行图嵌入
    g_mdl = Graph2Vec(dimensions=50, min_count=1)
    g_mdl.fit(graphs)
    g_emb = g_mdl.get_embedding()

    embedding_G1 = g_emb[0]
    embedding_G2 = g_emb[1]

    # 使用RBF核计算图之间的相似性
    similarity = rbf_kernel([embedding_G1], [embedding_G2])[0][0]

    return similarity


# 创建两个有向图示例
G1 = nx.DiGraph()
G1.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])

G2 = nx.DiGraph()
G2.add_edges_from([(0, 1), (1, 2), (0, 2), (1, 3)])

print(similar(G1, G2))