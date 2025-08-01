import networkx as nx
import random
from collections import defaultdict


def monte_carlo_simrank(G, source, target, walk_length, num_walks):
    """
    使用蒙特卡洛模拟方法估计有向图中两个节点的SimRank相似度。

    :param G: 一个NetworkX有向图。
    :param source: 源节点。
    :param target: 目标节点。
    :param walk_length: 每次随机游走的最大长度。
    :param num_walks: 从每个节点出发的随机游走次数。
    :return: 源节点和目标节点之间的估计SimRank相似度。
    """

    # 初始化相似度计数为0
    similarity_count = 0

    # 从源节点和目标节点出发执行随机游走
    for _ in range(num_walks):
        # 执行从源节点的随机游走
        walk_source = random_walk(G, source, walk_length, target)
        # 执行从目标节点的随机游走
        walk_target = random_walk(G, target, walk_length, source)

        # 检查两条随机游走路径是否相交（即在某个节点相遇）
        intersection = set(walk_source).intersection(set(walk_target))
        if intersection:
            index_source = walk_length
            index_target = walk_length
            for node in intersection:
                # 避免孤立节点的影响
                if node == source or node == target:
                    continue

                # 找最早相遇时间
                pos_node_source = walk_source.index(node)
                pos_node_target = walk_target.index(node)
                if pos_node_source < index_source:
                    index_source = pos_node_source
                if pos_node_target < index_target:
                    index_target = pos_node_target

            index = min(index_source, index_target)

            # 时间权重为0.25
            similarity_count_add = 0.8 + 0.2 * (walk_length - index) / walk_length
            similarity_count += similarity_count_add

        # 根据相交游走的比例估算相似度
    similarity = similarity_count / num_walks

    # print(similarity)
    # 返回估算的相似度
    return round(similarity, 4)


def random_walk(G, start, length, target, p=0.5):
    """
    在有向图上从起始节点执行一条指定长度的随机游走。

    :param G: 一个NetworkX有向图。
    :param start: 随机游走的起始节点。
    :param length: 随机游走的最大长度。
    :return: 随机游走过程中访问的节点列表。
    """

    # 初始化随机游走的节点列表，包含起始节点
    walk = [start]
    current_node = start

    # 继续游走，直到达到指定的长度
    for i in range(length - 1):
        # 获取当前节点的所有后继节点
        successors = list(G.successors(current_node))
        walked = False

        # 如果存在后继节点
        while successors:
            # 从后继节点中随机选择一个作为下一步
            current_node = random.choice(successors)

            # 如果随机游走到了target，就以p概率继续游走，1-p概率移除该候选节点，重新选择一个新节点
            if current_node == target:
                if random.random() > p:
                    successors.pop(successors.index(current_node))
                    continue
                else:
                    walk.append(current_node)
                    walked = True
                    break

            # 将选择的节点添加到游走列表中
            walk.append(current_node)
            walked = True
            break

        if not walked:
            # 如果当前节点没有后继节点，则游走结束
            break
        i += 1

    # 返回随机游走过程中访问的节点列表
    return walk


def local_simrank(graph, source, target, max_iterations=10, c=0.8):
    """
    计算有向图中两个相邻节点的局部SimRank相似度。

    :param graph: NetworkX有向图对象。
    :param source: 源节点。
    :param target: 目标节点。
    :param max_iterations: 最大迭代次数。
    :param c: 衰减因子。
    :return: source和target之间的局部SimRank相似度。
    """


    # 初始化相似度字典
    similarity = {node: {node: 1.0 for node in graph.nodes()} for node in graph.nodes()}

    # 对于每个迭代
    for iteration in range(max_iterations):
        # 准备下一个迭代的相似度字典
        next_similarity = {node: {} for node in graph.nodes()}

        # 遍历所有节点对
        for node in graph.nodes():
            for other in graph.nodes():
                if node == other:
                    # 同一个节点的相似度总是1
                    next_similarity[node][other] = 1.0
                else:
                    # 计算节点对的相似度
                    in_similarity = sum(similarity[neighbor][other] for neighbor in graph.predecessors(node))
                    out_similarity = sum(similarity[node][neighbor] for neighbor in graph.successors(other))
                    next_similarity[node][other] = (c * (in_similarity + out_similarity)) / (
                                graph.in_degree(node, loops=True) + graph.out_degree(other, loops=True))

                    # 替换当前相似度
        similarity = next_similarity

        # 返回源和目标节点之间的最终相似度
    return similarity[source][target]


def get_connected_components(G):
    """
    计算无向图G的连通分量个数及每个分量的大小。

    输入:
        G: NetworkX无向图对象

    返回:
        num_components: 连通分量个数
        component_sizes: 每个连通分量的大小(节点数)列表
    """

    # 找到所有连通分量
    components = nx.connected_components(G)

    # 计算连通分量个数
    num_components = len(list(components))

    # 计算每个连通分量的大小
    component_sizes = [len(component) for component in components]

    return num_components, component_sizes


# with open("./dataset/DeepHawkes/user_network.txt", "r") as f:
#     dataset = [line.strip() for line in f]
#
#
# user_network = []
# for d in dataset:
#     user_network.append(tuple(d.split(" ")))
#
# G = nx.DiGraph()
# G.add_edges_from(user_network)
#
#
# walk_length = 5
# num_walks = 5000
#
# # 测试
# with open("./output", "w") as f:
#     for p in user_network:
#         node1 = p[0]
#         node2 = p[1]
#         simrank_values = monte_carlo_simrank(G, node1, node2, walk_length, num_walks)
#         f.write(node1 + " " + node2 + " " + str(simrank_values) + "\n")


# node1 = "63369"
# node2 = "1960308"
# simrank_values = monte_carlo_simrank(G, node1, node2, walk_length, num_walks)
# print(simrank_values)
# max_iterations = 5
# simrank_values2 = local_simrank(G, node1, node2, max_iterations)
# print(simrank_values2)