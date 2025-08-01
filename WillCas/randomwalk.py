import numpy as np
import networkx as nx

# # 测试数据
# V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
# E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}
# Vc0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12}
# Ec0 = {(1, 4), (4, 2), (4, 3), (3, 5), (6, 4), (1, 6), (6, 7), (1, 9), (9, 8), (9, 11), (9, 12), (8, 12)}
# K = 4
# L = 3


def randomwalk(user_network, cascade_network, K=200, L=10, eps=0.01):
    sampleseq = list()

    # 先计算传播网络中各个节点的被选择概率
    Vc0 = list(cascade_network.nodes(data=True))
    Vc0_temp_time = [n[1]['time'] for n in Vc0]
    Vc0_temp_former = [n[1]['former'] for n in Vc0]
    Vc0_temp = [n[0] for n in Vc0]

    start_prop = list(range(cascade_network.number_of_nodes()))

    out_sum = 0

    for i in range(cascade_network.number_of_nodes()):
        out_sum += user_network.out_degree(Vc0_temp[i])

    for i in range(cascade_network.number_of_nodes()):
        start_prop[i] = (user_network.out_degree(Vc0_temp[i]) + eps) / out_sum

    total_start_prop = sum(start_prop)
    if not np.isclose(total_start_prop, 1.0):
        start_prop = [prob / total_start_prop for prob in start_prop]
        total_start_prop = sum(start_prop)

    # K个序列
    while sampleseq.__len__() != K:
        seq = list()
        # 1. 随机选择一个初始节点
        start = None
        while True:
            start = np.random.choice(Vc0_temp, p=start_prop)
            start_time = Vc0_temp_time[Vc0_temp.index(start)]
            start_former = Vc0_temp_former[Vc0_temp.index(start)]

            # 孤立节点要从备选节点中排除，防止再次选到
            if user_network.out_degree(start) == 0:
                start_prop.pop(Vc0_temp.index(start))
                Vc0_temp_time.pop(Vc0_temp.index(start))
                Vc0_temp_former.pop(Vc0_temp.index(start))
                Vc0_temp.pop(Vc0_temp.index(start))

                # 弹出后重新归一化
                total_start_prop = sum(start_prop)
                start_prop = [prob / total_start_prop for prob in start_prop]
                total_start_prop = sum(start_prop)

                # 重新选择start
                continue
            else:
                break

        seq.append((start, start_time, start_former))

        # 2. 开始随机游走
        # 记录当前节点
        now = start

        # 循环选择下一个节点
        while seq.__len__() != L:
            # 获取当前节点的出边邻居节点集合
            now_out = list(cascade_network.successors(now))
            now_out_time = [Vc0_temp_time[Vc0_temp.index(n)] for n in now_out]

            # 回溯：选择到了孤立节点且未到达长度，补+
            if len(now_out) == 0:
                seq.append(("+", -1, -1))
                continue

            # 构造当前节点的备选集合并计算概率
            now_choose_prop = list(range(len(now_out)))

            now_out_sum = 0

            for i in range(len(now_out)):
                now_out_sum += user_network.out_degree(now_out[i]) + eps

            for i in range(len(now_out)):
                now_choose_prop[i] = (user_network.out_degree(now_out[i]) + eps) / now_out_sum

            total_now_prop = sum(now_choose_prop)
            if not np.isclose(total_now_prop, 1.0):
                now_choose_prop = [prob / total_now_prop for prob in now_choose_prop]

            # 按概率，随机选择下一个节点
            next = np.random.choice(now_out, p=now_choose_prop)
            next_time = now_out_time[now_out.index(next)]
            next_former = now

            # 将下一节点加入序列，移动当前节点指向
            seq.append((next, next_time, next_former))
            now = next

        sampleseq.append(seq)

    return sampleseq


# 随机游走传播网络，获取K个长度为L的采样序列（回溯版）


# def randomwalk(V, E, Vc0, Ec0, K=200, L=10, eps=0.01):
#     sampleseq = list()
#
#     # 先计算传播网络中各个节点的被选择概率
#     Vc0_temp = list(Vc0)
#     start_prop = list(range(len(Vc0)))
#
#     out_sum = 0
#     for i in range(len(Vc0)):
#         out_sum += outdegree(V, E, Vc0_temp[i]) + eps
#     sum = 0
#     for i in range(len(Vc0)):
#         start_prop[i] = (outdegree(V, E, Vc0_temp[i]) + eps) / out_sum
#         sum += start_prop[i]
#     start = np.random.choice(Vc0_temp, p=start_prop)
#     print(start)
#
#     # K个序列
#     while sampleseq.__len__() != K:
#         seq = list()
#         # 1. 随机选择一个初始节点
#
#         # 按概率，随机选择一个起始节点
#         start = None
#         while True:
#             start = np.random.choice(Vc0_temp, p=start_prop)
#
#             # 孤立节点要从备选节点中排除，防止再次选到
#             if outdegree(V, E, start) == 0:
#                 start_prop.pop(Vc0_temp.index(start))
#                 Vc0_temp.pop(Vc0_temp.index(start))
#
#                 # 重新选择start
#                 continue
#             else:
#                 break
#
#         seq.append(start)
#
#         # 2. 开始随机游走
#         # 记录当前节点
#         now = start
#
#         # 记录备选集合和概率集合,序列末尾的集合是选择now节点那一步的备选集合和概率集合
#         out_temp = list()
#         choose_prop = list()
#
#         # 循环选择下一个节点
#         while seq.__len__() != L:
#             # 获取当前节点的出边邻居节点集合
#             # 如果备选概率的存储数量等于序列里节点数量，证明是从上一步回溯过来的，要用记忆的备选集合
#             now_out = None
#             if len(seq) == len(out_temp):
#                 # 清除记忆
#                 now_out = out_temp.pop()
#                 choose_prop.pop()
#             else:
#                 now_out = outneighbor(Vc0, Ec0, now)
#
#             # 回溯：选择到了孤立节点且未到达长度，或当前节点的所有待选择节点均不可选择（备选集合为空）
#             if len(now_out) == 0:
#
#                 # 判断是否为多次回溯，如果是多级回溯要删除记忆的备选集合
#                 if len(seq) == len(out_temp):
#                     out_temp.pop()
#                     choose_prop.pop()
#
#                 # 回到了初始节点，且从该初始节点出发永远不可能完成游走，则重启并丢弃这个节点
#                 if len(out_temp) == 0:
#                     start_prop.pop(Vc0_temp.index(start))
#                     Vc0_temp.pop(Vc0_temp.index(start))
#
#                     # 清空seq
#                     seq.pop()
#                     break
#
#                 # 从上一时刻记忆的备选集合中删除当前节点，避免再次游走到
#                 index = out_temp[-1].index(now)
#                 choose_prop[-1].pop(index)
#                 out_temp[-1].pop(index)
#
#                 # 回溯上一节点
#                 seq.pop()
#                 now = seq[-1]
#                 continue
#
#             # 构造当前节点的备选集合并计算概率
#             now_out_temp = list(now_out)
#             now_choose_prop = list(range(len(now_out)))
#
#             now_out_sum = 0
#
#             for i in range(len(now_out_temp)):
#                 now_out_sum += outdegree(V, E, now_out_temp[i]) + eps
#
#             for i in range(len(now_out_temp)):
#                 now_choose_prop[i] = (outdegree(V, E, now_out_temp[i]) + eps) / now_out_sum
#
#             # 按概率，随机选择下一个节点
#             next = np.random.choice(now_out_temp, p=now_choose_prop)
#
#             # 将当前节点的备选集合和概率集合存储，便于后续回溯
#             out_temp.append(now_out_temp)
#             choose_prop.append(now_choose_prop)
#
#             # 将下一节点加入序列，移动当前节点指向
#             seq.append(next)
#             now = next
#
#         # 重启
#         if len(seq) == 0:
#             continue
#
#         sampleseq.append(seq)
#
#         # 去重
#         # if not any(seq == subseq for subseq in sampleseq):
#         #     sampleseq.append(seq)
#
#     return sampleseq


# 获取vi的出边邻居节点集合

#
# def outneighbor(Vc0, Ec0, vi):
#     out = set()
#     for e in Ec0:
#         if e[0] == vi:
#             out.add(e[1])
#
#     return out
#
#
# # 计算vi的出度
# def outdegree(V, E, vi):
#
#     od = 0
#
#     for e in E:
#         if e[0] == vi:
#             od += 1
#     return od