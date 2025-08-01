import os
import random
import itertools
from itertools import product
from utils import willcasprop
from utils import willcas_pop
from utils import time_pop
from utils import activity
from utils import log_min_max_normal

# 测试数据
V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}
Vc0 = {1}
Ec0 = set()
te = 10

t0 = 1  # 时间片长度


def popsimmax(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks,  start_time, seeds):
    X = set(Vc0)  # 激活用户集合
    Xw = set()  # 待激活用户集合
    Xu = set(V - X)  # 未激活用户集合

    Res = set(Ec0)  # 初始传播网络
    Pe = 1  # 生成概率

    t = start_time  # 定义起始时刻（需要在初始传播网络的传播路径时刻之后）
    Px = set()  # 中间集合

    # 构造初始Xw和Px
    for vi in X:
        invi = set(G.predecessors(vi))
        for vj in invi:
            if vj in Xu and vj not in Xw:
                Xw.add(vj)
                Px.add((vi, vj, 1)) # 后续循环中还要更新
            else:
                continue

    while Xu.__len__() != 0:
        # 终止条件1：预测无法增长
        if Xw.__len__() == 0:
            return Res

        # 更新新时刻的Px中概率值
        new_Px = set()
        for p in list(Px):
            new_Px.add((p[0], p[1], willcasprop(V, E, G, max_degree, Vc0, Ec0, activities, pageranks, start_time, p[0], p[1], t)))
        Px = new_Px

        # 预测流行度增量
        # Rt = willcas(V, E, X, Res, t, t0)
        Rt = time_pop(start_time, t, t0)
        if Rt > Xw.__len__(): # 处理异常值
            Rt = Xw.__len__()

        # 将路径从大到小排序
        temp_Px = sorted(Px, key=lambda x:x[2], reverse=True)

        # 选择的用户和路径
        sl = set()
        nsk = set()

        # 从头遍历路径集合，不断加入概率大的待激活路径，每个用户只能由一个，直到达到Rt
        for p in temp_Px:
            if sl.__len__() == Rt:
                break
            if p[1] not in sl:
                sl.add(p[1])
                nsk.add(p)
                Res.add((p[0], p[1], t + seeds))


        # 移动时间片
        t += t0

        # 终止条件2：到达观测结束。终止条件3：预测不再增长
        if t >= te or Rt == 0:
            break

        # 暂时只模拟到原始大小
        if len(Res) >= origin_length:
            break

        # 修改相关集合
        Px = Px - nsk
        for vi in sl:
            X.add(vi)
            Xw.discard(vi)
            Xu.discard(vi)

            # 添加新的可能节点
            invi = set(G.predecessors(vi))
            for vj in invi:
                if vj in Xu and vj not in Xw:
                    Xw.add(vj)
                    Px.add((vi, vj, 1))  # 随机设一个概率值，在下次循环开始时会被更新
                else:
                    continue

    return Res

# def popsimmax(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks,  start_time, dp_pop):
#     X = set(Vc0)  # 激活用户集合
#     Xw = set()  # 待激活用户集合
#     Xu = set(V - X)  # 未激活用户集合
#
#     Res = set(Ec0)  # 初始传播网络
#     Pe = 1  # 生成概率
#
#     t = start_time  # 定义起始时刻（需要在初始传播网络的传播路径时刻之后）
#     Px = set()  # 中间集合
#
#     # 构造初始Xw和Px
#     for vi in X:
#         invi = set(G.predecessors(vi))
#         for vj in invi:
#             if vj in Xu and vj not in Xw:
#                 Xw.add(vj)
#                 Px.add((vi, vj, 1)) # 后续循环中还要更新
#             else:
#                 continue
#
#     while Xu.__len__() != 0:
#         # # 终止条件1：预测无法增长
#         if Xw.__len__() == 0:
#             return Res
#
#         # 更新新时刻的Px中概率值
#         new_Px = set()
#         for p in list(Px):
#             new_Px.add((p[0], p[1], willcasprop(V, E, G, max_degree, Vc0, Ec0, activities, pageranks, start_time, p[0], p[1], t)))
#         Px = new_Px
#
#         # 预测流行度增量
#         # Rt = willcas(V, E, X, Res, t, t0)
#         # Rt = time_pop(start_time, t, t0)
#         if t - int(start_time) in dp_pop.keys():
#             Rt = dp_pop[t - int(start_time)]
#         else:
#             Rt = 0
#
#         if Rt > Xw.__len__(): # 处理异常值
#             Rt = Xw.__len__()
#
#         # 将路径从大到小排序
#         temp_Px = sorted(Px, key=lambda x:x[2], reverse=True)
#
#         # 选择的用户和路径
#         sl = set()
#         nsk = set()
#
#         # 从头遍历路径集合，不断加入概率大的待激活路径，每个用户只能由一个，直到达到Rt
#         for p in temp_Px:
#             if sl.__len__() == Rt:
#                 break
#             if p[1] not in sl:
#                 sl.add(p[1])
#                 nsk.add(p)
#                 Res.add((p[0], p[1], t))
#
#
#         # 移动时间片
#         t += t0
#
#         # 终止条件2：到达观测结束。终止条件3：预测不再增长
#         if t >= te:
#             break
#
#         # # 暂时只模拟到结束
#         if len(Res) >= origin_length:
#             break
#
#         # 修改相关集合
#         Px = Px - nsk
#         for vi in sl:
#             X.add(vi)
#             Xw.discard(vi)
#             Xu.discard(vi)
#
#             # 添加新的可能节点
#             invi = set(G.predecessors(vi))
#             for vj in invi:
#                 if vj in Xu and vj not in Xw:
#                     Xw.add(vj)
#                     Px.add((vi, vj, 1))  # 随机设一个概率值，在下次循环开始时会被更新
#                 else:
#                     continue
#
#     return Res


def popsimmax_deephawkes(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks,  start_time, seeds, dp_pop):
    X = set(Vc0)  # 激活用户集合
    Xw = set()  # 待激活用户集合
    Xu = set(V - X)  # 未激活用户集合

    Res = set(Ec0)  # 初始传播网络
    Pe = 1  # 生成概率

    t = start_time  # 定义起始时刻（需要在初始传播网络的传播路径时刻之后）
    Px = set()  # 中间集合

    # 构造初始Xw和Px
    for vi in X:
        invi = set(G.predecessors(vi))
        for vj in invi:
            if vj in Xu and vj not in Xw:
                Xw.add(vj)
                Px.add((vi, vj, 1)) # 后续循环中还要更新
            else:
                continue

    while Xu.__len__() != 0:
        # # 终止条件1：预测无法增长
        if Xw.__len__() == 0:
            return Res

        # 更新新时刻的Px中概率值
        new_Px = set()
        for p in list(Px):
            new_Px.add((p[0], p[1], willcasprop(V, E, G, max_degree, Vc0, Ec0, activities, pageranks, start_time, p[0], p[1], t)))
        Px = new_Px

        # 预测流行度增量
        # Rt = willcas(V, E, X, Res, t, t0)
        # Rt = time_pop(start_time, t, t0)
        if t - int(start_time) in dp_pop.keys():
            Rt = dp_pop[t - int(start_time)]
        else:
            Rt = 0

        if Rt > Xw.__len__(): # 处理异常值
            Rt = Xw.__len__()

        # 将路径从大到小排序
        temp_Px = sorted(Px, key=lambda x:x[2], reverse=True)

        # 选择的用户和路径
        sl = set()
        nsk = set()

        # 从头遍历路径集合，不断加入概率大的待激活路径，每个用户只能由一个，直到达到Rt
        for p in temp_Px:
            if sl.__len__() == Rt:
                break
            if p[1] not in sl:
                sl.add(p[1])
                nsk.add(p)
                Res.add((p[0], p[1], t + seeds))


        # 移动时间片
        t += t0

        # 终止条件2：到达观测结束。终止条件3：预测不再增长
        if t >= te:
            break

        # # 暂时只模拟到结束
        if len(Res) >= origin_length:
            break

        # 修改相关集合
        Px = Px - nsk
        for vi in sl:
            X.add(vi)
            Xw.discard(vi)
            Xu.discard(vi)

            # 添加新的可能节点
            invi = set(G.predecessors(vi))
            for vj in invi:
                if vj in Xu and vj not in Xw:
                    Xw.add(vj)
                    Px.add((vi, vj, 1))  # 随机设一个概率值，在下次循环开始时会被更新
                else:
                    continue

    return Res

# def popsimmax(V, E, Vc0, Ec0, te, t0, cascades, pageranks, start_time):
#     X = set(Vc0)  # 激活用户集合
#     Xw = set()  # 待激活用户集合
#     Xu = set(V - X)  # 未激活用户集合
#
#     Res = set(Ec0)  # 初始传播网络
#     Pe = 1  # 生成概率
#
#     t = start_time  # 定义起始时刻（需要在初始传播网络的传播路径时刻之后）
#     Px = set()  # 中间集合
#
#     while Xu.__len__() != 0:
#         # 构造Xw和Px
#         for vi in X:
#             invi = getneighbor(V, E, vi)
#             for vj in invi:
#                 if vj in Xu and vj not in Xw:
#                     Xw.add(vj)
#                     Px.add((vi, vj, willcasprop(V, E, Vc0, Ec0, cascades, pageranks, start_time, vi, vj, t)))
#                 else:
#                     continue
#
#         # 预测流行度增量
#         # Rt = willcas(V, E, X, Res, t, t0)
#         Rt = time_pop(start_time, t, t0)
#         if Rt > Xw.__len__(): # 处理异常值
#             Rt = Xw.__len__()
#
#         # 将路径从大到小排序
#         temp_Px = sorted(Px, key=lambda x:x[2], reverse=True)
#
#         # 选择的用户和路径
#         sl = set()
#         nsk = set()
#
#         # 从头遍历路径集合，不断加入概率大的待激活路径，每个用户只能由一个，直到达到Rt
#         for p in temp_Px:
#             if sl.__len__() == Rt:
#                 break
#             if p[1] not in sl:
#                 sl.add(p[1])
#                 nsk.add(p)
#
#         # 计算条件概率
#         # p1为p(ab)k
#         p1 = 1
#         for p in nsk:
#             p1 *= p[2]
#         for p in Px - nsk:
#             p1 *= (1 - p[2])
#
#         # p2为p(a)
#         p2 = 0
#         # 所有可能的用户选择
#         combinations = itertools.combinations(Xw, Rt)
#         for s in combinations:
#
#             # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
#             mapping = {m: [tup for tup in Px if tup[1] == m] for m in s}
#
#             # 使用product生成所有组合
#             all_possible_combinations = [set(comb) for comb in product(*[mapping[m] for m in s])]
#
#             # 累积条件概率
#             for c in all_possible_combinations:
#                 temp_p = 1
#                 for p in c:
#                     temp_p *= p[2]
#                 for p in Px - c:
#                     temp_p *= (1 - p[2])
#
#                 p2 += temp_p
#
#         P = p1 / p2
#
#         # 归一化P，满足当前时刻所有的路径选择组合的条件概率之和为1
#         # 计算所有可能的路径组合的条件概率之和
#         P_sum = 0
#         combinations = itertools.combinations(Xw, Rt)
#         for s in combinations:
#             # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
#             mapping = {m: [tup for tup in Px if tup[1] == m] for m in s}
#
#             # 使用product生成所有组合
#             all_possible_combinations = [set(comb) for comb in product(*[mapping[m] for m in s])]
#
#             # 计算每种组合的条件概率
#             for c in all_possible_combinations:
#                 temp_p = 1
#                 for p in c:
#                     temp_p *= p[2]
#                 for p in Px - c:
#                     temp_p *= (1 - p[2])
#
#                 P_sum += temp_p / p2
#
#         P = P / P_sum
#
#         # 累积结果
#         Pe *= P
#         for p in nsk:
#             Res.add((p[0], p[1], t))
#
#         # 修改相关集合
#         Px = Px - nsk
#         for vi in sl:
#             X.add(vi)
#             Xw.discard(vi)
#             Xu.discard(vi)
#
#         # 移动时间片
#         t += t0
#
#         print(t, "popsimmax completed")
#
#         # 终止条件2：到达观测结束。终止条件3：预测不再增长
#         if t == te or Rt == 0:
#             break
#
#     return Res, Pe

# print(popsimmax(V, E, Vc0, Ec0, te))