import os
import numpy as np
import random
import itertools
from itertools import product
from utils import willcasprop
from utils import willcas_pop
from utils import time_pop
import math

# 测试数据
# V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
# E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}
# Vc0 = {1}
# Ec0 = set()
# te = 10
#
# t0 = 1  # 时间片长度


def popsim(V, E, G, max_degree, Vc0, Ec0, te, t0, activities, pageranks,  start_time, dp_pop):
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
                Px.add((vi, vj, 1))  # 后续循环中还要更新
            else:
                continue

    while Xu.__len__() != 0:
        # 终止条件1：预测无法增长
        if Xw.__len__() == 0:
            return Res, Pe

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

        # 生成所有用户组合
        # combinations = itertools.combinations(Xw, Rt)

        # 用于记录所有用户组合及对应条件概率
        userlist=[]
        userprop=[]

        # 用于记录所有路径组合

        # 用于累积所有情况的条件概率
        p2 = 0

        # 定义蒙特卡洛模拟次数
        num_simulations = 1000

        # 蒙特卡洛方法近似计算 p2
        for _ in range(num_simulations):
            # 随机选取一个用户组合
            s = random.sample(list(Xw), Rt)

            # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
            mapping = {m: [tup for tup in Px if tup[1] == m] for m in s}

            # 随机选择路径
            c = set()
            for m in s:
                if mapping[m]:  # 确保列表不为空
                    c.add(random.choice(mapping[m]))

            # 计算选定路径的条件概率
            temp_p = 1
            for p in c:
                temp_p *= p[2]
            for p in Px - c:
                temp_p *= (1 - p[2])

            # 累积到p2中
            p2 += temp_p

        # 求平均值获得p2的近似值
        p2 /= num_simulations

        # # 生成用户的所有路径组合
        # for s in combinations:
        #     # 记录用户组合
        #     userlist.append(s)
        #
        #     # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
        #     mapping = {m: [tup for tup in Px if tup[1] == m] for m in s}
        #
        #     # 使用product生成所有组合
        #     all_possible_combinations = [set(comb) for comb in product(*[mapping[m] for m in s])]
        #
        #     # 计算每种组合的概率，并求和
        #     ps = 0
        #     for c in all_possible_combinations:
        #         temp_p = 1
        #         for p in c:
        #             temp_p *= p[2]
        #         for p in Px - c:
        #             temp_p *= (1 - p[2])
        #
        #         ps += temp_p
        #         p2 += temp_p
        #
        #     # 记录用户组合的概率
        #     userprop.append(ps)

        # 组合概率归一化(便于选择一个用户组合，不是真正的归一化）
        # ps_sum = 0
        # for ps in userprop:
        #     ps_sum += ps
        #
        # for i in range(len(userprop)):
        #     userprop[i] = userprop[i] / ps_sum


        # 选一组用户
        # userlist_flatten = [i for i in range(len(userlist))]
        # 计算组合的总数，不实际创建它们（需要Xw是可知大小的）
        # total_combinations = math.comb(len(Xw), Rt)
        #
        # # 生成一个随机索引
        # random_index = random.randrange(total_combinations)
        #
        # sl = set()
        # # 遍历迭代器直到到达随机索引位置
        # for i, combination in enumerate(itertools.combinations(Xw, Rt)):
        #     if i == random_index:
        #         sl = combination
        #         break
        # sl = set(userlist[sl])
        sl = set(random.sample(list(Xw), Rt))


        # 每个用户选一个路径
        # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
        mapping = {m: [tup for tup in Px if tup[1] == m] for m in sl}

        # 使用product生成该用户所有路径组合
        # all_possible_combinations = [set(comb) for comb in product(*[mapping[m] for m in sl])]
        #
        # nsk = random.choice(list(all_possible_combinations))

        nsk = set(random.choice(mapping[m]) for m in sl)

        # # 用于记录该用户所有路径组合及对应概率
        # pathlist = []
        # pathprop = []
        #
        # # 计算条件概率
        # for c in all_possible_combinations:
        #     pathlist.append(c)
        #     temp_p = 1
        #     for p in c:
        #         temp_p *= p[2]
        #     for p in Px - c:
        #         temp_p *= (1 - p[2])
        #     pathprop.append(temp_p)
        #
        # # 路径概率归一化（便于选择一组路径，不是真正的概率归一化）
        # pp_sum = 0
        # for ps in pathprop:
        #     pp_sum += ps
        #
        # for i in range(len(pathprop)):
        #     pathprop[i] = pathprop[i] / pp_sum
        #
        # # 选一组路径
        # pathlist_flatten = [i for i in range(len(pathlist))]


        # nsk = random.choice(list(all_possible_combinations))
        # nsk = set(pathlist[nsk])


        # 计算该情况下的条件概率
        # p1为p(ab)k
        p1 = 1
        for p in nsk:
            p1 *= p[2]
        for p in Px - nsk:
            p1 *= (1 - p[2])

        # p2在前面计算了
        P = p1 / p2

        # # 归一化P，满足当前时刻所有的路径选择组合的条件概率之和为1
        # # 计算所有可能的路径组合的条件概率之和
        # P_sum = 0
        # for s in userlist:
        #
        #     # 创建一个映射，将s中的每个元素映射到Px中所有第二个元素等于该元素的元组列表
        #     mapping = {m: [tup for tup in Px if tup[1] == m] for m in s}
        #
        #     # 使用product生成所有组合
        #     all_possible_combinations = [set(comb) for comb in product(*[mapping[m] for m in s])]
        #
        #     # 计算每种组合的条件概率
        #     for c in all_possible_combinations:
        #         temp_p = 1
        #         for p in c:
        #             temp_p *= p[2]
        #         for p in Px - c:
        #             temp_p *= (1 - p[2])
        #
        #         P_sum += temp_p / p2
        #
        # P = P / P_sum

        # 累积结果
        Pe *= P
        for p in nsk:
            matching_tuples = [tup for tup in Px if tup[0] == p[0] and tup[1] == p[1]][0]
            print(p[0], "to", p[1], "at time", t-start_time, "for", matching_tuples[2])
            Res.add((p[0], p[1], t))

        # 移动时间片
        t += t0

        # 终止条件2：到达观测结束。终止条件3：预测不再增长
        if t >= te:
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

    Res |= Ec0
    print(Res, Pe)
    return Res, Pe

# print(popsim(V, E, Vc0, Ec0, te))