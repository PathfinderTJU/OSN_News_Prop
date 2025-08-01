import os
import random
import itertools
from itertools import product
from utils import willcasprop
from utils import activity
from utils import willcas_pop
from utils import log_min_max_normal

# 测试数据
V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
E = {(1, 2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}
Vc0 = {1}
Ec0 = set()
te = 10

t0 = 60  # 时间片长度


# 完全随机的传播过程模拟，返回一次模拟结果
def randomsim(V, E, G, max_degree, origin_length, Vc0, Ec0, te, t0, activities, pageranks, start_time, seeds):
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

        # 将路径从大到小排序
        # temp_Px = sorted(Px, key=lambda x:x[2], reverse=True)

        # 生成一个概率随机值
        prop_standard = random.random()

        # 选择的用户和路径
        sl = set()
        nsk = set()

        # 从头遍历路径集合，不断加入概率大的待激活路径，每个用户只能由一个，直到达到Rt
        for p in Px:
            if p[2] >= prop_standard:
                nsk.add(p)
                sl.add(p[1])
                Res.add((p[0], p[1], t + seeds))
            else:
                continue

        # 移动时间片
        t += t0

        # 终止条件2：到达观测结束。
        if t >= te:
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
                    Px.add((vi, vj, 1)) # 随机设一个概率值，在下次循环开始时会被更新
                else:
                    continue

    return Res


# print(randomsim(V, E, Vc0, Ec0, te))