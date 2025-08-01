import itertools

# 测试数据
V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
E = {(1,2), (2, 4), (3, 4), (5, 3), (4, 1), (4, 6), (6, 1), (7, 6), (7, 9), (9, 1), (10, 1), (11, 9), (12, 9), (12, 8), (8, 9)}
Vc0 = {1}
Ec0 = set()
RES = [{(9, 8, 4), (9, 11, 1), (1, 6, 3), (1, 9, 0), (9, 7, 3), (1, 10, 0), (4, 3, 4), (9, 12, 2), (3, 5, 5), (1, 4, 1), (4, 2, 2)},
       {(9, 11, 1), (1, 9, 0), (9, 7, 3), (1, 10, 0), (4, 3, 4), (3, 5, 5), (1, 4, 2), (4, 2, 3), (1, 6, 1), (9, 8, 2), (9, 12, 4)},
       {(1, 10, 1), (9, 8, 4), (1, 6, 0), (9, 11, 5), (6, 7, 2), (1, 4, 0), (4, 2, 1), (9, 12, 3), (1, 9, 2), (4, 3, 3), (3, 5, 4)},
       {(9, 12, 1), (9, 7, 1), (1, 6, 3), (4, 2, 4), (1, 9, 0), (4, 3, 4), (1, 10, 0), (1, 4, 3), (3, 5, 5), (9, 8, 2), (9, 11, 2)},
       {(6, 7, 3), (1, 6, 0), (4, 3, 1), (9, 11, 4), (1, 4, 0), (4, 2, 1), (1, 9, 3), (3, 5, 2), (1, 10, 2), (9, 8, 5), (9, 12, 4)}
       ]
PE = [1.2872326716860331e-06, 1.9844974031927947e-06, 1.732295208935903e-06, 0.33147320608950215, 1.153672625708382e-06]


def pronet(RES, PE):
    VP = list()
    EP = list()

    # 遍历每次结果
    for i in range(RES.__len__()):
        res = RES[i]

        # 每次结果中多次出现的节点只计算一次概率
        temp_vp = list()

        # 遍历每条路径
        for p in res:

            # 处理vi
            if p[0] not in temp_vp:
                temp_vp.append(p[0])

                px_vi = next((x for x in VP if x[0] == p[0]), None)
                if px_vi is not None:
                    px_vi[1] += PE[i]
                else:
                    VP.append([p[0], PE[i]])

            # 处理vj
            if p[1] not in temp_vp:
                temp_vp.append(p[1])

                px_vj = next((x for x in VP if x[0] == p[1]), None)
                if px_vj is not None:
                    px_vj[1] += PE[i]
                else:
                    VP.append([p[1], PE[i]])

            # 处理eij
            px = next((x for x in EP if x[0] == p[0] and x[1] == p[1]), None)
            if px is not None:
                px[2] += PE[i]
            else:
                EP.append([p[0], p[1], PE[i]])

    # 归一化
    sum_pe = sum(PE)
    for ep in EP:
        ep[2] /= sum_pe
    for vp in VP:
        vp[1] /= sum_pe

    return EP, VP

# print(pronet(RES, PE))