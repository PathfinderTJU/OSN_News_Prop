import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import pytz
import time


def deephawkes():
    """
    读取deephawkes数据集，返回传播网络和用户网络
    传播网络casecade_network: list，代表多个传播级联
        对于每个级联，数据结构如下：
        - Vc: set 节点集合，元素为用户ID: string
        - Ec: set 边集合，元素为传播路径: tuple(start: string, end: string, time: string)，代表从start传播到end
        - start_time: string 起始时间
        - start_user: string 起始用户
    用户网络user_network: dict，代表构建的用户网络
    - V: 节点集合，元素为用户ID: string
    - E: 边集合，元素为关注关系: tupe(start: string, end:string)，代表start关注了end
    """

    filepath = "./dataset/DeepHawkes/last_data.txt"
    with open(filepath, "r") as f:
        dataset = [line.strip() for line in f]

    # with open("./deephawkes_id.txt") as f:
    #     dataids = [line.strip() for line in f]

    # specials = [
    #     dataset[86987],
    #     dataset[45986],
    #     dataset[97516],
    #     dataset[70223],
    #     dataset[85883],
    #     dataset[64820],
    #     dataset[94763],
    #     dataset[50033],
    #     dataset[65917],
    #     dataset[49968]
    # ]
    # #
    # specials = [dataset[88]]
    # #
    #
    # dataset = random.sample(dataset, 5000) # 抽选数据集
    # dataset.extend(specials)
    # print(dataset)

    # with open("./last_data.txt", "w") as f:
    #     for d in dataset:
    #         f.write(str(d) + "\n")

    # 传播网络集合
    cascade_network = []

    # 用户网络
    V = set()
    E = set()
    user_network = dict()

    for line in dataset:
        # 存储一个传播网络
        Vc = set()
        Ec = set()
        cascade_data = dict()

        temp = line.split("\t")

        start_time = temp[2]
        start_user = temp[1]
        ID = temp[0]

        start_time_unix = int(start_time)

        cascades = temp[4].split()

        cascades.pop(0) # 去除初始

        for cas in cascades:
            temp_cas = cas.split(":")
            time = temp_cas[1]
            path = temp_cas[0].split("/")

            index = 0
            while index < len(path) - 1:
                start = path[index]
                end = path[index + 1]

                if start == -1 or end == -1 or start == end:
                    index += 1
                    continue

                V.add(start)
                V.add(end)
                E.add((end, start))

                # 只观测24小时内，因此不需要存储超过24小时的转发
                if int(time) > 24 * 3600:
                    index += 1
                    continue

                Vc.add(start)
                Vc.add(end)
                Ec.add((start, end, str(int(start_time) + int(time))))
                index += 1

        # 筛选传播网络数据集，但仍然使用被筛选的传播网络构造用户网络
        # 丢弃大小小于10或者大于1000的传播网络
        if len(Ec) > 1000 or len(Ec) < 10:
            continue

        # 丢弃发布时间在24:00-8:00的传播网络
        # if start_hour < 8:
        #     continue

        # 有的傻逼全是自己转发自己
        if Ec.__len__() == 0:
            continue

        # if ID not in dataids:
        #     continue

        cascade_data["Vc"] = Vc
        cascade_data["Ec"] = Ec
        cascade_data["start_time"] = start_time
        cascade_data["start_user"] = start_user
        cascade_data["ID"] = ID
        cascade_network.append(cascade_data)

    user_network["V"] = V
    user_network["E"] = E

    # # 生成用户网络数据文件
    # with open("./dataset/DeepHawkes/user_network.txt", "w") as f:
    #     for e in E:
    #         f.write(e[0] + " " + e[1] + "\n")
    print(user_network["V"].__len__())
    print(user_network["E"].__len__())

    # 生成传播网络数据文件
    print(cascade_network.__len__())
    # with open("./dataset/DeepHawkes/cascade_network.txt", "w") as f:
    #     for cascade in cascade_network:
    #         f.write(cascade["start_time"] + "\t" + str(cascade["Ec"].__len__()) + "\t")
    #         for e in cascade["Ec"]:
    #             f.write(e[0] + "/" + e[1] + ":" + e[2] + " ")
    #         f.write("\n")


    # G = nx.DiGraph()
    # for edge in user_network["E"]:
    #     G.add_edge(edge[0], edge[1])
    # nx.draw(G, with_labels=True)
    # plt.show()

    return cascade_network, user_network


def deephawkes_one():
    """
    读取deephawkes数据集，返回传播网络和用户网络
    传播网络casecade_network: list，代表多个传播级联
        对于每个级联，数据结构如下：
        - Vc: set 节点集合，元素为用户ID: string
        - Ec: set 边集合，元素为传播路径: tuple(start: string, end: string, time: string)，代表从start传播到end
        - start_time: string 起始时间
        - start_user: string 起始用户
    用户网络user_network: dict，代表构建的用户网络
    - V: 节点集合，元素为用户ID: string
    - E: 边集合，元素为关注关系: tupe(start: string, end:string)，代表start关注了end
    """

    filepath = "./dataset/DeepHawkes/dataset_weibo.txt"
    with open(filepath, "r") as f:
        dataset = [line.strip() for line in f]

    # with open("./deephawkes_id.txt") as f:
    #     dataids = [line.strip() for line in f]

    # special = dataset[25582]
    #
    # dataset = random.sample(dataset, 1000) # 抽选数据集
    # dataset.append(special)

    # 传播网络集合
    cascade_network = []

    # 用户网络
    V = set()
    E = set()
    user_network = dict()

    for line in dataset:
        # 存储一个传播网络
        Vc = set()
        Ec = set()
        cascade_data = dict()

        temp = line.split("\t")

        start_time = temp[2]
        start_user = temp[1]
        ID = temp[0]

        start_time_unix = int(start_time)

        cascades = temp[4].split()

        cascades.pop(0) # 去除初始

        for cas in cascades:
            temp_cas = cas.split(":")
            time = temp_cas[1]
            path = temp_cas[0].split("/")

            index = 0
            while index < len(path) - 1:
                start = path[index]
                end = path[index + 1]

                if start == -1 or end == -1 or start == end:
                    index += 1
                    continue

                V.add(start)
                V.add(end)
                E.add((end, start))

                # 只观测24小时内，因此不需要存储超过24小时的转发
                if int(time) > 24 * 3600:
                    index += 1
                    continue
                if ID == '25589':
                    Vc.add(start)
                    Vc.add(end)
                    Ec.add((start, end, str(int(start_time) + int(time))))
                index += 1

        # 筛选传播网络数据集，但仍然使用被筛选的传播网络构造用户网络
        # 丢弃大小小于10或者大于1000的传播网络
        if len(Ec) > 1000 or len(Ec) < 10:
            continue

        # 丢弃发布时间在24:00-8:00的传播网络
        # if start_hour < 8:
        #     continue

        # 有的傻逼全是自己转发自己
        if Ec.__len__() == 0:
            continue

        # if ID not in dataids:
        #     continue

        cascade_data["Vc"] = Vc
        cascade_data["Ec"] = Ec
        cascade_data["start_time"] = start_time
        cascade_data["start_user"] = start_user
        cascade_data["ID"] = ID
        cascade_network.append(cascade_data)

    user_network["V"] = V
    user_network["E"] = E

    # # 生成用户网络数据文件
    # with open("./dataset/DeepHawkes/user_network.txt", "w") as f:
    #     for e in E:
    #         f.write(e[0] + " " + e[1] + "\n")
    print(user_network["V"].__len__())
    print(user_network["E"].__len__())

    # 生成传播网络数据文件
    print(cascade_network.__len__())
    # with open("./dataset/DeepHawkes/cascade_network.txt", "w") as f:
    #     for cascade in cascade_network:
    #         f.write(cascade["start_time"] + "\t" + str(cascade["Ec"].__len__()) + "\t")
    #         for e in cascade["Ec"]:
    #             f.write(e[0] + "/" + e[1] + ":" + e[2] + " ")
    #         f.write("\n")


    # G = nx.DiGraph()
    # for edge in user_network["E"]:
    #     G.add_edge(edge[0], edge[1])
    # nx.draw(G, with_labels=True)
    # plt.show()

    return cascade_network, user_network
