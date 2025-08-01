import random

import networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import pytz
import time


def deephawkes(ob_time):
    """
    读取deephawkes数据集，返回传播网络和用户网络
    传播网络casecade_network: list，代表多个传播级联
        对于每个级联，数据结构如下：
        - Vc: set 节点集合，元素为用户ID: string
        - Ec: set 边集合，元素为传播路径: tuple(start: string, end: string, time: string)，代表从start传播到end
        - start_time: string 起始时间
    用户网络user_network: dict，代表构建的用户网络
    - V: 节点集合，元素为用户ID: string
    - E: 边集合，元素为关注关系: tupe(start: string, end:string)，代表start关注了end
    """

    filepath = "./dataset/DeepHawkes/dataset_weibo.txt"
    with open(filepath, "r") as f:
        dataset = [line.strip() for line in f]

    dataset = random.sample(dataset, 1000)

    # with open ("./dataset/DeepHawkes/last_data.txt", "w") as f:
    #     for line in dataset:
    #         f.write(str(line) + "\n")

    # 传播网络集合
    cascade_network = []

    # 用户网络
    V = []
    E = []
    user_network = dict()

    for line in dataset:
        # 存储一个传播网络
        Vc = set()
        Ec = set()
        temp_V = []
        temp_E = []

        cascade_data = dict()

        temp = line.split("\t")

        start_user = temp[1]

        start_time = temp[2]

        start_time_unix = int(start_time)

        # 获取本地时间
        start_hour = datetime.datetime.fromtimestamp(start_time_unix, tz=pytz.timezone('Etc/GMT-8'))

        # 获取小时
        start_hour = start_hour.hour

        cascades = temp[4].split()

        cascades.pop(0) # 去除初始

        for cas in cascades:
            temp_cas = cas.split(":")
            time = temp_cas[1]
            path = temp_cas[0].split("/")

            if int(time) > ob_time:
                continue

            index = 0
            while index < len(path) - 1:
                start = path[index]
                end = path[index + 1]

                if start == -1 or end == -1 or start == end:
                    index += 1
                    continue

                temp_V.append(start)
                temp_V.append(end)
                temp_E.append((end, start))

                Vc.add(start)
                Vc.add(end)
                Ec.add((start, end, str(int(start_time) + int(time))))
                index += 1

        # 筛选传播网络数据集，但仍然使用被筛选的传播网络构造用户网络
        # 丢弃大小小于10或者大于1000的传播网络
        if Ec.__len__() > 1000 or Ec.__len__() < 10:
            continue

        # # 丢弃发布时间在24:00-8:00的传播网络
        if start_hour < 8 or start_hour > 18:
            continue

        temp_Ec = [(e[0], e[1]) for e in Ec]
        G = networkx.DiGraph()
        G.add_edges_from(temp_Ec)

        # 使用单源最短路径算法计算从起点到所有其他节点的最短路径
        lengths = nx.single_source_shortest_path_length(G, start_user)

        # 计算平均深度
        avg_path_length = sum(lengths.values()) / (len(lengths) - 1) + 1

        cascade_data["Vc"] = Vc
        cascade_data["Ec"] = Ec
        cascade_data["start_time"] = start_time
        cascade_data["start_user"] = start_user
        cascade_data["average_path_length"] = avg_path_length
        cascade_network.append(cascade_data)

        V.extend(temp_V)
        E.extend(temp_E)

    user_network["V"] = set(V)
    user_network["E"] = set(E)

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

# print(deephawkes(3600))