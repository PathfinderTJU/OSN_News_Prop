import math
import numpy as np
import networkx as nx
from dataloader import deephawkes
from randomwalk import randomwalk
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from weight import cluster_coefficient
from weight import zombie_follower
from weight import activity
from weight import log_min_max_normal
from simrank import monte_carlo_simrank


# 模型
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, V, bidirectional=True):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 定义嵌入层
        self.embedding = nn.Embedding(num_embeddings=V, embedding_dim=input_size)

        # 定义双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                          dropout=dropout_rate if num_layers > 1 else 0)

        # 定义多头注意力层
        self.attention = SumAttention()

        # 定义MLP层
        self.fc1 = nn.Linear(2 * hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)  # 偏置初始化为0

        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)  # 偏置初始化为0

        nn.init.kaiming_normal_(self.out.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.out.bias, 0)  # 偏置初始化为0

        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, weight_w, weight_i, weight_t):
        # x的形状：(batch_size, K, L)
        # 将x变形为(batch_size * K, L)以便于embedding层处理
        batch_size, K, L = x.size()
        x = x.view(batch_size * K, L)

        # 嵌入节点
        x = self.embedding(x)   # x形状：(batch_size * K, L, input_size)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size * K, self.hidden_size).to(x.device)

        # 双向GRU学习表示
        x, _ = self.gru(x, h0)  # x形状：(batch_size * K, L, 2*hidden_size)

        # 将x变形回(batch_size, K, L, 2*hidden_size)
        x = x.view(batch_size, K, L, -1)
        # print(x.size())

        # 计算注意力权重
        sequence_h = self.attention(x, weight_w, weight_i, weight_t)
        # print(sequence_h.size())

        # 平均池化
        average_sequence_h = torch.mean(sequence_h, dim=1)

        # dropout
        average_sequence_h = self.dropout(average_sequence_h)

        # 输入MLP，得到预测结果
        average_sequence_h = F.relu(self.fc1(average_sequence_h))
        average_sequence_h = self.dropout(average_sequence_h)
        average_sequence_h = F.relu(self.fc2(average_sequence_h))
        result = self.out(average_sequence_h)

        return result


# 多头注意力机制，用于组合权重
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.combine = nn.Linear(2, 1, bias=False)
        # self.w_combine = nn.Linear(3, 1, bias=False)

        nn.init.kaiming_normal_(self.combine.weight, mode='fan_in', nonlinearity='relu')

        # nn.init.kaiming_normal_(self.w_combine.weight, mode='fan_in', nonlinearity='relu')

        # self.time_caculate = TimeAttention(initial_a=0.0884, initial_b=-0.8477)

    def forward(self, w, i ,t):
        # 假设 w、i、t 的形状为 (batch_size, seq_len, 1)
        # w = torch.cat((wa, wh, wt), dim=-1) # 拼接得到形状(batch_size, seq_len, 3)
        # w = self.w_combine(w)
        # t = self.time_caculate(t)

        # weights = torch.cat((w, i, t), dim=-1)  # 拼接得到形状 (batch_size, seq_len, 3)
        weights = torch.cat((i, t), dim=-1)
        # weights = torch.cat((w, t), dim=-1)
        # weights = torch.cat((w, i), dim=-1)
        combined_weights = self.combine(weights)  # 线性变换得到形状 (batch_size, seq_len, 1)
        attention_weights = F.softmax(combined_weights, dim=-1)  # 对每个序列应用softmax
        return attention_weights


# 传统注意力机制，用于求加权和
class SumAttention(nn.Module):
    def __init__(self):
        super(SumAttention, self).__init__()
        self.attention_weight_combination = MultiHeadAttention()

    def forward(self, node_vectors, w, i, t):
        # node_vectors 的形状为 (batch_size, seq_len, 2*H)
        # w, i, t 的形状为 (batch_size, seq_len, 1)
        attention_weights = self.attention_weight_combination(w, i, t)  # 得到形状为 (batch_size, seq_len, 1)
        attention_weights = attention_weights.expand_as(node_vectors)  # 扩展权重以匹配节点向量的形状
        attended_vectors = attention_weights * node_vectors  # 应用注意力权重
        sequence_vector = torch.sum(attended_vectors, dim=1)  # 按序列求和
        return sequence_vector


# 用于拟合时间幂律参数y=a * x ^ b的注意力层
# class TimeAttention(nn.Module):
#     def __init__(self, initial_a=0.0884, initial_b=-0.8477): # 统计的初始值initial_a=0.0884, initial_b=-0.8477
#         super(TimeAttention, self).__init__()
#         # 将a和b都定义为模型参数，并且需要梯度更新
#         self.a = nn.Parameter(torch.tensor([initial_a]), requires_grad=True)
#         self.b = nn.Parameter(torch.tensor([initial_b]), requires_grad=True)
#
#     # 前向传播方法，接受原始特征x作为输入
#     def forward(self, x):
#         # 应用幂律函数y=a*x^b，确保x为非负数
#         weight_t = self.a * torch.pow(torch.clamp(x, min=0), self.b)
#
#         return weight_t

# 定义MSLE计算函数
def msle_loss(pred, true):
    return torch.mean((torch.log(pred + 1) - torch.log(true + 1)) ** 2)


# 数据集
class SampleDataset(Dataset):
    def __init__(self, sequences, labels, weight_w, weight_i, weight_t):
        self.sequences = sequences
        self.labels = labels
        self.weight_w = weight_w
        self.weight_i = weight_i
        self.weight_t = weight_t

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item], self.labels[item], self.weight_w[item], self.weight_i[item], self.weight_t[item]


# 训练模型
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_mae_loss = 0.0
    # total_msle_loss = 0.0
    total_mse_loss = 0.0
    mse_func = nn.MSELoss()

    for data, target, w, i, t in train_loader:
        data, target, w, i, t = data.to(device), target.to(device), w.to(device), i.to(device), t.to(device)
        optimizer.zero_grad()
        output = model(data, w, i, t)
        loss_mae = criterion(output, target)
        loss_mae.backward()

        # for name, parameter in model.named_parameters():
        #     if parameter.grad is not None:
        #         grad_value = parameter.grad.data
        #         if torch.isnan(grad_value).any() or torch.isinf(grad_value).any():
        #             print(f"Gradient explosion detected in {name}")

        optimizer.step()
        # 累加MAE损失
        total_mae_loss += loss_mae.item()

        # 计算MSLE损失（不用于反向传播）
        with torch.no_grad():
            # loss_msle = msle_loss(output, target)
            # total_msle_loss += loss_msle.item()
            loss_mse = mse_func(output, target)
            total_mse_loss += loss_mse

    # 计算平均MAE损失
    avg_mae_loss = total_mae_loss / len(train_loader)
    # 计算平均MSLE损失
    # avg_msle_loss = total_msle_loss / len(train_loader)
    # # 计算MSE
    avg_mse_loss = total_mse_loss / len(train_loader)
    # # 计算RMSE
    # avg_rmse_loss = avg_mse_loss ** 0.5

    print("Train MAE: {:.3f}, MSE: {:.3f}".format(avg_mae_loss, avg_mse_loss))

    return avg_mae_loss, avg_mse_loss


# 验证模型
def validate(model, val_loader, criterion, device):
    model.eval()
    total_mae_loss = 0.0
    # total_msle_loss = 0.0
    total_mse_loss = 0.0
    mse_func = nn.MSELoss()

    with torch.no_grad():
        for data, target, w, i, t in val_loader:
            data, target, w, i, t = data.to(device), target.to(device), w.to(device), i.to(device), t.to(device)
            output = model(data, w, i, t)
            mae_loss = criterion(output, target)
            total_mae_loss += mae_loss.item()

            # loss_msle = msle_loss(output, target)
            # total_msle_loss += loss_msle.item()

            loss_mse = mse_func(output, target)
            total_mse_loss += loss_mse

    # 计算平均MAE损失
    avg_mae_loss = total_mae_loss / len(train_loader)
    # 计算平均MSLE损失
    # avg_msle_loss = total_msle_loss / len(train_loader)
    # 计算MSE
    avg_mse_loss = total_mse_loss / len(train_loader)
    # # 计算RMSE
    # avg_rmse_loss = avg_mse_loss ** 0.5

    print("Train MAE: {:.3f}, MSLE: {:.3f}".format(avg_mae_loss, avg_mse_loss))

    return avg_mae_loss, avg_mse_loss


# 评估模型
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_mae_loss = 0.0
    # total_msle_loss = 0.0
    total_mse_loss = 0.0
    mse_func = nn.MSELoss()

    with torch.no_grad():
        for data, target, w, i, t in test_loader:
            data, target, w, i, t = data.to(device), target.to(device), w.to(device), i.to(device), t.to(device)
            output = model(data, w, i, t)
            mae_loss = criterion(output, target)
            total_mae_loss += mae_loss.item()

            # loss_msle = msle_loss(output, target)
            # total_msle_loss += loss_msle.item()

            loss_mse = mse_func(output, target)
            total_mse_loss += loss_mse

    # 计算平均MAE损失
    avg_mae_loss = total_mae_loss / len(train_loader)
    # 计算平均MSLE损失
    # avg_msle_loss = total_msle_loss / len(train_loader)
    # 计算MSE
    avg_mse_loss = total_mse_loss / len(train_loader)
    # # 计算RMSE
    # avg_rmse_loss = avg_mse_loss ** 0.5

    return avg_mae_loss, avg_mse_loss


# 定义提前退出机制
class EarlyStopping:
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# 读取数据
observation_time = 3600 * 1     # 观测时间：1小时
cascade_network, user_network = deephawkes(observation_time)

user_network_G = nx.DiGraph()
user_network_G.add_edges_from(user_network["E"])

max_degree = max(dict(user_network_G.degree()).values())

# 计算pagerank缓存
pageranks = nx.pagerank(user_network_G, alpha=0.85)
epsilon = 1e-10
pageranks = {node: np.log1p(pr) for node, pr in pageranks.items()}

# 准备一个用于计算用户活跃度的缓存
all_cascades_temp = []
all_cascades = []   # 这个存储所有的级联
all_start_user = []     # 这个存储所有的发起者，因为发起者也要算入1活跃度
for cas in cascade_network:
    all_cascades_temp.extend(cas["Ec"])
    all_start_user.append((cas["start_user"], int(cas["start_time"])))

for cas in all_cascades_temp:
    all_cascades.append((cas[0], cas[1], int(cas[2])))

activities = log_min_max_normal(activity(all_cascades, all_start_user))

# 按时间从小到大排序
# all_cascades = sorted(all_cascades, key=lambda x: x[2])
# all_start_user = sorted(all_start_user, key=lambda x: x[1])


V = len(user_network['V']) + 1  # 加上一个填充节点“+”
K = 50                         # 采样序列数
L = 5                          # 采样长度
H = 60                          # 隐藏状态维度
batch_size = 32                 # 批处理大小
learning_rate = 1e-3            # 学习率
learning_rate_embedding = 5e-3  # 嵌入层学习率
dropout_rate = 0.25             # dropout率
num_epochs = 1000000                # 默认的迭代次数
patience = 10000                    # 提前退出的耐心值

# 采样
# 存储所有样本的采样序列集合S0
cascade_sample_origin = []
# 存储所有样本的级联起始时间
cascade_start_time = []
# 标签（0~观测时间内的流行度增量）
labels = []
# 平均路径长度
average_path_length = 0

# 每个样本采样K个长度为L的序列，称为S，对应样本的级联大小（在观测时间内）
for cascade in cascade_network:
    cascade_start_time.append(int(cascade["start_time"]))
    cascade_network_G = nx.DiGraph()

    node_times = {cascade["start_user"]: int(cascade["start_time"])}
    node_formers = {cascade["start_user"]: cascade["start_user"]}
    for edge in cascade["Ec"]:
        A, B, T = edge
        cascade_network_G.add_edge(A, B)
        if B not in node_times:
            node_times[B] = float('inf')

        if B not in node_formers:
            node_formers[B] = -1

        # 更新节点B的time为最早出现时间
        if int(T) < node_times[B]:
            node_times[B] = int(T)
            node_formers[B] = A

    # 设置每个节点的time属性
    for node, time in node_times.items():
        cascade_network_G.nodes[node]['time'] = time

    for node, former in node_formers.items():
        cascade_network_G.nodes[node]['former'] = former

    sampleseq = randomwalk(user_network_G, cascade_network_G, K=K, L=L)
    cascade_sample_origin.append(sampleseq)

    label = 0
    for e in cascade["Ec"]:
        if int(e[2]) - int(cascade['start_time']) <= observation_time:
            label += 1
    labels.append(label)

    average_path_length += cascade["average_path_length"]

print("Average Cascade Size:", np.mean(labels))
print("Average Path Length:", average_path_length / len(cascade_network))

cascade_sample = [[[node[0] for node in seq] for seq in sample] for sample in cascade_sample_origin]

# print(cascade_sample)
# print(cascade_network)

# 计算注意力权重
weight_w = []
weight_i = []
weight_t = []

activities_temp = {}

print("caculate weight...")

# 每个样本
for i, sample in enumerate(cascade_sample):
    weight_w_i = []
    weight_i_i = []
    weight_t_i = []

    cascade = cascade_network[i]
    start_time = int(cascade["start_time"])

    # 每条序列
    for k, seq in enumerate(sample):
        weight_w_k = []
        weight_i_k = []
        weight_t_k = []

        # 每个节点
        for l, node in enumerate(seq):
            # 处理填充节点，权重均为0
            if node == '+':
                weight_w_k.append(0)
                weight_i_k.append(0)
                weight_t_k.append(-1) # 时间权重还要学习，因此传入-1，在模型中进行判断
                continue

            # 节点时间
            node_time = int(cascade_sample_origin[i][k][l][1])

            # 前驱结点
            former_node = cascade_sample_origin[i][k][l][2]

            # 活跃度
            # 活跃度做对数+min-max归一化
            if node in activities.keys():
                ai = activities[node]
            else:
                ai = 1e-12

            if former_node in activities.keys():
                aj = activities[former_node]
            else:
                aj = 1e-12

            # 1. 传播意愿w_i
            # 1.1 活跃度wa_i
            wa_i = ai

            # 1.2 前驱结点影响力wh_j
            # 1.2.1 全局影响力ig_j
            ig_j = pageranks[former_node]

            # 1.2.2 拓扑连通性it_j
            it_j = cluster_coefficient(user_network_G, former_node)


            # 1.2.3 虚假关注者得分in_j
            in_j = zombie_follower(user_network_G, former_node, pageranks) / math.sqrt(max_degree)
            # print(ig_j, it_j, in_j)

            # 1.2.4 计算
            wh_j = ig_j * it_j * in_j

            # 1.3 信任程度t_ij
            if former_node == node:
                t_ij = 1
            else:
                # 1.3.1 拓扑结构相似度SN_ij
                sn_ij = monte_carlo_simrank(user_network_G, former_node, node, num_walks=1000, walk_length=5)

                # 1.3.2 历史行为相似度SC_ij
                sc_ij = 1 - math.fabs(ai - aj)

                # 1.3.3 计算
                t_ij = 0.5 * sn_ij + 0.5 * sc_ij

            # 1.4 计算
            # w = wa_i * wh_j * t_ij
            w = wh_j * t_ij
            # w = wa_i * t_ij
            # w = wa_i * wh_j
            weight_w_k.append(w)

            # 2. 时间特征t_i
            t_i = node_time - cascade_start_time[i]
            if t_i < 600:
                t_i = 0.0122
            else:
                t_i = 0.0884 * (t_i ** -0.8477)
            weight_t_k.append(t_i)

            # 3. 网络特征i_i
            # 3.1 全局影响力ig_i
            ig_i = pageranks[node]

            # 3.2 拓扑连通性it_i
            it_i = cluster_coefficient(user_network_G, node)

            # 3.3 虚假关注者得分in_i
            in_i = zombie_follower(user_network_G, node, pageranks) / math.sqrt(max_degree)

            # 3.4 计算
            ii = ig_i * it_i * in_i
            weight_i_k.append(ii)

        weight_w_i.append(weight_w_k)
        weight_i_i.append(weight_i_k)
        weight_t_i.append(weight_t_k)

        # print(i + 1, "sample", k + 1, "sequence completed")

    weight_w.append(weight_w_i)
    weight_i.append(weight_i_i)
    weight_t.append(weight_t_i)

    print(i + 1, "sample completed")

print("weight caculate completed")

# 嵌入与编码
# 1. 创建一个从节点名称到整数索引的映射
unique_nodes = set(node for group in cascade_sample for seq in group for node in seq if node != '+')
node_to_index = {node: i+1 for i, node in enumerate(unique_nodes)}
node_to_index["+"] = 0

# 2. 将序列中的节点名称映射到整数索引
index_cascade_sample = [[[node_to_index[node] for node in seq] for seq in group] for group in cascade_sample]
# 3. 将索引序列转化为张量
index_cascade_sample_tensors = [torch.tensor(group, dtype=torch.long) for group in index_cascade_sample]
# 4. 将标签转化为张量
tensor_labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
# 5. 将注意力权重转换为张量
weight_w_tensors = [torch.tensor(group, dtype=torch.float32).unsqueeze(-1) for group in weight_w]
weight_i_tensors = [torch.tensor(group, dtype=torch.float32).unsqueeze(-1) for group in weight_i]
weight_t_tensors = [torch.tensor(group, dtype=torch.float32).unsqueeze(-1) for group in weight_t]


# 实例化数据集，划分训练集、验证集、测试集
dataset = SampleDataset(index_cascade_sample_tensors, tensor_labels, weight_w_tensors, weight_i_tensors, weight_t_tensors)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
# test_size = len(dataset) - train_size - val_size
# train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
train_set, val_set = random_split(dataset, [train_size, val_size])

print("Size:", train_size, val_size)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 定义模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiGRU(input_size=H, hidden_size=H, num_layers=1, dropout_rate=dropout_rate, V=V).to(device)
criterion = nn.L1Loss()

# 将嵌入层参数与模型的其余参数分开
embedding_params = model.embedding.parameters()
other_params = [p for n, p in model.named_parameters() if 'embedding' not in n]

# 创建优化器，为不同的参数组设置不同的学习率
optimizer = optim.Adam([
    {'params': embedding_params, 'lr': learning_rate_embedding},
    {'params': other_params, 'lr': learning_rate}
])
early_stop = EarlyStopping(patience, verbose=True)

# 开始训练
best_mae = float('inf')
best_msle = float('inf')
patience_count = 0
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    train_loss_mae, train_loss_msle = train(model, train_loader, criterion, optimizer, device)
    val_loss_mae, val_loss_msle = validate(model, val_loader, criterion, device)

    # early_stop(val_loss)

    if train_loss_mae < best_mae:
        best_mae = train_loss_mae
        best_msle = train_loss_msle
        patience_count = 0

    if val_loss_mae < best_mae:
        best_mae = val_loss_mae
        best_msle = val_loss_msle
        patience_count = 0

    patience_count += 1
            # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), 'best_model.pth')
    #     print(f"Model saved at epoch: {epoch + 1}")
    #
    # if early_stop.early_stop:
    #     print("Early Stopped")
    #     print("-  " * 30)
    #     break
    print("best mae: {:.3f}, best msle: {:.3f}".format(best_mae, best_msle))
    print("-" * 30)

    if patience_count >= patience:
        break

print("best mae: {:.3f}, best msle: {:.3f}".format(best_mae, best_msle))

# model.load_state_dict(torch.load('best_model.pth'))
# test_mae, test_msle, test_mse, test_rmse = evaluate(model, test_loader, criterion, device)
# print("Test MAE: {:.3f}, MSLE: {:.3f}, MSE: {:.3f}, RMSE: {:.3f}".format(test_mse, test_rmse, test_mse, test_rmse))
