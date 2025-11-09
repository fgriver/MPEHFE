import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

class Data(Dataset):
    def __init__(self, data, adj, last_len=2, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)

        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        # length: 整个数据集的大小
        self.length = len(data[0])
        # 数据集中序列的最大长度
        self.max_len = max_len
        # max_n_node: main 中的 n_node
        # self.max_n_node = max_n_node

    def __getitem__(self, index):
        # u_input中已经有0了
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        max_sess_len = self.max_len
        # max_n_node = self.max_n_node
        # node中第一个一定是0
        node = np.unique(u_input)
        items = node.tolist() + (max_sess_len - len(node)) * [0]

        adj = np.zeros((max_sess_len, max_sess_len))
        # 1: self 2:out 3:in 4:in&out
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            # 添加自环
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        u_A = np.zeros((max_sess_len, max_sess_len))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        # u_A = u_A
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        u_A = np.concatenate([u_A_in, u_A_out]).transpose()

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(u_A), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length
