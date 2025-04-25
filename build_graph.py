import pickle
import argparse
import numpy as np
from networkx.classes import neighbors

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Nowplaying', help='diginetica/Tmall/Nowplaying/RetailRocket')
# sample_num: 邻居的数量
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seqs = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
elif opt.dataset == 'RetailRocket':
    num = 60965
else:
    num = 3

# relation: 保存交互信息, 存储 e-Neighbor 的双向信息以计算权重
relation = []
"""
    adj1: [
            { vj1: 2, vj2: 1, ...  vjl: xx  }  -> 顶点1
            { vj1: 2, vj2: 1, ...  vjl: xx  }  -> 顶点2
            ......
            {     }
           ]
"""
# adj1:  保存 边的权重, 表现形式为 (vi, vj)的交互数量
adj1 = [dict() for _ in range(num)]
# adj: 筛选出来权值最大的邻居
adj = [[] for _ in range(num)]


for i in range(len(seqs)):
    data = seqs[i]
    # e = 3, find e-Neighbor
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    # 由于sorted默认从小到大排列->所以reverse使其从大到小排列
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    # adj[t]: 第t个顶点邻居节点
    adj[t] = [v[0] for v in x]
    # 权重
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

# adj: 全局图邻居节点
# weight: 对应的权重
pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))


negative_samples = []
for i in range(num):
    neighbors = adj[i]
    sample_negative = np.setdiff1d(np.arange(num), neighbors)
    negative_samples.append(np.random.choice(sample_negative, 1, replace=False))
pickle.dump(negative_samples, open('datasets/' + dataset + '/neg_' + str(sample_num) + '.pkl', 'wb'))
