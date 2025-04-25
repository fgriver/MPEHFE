import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GGNN(nn.Module):
    def __init__(self, hidden_size):
        super(GGNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 2 * self.hidden_size
        self.gate_size = 3 * self.hidden_size  # 同时处理 更新门i 重置门r 新状态n
        # 注意是 Tensor 不是 tensor
        self.wi = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.bi = nn.Parameter(torch.Tensor(self.gate_size))

        self.wh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.bh = nn.Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

    def GRUCell(self, A, hidden):
        """
        :param A:
        :param hidden:
        :return:
        """
        # A = A
        # hidden = hidden
        input_in = torch.matmul(A[:, :, :A.shape[1]].float(), self.linear_edge_in(hidden).float()) + self.b_iah.float()
        input_out = torch.matmul(A[:, :, A.shape[1]:].float(), self.linear_edge_out(hidden).float()) + self.b_oah.float()
        input_emb = torch.cat([input_in, input_out], 2)
        # 处理 inputs-> [b x l x 2*h]
        gi = F.linear(input_emb, self.wi, self.bi)
        # 处理 hidden_emb-> [b x l x h]
        gh = F.linear(hidden, self.wh, self.bh)
        # GRU需要按固定顺序拆分
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        update_gate = torch.sigmoid(i_i + h_i)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(reset_gate * h_n + i_n)

        hidden_state = (1 - update_gate) * hidden + update_gate * new_gate

        return hidden_state

    def forward(self, A, hidden):
        output = self.GRUCell(A, hidden)
        output = F.normalize(output, p=2, dim=-1)
        hidden = hidden + output
        return hidden

class PGAT(nn.Module):
    def __init__(self, hidden_size):
        super(PGAT, self).__init__()

        self.hidden_size = hidden_size
        self.query_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.query_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.layer_norm = nn.LayerNorm(self.hidden_size)

    def mask_softmax(self, logits, mask):
        mask_bool = (mask == 0)
        logits[mask_bool] = float('-inf')
        return torch.softmax(logits, -1)

    def forward(self, items_emb, star_node, item_mask=None):
        q_items = self.query_1(items_emb)
        k_star = self.key_1(star_node)

        attn_weights = torch.matmul(q_items, k_star.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        sate_hidden = (1. - attn_weights) * items_emb + attn_weights * star_node

        # 更新星节点
        q_star = self.query_2(star_node)
        k_items = self.key_2(sate_hidden)
        attn_weights_star = torch.matmul(q_star, k_items.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_weights = self.mask_softmax(attn_weights_star, item_mask.unsqueeze(1))
        star_node = torch.matmul(attn_weights, sate_hidden)

        return sate_hidden, star_node

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        # a_0, a_1, a_2, a_3: 四种关系的可训练权重向量
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj):
        # hidden: [batch_size x nodes_num x emb_dim]
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        # .repeat(1, 1, N).view(batch_size, N*N, self.dim) -> 表示每个节点的特征扩展到了与每一个其他节点交互
        # .repeat(1, N, 1) -> 使每个节点复制N次, 方便与其他节点配对
        # 生成每个节点与其他所有节点之间的交互特征, N_1: 顶点数; N_2: 其他所有节点数
        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)  # batch_size x N x N x 1
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        # 掩码采用-9的15次方达到极小值
        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        # resnet
        output = F.normalize(output, p=2, dim=-1)
        output = output + hidden

        return output

class LongShortAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.1, gnn_step=2, star_step=3):
        super(LongShortAggregator, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.dropout = dropout
        self.gnn_step = gnn_step
        self.star_step = star_step
        self.gnn = GGNN(self.dim)
        self.local_agg = LocalAggregator(dim, alpha)
        self.pgat = PGAT(self.dim)

        self.linear_weight = nn.Linear(2 * self.dim, self.dim)

    def forward(self, A, adj, hidden, s_node, item_mask, no_hybrid_gnn = False):
        """
        :param adj:
        :param no_hybrid_gnn:
        :param item_mask: mask for item level hidden (not sequence hidden)
        :param s_node: star_node
        :param A: GGNN adjacency matrix
        :param hidden: item embedding
        :return:
        """
        sate_hidden = hidden
        for _ in range(self.star_step):
            for _ in range(self.gnn_step):
                sate_hidden = self.gnn(A, sate_hidden)
                # sate_hidden = self.local_agg(sate_hidden, adj)
                if not no_hybrid_gnn:
                    sate_hidden, s_node = self.pgat(sate_hidden, s_node, item_mask)

            if not no_hybrid_gnn:
                weight = torch.sigmoid(self.linear_weight(torch.cat([hidden, sate_hidden], dim=-1)))
                sate_hidden = weight * hidden + (1 - weight) * sate_hidden

                return sate_hidden, s_node

        return sate_hidden, 0

class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        # self.bias = nn.Parameter(torch.Tensor(self.dim))

    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.0001)
        return value

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        """
        :param self_vectors:
        :param neighbor_vector:
        :param batch_size:
        :param masks:
        :param neighbor_weight:
        :param extra_vector: session mean Embedding b * d -> b * neighbor_num * d
        :return:
        """
        # Propagation
        if extra_vector is not None:
            sample_num = neighbor_vector.shape[-2]
            neighbor_emb = torch.div(torch.sum(neighbor_vector, dim=-2), sample_num)

            alpha = torch.matmul(
                torch.cat([self_vectors.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1) * neighbor_vector,
                            neighbor_weight.unsqueeze(-1)], -1
                            ),
                self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            # try to exchange to the 'entmax' function
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)

            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        # Aggregation
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        # output: h^{g, (k)}_v: global item embedding
        return output, neighbor_vector

class LastAggregator(nn.Module):
    def __init__(self, dim, dropout=0.4, name=None):
        super(LastAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.q_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.k_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.linear_one = nn.Linear(self.dim, self.dim)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_weight = nn.Linear(self.dim, 1, bias=False)

    def forward(self, item_hidden, adj_hidden):
        # calculate importance
        query = torch.matmul(adj_hidden, self.q_1)   # b x n x m x d
        key = torch.matmul(item_hidden.unsqueeze(-2), self.k_1)    # b x n x 1 x d
        alpha = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)    # b x n x m x 1
        # adj info aggregation
        agg_hidden = torch.sum(alpha * adj_hidden, dim=-2)   # b x n x d
        agg_hidden = F.dropout(agg_hidden, self.dropout, training=self.training)
        weight = torch.sigmoid(torch.matmul(item_hidden, self.w_1) + torch.matmul(agg_hidden, self.w_2))    # b x n x d
        final_hidden = (1 - weight) * item_hidden + weight * agg_hidden  # b x n x d
        # generate short-term preference
        avg_hidden = torch.sum(final_hidden, dim=1, keepdim=True) / final_hidden.shape[1]  # b x 1 x d
        beta = self.linear_weight(torch.sigmoid(self.linear_one(final_hidden) + self.linear_two(avg_hidden))) # b x n x 1
        last_hidden = torch.sum(beta * final_hidden, dim=1)    # b x d
        return last_hidden