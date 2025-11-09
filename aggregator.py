import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import scaled_dot_product_attention as sdpa


class GGNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = 2 * hidden_size
        self.gate_size = 3 * hidden_size

        self.wi = nn.Parameter(torch.empty(self.gate_size, self.input_size))
        self.bi = nn.Parameter(torch.empty(self.gate_size))
        self.wh = nn.Parameter(torch.empty(self.gate_size, self.hidden_size))
        self.bh = nn.Parameter(torch.empty(self.gate_size))

        self.linear_edge_in  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # 初始化别忘了
        nn.init.xavier_uniform_(self.wi); nn.init.zeros_(self.bi)
        nn.init.xavier_uniform_(self.wh); nn.init.zeros_(self.bh)

    def GRUCell(self, A, hidden):
        # A: [B, L, 2L], hidden: [B, L, H]
        B, L, H = hidden.shape
        A_in, A_out = A[:, :, :L], A[:, :, L:]                 # 只切一次

        h_in  = self.linear_edge_in(hidden)                    # [B, L, H]
        h_out = self.linear_edge_out(hidden)                   # [B, L, H]

        # 邻接汇聚（批量矩阵乘，快于 broadcast）
        input_in  = torch.bmm(A_in,  h_in)                     # [B, L, H]
        input_out = torch.bmm(A_out, h_out)                    # [B, L, H]
        input_emb = torch.cat([input_in, input_out], dim=-1)   # [B, L, 2H]

        gi = F.linear(input_emb, self.wi, self.bi)             # [B, L, 3H]
        gh = F.linear(hidden,   self.wh, self.bh)              # [B, L, 3H]
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)

        update_gate = torch.sigmoid(i_i + h_i)
        reset_gate  = torch.sigmoid(i_r + h_r)
        new_gate    = torch.tanh(reset_gate * h_n + i_n)

        return (1. - update_gate) * hidden + update_gate * new_gate

    def forward(self, A, hidden):
        out = self.GRUCell(A, hidden)
        # L2 normalize 每步做一次很花时间，建议挪到外层需要的地方再做
        # out = F.normalize(out, p=2, dim=-1)
        return hidden + out

class PGAT(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        H = hidden_size
        self.query_1 = nn.Linear(H, H, bias=False)
        self.key_1   = nn.Linear(H, H, bias=False)
        self.query_2 = nn.Linear(H, H, bias=False)
        self.key_2   = nn.Linear(H, H, bias=False)

    def forward(self, items_emb, star_node, item_mask=None):
        # items <- star
        q = self.query_1(items_emb)  # [B, L, H]
        k = self.key_1(star_node)  # [B, 1, H]
        v = star_node  # [B, 1, H]
        alpha = sdpa(q, k, v, attn_mask=None, is_causal=False)  # [B, L, H]
        sate_hidden = (1. - alpha) * items_emb + alpha * star_node

        # star <- items
        q2 = self.query_2(star_node)  # [B, 1, H]
        k2 = self.key_2(sate_hidden)  # [B, L, H]
        v2 = sate_hidden  # [B, L, H]
        attn_mask_star = None
        if item_mask is not None:
            attn_mask_star = ~(item_mask.bool()).unsqueeze(1)  # [B, 1, L]

        star_node = sdpa(q2, k2, v2, attn_mask=attn_mask_star, is_causal=False)  # [B, 1, H]
        return sate_hidden, star_node


class LongShortAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.1, gnn_step=2, star_step=3):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.gnn_step = gnn_step
        self.star_step = star_step
        self.gnn = GGNN(dim)
        self.pgat = PGAT(dim)
        self.linear_weight = nn.Linear(2*dim, dim)

    def forward(self, A, adj, hidden, s_node, item_mask, no_hybrid_gnn=False):
        sate_hidden = hidden

        for _ in range(self.star_step):
            for _ in range(self.gnn_step):
                sate_hidden = self.gnn(A, sate_hidden)
                if not no_hybrid_gnn:
                    sate_hidden, s_node = self.pgat(sate_hidden, s_node, item_mask)  # PGAT

            if not no_hybrid_gnn:
                gate = torch.sigmoid(self.linear_weight(torch.cat([hidden, sate_hidden], dim=-1)))
                sate_hidden = gate * hidden + (1. - gate) * sate_hidden
        return sate_hidden, s_node

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
        # Propagation
        if extra_vector is not None:
            # OPTIMIZATION: Use .expand() for memory efficiency instead of .repeat().
            expanded_self = self_vectors.unsqueeze(2).expand(-1, -1, neighbor_vector.shape[2], -1)

            # Compute attention scores.
            alpha = torch.matmul(
                torch.cat([expanded_self * neighbor_vector,
                           neighbor_weight.unsqueeze(-1)], -1),
                self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)

            # Aggregate neighbor vectors using the computed attention.
            aggregated_neighbors = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            # If no extra_vector is provided, fall back to mean aggregation.
            aggregated_neighbors = torch.mean(neighbor_vector, dim=2)

        # Aggregation with self-vectors
        output = torch.cat([self_vectors, aggregated_neighbors], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)

        # output: h^{g, (k)}_v: global item embedding
        # Return the main output and the aggregated neighbor representation.
        return output, aggregated_neighbors

  
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