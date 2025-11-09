import datetime
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from tqdm import tqdm
from aggregator import GlobalAggregator, LongShortAggregator, LastAggregator
import os
import pickle
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import sys

class AutomaticSearchModuleWithLengths(nn.Module):
    def __init__(self, dim, candidate_lengths):
        super(AutomaticSearchModuleWithLengths, self).__init__()
        self.register_buffer('candidate_lengths', torch.tensor(candidate_lengths, dtype=torch.long))
        self.additive_layer = nn.Linear(dim, 1)

    def forward(self, embeddings, actual_lengths):

        batch, seq_len, dim = embeddings.shape
        num_candidates = len(self.candidate_lengths)

        padded_embeddings = F.pad(embeddings, (0, 0, 1, 0))  # shape: (batch, max_len+1, dim)
        cumsum_embed = torch.cumsum(padded_embeddings, dim=1)

        # actual_lengths: [5, 8], candidate_lengths: [1, 2, 3]
        # end_indices: [[5], [8]], shape: (batch, 1)
        end_indices = actual_lengths.unsqueeze(1)
        # start_indices: [[4, 3, 2], [7, 6, 5]], shape: (batch, num_candidates)
        start_indices = end_indices - self.candidate_lengths.unsqueeze(0)
        start_indices = torch.clamp(start_indices, min=0)  # 防止索引为负

        end_indices_exp = end_indices.unsqueeze(-1).expand(batch, num_candidates, dim)
        start_indices_exp = start_indices.unsqueeze(-1).expand(batch, num_candidates, dim)

        sum_at_end = cumsum_embed.gather(1, end_indices_exp)
        sum_at_start = cumsum_embed.gather(1, start_indices_exp)
        sum_embeddings = sum_at_end - sum_at_start  # shape: (batch, num_candidates, dim)
        valid_lengths = self.candidate_lengths.unsqueeze(0).repeat(batch, 1)
        valid_lengths = torch.min(valid_lengths, actual_lengths.unsqueeze(1))
        valid_lengths = torch.clamp(valid_lengths.float().unsqueeze(-1), min=1e-9)
        pooled_embeddings = sum_embeddings / valid_lengths  # shape: (batch, num_candidates, dim)
        weights = self.additive_layer(pooled_embeddings).squeeze(-1)  # shape: (batch, num_candidates)
        attention_weights = F.softmax(weights, dim=1)  # shape: (batch, num_candidates)

        final_representation = torch.sum(attention_weights.unsqueeze(-1) * pooled_embeddings, dim=1)

        return final_representation


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num, neg, max_seq_len):
        super(CombineGraph, self).__init__()
        # 消融
        self.no_skip_connection = opt.no_skip_connection
        self.agg_model = opt.agg_model
        self.no_hybrid_gnn = opt.no_hybrid_gnn
        self.no_global = opt.no_global
        self.no_local = opt.no_local
        # 调参
        self.k = opt.k
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        # global hop
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.neg = trans_to_cuda(neg).long()

        self.long_short_agg = LongShortAggregator(self.dim, opt.alpha, dropout=0.0, gnn_step=opt.gnn_step, star_step=opt.star_step)
        # Aggregator
        # self.local_agg = LocalAggregator(self.dim, opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item original representation & Position representation
        self.pos_embedding = nn.Embedding(300, self.dim)
        self.embedding = nn.Embedding(num_node, self.dim, padding_idx=0)
        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.soft_func = nn.Softmax(dim=1)
        self.scale = opt.scale
        self.norm = opt.norm
        self.tau = opt.tau
        self.loss_function = nn.CrossEntropyLoss()

        self.candidate_lengths = list(range(1, max_seq_len + 1))
        self.ASWL = AutomaticSearchModuleWithLengths(self.dim, self.candidate_lengths)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.local_ln = nn.LayerNorm(self.dim)
        self.global_ln = nn.LayerNorm(self.dim)
        self.u_1 = nn.Parameter(torch.Tensor(self.hop, 1))
        self.u_2 = nn.Parameter(torch.Tensor(2, 1))

        self.w_3 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, self.dim))

        self.gate = nn.Linear(2*self.dim, self.dim)
        self.reset_parameters()

        self.att_weight_layer = nn.Linear(self.dim, 1, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        a = target.view(-1)
        b = self.adj_all[a]
        c = self.num[a]
        return b, c

    def get_agg_model(self):
        return self.agg_model

    def hidden_fusion(self, hidden, masks, return_attention=False):
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        pos_emb = self.pos_embedding.weight[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        if self.agg_model == 'df':
            actual_length = torch.sum(masks.squeeze(), dim=-1).long()
            short_term_intent = self.ASWL(hidden, actual_length)
            short_term_intent = F.normalize(short_term_intent, p=2, dim=-1)
            short_term_intent = short_term_intent.unsqueeze(-2).repeat(1, seq_len, 1)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(short_term_intent))
        elif self.agg_model == 'avg':
            hs = torch.sum(hidden * masks, -2) / torch.sum(masks, 1)
            hs = hs.unsqueeze(-2).repeat(1, seq_len, 1)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        else:
            ht = hidden[torch.arange(masks.shape[0]).long(), torch.sum(masks.squeeze(), 1).long() - 1]
            ht = ht.unsqueeze(-2).repeat(1, seq_len, 1)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(ht))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * masks
        select = torch.sum(beta * hidden, 1)

        if return_attention:
            attn_scores = self.att_weight_layer(nh).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            return select, attn_weights
        else:
            return  select

    def find_neighbor(self, x, k, model=0):
        # x: [N, D]
        x = F.normalize(x, p=2, dim=1)  # [N, D]
        sim = x @ x.T  # [N, N]，等价于 cosine similarity
        sim.fill_diagonal_(-float('inf'))  # 屏蔽自身

        # model==0: 取最大k；model!=0: 取最小k（对 -sim 取最大k）
        if model == 0:
            vals, idx = torch.topk(sim, k, dim=-1, largest=True)
        else:
            vals, idx = torch.topk(-sim, k, dim=-1, largest=True)
            vals = -vals

        attn = F.softmax(vals, dim=-1)  # [N, k]
        # 使用高级索引取邻居向量 -> [N, k, D]
        neigh = x[idx]
        # 加权求和 -> [N, D]
        out = torch.sum(attn.unsqueeze(-1) * neigh, dim=1)
        return out

    def compute_cl_loss(self, final_sess_emb, sim_sess_emb, dissim_sess_emb):

        global_emb_flat = F.normalize(final_sess_emb.view(-1, final_sess_emb.shape[-1]), dim=-1)  # [batch*len, dim]
        positive_emb_flat = F.normalize(sim_sess_emb.view(-1, sim_sess_emb.shape[-1]), dim=-1)  # [batch*len, dim]
        negative_emb_flat = F.normalize(dissim_sess_emb.view(-1, dissim_sess_emb.shape[-1]), dim=-1)  # [batch*len, dim]

        neg_sample_size = 128
        neg_indices = torch.randint(0, negative_emb_flat.shape[0], (neg_sample_size,))
        negative_emb_sampled = negative_emb_flat[neg_indices]

        pos_sim = torch.sum(global_emb_flat * positive_emb_flat, dim=-1)
        neg_sim = torch.matmul(global_emb_flat, negative_emb_sampled.T)

        test_pos_sim = F.cosine_similarity(global_emb_flat, positive_emb_flat, dim=-1)
        test_neg_sim = F.cosine_similarity(global_emb_flat, negative_emb_flat, dim=-1)

        temperature = 0.07
        pos_exp = torch.exp(pos_sim / temperature)  # [batch*len]
        neg_exp = torch.sum(torch.exp(neg_sim / temperature), dim=1)  # [batch*len]
        contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        test_pos_sim = torch.exp(test_pos_sim / temperature)
        test_neg_exp = torch.exp(test_neg_sim / temperature)
        test_contrastive_loss = -torch.log(test_pos_sim / (test_pos_sim + test_neg_exp)).mean()
        return test_contrastive_loss

    def compute_scores(self, sess_emb, targets, is_test=False):
        # self.embedding.weight[0] = torch.Tensor[0,0,0..., 0]
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        if self.norm:
            sess_emb = F.normalize(sess_emb, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)
            # b = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)

        scores = torch.matmul(sess_emb, b.transpose(1, 0))
        if self.scale:
            scores = self.tau * scores  # tau is the sigma factor

        if is_test:
            main_output = self.soft_func(scores)
        else:
            targets = trans_to_cuda(targets).long()
            main_output = self.loss_function(scores, targets - 1)

        output = main_output

        return output

    def global_graph_nn(self, items, mask_item, inputs):
        batch_size = items.shape[0]
        seqs_len = items.shape[1]

        seq_item_emb = self.embedding(inputs) * mask_item.float().unsqueeze(-1)
        item_neighbors = [items]
        weight_neighbors = []
        support_size = seqs_len

        # This sampling logic remains unchanged.
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors
        session_info = []

        sum_item_emb = torch.sum(seq_item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        # OPTIMIZATION: Use a single variable to store the last neighbor embedding
        # instead of appending to a list in every iteration.
        last_neighbor_emb = None

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                # vector: the updated representation for the current hop's nodes.
                # neigh_emb: the aggregated representation of the neighbors.
                vector, neigh_emb = aggregator(self_vectors=entity_vectors[hop],
                                               neighbor_vector=entity_vectors[hop + 1].view(shape),
                                               masks=mask_item,
                                               batch_size=batch_size,
                                               neighbor_weight=weight_vectors[hop].view(batch_size, -1,
                                                                                        self.sample_num),
                                               extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)

                # OPTIMIZATION: Overwrite the variable, storing only the latest embedding.
                last_neighbor_emb = neigh_emb
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # The final neighbor embedding is the one captured from the last aggregator call.
        neighbor_emb = last_neighbor_emb
        return h_global, neighbor_emb

    def forward(self, items, adj, u_A, mask_item, inputs, alias_inputs):
        # batch_size x node x emb  -->  GNN
        h = self.embedding(items)
        h_mask = torch.sign(items)

        mean_item = torch.sum(h, dim=1) / torch.sum(h_mask, dim=-1).unsqueeze(-1)
        mean_item = mean_item.unsqueeze(1)

        h_local, s_node = self.long_short_agg(u_A, adj, h, mean_item, h_mask, self.no_hybrid_gnn)

        h_global_list = []
        pos_neigh_list = []
        for _ in range(self.hop):
            h_global_hidden, neigh_emb = self.global_graph_nn(items, mask_item, inputs)
            h_global_list.append(h_global_hidden)
            pos_neigh_list.append(neigh_emb)
        h_global_hidden = torch.stack(h_global_list, dim=-1)
        h_global = torch.matmul(h_global_hidden, self.u_1).squeeze()
        weight = torch.sigmoid(torch.matmul(h, self.w_3) + torch.matmul(h_global, self.w_4))
        h_global = (1 - weight) * h + weight * h_global

        pos_emb = pos_neigh_list[-1].masked_fill(h_mask.unsqueeze(-1)==0,0.)
        neg_emb = self.embedding(self.neg[items]).squeeze().masked_fill(h_mask.unsqueeze(-1)==0,0.)

        if self.no_global:
            h_local = F.dropout(h_local, self.dropout_local, training=self.training)
            item_output = h_local
        elif self.no_local:
            item_output = h_global
        else:
            h_local = F.normalize(h_local, p=2, dim=-1)
            h_global = F.normalize(h_global, p=2, dim=-1)

            # combine
            h_local = F.dropout(h_local, self.dropout_local, training=self.training)
            h_global = F.dropout(h_global, self.dropout_global, training=self.training)
            h_final = torch.stack((h_local, h_global), dim=2)
            item_output = torch.matmul(h_final.transpose(-1, -2), self.u_2)
            item_output = item_output.squeeze()

        batch_size, seq_len = alias_inputs.shape
        dim = item_output.shape[-1]

        # 扩展 alias_inputs 以用于 gather
        # new shape: (batch_size, seq_len, dim)
        ali_shape = alias_inputs.unsqueeze(-1).expand(batch_size, seq_len, dim)

        # 从 item_output 中高效地收集嵌入
        # gather(input, dim, index)
        seq_items_emb = item_output.gather(1, ali_shape)

        original_item = self.embedding(items.long())
        original_items_emb = original_item.gather(1, ali_shape)
        original_items_emb = original_items_emb * mask_item.float().unsqueeze(-1)
        # hs: 全局节点归一化特征

        combined_att_weight = None
        if self.training:
            original_sess_emb = self.hidden_fusion(original_items_emb, mask_item.float().unsqueeze(-1))
            combined_sess_emb = self.hidden_fusion(seq_items_emb, mask_item.float().unsqueeze(-1))
        else:
            original_sess_emb = self.hidden_fusion(original_items_emb, mask_item.float().unsqueeze(-1))
            combined_sess_emb, combined_att_weight = self.hidden_fusion(seq_items_emb, mask_item.float().unsqueeze(-1), return_attention=True)

        # skip-connection
        if self.no_skip_connection:
            final_sess_emb = combined_sess_emb
        else:
            final_sess_emb = original_sess_emb + combined_sess_emb

        sim_sess_emb = self.find_neighbor(final_sess_emb, k=self.k)

        combined = torch.cat([final_sess_emb, sim_sess_emb], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        final_sess_emb = gate * final_sess_emb + (1.0 - gate) * sim_sess_emb

        if self.training:
            return final_sess_emb, h_global, pos_emb, neg_emb
        else:
            return final_sess_emb, h_global, pos_emb, neg_emb, combined_att_weight

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
