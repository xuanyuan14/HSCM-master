# coding=utf-8
'''
@ref: A Hybrid Framework for Session Context Modeling
@desc: Implementation of each module in HSCM
@appreciation: Some scripts are borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch
'''

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

# 4-D scaled self-attention
class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v, mask=None):

        # b x q_num x len x embed
        # b x q_num x embed x len
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature # batch dot product : b x len x len

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v) # b x q_num x len x len, b x q_num x len x embed = b x q_num x len x embed

        return output, attn

# 4-D multi-head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k)
        self.w_ks = nn.Linear(d_model, d_k)
        self.w_vs = nn.Linear(d_model, d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = self.attention.cuda() if use_cuda else self.attention
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, num_q, len_q, _ = q.size()
        sz_b, num_q, len_k, _ = k.size()
        sz_b, num_q, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, num_q, len_q, n_head, d_k // n_head)
        k = self.w_ks(k).view(sz_b, num_q, len_k, n_head, d_k // n_head)
        v = self.w_vs(v).view(sz_b, num_q, len_v, n_head, d_v // n_head)

        q = q.permute(3, 0, 1, 2, 4).contiguous().view(-1, num_q, len_q, d_k // n_head) # (n*b) x lq x dk
        k = k.permute(3, 0, 1, 2, 4).contiguous().view(-1, num_q, len_k, d_k // n_head) # (n*b) x lk x dk
        v = v.permute(3, 0, 1, 2, 4).contiguous().view(-1, num_q, len_v, d_v // n_head) # (n*b) x lv x dv

        if mask:
            mask = mask.repeat(n_head, 1, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention.forward(q, k, v, mask=mask)   # b x nq x lq x dv
        del q, k, v

        output = output.view(n_head, sz_b, num_q, len_q, d_v // n_head)
        output = output.permute(1, 2, 3, 0, 4).contiguous().view(sz_b, num_q, len_q, -1) # b x nq x lq x dv

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

# position encoding
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):  # pos, index
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


# content encoder, query/document encoder
# for query encoder, encode a sequence (batch) of queries at a time
# for doc encoder, encoder a document at a time, using "topic structure"
class ContentEncoder(nn.Module):
    def __init__(self, args, doc_enc=False):
        super(ContentEncoder, self).__init__()
        self.hidden_size = args.hidden_size  # model hidden size
        self.embed_size = args.embed_size  # embedding size
        self.topic_len = args.topic_len  # 24
        self.n_head = args.head_num
        self.doc_enc = doc_enc

        if self.doc_enc:
            self.max_len = self.topic_len + 1 # default=24
        else:
            self.max_len = args.max_q_len + 1  # max query length

        self.multi_attn = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_size, d_k=self.embed_size, d_v=self.embed_size)
        self.self_attn = ScaledDotProductAttention(temperature=self.hidden_size)
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.max_len, self.embed_size,
                                                                                     padding_idx=0), freeze=True)

        self.w1 = nn.Linear(self.embed_size, self.embed_size)
        self.w2 = nn.Linear(self.embed_size, self.embed_size)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)
        self.tanh = nn.Tanh()
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

        if self.doc_enc:
            self.w_seg = nn.Linear(self.embed_size, self.embed_size)
            self.w_q = nn.Linear(self.embed_size, self.embed_size)
            self.w_out = nn.Linear(self.embed_size, self.embed_size)
            nn.init.xavier_normal_(self.w_seg.weight)
            nn.init.xavier_normal_(self.w_q.weight)
            nn.init.xavier_normal_(self.w_out.weight)

    # query sequence, the last one is the current one, [q_1, q_2, ..., q_L], q_k = [w_1, w_2, ...w_N]
    # one session one batch, query sequence should be padded according to the longest query length

    # query encoder: batch_content: b x q_num x max_len_q x embed_sz (d_v=d_k); batch_pos: b x q_num x max_len_q
    # doc encoder: batch_content: d_num x seg_num x 32 x embed_sz; batch_pos: d_num x seg_num x seg_len; current_q: b x (10 or 60) x embed_sz

    # query once forward one batch, doc once forward 60
    def forward(self, batch_content, batch_pos, current_q=None):

        # local interaction, multi-headed
        sz_b, n_q, len_q, sz_embed = batch_content.size()   # dr: 10 x 20 x 24 x 256
        local_interacted_content, m_attn = self.multi_attn.forward(batch_content, batch_content, batch_content)  # batch : b x nq x lq x embed_sz (d_v=d_k)
        local_content_output = local_interacted_content + self.position_enc(batch_pos)  # add position encoding

        # content encoding
        local_attn = self.softmax1(self.w2(self.tanh(self.w1(local_content_output)))) # b x nq x lq x embed_sz, dim=2
        local_attn_content_output = torch.mul(local_content_output, local_attn).sum(dim=2) # b x nq x embed_sz

        this_q = local_attn_content_output.view(1, sz_b, n_q, sz_embed)
        global_content_output, global_attn = self.self_attn.forward(this_q, this_q, this_q)  #  1 x b x nq x embed_sz
        global_content_output = global_content_output.squeeze()  # b x nq x embed_sz
        del this_q, global_attn

        if self.doc_enc:
            global_content_output = global_content_output  # nd x seg_num x embed_sz
            global_attn = self.softmax2(self.w_out(self.tanh(self.w_seg(global_content_output) + self.w_q(current_q))))  # nd x seg_num x embed_sz
            global_content_output = torch.mul(global_content_output, global_attn)
            global_content_output = global_content_output.sum(dim=1)  # nd x embed_sz
            del global_attn

        return local_attn_content_output, global_content_output


# query candidates encoder
class CandidateEncoder(nn.Module):
    def __init__(self, args):
        super(CandidateEncoder, self).__init__()
        self.hidden_size = args.hidden_size  # model hidden size
        self.embed_size = args.embed_size  # embedding size
        self.topic_len = args.topic_len  # 64

        self.max_len = args.max_q_len + 1  # max query length
        self.n_head = 4

        self.multi_attn = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_size, d_k=self.embed_size, d_v=self.embed_size)
        self.self_attn = ScaledDotProductAttention(temperature=self.hidden_size)
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.max_len, self.embed_size,
                                                                                     padding_idx=0), freeze=True)
        self.w1 = nn.Linear(self.embed_size, self.embed_size)
        self.w2 = nn.Linear(self.embed_size, self.embed_size)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.freq_enc = nn.Embedding(21, 4)
        self.w_out = nn.Linear(self.embed_size + 4, self.hidden_size)

    # query sequence, the last one is the current one, [q_1, q_2, ..., q_L], q_k = [w_1, w_2, ...w_N]
    # one session one batch, query sequence should be padded according to the longest query length

    # query encoder: batch_content: b x nq x lq x embed_sz (d_v=d_k); batch_pos: b x nq x lq
    def forward(self, batch_content, batch_pos, current_q=None):

        # local interaction, multi-headed
        sz_b, nq, lq, sz_embed = batch_content.size()
        local_interacted_content, m_attn = self.multi_attn.forward(batch_content, batch_content, batch_content)  # batch : b x nq x lq x embed_sz (d_v=d_k)
        local_content_output = local_interacted_content + self.position_enc(batch_pos)  # add position encoding
        del local_interacted_content, m_attn
        # content encoding
        local_attn = self.softmax(self.w2(self.tanh(self.w1(local_content_output)))) # b x nq x lq x embed_sz, dim=2
        local_attn_content_output = torch.mul(local_content_output, local_attn).sum(dim=2) # b x nq x embed_sz

        # freq_encoding
        freqs = Variable(torch.LongTensor(np.array([[i + 1 for i in range(nq)] for _ in range(sz_b)]))).view(sz_b, -1, 1)
        freqs = freqs.cuda() if use_cuda else freqs
        freqs_embeddings = self.freq_enc(freqs).view(sz_b, nq, 4)  # b x nq x 4
        del freqs, local_attn, local_content_output
        cands_embeddings = self.tanh(self.w_out(torch.cat((local_attn_content_output, freqs_embeddings), dim=2))) # b x nq x embed_sz

        return cands_embeddings


# query history aggregation
class QueryHistoryAggregator(nn.Module):
    def __init__(self, args):
        super(QueryHistoryAggregator, self).__init__()
        self.hidden_size = args.hidden_size
        self.embed_size = args.embed_size
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(8, self.hidden_size,
                                                                                     padding_idx=0), freeze=True)
        self.w1 = nn.Linear(self.embed_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, query_history):  # query_history: b x lq x hidden_sz, query_pos = b x lq
        sz_b, len_q, embed_sz = query_history.size()
        query_pos = [[pos + 1 for pos in range(len_q)] for _ in range(sz_b)]
        query_pos = Variable(torch.LongTensor(query_pos).view(sz_b, len_q))
        query_pos = query_pos.cuda() if use_cuda else query_pos
        query_pos_history = query_history + self.position_enc(query_pos)
        del query_pos

        history_attn = self.softmax(self.w2(self.tanh(self.w1(query_pos_history))))  # b x len_q x hidden_sz
        query_history_aggregation = torch.mul(history_attn, query_pos_history) # b x len_q x hidden_sz
        query_history_aggregation = query_history_aggregation.sum(dim=1) # b x hidden_sz

        return query_history_aggregation


# interaction aggregator, calculate the interaction in a feedback-style
class InteractionAggregator(nn.Module):
    def __init__(self, args, rank_mode=False):
        super(InteractionAggregator, self).__init__()
        self.hidden_size = args.hidden_size
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(11, self.hidden_size,
                                                                                     padding_idx=0), freeze=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=1)
        self.rank_mode = rank_mode
        self.M = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.M.weight)

        self.w_attn1 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.w_attn1.weight)
        if self.rank_mode:
            self.w_attn2 = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.xavier_normal_(self.w_attn2.weight)

    # qs/docs: b x 60 x hidden_sz, clicks: b x 60 x 1，click value: {1, -1}, current_q: b x hidden_sz
    # q, d, click: interaction = tanh (c * (q'Md)), q' = qT
    def forward(self, qs, docs, clicks, current_q=None):
        # interaction representation
        sz_b, inter_len, _ = qs.size()
        prev_len = inter_len // 10
        inter_pos = np.array([[[i + 1 for i in range(10)] for _ in range(prev_len)] for _ in range(sz_b)])
        inter_pos = Variable(torch.LongTensor(inter_pos))
        clicks = clicks.view(sz_b, -1, 1)
        clicks = (clicks - 0.5) * 2
        inter_pos = inter_pos.cuda() if use_cuda else inter_pos

        # b x 60 x h_sz ==  h_sz x 60 == 60 x h_sz
        # b x 60 x 60 ==> b x 60 x h_sz
        interactions = torch.mul(clicks, self.relu(self.tanh(self.M(qs, docs))))  # b x 60 x hidden_sz
        interactions = interactions + self.position_enc(inter_pos).view(sz_b, inter_len, self.hidden_size)
        del inter_pos, qs, docs, clicks

        # interaction aggregation
        if self.rank_mode: # document ranking aggregation: query-aware
            current_q = current_q.expand(inter_len, sz_b, self.hidden_size).permute(1, 0, 2).contiguous().view(sz_b, -1, self.hidden_size)
            interaction_attn = self.softmax(self.tanh(self.w_attn1(interactions) + self.w_attn2(current_q)))
        else: # query suggestion aggregation: query-free
            interaction_attn = self.softmax(self.tanh(self.w_attn1(interactions)))  # b x inter_len x hidden_sz
        interaction_agg = torch.mul(interaction_attn, interactions).sum(dim=1)  # b x hidden_sz

        return interaction_agg


# output the probability of each candidate queries
class QueryPredictor(nn.Module):
    def __init__(self, args, w_share):
        super(QueryPredictor, self).__init__()
        self.hidden_size = args.hidden_size
        self.w = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.w_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_share = w_share
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_c.weight)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # query_history: b x h_sz; interaction: b x h_sz; candidates:b x nq x h_sz; last_query: b x h_sz
    def forward(self, query_history, interaction_aggregation, sample_interaction_aggregation, candidates, last_query):
        # fuse the contexts
        sz_b, nq, _ = candidates.size()
        context = self.tanh(self.w(torch.cat((query_history, interaction_aggregation, sample_interaction_aggregation, last_query), dim=1)) +
                            self.w_share(torch.cat((query_history, interaction_aggregation, sample_interaction_aggregation, last_query),
                                dim=1)))  # b x h_sz
        context = context.view(sz_b, -1, 1)
        query_probs = self.sigmoid(torch.matmul(candidates, context)).view(sz_b, nq) # b x nq

        return query_probs


class ClickPredictor(nn.Module):
    def __init__(self, args, w_share):
        super(ClickPredictor, self).__init__()
        self.hidden_size = args.hidden_size
        self.w = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.w_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_share = w_share
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_c.weight)
        self.match = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

    # candidates: b x nd x h_sz
    # b x h_sz
    def forward(self, query_history, interaction_aggregation, sample_interaction_aggregation, candidates, current_query):
        # fuse the contexts
        sz_b, nd, _ = candidates.size()
        q_context = self.sigmoid(self.w(torch.cat((query_history, interaction_aggregation, sample_interaction_aggregation, current_query), dim=1)) +
                                 self.w_share(torch.cat((query_history, interaction_aggregation,
                                                   sample_interaction_aggregation, current_query), dim=1)))  # b x h_sz
        q_context = q_context.view(sz_b, -1, 1)
        click_probs = self.sigmoid(torch.matmul(self.match(candidates), q_context)).view(sz_b, nd)  # b x nd x 1

        return click_probs

