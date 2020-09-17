# coding=utf-8
'''
@ref: A Hybrid Framework for Session Context Modeling
@desc: The implementation of HSCM
'''

import torch
import random
import logging
import numpy as np
from torch import nn
from config import PAD_ID
from torch.autograd import Variable
from modules import ContentEncoder, QueryHistoryAggregator, InteractionAggregator, QueryPredictor, ClickPredictor, CandidateEncoder

MINF = 1e-30
use_cuda = torch.cuda.is_available()


# Hybrid Context Session Model
class HCSMN(nn.Module):
    def __init__(self, args):
        super(HCSMN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.topic_len = args.topic_len

        self.logger = logging.getLogger("HCSM")
        self.dropout_rate = args.dropout_rate
        self.softmax = nn.Softmax(dim=0)

        # modules
        self.w_share = nn.Linear(4 * self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.w_share.weight)
        self.query_encoder = ContentEncoder(self.args, doc_enc=False)
        self.doc_encoder = ContentEncoder(self.args, doc_enc=True)
        self.cand_encoder = CandidateEncoder(self.args)
        self.query_history_aggregator = QueryHistoryAggregator(self.args)
        self.interaction_aggregator_qs = InteractionAggregator(self.args, rank_mode=False)
        self.interaction_aggregator_dr = InteractionAggregator(self.args, rank_mode=True)
        self.query_predictor = QueryPredictor(self.args, self.w_share)
        self.click_predictor = ClickPredictor(self.args, self.w_share)


    def sample_from_graph(self, q_nid, node2node, qid2query, qid2freq):
        cross_qs = self.args.cross_qs
        this_q_vec = node2node[q_nid]['vec']
        visted = {}
        cand_nids = []
        pos = 0
        depth = 0
        visted[q_nid] = 1
        nid_depth = [q_nid, 0]
        cand_nids.append(nid_depth)
        while depth <= 3:
            nid = nid_depth[0]
            childs = set(node2node[nid]["1"])  # q or doc
            for child in childs:
                if child not in visted:
                    visted[child] = 1
                    cand_nids.append([child, depth + 1])
            pos += 1
            if pos >= len(cand_nids):
                break
            depth = cand_nids[pos][1]
            nid_depth = cand_nids[pos]

        cand_q_nids = []
        for nid in cand_nids:
            if nid[0].startswith('q') and nid[0] != q_nid:  # avoid repetition
                cand_q_nids.append([nid[0], node2node[nid[0]]['vec']])

        sample_len = len(cand_q_nids)
        sample_p = np.array([torch.cosine_similarity(this_q_vec, cand[1], dim=0).data.numpy() for cand in cand_q_nids])
        sample_p = np.clip(sample_p, a_min=0., a_max=None)

        all_list = []
        for i in range(sample_len):
            all_list.append([cand_q_nids[i][0], sample_p[i]])
        all_len = len(all_list)

        valid_list = sorted(all_list, key=lambda x: x[1], reverse=True)[: min(20, all_len)]
        new_len = len(valid_list)
        valid_index = new_len
        for i in range(new_len - 1, -1, -1):
            if valid_list[i][1] >= 0.8:
                valid_index = i
                break
        valid_list = valid_list[:valid_index]

        valid_q_nids = [e[0] for e in valid_list]
        sample_p = [e[1] for e in valid_list]
        if np.sum(sample_p) != 0:
            sample_p /= np.sum(sample_p)
        nonzero_num = len(np.nonzero(sample_p)[0])
        if nonzero_num != 0:
            sample_q_nids = np.random.choice(valid_q_nids, min(cross_qs, nonzero_num), replace=False,  p=sample_p)
        else:
            sample_q_nids = []

        this_qid = q_nid[1: ]

        tmp_qs = []
        for sample_qid in sample_q_nids:
            tmp_qid = sample_qid[1: ]
            tmp_query = qid2query[tmp_qid]
            tmp_qs.append(tmp_query)

        return sample_q_nids

    # inputs are 2D-lists
    # queries: b x query_num x query_len ==> b x 6 x 24
    # this_q: b x 24
    # documents: b x 10 * query_num x doc_len ==> b x 60 x 480
    # click: b x 10 * (query_num - 1) x 1 ==> b x 60 x 1
    # candidate_queries: b x 20 x query_len ==> b x 20 x 24
    # candidate_docs: b x 10 x doc_len ==> b x 10 x 480
    # forward one batch at a time

    def forward(self, prev_qs, prev_docs, prev_clicks, this_q, this_clicks, this_qid, last_qid, cand_docs, cand_qs, target_cands, dr_modes, qs_modes, data, test_mode=False):
        # Task1: Query Suggestion

        # query encoder: batch_content: b x max_len_q x embed_sz (d_v=d_k); batch_pos: b x max_len_q
        # doc encoder: batch_content: seg_num x 32 x embed_sz; batch_pos: seg_num x seg_len; current_q: embed_sz

        # initialization
        word_embed = data.word_embed
        qid2query = data.qid2query
        qid2freq = data.qid2freq
        topic_num = self.args.topic_num

        # query history
        prev_qs = np.array(prev_qs)
        query_variable = word_embed(Variable(torch.from_numpy(prev_qs)))  # b x query_num x max_query_len x embed_sz
        b_sz, q_num, _, _ = query_variable.size()
        batch_pos = np.array([[[k + 1 if prev_qs[i][j][k] != PAD_ID else 0
                                for k in range(self.topic_len)] for j in range(q_num)] for i in range(b_sz)])
        batch_pos = Variable(torch.LongTensor(np.array(batch_pos)))
        if use_cuda:
            query_variable, batch_pos = query_variable.cuda(), batch_pos.cuda()
        _, context_query_vectors = self.query_encoder.forward(query_variable, batch_pos)  # ==> b x nq x hidden_sz

        # print(context_query_vectors)
        context_query_vectors = context_query_vectors.view(b_sz, q_num, self.embed_size)
        query_history = self.query_history_aggregator.forward(context_query_vectors).view(b_sz, self.hidden_size)  # b x hidden_sz

        # consider current query for document ranking
        this_q = np.array(this_q).reshape((b_sz, 1, self.topic_len))
        all_qs = np.concatenate((prev_qs, this_q), axis=1)
        dr_batch_pos = Variable(torch.LongTensor(np.array([[[k + 1 if all_qs[i][j][k] != PAD_ID else 0 for k in
                                                             range(self.topic_len)] for j in range(q_num + 1)] for i in
                                                           range(b_sz)])))
        query_variable = word_embed(Variable(torch.from_numpy(all_qs)))  # b x (nq+1) x max_query_len x embed_sz
        if use_cuda:
            query_variable, dr_batch_pos = query_variable.cuda(), dr_batch_pos.cuda()
        _, dr_context_query_vectors = self.query_encoder.forward(query_variable, dr_batch_pos)
        dr_context_query_vectors = dr_context_query_vectors.view(b_sz, (q_num + 1), self.embed_size)  # b x (nq+1) x embed_sz
        del query_variable, batch_pos, dr_batch_pos

        # doc encoding
        # b x 60 x topic_num * topic_len
        this_prev_docs = np.array(prev_docs).reshape((b_sz * 60, topic_num, self.topic_len))  # 60b
        tmp_doc_variable = word_embed(
            Variable(torch.from_numpy(this_prev_docs)))  # 60b x topic_num x seg_len x embed_sz
        tmp_doc_pos = Variable(torch.LongTensor(np.array([[[k + 1 if this_prev_docs[inter][j][k] != PAD_ID else 0
                                                            for k in range(self.topic_len)] for j in range(topic_num)]
                                                            for inter in range(60 * b_sz)])))
        # get current query representation
        # b x 6 x embed_sz
        cur_query = context_query_vectors.expand(10, b_sz, 6, self.embed_size).permute(1, 2, 0, 3).contiguous().view(
            b_sz * 6 * 10, self.embed_size)  # 60b x embed_sz
        cur_query = cur_query.expand(topic_num, 60 * b_sz, self.embed_size).permute(1, 0, 2)  # 60b x topic_num x embed_sz
        if use_cuda:
            tmp_doc_variable, tmp_doc_pos = tmp_doc_variable.cuda(), tmp_doc_pos.cuda()
        _, this_doc_embed = self.doc_encoder.forward(tmp_doc_variable, tmp_doc_pos, cur_query)  # 60b x embed_sz
        batch_prev_docs_variable = this_doc_embed.view(b_sz, 60, self.embed_size)

        # candidate doc encoding
        # b x 10 x 480
        # batch_cand_docs = []
        this_cand_docs = np.array(cand_docs).reshape((b_sz * 10, topic_num, self.topic_len))
        tmp_doc_variable = word_embed(
            Variable(torch.from_numpy(this_cand_docs)))  # 10b x topic_num x seg_len x embed_sz
        tmp_doc_pos = Variable(torch.LongTensor(np.array([[[k + 1 if this_cand_docs[inter][j][k] != PAD_ID else 0
                                                            for k in range(self.topic_len)] for j in range(topic_num)]
                                                            for inter in range(b_sz * 10)])))
        # get current query representation
        cur_query = dr_context_query_vectors[:, -1]
        # print(cur_query.size())
        cur_query = cur_query.expand(10, b_sz, self.embed_size).permute(1, 0, 2).contiguous().view(b_sz * 10, self.embed_size)  # 10b x embed_sz
        cur_query = cur_query.expand(topic_num, 10 * b_sz, self.embed_size).permute(1, 0, 2)  # 10b x topic_num x embed_sz
        if use_cuda:
            tmp_doc_variable, tmp_doc_pos = tmp_doc_variable.cuda(), tmp_doc_pos.cuda()
        _, this_doc_embed = self.doc_encoder.forward(tmp_doc_variable, tmp_doc_pos, cur_query)  # 10b x embed_sz
        batch_cand_docs = this_doc_embed.view(b_sz, 10, self.embed_size)

        # interaction aggregation
        # b x nq x embed_sz
        batch_prev_query_variable = context_query_vectors.expand(10, b_sz, q_num, self.embed_size).permute(1, 2, 0, 3).reshape(b_sz, -1, self.embed_size)  # b x 60 x embed_sz
        prev_clicks = Variable(torch.FloatTensor(prev_clicks))  # b x 60 x 1
        if use_cuda:
            prev_clicks = prev_clicks.cuda()
        interaction_qs = self.interaction_aggregator_qs.forward(batch_prev_query_variable, batch_prev_docs_variable,
                                                                prev_clicks)  # b x embed_sz
        del batch_prev_query_variable

        # sample from cross-session interactions
        node2node = data.graph.node2node
        qid2wids = data.qid2wids
        did2wids = data.did2wids
        cross_qs = self.args.cross_qs
        sample_q_qs, sample_q_dr = [], []  # b x 3 x 24
        sample_doc_qs, sample_doc_dr = [], []  # b x 30 x 1
        sample_click_qs, sample_click_dr = [], []  # b x 30 x 1
        for i in range(b_sz):
            this_qid_node = this_qid[i]
            last_qid_node = last_qid[i]
            tmp_sample_q_qs, tmp_sample_q_dr = [], []  # 3 x 24
            tmp_sample_doc_qs, tmp_sample_doc_dr = [], []  # 30 x 1
            tmp_sample_click_qs, tmp_sample_click_dr = [], []  # 30 x 1

            # doc rank
            sample_q_nids = self.sample_from_graph('q' + str(this_qid_node), node2node, qid2query, qid2freq)

            for sample_q_nid in sample_q_nids:
                tmp_sample_q_dr.append(sample_q_nid[1:])

                # 每个query抽样10个interaction
                interactions = []
                for nid in node2node[sample_q_nid]["1"]:
                    if nid.startswith('d'):
                        interactions.append([nid, 1])
                for nid in node2node[sample_q_nid]["0"]:
                    if nid.startswith('d'):
                        interactions.append([nid, 0])

                inter_num = len(interactions)
                sample_interactions = random.sample(interactions, min(10, inter_num))
                tmp_sample_doc_dr.extend([interaction[0][1:] for interaction in sample_interactions])
                tmp_sample_click_dr.extend([interaction[1] for interaction in sample_interactions])

            # query suggestion
            if not qs_modes[i] or last_qid_node == PAD_ID:
                tmp_sample_q_qs = [PAD_ID] * cross_qs
                tmp_sample_doc_qs = [PAD_ID] * (cross_qs * 10)
                tmp_sample_click_qs = [PAD_ID] * (cross_qs * 10)
            else:
                sample_q_nids = self.sample_from_graph('q' + str(last_qid_node), node2node, qid2query, qid2freq)

                for sample_q_nid in sample_q_nids:
                    tmp_sample_q_qs.append(sample_q_nid[1:])

                    # 每个query抽样10个interaction
                    interactions = []
                    for nid in node2node[sample_q_nid]["1"]:
                        if nid.startswith('d'):
                            interactions.append([nid, 1])
                    for nid in node2node[sample_q_nid]["0"]:
                        if nid.startswith('d'):
                            interactions.append([nid, 0])

                    inter_num = len(interactions)
                    sample_interactions = random.sample(interactions, min(10, inter_num))
                    tmp_sample_doc_qs.extend([interaction[0][1:] for interaction in sample_interactions])
                    tmp_sample_click_qs.extend([interaction[1] for interaction in sample_interactions])

            # padding
            if len(tmp_sample_q_dr) < cross_qs:
                this_len = len(tmp_sample_q_dr)
                for _ in range(cross_qs - this_len):
                    tmp_sample_q_dr.append(PAD_ID)

            if len(tmp_sample_q_qs) < cross_qs:
                this_len = len(tmp_sample_q_qs)
                for _ in range(cross_qs - this_len):
                    tmp_sample_q_qs.append(PAD_ID)

            if len(tmp_sample_doc_dr) < cross_qs * 10:
                this_len = len(tmp_sample_doc_dr)
                for _ in range(cross_qs * 10 - this_len):
                    tmp_sample_doc_dr.append(PAD_ID)

            if len(tmp_sample_doc_qs) < cross_qs * 10:
                this_len = len(tmp_sample_doc_qs)
                for _ in range(cross_qs * 10 - this_len):
                    tmp_sample_doc_qs.append(PAD_ID)

            if len(tmp_sample_click_dr) < cross_qs * 10:
                this_len = len(tmp_sample_click_dr)
                for _ in range(cross_qs * 10 - this_len):
                    tmp_sample_click_dr.append(PAD_ID)

            if len(tmp_sample_click_qs) < cross_qs * 10:
                this_len = len(tmp_sample_click_qs)
                for _ in range(cross_qs * 10 - this_len):
                    tmp_sample_click_qs.append(PAD_ID)

            tmp_sample_q_dr_ws = [qid2wids[str(q)] for q in tmp_sample_q_dr]
            tmp_sample_q_qs_ws = [qid2wids[str(q)] for q in tmp_sample_q_qs]
            tmp_sample_doc_dr_ws = [did2wids[str(d)] for d in tmp_sample_doc_dr]
            tmp_sample_doc_qs_ws = [did2wids[str(d)] for d in tmp_sample_doc_qs]

            sample_q_dr.append(tmp_sample_q_dr_ws)
            sample_click_dr.append(tmp_sample_click_dr)
            sample_doc_dr.append(tmp_sample_doc_dr_ws)
            sample_q_qs.append(tmp_sample_q_qs_ws)
            sample_click_qs.append(tmp_sample_click_qs)
            sample_doc_qs.append(tmp_sample_doc_qs_ws)

        # represent cross-session interaction
        # print(sample_q_dr, sample_q_qs)
        sample_q_dr = np.array(sample_q_dr).reshape((b_sz, cross_qs, self.topic_len))
        sample_q_qs = np.array(sample_q_qs).reshape((b_sz, cross_qs, self.topic_len))
        sample_batch_pos = Variable(torch.LongTensor(np.array([[[k + 1 if sample_q_dr[i][j][k] != PAD_ID else 0 for k in
                                                                 range(self.topic_len)] for j in range(cross_qs)] for i
                                                                in range(b_sz)])))
        sample_q_dr_variable = word_embed(Variable(torch.from_numpy(sample_q_dr)))  # b x 3 x max_query_len x embed_sz
        sample_q_qs_variable = word_embed(Variable(torch.from_numpy(sample_q_qs)))  # b x 3 x max_query_len x embed_sz
        if use_cuda:
            sample_q_dr_variable, sample_q_qs_variable, sample_batch_pos = sample_q_dr_variable.cuda(), sample_q_qs_variable.cuda(), sample_batch_pos.cuda()
        sample_query_vectors_dr, _ = self.query_encoder.forward(sample_q_dr_variable, sample_batch_pos)
        sample_query_vectors_qs, _ = self.query_encoder.forward(sample_q_qs_variable, sample_batch_pos)
        sample_query_vectors_dr = sample_query_vectors_dr.view(b_sz, cross_qs, self.embed_size)  # b x 3 x embed_sz
        sample_query_vectors_qs = sample_query_vectors_qs.view(b_sz, cross_qs, self.embed_size)  # b x 3 x embed_sz

        # b x 30 x (topic_num * topic_len)
        sample_doc_dr = np.array(sample_doc_dr).reshape((b_sz * 10 * cross_qs, topic_num, self.topic_len))  # 30b
        sample_doc_qs = np.array(sample_doc_qs).reshape((b_sz * 10 * cross_qs, topic_num, self.topic_len))  # 30b
        sample_doc_dr_variable = word_embed(
            Variable(torch.from_numpy(sample_doc_dr)))  # 30b x topic_num x seg_len x embed_sz
        sample_doc_qs_variable = word_embed(Variable(torch.from_numpy(sample_doc_qs)))
        tmp_doc_pos = Variable(torch.LongTensor(np.array([[[k + 1 if this_prev_docs[inter][j][k] != PAD_ID else 0
                                                            for k in range(self.topic_len)] for j in range(topic_num)]
                                                          for inter in range(10 * cross_qs * b_sz)])))
        # get current query representation
        # b x 6 x embed_sz
        cur_query_dr = sample_query_vectors_dr.expand(10, b_sz, cross_qs, self.embed_size).permute(1, 2, 0,3).contiguous().view(b_sz * cross_qs * 10, self.embed_size)  # 30b x embed_sz
        cur_query_dr_seg = cur_query_dr.expand(topic_num, 10 * cross_qs * b_sz, self.embed_size).permute(1, 0, 2)  # 30b x topic_num x embed_sz
        cur_query_qs = sample_query_vectors_qs.expand(10, b_sz, cross_qs, self.embed_size).permute(1, 2, 0, 3).contiguous().view(b_sz * cross_qs * 10, self.embed_size)  # 30b x embed_sz
        cur_query_qs_seg = cur_query_qs.expand(topic_num, 10 * cross_qs * b_sz, self.embed_size).permute(1, 0, 2)  # 30b x topic_num x embed_sz
        if use_cuda:
            sample_doc_dr_variable, sample_doc_qs_variable, tmp_doc_pos = sample_doc_dr_variable.cuda(), sample_doc_qs_variable.cuda(), tmp_doc_pos.cuda()
        _, sample_doc_dr_embed = self.doc_encoder.forward(sample_doc_dr_variable, tmp_doc_pos,
                                                          cur_query_dr_seg)  # 30b x embed_sz
        _, sample_doc_qs_embed = self.doc_encoder.forward(sample_doc_qs_variable, tmp_doc_pos,
                                                          cur_query_qs_seg)  # 30b x embed_sz

        sample_doc_dr_embed = sample_doc_dr_embed.view(b_sz, 10 * cross_qs, self.embed_size)
        sample_doc_qs_embed = sample_doc_qs_embed.view(b_sz, 10 * cross_qs, self.embed_size)

        sample_click_dr = Variable(torch.FloatTensor(sample_click_dr))  # b x 30 x 1
        sample_click_qs = Variable(torch.FloatTensor(sample_click_qs))  # b x 30 x 1
        cur_query_qs = cur_query_qs.view(b_sz, -1, self.embed_size)
        cur_query_dr = cur_query_dr.view(b_sz, -1, self.embed_size)
        if use_cuda:
            sample_click_dr, sample_click_qs = sample_click_dr.cuda(), sample_click_qs.cuda()
        sample_interaction_qs = self.interaction_aggregator_qs.forward(cur_query_qs, sample_doc_qs_embed,
                                                                       sample_click_qs)  # b x embed_sz
        sample_interaction_dr = self.interaction_aggregator_dr.forward(cur_query_dr, sample_doc_dr_embed,
                                                                       sample_click_dr,
                                                                       current_q=sample_query_vectors_dr[:,
                                                                                 -1])  # b x embed_sz

        # candidate query encoding
        # b x nq x lq
        # print(np.array(cand_qs).shape)
        candidate_query_variable = word_embed(Variable(torch.from_numpy(np.array(cand_qs))))  # b x nq x lq x embed_sz
        cand_batch_pos = Variable(
            torch.LongTensor(np.array([[[j + 1 if cand_qs[b][i][j] != PAD_ID else 0 for j in range(self.topic_len)]
                                        for i in range(20)] for b in range(b_sz)])))
        if use_cuda:
            cand_batch_pos, candidate_query_variable = cand_batch_pos.cuda(), candidate_query_variable.cuda()
        local_candidates = self.cand_encoder.forward(candidate_query_variable, cand_batch_pos)  # b x nq x embed_sz
        last_query = context_query_vectors[:, -1::].view(b_sz, -1)  # b x h_sz
        # print(query_history.size(), last_query.size())
        candidate_probs = self.query_predictor.forward(query_history, interaction_qs, sample_interaction_qs,
                                                       local_candidates, last_query)  # b x embed_sz, b x nq x embed_sz
        del cand_batch_pos, candidate_query_variable, local_candidates, interaction_qs, query_history

        # Task 2: document ranking
        current_query = dr_context_query_vectors[:, -1]  # b x embed_sz
        previous_query = dr_context_query_vectors[:, :-1]  # b x 6 x embed_sz
        batch_query_variable = previous_query.expand(10, b_sz, q_num, self.embed_size).permute(1, 2, 0, 3).reshape(b_sz,
                                                                                                                   -1,
                                                                                                                   self.embed_size)  # b x 70 x embed_sz
        dr_query_history = self.query_history_aggregator.forward(dr_context_query_vectors).view(b_sz,
                                                                                                self.hidden_size)  # b x hidden_sz
        interaction_dr = self.interaction_aggregator_dr.forward(batch_query_variable, batch_prev_docs_variable,
                                                                prev_clicks, current_q=current_query)
        click_probs = self.click_predictor.forward(dr_query_history, interaction_dr, sample_interaction_dr,
                                                   candidates=batch_cand_docs, current_query=current_query)
        del dr_query_history, interaction_dr, cand_docs, current_query, dr_context_query_vectors, \
            prev_qs, prev_docs, previous_query, prev_clicks

        dr_loss, qs_loss = 0., 0.
        if not test_mode:
            # query suggestion
            for b_idx, batch in enumerate(target_cands):
                pos_list = []
                neg_list = []
                if not qs_modes[b_idx]:
                    continue

                # cross-entropy point-wise
                for position_idx, cand in enumerate(batch):
                    if cand == 0:
                        neg_list.append(candidate_probs[b_idx][position_idx].view(1))
                        qs_loss -= torch.log(1. - candidate_probs[b_idx][position_idx].view(1) + 1e-30)
                    else:
                        pos_list.append(candidate_probs[b_idx][position_idx].view(1))
                        qs_loss -= torch.log(candidate_probs[b_idx][position_idx].view(1) + 1e-30)

            # document ranking
            target_clicks = Variable(torch.FloatTensor(this_clicks))
            target_clicks = target_clicks.cuda() if use_cuda else target_clicks
            for b_idx, batch in enumerate(target_clicks):
                if not dr_modes[b_idx]:
                    continue
                for position_idx, score in enumerate(batch):
                    if score == 0:
                        dr_loss -= torch.log(1. - click_probs[b_idx][position_idx].view(1) + 1e-30)
                    else:
                        dr_loss -= torch.log(click_probs[b_idx][position_idx].view(1) + 1e-30)

            loss = 0.9 * dr_loss + 0.1 * qs_loss
            if loss != 0.:
                loss.backward()

        return candidate_probs, click_probs