# coding=utf-8
'''
@ref: A Hybrid Framework for Session Context Modeling
@desc: Loading dataset, build cross-session graph
'''
import json
import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
from mip_tree import MIPTree
from torch.autograd import Variable
from config import PAD_ID

use_cuda = torch.cuda.is_available()


class Graph(object):
    '''
    This module implements the APIs for constructing click-semantic graph on dataset
    '''
    def __init__(self, args, word_embed, qid2wids, wid2qid):
        self.args = args
        self.word_embed = word_embed
        self.qid2wids = qid2wids
        self.wid2qid = wid2qid
        self.q_nids = []

        # qid2did, did2qid, qid2qid -- click bipartite graph
        # directed graph q->d d->q q->q
        self.node2node = {}

    # build query-document click bipartite graph
    def build(self, this_q, this_docs=None, this_clicks=None):

        q_nid = 'q' + str(this_q)
        if q_nid not in self.node2node:
            self.node2node[q_nid] = {"0": [], "1": []}
            self.q_nids.append(q_nid)
        if this_docs and this_clicks:
            for i in range(10):
                d_nid = 'd' + str(this_docs[i])
                if d_nid not in self.node2node:
                    self.node2node[d_nid] = {"1": []}
                this_click = this_clicks[i]
                if this_click == 1 and d_nid not in self.node2node[q_nid]["1"]:
                    self.node2node[q_nid]["1"].append(d_nid)
                    if q_nid not in self.node2node[d_nid]["1"]:
                        self.node2node[d_nid]["1"].append(q_nid)
                elif this_click == 0 and d_nid not in self.node2node[q_nid]["0"]:
                    self.node2node[q_nid]["0"].append(d_nid)

    # add semantic dependencies
    def add_semantic(self):
        qcnt = len(self.q_nids)
        print('Query nums:%s' % qcnt)
        print('Building the graph with semantic similarity...')
        q_vector_list = []

        # build query graph with its vector
        for q_nid in self.q_nids:
            qid = q_nid[1:]
            wids = self.qid2wids[qid]
            wids_variable = Variable(torch.LongTensor(np.array(wids).reshape(1, self.args.topic_len)))
            q_vector = np.sum(self.word_embed(wids_variable).view(self.args.topic_len, self.args.embed_size).numpy(),
                              axis=0) / self.args.topic_len  # 1 x 24 x 256
            q_vector = q_vector / np.linalg.norm(q_vector)
            if q_nid in self.node2node:
                self.node2node[q_nid]["vec"] = q_vector
            else:
                self.node2node[q_nid] = {"0": [], "1": [], "vec": q_vector}
            q_vector_list.append(q_vector.tolist())
        print('Complete build query with its vector')
        vec2q_nid = {}
        for k in self.node2node:
            if 'vec' in self.node2node[k]:
                vec = json.dumps(self.node2node[k]['vec'].tolist())
                if vec not in vec2q_nid:
                    vec2q_nid[vec] = k
        q_vectors = np.array(q_vector_list)

        N0 = 20
        t = MIPTree(q_vectors, N0=N0, reorder=True)
        print('Build MIP tree done!')

        # find the top3 similar queries in cos_sim
        cnt = 0
        for q_nid in self.q_nids:
            cnt += 1
            if cnt % 500 == 0:
                print(cnt)
            if 'semantic' in self.node2node[q_nid]:
                continue
            vector = self.node2node[q_nid]['vec']
            cand_vectors, _ = t.match(vector, 4)

            for cand_vector in cand_vectors:
                cand_q_nid = vec2q_nid[json.dumps(q_vector_list[cand_vector[0]])]
                if cand_q_nid != q_nid:
                    self.node2node[q_nid]["1"].append(cand_q_nid)
            if 'semantic' not in self.node2node[q_nid]:
                self.node2node[q_nid]['semantic'] = True


class Dataset(object):
    """
    This module implements the APIs for loading dataset
    """
    def __init__(self, args, train_dirs=[], dev_dirs=[], test_dirs=[], test_mode=False):
        self.logger = logging.getLogger("HCSM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.args = args
        self.topic_len = args.topic_len
        self.num_train_files = args.num_train_files
        self.num_dev_files = args.num_dev_files
        self.num_test_files = args.num_test_files
        self.embed_size = args.embed_size

        self.word_emd_list = []
        # set padding embeddings
        self.word_emd_list.append([0.0] * self.embed_size)

        with open('../data/vectors.txt', 'r') as f_embed:
            for line in f_embed:
                data = line.strip().split()
                assert (len(data) == args.embed_size + 1)
                self.word_emd_list.append([float(x) for x in data[1:]])
        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.word_emd_list), freeze=True)

        # load dicts
        with open('../data/candidate_qids.json') as f1:
            self.candidates = json.load(f1)

        with open('../data/qid2query.json') as f2:
            self.qid2query = json.load(f2)

        self.query2qid = {}
        for key in self.qid2query.keys():
            query = self.qid2query[key]
            self.query2qid[query] = key

        # unify the q/doc len, q_len = topic_len, doc_len = topic_num * topic_len
        with open('../data/qid2wids.json') as f3:
            self.qid2wids = json.load(f3)
        for qid in self.qid2wids.keys():
            this_len = len(self.qid2wids[qid])
            if this_len > self.topic_len:
                self.qid2wids[qid] = self.qid2wids[qid][: self.topic_len]
            else:
                for _ in range(self.topic_len - this_len):
                    self.qid2wids[qid].append(PAD_ID)
        self.qid2wids["0"] = [0] * self.topic_len

        # wid2qid -- reverse index
        self.wid2qid = {}
        for qid in self.qid2wids.keys():
            wids = self.qid2wids[qid]
            for wid in wids:
                if wid == PAD_ID:
                    break
                if wid not in self.wid2qid:
                    self.wid2qid[wid] = []
                self.wid2qid[wid].append(qid)

        if os.path.exists(self.graph_dir):
            with open(self.graph_dir, 'rb') as fr:
                self.graph = pickle.load(fr)
        else:
            self.graph = Graph(args=self.args, word_embed=self.word_embed, qid2wids=self.qid2wids, wid2qid=self.wid2qid)

        with open('../data/did2wids.json') as f4:
            self.did2wids = json.load(f4)
        doc_len = self.topic_num * self.topic_len
        for did in self.did2wids.keys():
            this_len = len(self.did2wids[did])
            if this_len > doc_len:
                self.did2wids[did] = self.did2wids[did][: doc_len]
            else:
                for _ in range(doc_len - this_len):
                    self.did2wids[did].append(PAD_ID)
        self.did2wids["0"] = [0] * doc_len

        print('load dicts completed!')

        self.train_set, self.dev_set, self.test_set = [], [], []
        self.qid2freq = {}

        if train_dirs:
            for train_dir in train_dirs:
                self.train_set += self.load_dataset(train_dir, num=self.num_train_files, mode='train')
            self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
        if dev_dirs:
            for dev_dir in dev_dirs:
                self.dev_set += self.load_dataset(dev_dir, num=self.num_dev_files, mode='dev')
            self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))
        if test_dirs:
            for test_dir in test_dirs:
                self.test_set += self.load_dataset(test_dir, num=self.num_test_files, mode='test')
            self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))

        if args.train:
            self.graph.add_semantic()
            with open(self.graph_dir, 'wb') as fw:  # train时存储
                pickle.dump(self.graph, fw)


    def load_dataset(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set, files = [], [data_path]
        if num > 0:
            files = files[0: num]

        sess_id = 1
        for dir in files:
            fn = open(dir, 'r')
            sess = fn.read().strip().split('\n\n')
            for s in sess:
                qs, docs, clicks = [], [], []
                lines = s.strip().split('\n')
                for line in lines:
                    this_line = json.loads(line.strip())
                    this_q, this_docs, this_clicks = this_line[0], this_line[1], this_line[2]
                    if this_q not in self.qid2freq:
                        self.qid2freq[this_q] = 0
                    self.qid2freq[this_q] += 1

                    if mode == 'train':
                        self.graph.build(this_q, this_docs, this_clicks)
                    else:
                        self.graph.build(this_q)

                    qs.append(this_q)
                    docs.extend(this_docs)
                    clicks.extend(this_clicks)

                    # prev_qs: 6, prev_docs: 60, prev_clicks: 60, cur_q: 1, cand_docs: 10, cand_qs: 20
                    prev_qs = qs[:-1]
                    prev_docs = docs[:-10]
                    prev_clicks = clicks[:-10]

                    this_qid = this_q
                    try:
                        last_q = qs[-2]
                        last_qid = qs[-2]
                        cand_qs = self.candidates[str(last_q)]
                    except:
                        cand_qs = [0] * 20
                        last_qid = PAD_ID

                    # padding
                    if len(prev_qs) > 6:
                        prev_qs = prev_qs[-6: ]
                    else:
                        for _ in range(6 - len(prev_qs)):
                            prev_qs.insert(0, PAD_ID)

                    if len(prev_docs) > 60:
                        prev_docs = prev_docs[-60: ]
                    else:
                        for _ in range(60 - len(prev_docs)):
                            prev_docs.insert(0, PAD_ID)

                    if len(prev_clicks) > 60:
                        prev_clicks = prev_clicks[-60: ]
                    else:
                        for _ in range(60 - len(prev_clicks)):
                            prev_clicks.insert(0, PAD_ID)

                    if len(cand_qs) < 20:
                        for _ in range(20 - len(cand_qs)):
                            cand_qs.append(PAD_ID)

                    # replace qs and docs with wids
                    prev_qs = [self.qid2wids[str(q)] for q in prev_qs]
                    prev_docs = [self.did2wids[str(d)] for d in prev_docs]
                    this_q = self.qid2wids[str(this_q)]
                    cand_docs = [self.did2wids[str(d)] for d in this_docs]
                    cand_qs = [self.qid2wids[str(cand)] for cand in cand_qs]

                    data_set.append({'prev_qs': prev_qs,
                                     'prev_docs': prev_docs,
                                     'prev_clicks': prev_clicks,
                                     'this_q': this_q,
                                     'this_clicks': this_clicks,
                                     'this_qid': this_qid,
                                     'last_qid': last_qid,
                                     'cand_docs': cand_docs,
                                     'cand_qs': cand_qs,
                                     'sess_id': sess_id})
                sess_id += 1

        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'prev_qs': [],
                      'prev_docs': [],
                      'prev_clicks': [],
                      'this_q': [],
                      'this_clicks': [],
                      'this_qid': [],
                      'last_qid': [],
                      'cand_docs': [],
                      'cand_qs': [],}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['prev_qs'].append(sample['prev_qs'])
            batch_data['prev_docs'].append(sample['prev_docs'])
            batch_data['prev_clicks'].append(sample['prev_clicks'])
            batch_data['this_q'].append(sample['this_q'])
            batch_data['this_qid'].append(sample['this_qid'])
            batch_data['last_qid'].append(sample['last_qid'])
            batch_data['this_clicks'].append(sample['this_clicks'])
            batch_data['cand_docs'].append(sample['cand_docs'])
            batch_data['cand_qs'].append(sample['cand_qs'])
        return batch_data


    def convert_to_ids(self, text, vocab):
        """
        Convert the question and paragraph in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        return vocab.convert_to_ids(text, 'vocab')

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()

        indices += indices[:(self.gpu_num - data_size % self.gpu_num)%self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)