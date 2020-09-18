# coding=utf-8
'''
@ref: A Hybrid Framework for Session Context Modeling
@desc: Modeling training, testing, saving and loading
'''
import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import nn
from HSCMN import HSCMN

use_cuda = torch.cuda.is_available()
MINF = 1e-30


class Model(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, args):
        self.args = args

        # logging
        self.logger = logging.getLogger("HSCM")

        # basic config
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.max_d_num = args.max_d_num
        self.topic_len = args.topic_len
        if args.train:
            self.writer = SummaryWriter(self.args.summary_dir)

        self.model = HSCMN(self.args)

        if args.data_parallel:
            self.model = nn.DataParallel(self.model)

        if use_cuda:
            self.model = self.model.cuda()
        self.optimizer = self.create_train_op()
        self.criterion = nn.MSELoss()

    def create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'rprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, data, max_metric_value, metric_save, patience, step_pbar):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        evaluate = True
        exit_tag = False
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.model_dir, self.args.algo

        # loss = 0.
        self.model.train()
        self.optimizer.zero_grad()

        for bitx, batch in enumerate(train_batches):
            prev_qs = batch['prev_qs']  # 6
            prev_docs = batch['prev_docs']  # 60
            prev_clicks = batch['prev_clicks']  # 60
            this_q = batch['this_q']  # 1
            this_clicks = batch['this_clicks']  # 10
            this_qid = batch['this_qid']
            last_qid = batch['last_qid']
            cand_docs = batch['cand_docs']  # 10
            cand_qs = batch['cand_qs']  # 20

            batch_n = len(prev_qs)
            qs_modes = [False] * batch_n
            dr_modes = [True] * batch_n
            cand_probs = np.zeros((batch_n, 20))
            q_ranks = [-1] * batch_n

            for i in range(batch_n):

                # decide dr mode
                if len(set(this_clicks[i])) == 1 and this_clicks[i][-1] == 0:
                    dr_modes[i] = False

                # decide qs mode
                cand_num = len(cand_qs[i])
                pad_cnt = 0
                for j in range(cand_num):
                    if len(set(cand_qs[i][j])) == 1 and cand_qs[i][j][0] == 0:
                        pad_cnt += 1
                if pad_cnt < 19:
                    for j in range(cand_num):
                        if cand_qs[i][j] == this_q[i]:
                            qs_modes[i] = True
                            q_ranks[i] = j
                            cand_probs[i][q_ranks[i]] = 1.0
                            break

            _, _ = self.model(prev_qs, prev_docs, prev_clicks, this_q, this_clicks, this_qid, last_qid, cand_docs, cand_qs, cand_probs, dr_modes, qs_modes, data)

            if (bitx + 1) % 32 == 0:
                self.global_step += 1
                step_pbar.update(1)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                if evaluate and self.global_step % self.eval_freq == 0:
                    if data.dev_set is not None:
                        test_batchs = data.gen_mini_batches('dev', self.args.batch_size, shuffle=True)
                        eval_loss = self.evaluate(test_batchs, data, result_dir=self.args.result_dir, t=-1,
                                                  result_prefix='train_dev.predicted.{}.{}'.format(self.args.algo,
                                                                                                   self.global_step))
                        self.writer.add_scalar("dev/loss", eval_loss, self.global_step)
                        if eval_loss < metric_save:
                            metric_save = eval_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience >= self.patience:
                            self.adjust_learning_rate(self.args.lr_decay)
                            self.learning_rate *= self.args.lr_decay
                            self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                            metric_save = eval_loss
                            patience = 0
                            self.patience += 1
                    else:
                        self.logger.warning('No dev set is loaded for evaluation in the dataset!')
                if check_point > 0 and self.global_step % check_point == 0:
                    self.save_model(save_dir, save_prefix)
                if self.global_step >= num_steps:
                    exit_tag = True

        return max_metric_value, exit_tag, metric_save, patience


    def train(self, data):
        max_metric_value, epoch, patience, metric_save = 0., 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
        self.global_step += 1
        while not exit_tag:
            print('**************************Epoch %s*************************' % epoch)
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            max_metric_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data, max_metric_value, metric_save,
                                                                                  patience, step_pbar)

    def evaluate(self, test_batchs, data, result_dir=None, result_prefix=None, t=-1):
        eval_ouput = []
        total_loss, total_num = 0., 0

        total_num = 0
        with torch.no_grad():
            for b_itx, batch in enumerate(test_batchs):
                total_num += 1
                if b_itx == t:
                    break
                if b_itx % 100 == 0: # 2000
                    self.logger.info('Evaluation step {}.'.format(b_itx))

                self.model.eval()
                prev_qs = batch['prev_qs']  # 6
                prev_docs = batch['prev_docs']  # 60
                prev_clicks = batch['prev_clicks']  # 60
                this_q = batch['this_q']  # 1
                this_clicks = batch['this_clicks']  # 10
                this_qid = batch['this_qid']
                last_qid = batch['last_qid']
                cand_docs = batch['cand_docs']  # 10
                cand_qs = batch['cand_qs']  # 20


                batch_n = len(prev_qs)
                qs_modes = [False] * batch_n
                dr_modes = [True] * batch_n
                cand_probs = np.zeros((batch_n, 20))
                q_ranks = [-1] * batch_n
                for i in range(batch_n):
                    # decide dr mode
                    if len(set(this_clicks[i])) == 1 and this_clicks[i][-1] == 0:
                        dr_modes[i] = False

                    # decide qs mode
                    cand_num = len(cand_qs[i])
                    pad_cnt = 0
                    for j in range(cand_num):
                        if len(set(cand_qs[i][j])) == 1 and cand_qs[i][j][0] == 0:
                            pad_cnt += 1
                    if pad_cnt < 19:
                        for j in range(cand_num):
                            if cand_qs[i][j] == this_q[i]:
                                qs_modes[i] = True
                                q_ranks[i] = j
                                cand_probs[i][q_ranks[i]] = 1.0

                candidate_probs, click_probs = self.model(prev_qs, prev_docs, prev_clicks, this_q, this_clicks, this_qid,
                                                          last_qid, cand_docs, cand_qs, cand_probs, dr_modes, qs_modes, data, test_mode=True)

                # b x 20 x 1
                if use_cuda:
                    click_probs_list = click_probs.data.cpu().numpy().tolist()
                    candidate_probs = candidate_probs.data.cpu().numpy().tolist()
                else:
                    click_probs_list = click_probs.numpy().tolist()
                    candidate_probs = candidate_probs.numpy().tolist()
                target_clicks_list = this_clicks
                for i in range(batch_n):
                    final_rank = -1
                    if not qs_modes[i]:
                        eval_ouput.append(
                            [click_probs_list[i], target_clicks_list[i], final_rank])
                    else:
                        this_candidate_probs = [[qs_rank, prob] for qs_rank, prob in enumerate(candidate_probs[i])]
                        sorted_probs = sorted(this_candidate_probs, key=lambda x: x[1], reverse=True)
                        cand_num = len(sorted_probs)
                        for j in range(cand_num):
                            origin_rank = sorted_probs[j][0]
                            if origin_rank == q_ranks[i]:
                                final_rank = j + 1
                        eval_ouput.append([click_probs_list[i], target_clicks_list[i], final_rank])

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                for sample in eval_ouput:
                    fout.write('\t'.join(map(str, sample)) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_span_loss = 1.0 * total_loss / total_num
        return ave_span_loss


    def save_model(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix+'_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                      model_prefix,
                                                                                                      self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
        if not os.path.isfile(optimizer_path):
            optimizer_path = os.path.join(model_dir, model_prefix + '_best_{}.optimizer'.format(global_step))
        if os.path.isfile(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                     model_prefix,
                                                                                                     global_step))
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, model_prefix + '_best_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
