# coding=utf-8
'''
@ref: A Hybrid Framework for Session Context Modeling
@desc: Startup
'''

import os
import time
import argparse
import logging
from dataset import Dataset
from model import Model


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('HCSM')  # Hybrid Context Session Model
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--test', action='store_true',
                        help='test on test set')
    parser.add_argument('--gpu', type=str, default='2',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.2,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=2,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=200000,
                                help='number of training steps')
    train_settings.add_argument('--num_train_files', type=int, default=1,
                                help='number of training files')
    train_settings.add_argument('--num_dev_files', type=int, default=1,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=1,
                                help='number of test files')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='HCSM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=256,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')
    model_settings.add_argument('--topic_len', type=int, default=32,
                                help='number of words in a document segment')
    model_settings.add_argument('--topic_num', type=int, default=1,
                                help='number of document segments')
    model_settings.add_argument('--max_sess_length', type=int, default=10,
                                help='max session length')
    model_settings.add_argument('--head_num', type=int, default=1,
                                help='number of the heads in MHA')
    model_settings.add_argument('--similar_qs', type=int, default=3,
                                help='number of expanded query nodes in click bipartite graph')
    model_settings.add_argument('--cross_qs', type=int, default=1,
                                help='number of cross-session query nodes for aggregation')
    model_settings.add_argument('--bfs_depth', type=int, default=1,
                                help='the depth for bfs')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--data_type', type=str,
                               default='data',
                               help='type of dataset, aol sessions or sogou sessions')
    path_settings.add_argument('--train_dirs', nargs='+',
                               default=['../../data/train_sess.txt'],
                               help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                               default=['../../data/test_sess.txt'],
                               help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                               default=['../../data/test_sess.txt'],
                               help='list of dirs that contain the preprocessed test data')
    path_settings.add_argument('--graph_dir', type=str,
                               default='../aol_data/graph.pkl',
                               help='list of constructed graph')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=1000,
                               help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=1000,
                               help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                               help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                               help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                               help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                               help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')

    return parser.parse_args()


def test(args):
    """
    test on the testing set
    """
    logger = logging.getLogger("HCSM")
    logger.info('Checking the data files...')
    for data_path in args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs, test_mode=True)
    logger.info('Initialize the model...')
    max_q_len = -1
    for key in dataset.qid2wids.keys():
        max_q_len = max(len(dataset.qid2wids[key]), max_q_len)
    args.max_q_len = max_q_len
    model = Model(args)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    test_batchs = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    model.evaluate(test_batchs, dataset, result_dir=args.result_dir,
                   result_prefix='rank.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Done with model ranking!')


def train(args):
    """
    training on the training set
    """
    logger = logging.getLogger("HCSM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    max_q_len = -1
    for key in dataset.qid2wids.keys():
        max_q_len = max(len(dataset.qid2wids[key]), max_q_len)
    logger.info('Initialize the model...')
    args.max_q_len = max_q_len
    model = Model(args)
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')

    model.train(dataset)
    logger.info('Done with model training!')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0

    # create a logger
    logger = logging.getLogger("HCSM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info('Checking the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    if args.train:
        train(args)
    if args.test:
        test(args)
    logger.info('run done.')


if __name__ == '__main__':
    run()
