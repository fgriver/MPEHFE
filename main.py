import argparse
import datetime
import logging
import os
import time
import torch
import numpy as np
from trainer import train_test
from model import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_parameter_memory(model):
    """Calculates the static memory usage of model parameters."""
    total_params = 0
    for param in model.parameters():
        total_params += param.nelement()

    # Parameters are typically float32 (4 bytes)
    bytes_per_param = 4
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024 ** 2)
    print(f"Model Parameters Memory: {total_mb:.2f} MB")
    return total_mb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Nowplaying/Tmall/RetailRocket')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--log', type=str, default='log')
parser.add_argument('--activate', type=str, default='relu')

parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=2)                                    # [1, 2] 这是全局图的聚合数
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--gnn_step', type=int, default=3, help='control step of ggnn in 1 xx layer')
parser.add_argument('--star_step', type=int, default=1, help='control step of star gcn in 1 xx layer')
parser.add_argument('--k', type=int, default=2, help='the number of k')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--scale', action='store_true')
parser.add_argument('--lamda', type=float, default=0.3, help='lambda parameter')
# ablation
parser.add_argument('--no_skip_connection', action='store_true', help='do not use skip_connection?')
parser.add_argument('--no_cl_loss', action='store_true', help='do not use cl loss')
parser.add_argument('--agg_model', type=str, default='df', help='df/last/avg')
parser.add_argument('--no_hybrid_gnn', action='store_true', help='only use GGNN without pgat and highway')
parser.add_argument('--no_global', action='store_true', help='only use local')
parser.add_argument('--no_local', action='store_true', help='only use global')
opt = parser.parse_args()

def main(seed):
    init_seed(seed)

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    neg = pickle.load(open('datasets/' + opt.dataset + '/neg_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    adj, num = handle_adj(adj, num_node, opt.n_sample, num)
    neg = np.asarray(neg)
    neg = torch.from_numpy(neg)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num, neg, max_seq_len))

    # get_model_parameter_memory(model)

    train_data = Data(train_data, adj)
    test_data = Data(test_data, adj)

    start_time = datetime.datetime.now()
    log_dir = f"log/{opt.dataset}/{start_time}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = (f"{opt.dataset}-k-{opt.k}-no_skip_connection-{opt.no_skip_connection}-"
                    f"agg_model-{opt.agg_model}-no_hybrid_gnn-{opt.no_hybrid_gnn}-no_global-{opt.no_global}-"
                    f"no_local-{opt.no_local}-no_cl_loss-{opt.no_cl_loss}.log")
    log_filename = os.path.join(log_dir, log_filename)


    logging.basicConfig(
        filename=log_filename,  # 动态生成的日志文件名
        level=logging.INFO,  # 日志级别
        format='%(asctime)s - %(message)s',  # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
    )
    start = time.time()
    best_result = {5: [0, 0], 10: [0, 0], 20: [0, 0]}
    best_epoch = {5: [0, 0], 10: [0, 0], 20: [0, 0]}
    bad_counter = 0

    print(opt)
    logging.info(opt)

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        logging.info('-------------------------------------------------------')
        print('epoch: ', epoch)
        logging.info(f'epoch: {epoch}')
        results = train_test(model, train_data, test_data, opt.no_cl_loss, opt.lamda)
        flag = 0
        for k in [5, 10, 20]:
            if results[k][0] >= best_result[k][0]:  # Recall@k
                best_result[k][0] = results[k][0]
                best_epoch[k][0] = epoch
                flag = 1
            if results[k][1] >= best_result[k][1]:  # MRR@k
                best_result[k][1] = results[k][1]
                best_epoch[k][1] = epoch
                flag = 1
            print('Current Result:')
            logging.info('Current Result:')
        for k in [5, 10, 20]:
            print(f'\tRecall@{k}:\t{results[k][0]:.4f}\tMRR@{k}:\t{results[k][1]:.4f}')
            logging.info(f'\tRecall@{k}:\t{results[k][0]}\tMRR@{k}:\t{results[k][1]}')
        # 输出最佳结果
        print('Best Result:')
        logging.info('Best Result:')
        for k in [5, 10, 20]:
            print(
                f'\tRecall@{k}:\t{best_result[k][0]:.4f}\tMRR@{k}:\t{best_result[k][1]:.4f}\tEpoch:\t{best_epoch[k][0]},\t{best_epoch[k][1]}')
            logging.info(
                f'\tRecall@{k}:\t{best_result[k][0]}\tMRR@{k}:\t{best_result[k][1]}\tEpoch:\t{best_epoch[k][0]},\t{best_epoch[k][1]}')
        # 更新早停计数器
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    logging.info('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    logging.info(f'Run time: {(end - start)} s')

if __name__ == '__main__':

    main(2020)
