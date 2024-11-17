import os
import argparse
import datetime
import torch
from src.Load import pro_dataset, load_data_wise
from src import ISCor
from src.Train import run_model
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ISCor', help='name of model')
    parser.add_argument('--dataset', type=str, default='Enron', help='name of dataset')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--ifDecay', action='store_false', help='Default is true. If true, the learning rate decays. If false, the learning rate is constant.')

    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--EarlyStop', action='store_true', help='Default is false. If true, enable early stopping strategy')
    parser.add_argument('--patience', type=int, default=50, help='early stop')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train data')
    parser.add_argument('--dim', type=int, default=16, help='dims of node embedding')

    parser.add_argument('--onlyTest', action='store_true', help='Default is false. If true, try to load an existing model.')
    parser.add_argument('--log', type=str, default='', help='record file path')

    args = parser.parse_args()
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    path = './log/' + args.dataset + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('current running device:', args.device)

    print('***************************')
    print('The program starts running.')
    print('***************************')

    args.log = path + args.model + '-' + args.dataset + '-Result.txt'
    print(args)

    log = open(args.log, 'a', encoding='utf-8')
    begin = datetime.datetime.now()
    time = str(begin.year) + '-' + str(begin.month) + '-' + str(begin.day) + '-' + str(begin.hour) + '-' + str(
        begin.minute) + '-' + str(begin.second)
    write_infor = '\n==========================================================\n' + time + '\n'
    log.write(write_infor)
    write_infor = "model:{}, Batch_size:{}, Dim:{}".format(args.model, args.batch_size, args.dim) + '\n'
    log.write(write_infor)
    write_infor = "lr:{}, IfDecay:{}, Epochs:{}, IfStop:{}, Patience:{}".format(args.learning_rate, args.ifDecay, args.epochs, args.EarlyStop, args.patience) + '\n'
    log.write(write_infor)

    network_total, layers_pds, node_nums = pro_dataset(args.dataset)
    # target layer
    for target_id in range(network_total):
        write_infor = '--- target_network: layer ' + str(target_id) + ' --- '
        print(write_infor)
        log.write('\n' + write_infor)
        auxiliary_ids = [id for id in range(network_total)]
        auxiliary_ids.pop(target_id)
        print('--- auxiliary_networks: layer ', auxiliary_ids, ' --- ')
        gcn_data, train, valid, test = load_data_wise(target_id, auxiliary_ids, layers_pds, node_nums, args.batch_size)

        for i in range(network_total):
            gcn_data[i].x = gcn_data[i].x.to(args.device)
            gcn_data[i].edge_index = gcn_data[i].edge_index.to(args.device)
        model = eval(args.model).Model_Net(args.dim, network_total, node_nums, gcn_data, target_id, args.device)
        model = model.to(args.device)
        run_model(train, valid, test, model, target_id, args, log)
    log.close()

