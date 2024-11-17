import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

def obtain_sample(inter):
    layer_node_nums = list(set(np.array(inter).reshape(-1)))
    # Consider undirected
    df_extended = pd.DataFrame({'node': inter['pos'], 'pos': inter['node']})
    inter = pd.concat([inter, df_extended], ignore_index=True)
    inter = inter.drop_duplicates().sort_values(by='node').reset_index(drop=True)
    # Negative sample sampling
    if len(layer_node_nums) >= 999:
        # undersampling
        group = inter.groupby('node')
        neg_samples = []
        for node, pos_nodes in group:
            pos_list = pos_nodes['pos'].tolist()
            sample_nums = len(pos_list)
            pos_list.append(node)  # Node itself
            # Optional<object sampling with replacement
            if len(layer_node_nums)-sample_nums-1 < sample_nums:
                neg_samples += random.choices(list(filter(lambda x: x not in pos_list, layer_node_nums)), k=sample_nums)
            else:  # Unrepeated sampling is preferred whenever possible
                neg_samples += random.sample(list(filter(lambda x: x not in pos_list, layer_node_nums)), sample_nums)
        inter['neg'] = neg_samples
    else:  # All unobserved links are used as negative samples
        group = inter.groupby('node')
        node_samples = []
        pos_samples = []
        neg_samples = []
        for node, pos_nodes in group:
            pos_list = pos_nodes['pos'].tolist()
            neg_list = list(filter(lambda x: x not in pos_list+[node], layer_node_nums))
            if len(pos_list) < len(neg_list):
                pos_list = pos_list + random.choices(pos_list, k=len(neg_list) - len(pos_list))
            else:
                neg_list = neg_list + random.choices(neg_list, k=len(pos_list) - len(neg_list))
            node_samples += [node] * len(neg_list)
            pos_samples += pos_list
            neg_samples += neg_list
        inter = pd.DataFrame({'node': node_samples, 'pos': pos_samples, 'neg': neg_samples})
    return inter

def obtain_train_edge(inter):
    inter = obtain_sample(inter)
    # 8:1:1
    df_train = inter.sample(frac=0.8)
    df_temp = inter.drop(df_train.index)
    df_valid = df_temp.sample(frac=0.5)
    df_test = df_temp.drop(df_valid.index)
    return df_train, df_valid, df_test


def gcndata_load(inters, node_nums):
    all_nodes = [i for i in range(node_nums)]
    pos_edge = np.array(inters[['node', 'pos']]).tolist()
    g = nx.Graph(pos_edge)
    g.add_nodes_from(all_nodes)
    adj = nx.to_scipy_sparse_matrix(g, nodelist=all_nodes, dtype=int, format='coo')
    edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))
    x = torch.unsqueeze(torch.FloatTensor(all_nodes), 1)
    gcn_data = Data(x=x, edge_index=edge_index)
    return gcn_data


def pro_dataset(dataset):
    print('-----------------------------------')
    print('Dataset: ', dataset)
    datadir = 'data/' + dataset + '/'
    layerfiles = os.listdir(datadir)
    network_total = len(layerfiles)
    change = []
    all_edges_num = 0
    for i in range(network_total):
        now_layer = datadir + dataset + '_layer_' + str(i) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None)
        now_nodes = list(set(np.array(now_inter).reshape(-1)))
        print('-----------------------------------')
        print('Nodes of layer ' + str(i) + ": " + str(len(now_nodes)))
        print('Edges of layer ' + str(i) + ": " + str(now_inter.shape[0]))
        all_edges_num += now_inter.shape[0]
        change += now_nodes
    change = list(set(change))
    change_dict = {}
    for i in range(len(change)):
        change_dict[change[i]] = i
    all_node_nums = len(change)
    print('-----------------------------------')
    print('Nodes of all layers: ', all_node_nums)
    print('Edges of all layers: ', all_edges_num)
    print('-----------------------------------')
    layers_pds = []
    for i in range(network_total):
        layer_path = datadir + dataset + '_layer_' + str(i) + '.txt'
        layer = pd.read_csv(layer_path, sep=' ', header=None, names=['node', 'pos'])
        layer['node'] = layer['node'].map(change_dict)  # ID map
        layer['pos'] = layer['pos'].map(change_dict)  # ID map
        layers_pds.append(layer)
    return network_total, layers_pds, all_node_nums


def load_data_wise(tar_layer_id, aux_layer_ids, layers_pds, node_nums, batch_size):
    # "layer-wise prediction"
    # The training dataset consists of links on the target layer and auxiliary layers
    # on the target layer --> special features
    # on the target layer and auxiliary layers --> transferable features
    # The evaluation dataset has only links on the single layer of the target layer
    target_inter = layers_pds[tar_layer_id]
    target_train, target_valid, target_test = obtain_train_edge(target_inter)

    gcn_data = {}
    gcn_data[tar_layer_id] = gcndata_load(target_train, node_nums)
    # auxiliary layers
    for id in aux_layer_ids:
        aux_layer = layers_pds[id]
        gcn_data[id] = gcndata_load(aux_layer, node_nums)

    target_train, target_valid, target_test = get_dataloader(target_train, target_valid, target_test, batch_size)
    return gcn_data, target_train, target_valid, target_test


def pro_loader(df):
    pos_inter = df[['node', 'pos']]
    pos_inter['link'] = 1
    neg_inter = df[['node', 'neg']]
    neg_inter['link'] = 0
    result = np.concatenate((np.array(pos_inter), np.array(neg_inter)), axis=0)
    return result

def pro_dataloader(data, batch_size):
    data = torch.LongTensor(data)
    data_set = TensorDataset(data)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def get_dataloader(train, valid, test, batch_size):
    train = pro_loader(train)
    valid = pro_loader(valid)
    test = pro_loader(test)
    print('Train Links (Pos+Neg):', train.shape[0])
    print('Valid Links (Pos+Neg):', valid.shape[0])
    print('Test Links (Pos+Neg):', test.shape[0])
    train_loader = pro_dataloader(train, batch_size)
    valid_loader = pro_dataloader(valid, batch_size)
    test_loader = pro_dataloader(test, batch_size)
    return train_loader, valid_loader, test_loader
