#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:27:55 2022

@author: pnaddaf
"""


import numpy as np
import scipy
import ast
import pickle
from utils import *
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
import scipy.sparse as ssp
import math
import pickle as pkl
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import copy

save_path = './datasets_LLGF/'
input_dataset = 'cora'
output_dataset = input_dataset+ '_new'
ind = ''
semi_inductive = False
train_path = save_path + input_dataset + "/train.txt"

def load_citeseer():
    dataSet = 'citeseer'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/localhome/pnaddaf/Desktop/parmis/inductive_learning/citeseer/ind.{}.{}".format(dataSet, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/localhome/pnaddaf/Desktop/parmis/inductive_learning/citeseer/ind.{}.test.index".format(dataSet))
    test_idx_range = np.sort(test_idx_reorder)


    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    xx = np.where(labels == 1)
    for i in range(labels.shape[0]):
        if i not in xx[0]:
            labels[i][0] = 1
    labels = np.where(labels == 1)[1]
    
    return features.toarray(), sp.csr_matrix(adj.toarray().astype(np.float32))


def load_cora():
    dataSet='cora'
    cora_content_file = '/localhome/pnaddaf/Desktop/parmis/inductive_learning/inductive_learning/cora/cora.content'
    cora_cite_file = '/localhome/pnaddaf/Desktop/parmis/inductive_learning/inductive_learning/cora/cora.cites'

    feat_data = []
    labels = [] # label sequence of node
    node_map = {} # map node to Node_ID
    label_map = {} # map label to Label_ID
    with open(cora_content_file) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data.append([float(x) for x in info[1:-1]])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels.append(label_map[info[-1]])                
    feat_data = np.asarray(feat_data)
    labels = np.asarray(labels, dtype=np.int64)
    
    # get adjacency matrix                
    adjacency_matrix = scipy.sparse.lil_matrix((len(feat_data), len(feat_data)))
    with open(cora_cite_file) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            assert len(info) == 2
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adjacency_matrix[paper1, paper2] = 1
            adjacency_matrix[paper2, paper1] = 1
            
    assert len(feat_data) == len(labels) == adjacency_matrix.shape[0]
    return feat_data , adjacency_matrix
    



def load_ACM():
    obj = []
    adj_file_name = "/localhome/pnaddaf/Desktop/parmis/inductive_learning/inductive_learning/ACM/edges.pkl"

    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj:
        adj +=matrix[0]


    obj = []
    with open("/localhome/pnaddaf/Desktop/parmis/inductive_learning/inductive_learning/ACM/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0]).todense()


    index = -1
    aa = adj[:index,:index].todense()
    at = aa.transpose((1, 0))
    adjacency_matrix = aa + at

    return np.asarray(feature[:index]) , sp.csr_matrix(adjacency_matrix)




def load_IMDB():
    adj_file_name = "/localhome/pnaddaf/Desktop/parmis/inductive_learning/IMDB/edges.pkl"
    obj = []
    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    # merging diffrent edge type into a single adj matrix
    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj[0]:
        adj +=matrix
    
    obj = []
    with open("/localhome/pnaddaf/Desktop/parmis/inductive_learning//IMDB/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])
    
    return feature.todense(), adj
    


def split_data(num_nodes, test_split = 0.2, val_split = 0.1):
    np.random.seed(123)
    rand_indices = np.random.permutation(num_nodes)
    
    test_size = int(num_nodes * test_split)
    val_size = int(num_nodes * val_split)
    train_size = num_nodes - (test_size + val_size)
    # print(num_nodes, train_size, val_size, test_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size+val_size)]
    train_indexs = rand_indices[(test_size+val_size):]
    
    return test_indexs, val_indexs, train_indexs




def remove_in_between_link(org_adj, testId, trainId):
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId) &  np.isin(res[1], trainId))
    i_list = res[0][index]
    j_list = res[1][index]
    org_adj[i_list, j_list] = 0
    org_adj[j_list, i_list] = 0
    return org_adj
    # org_adj[res[index]] = 0
     




def get_test_edges(org_adj, testId):
    res = org_adj.nonzero()

    index = np.where(np.isin(res[0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    idd_list = res[0][index]
    neighbour_list = res[1][index]
    sample_list = random.sample(range(0, len(idd_list)), 100)
    false_list = []
    true_list = []
    for i in sample_list:
        idd = idd_list[i]
        neighbour_id = neighbour_list[i]
        adj_list_copy = copy.deepcopy(org_adj)
        true_multi_link = org_adj[idd].nonzero()
        false_multi_link = random.sample(list(np.nonzero(org_adj[idd] == 0)[0]), len(true_multi_link[0]))
        true_list.append([[idd, i] for i in list(true_multi_link[0])])
        false_list.append([[idd, i] for i in list(false_multi_link)])
    
    return true_list, false_list
    




def make_semi_inductive_edges(adjacency_matrix, inductive_test_edges):
    if "cora" in input_dataset:
        _ , adj = load_cora()
    elif "ACM" in input_dataset:
        _ , adj = load_ACM()
    elif "IMDB" in input_dataset:
        _ , adj = load_IMDB()
    elif "citeseer" in input_dataset:
        _ , adj = load_citeseer()
    i_list = adj.nonzero()[0]
    j_list = adj.nonzero()[1]
    edge_list = np.column_stack((i_list, j_list))
    # indutive_train_edges = []
    # for edge in edge_list:
    #     if edge not in inductive_test_edges and edge not in inductive_val_edges:
    #         indutive_train_edges.append(edge)
    res = np.array(list(set(map(tuple, edge_list)) - set(map(tuple, inductive_test_edges))))

    return res
    
    
    


def save_splits_to_file(inductive_test_edges, inductive_test_edges_false, adjacency_matrix, feat_data):
    print("Saving to files")
    global output_dataset
    if semi_inductive:
        output_dataset += "_semi"
    for i in range(len(inductive_test_edges)):
                
        if semi_inductive:
            inductive_train_edges = make_semi_inductive_edges(adjacency_matrix, inductive_test_edges[i])
            
        np.save(save_path + 'LLGF_' + output_dataset  + '_ind_' + str(i) +'_test_pos.npy', np.array(inductive_test_edges[i]))
        np.save(save_path + 'LLGF_' + output_dataset + '_ind_' + str(i) +'_test_neg.npy', np.array(inductive_test_edges_false[i]))
        
        np.save(save_path + 'LLGF_' + output_dataset +  '_ind_' + str(i) +'_val_pos.npy', np.array(inductive_test_edges[i]))
        np.save(save_path + 'LLGF_' + output_dataset + '_ind_' + str(i) +'_val_neg.npy', np.array(inductive_test_edges_false[i]))

        
        
        np.save(save_path + 'LLGF_' + output_dataset + '_ind_' + str(i) +'_train_pos.npy', np.array(inductive_train_edges))
        
        np.save(save_path + 'LLGF_' + output_dataset + '_ind_' + str(i) +'_x.npy', np.array(feat_data))

if "cora" in input_dataset:
    feat_data , adjacency_matrix = load_cora()
elif "ACM" in input_dataset:
    feat_data , adjacency_matrix = load_ACM()
elif "IMDB" in input_dataset:
    feat_data , adjacency_matrix = load_IMDB()
elif "citeseer" in input_dataset:
    feat_data , adjacency_matrix = load_citeseer()

testId, valId, trainId = split_data(feat_data.shape[0])
adjacency_matrix = remove_in_between_link(adjacency_matrix, testId, trainId)
inductive_test_edges, inductive_test_edges_false = get_test_edges(adjacency_matrix, testId)
save_splits_to_file(inductive_test_edges, inductive_test_edges_false, adjacency_matrix, feat_data)