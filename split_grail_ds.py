#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:50:17 2022

@author: pnaddaf
"""


import numpy as np
import ast
import pickle
import scipy.sparse as ss
import torch
import math
from utils import *

save_path = '/localhome/pnaddaf/Desktop/parmis/SEAl_miror/datasets_LLGF/'

input_dataset = 'FB15K237'
output_dataset = 'FB_new'
ind = ''

x_path = '/localhome/pnaddaf/Desktop/parmis/grail-master/data/' + input_dataset + '/entity2vec.txt'
train_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + input_dataset + "/train.txt"
test_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + input_dataset + "/test.txt"
val_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + input_dataset + "/valid.txt"

nodes_dict_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + input_dataset + "/entities.dict"

def read_edges(edge_path):
    edge_list = []
    with open(edge_path) as f:
        for line in f:
            line = line.split("\t")
            source = nodes_dict[line[0]]
            target = nodes_dict[line[2][:-1]]        
            
            edge_list.append([source, target])
            edge_list.append([target, source])
            
            all_nodes.add(source)
            all_nodes.add(target)
    return edge_list
  
    
  
# function to return key for any value
def get_key(val):
    for key, value in nodes_dict.items():
         if val == value:
             return key





def make_compatible_to_grail(edge_list):
    list_new = []
    for edge in edge_list:
        list_new.append([get_key(edge[0]), 0, get_key(edge[1])])
    return np.array(list_new)








def vis_subgraph(x, adjacency_matri):
    f = plt.figure(figsize=(20, 20))
    limits = plt.axis('off')
    subgraph = get_subgraph(x, adjacency_matrix)
    g = torch_geometric.utils.to_networkx(subgraph[0], to_undirected=True)
    nx.draw(g)
    f.savefig('temp1.png')


def get_nodes_neighbourhood(adjacency_matrix, x):
    adjacency_matrix = adjacency_matrix.todense()
    #root_list = random.sample(range(0, adjacency_matrix.shape[0]), math.floor(1/50 *  adjacency_matrix.shape[0]) )
    trans_neighbours = []
    nodes_to_be_removed = set()
    ind_nodes = [i for i in range(adjacency_matrix.shape[0])]
    i = 0 
    total = 1/3*(len(ind_nodes))
    while len(nodes_to_be_removed) < total:
        if i % 1000 == 0:
            print(str(i)+"/"+str(total))
        i+=1
        root = ind_nodes[random.sample(range(0,len(ind_nodes)),1)[0]]
        # root = random.sample(ind_nodes,1)[0]
        nodes_to_be_removed.add(root)
        ind_nodes.remove(root)
        # adjacency_matrix[: ][ root] = 0
        # adjacency_matrix[root][:] = 0 
        neighbour_list = adjacency_matrix[root].nonzero()[1]
        for neighbour in neighbour_list:
            trans_neighbours.append([root, neighbour ])
            trans_neighbours.append([neighbour, root ])
            nodes_to_be_removed.add(neighbour)
            if neighbour in ind_nodes:
                ind_nodes.remove(neighbour)
            # adjacency_matrix[:][ neighbour] = 0
            # adjacency_matrix[neighbour][ :] = 0 
            neighbour_of_neighbour = adjacency_matrix[neighbour].nonzero()[1]
            for nn in neighbour_of_neighbour :
                if nn in neighbour_list and nn != root:
                    trans_neighbours.append([neighbour, nn ])
                                        
   
    # adjacency_matrix[: ][ np.array(list(nodes_to_be_removed))] = 0
    # adjacency_matrix[np.array(list(nodes_to_be_removed))][ :] = 0 
    
    # vis_subgraph(x, adjacency_matrix)
    
    

    all_nodes = [i for i in range(adjacency_matrix.shape[0])]
    trans_nodes =  np.array(list(nodes_to_be_removed))
    
    # ind_nodes = []
    # for node in all_nodes:
    #     if node not in trans_nodes:
    #         ind_nodes.append(node)

    ind_neighbours = []        
    for node in ind_nodes:
        for j in adjacency_matrix[node].nonzero()[1]:
            if j not in trans_nodes:
                ind_neighbours.append([node, j ])

    
    return ind_neighbours, trans_neighbours, ind_nodes, trans_nodes
    

def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)




def get_false_new(test_edges, train_edges, val_edges, not_included_nodes, feat):
    edge_list = np.asarray( test_edges.tolist() + train_edges.tolist() + val_edges.tolist(),  dtype=np.float32)
    adj = fill_adj(edge_list, feat).todense()
    
    print(len(not_included_nodes))
    # i = 0
    # for node in not_included_nodes:
    #     if i % 1000 == 0:
    #         print(i)
    #     i += 1
    #     adj[:][ node] = 1
    #     adj[node][:] = 1
        
    print("done")
    
    not_included_nodes = np.array(list(not_included_nodes))
    
    adj[: ][ not_included_nodes] = 1
    adj[not_included_nodes][ :] = 1
    false_edges = np.argwhere(adj == 0)
    random.shuffle(false_edges)
    test_num , train_num, val_num = len(test_edges), len(train_edges), len(val_edges)
    test_edges_false = false_edges[:test_num]
    train_edges_false = false_edges[test_num: test_num + train_num]
    val_edges_false = false_edges[test_num + train_num: test_num + train_num + val_num]
    return  train_edges_false, test_edges_false, val_edges_false





def get_false_edges(test_edges, train_edges, val_edges , nodes):
    
    edges_all = np.asarray( test_edges.tolist() + train_edges.tolist() + val_edges.tolist(),  dtype=np.float32)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        if len(test_edges_false) % 1000 == 0:
            print( len(test_edges_false))
        idx_i = random.choice(nodes)
        idx_j = random.choice(nodes)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    print("Finished False Test")
    
    
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        if len(val_edges_false) % 1000 == 0:
            print( len(val_edges_false))
        idx_i = random.choice(nodes)
        idx_j = random.choice(nodes)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
        
    print("Finished False Val")

    train_edges_false = []
    # while len(train_edges_false) < len(train_edges):
        # idx_i = random.choice(nodes)
        # idx_j = random.choice(nodes)
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #         continue
    #     if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #         continue
    #     if train_edges_false:
    #         if ismember([idx_j, idx_i], np.array(train_edges_false)):
    #             continue
    #     train_edges_false.append([idx_i, idx_j])


    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    
    return train_edges_false, test_edges_false, val_edges_false
        

def get_pos_edges_from_edges(edges):
    edges = np.array(edges)    
    num_test = int(np.floor(edges.shape[0] / 5.))
    num_val = int(np.floor(edges.shape[0] / 10.))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    return train_edges, val_edges, test_edges
    


def fill_adj(edge_list, feat_data):
    edge_weight = torch.ones(edge_list.shape[0], dtype=int)
    adjacency_matrix = ss.csr_matrix(
        (edge_weight, (torch.tensor(edge_list[:,0]), torch.tensor(edge_list[:,1]))), 
        shape=(len(feat_data), len(feat_data)))
    return adjacency_matrix 


nodes_dict = dict()
with open(nodes_dict_path) as f:
    for line in f:
       line = line.split("\t")
       # line_0 = line[0].replace("_", ":",2)
       # nodes_dict[line_0] = int(line[1][:-1])
       nodes_dict[line[1][:-1]] = int(line[0])






feat_data = []
with open(x_path) as f:
    for line in f:
        feat_data.append([float(i) for i in line.split("\t")[:-1]])

np.save(save_path + 'LLGF_' + output_dataset + ind + '_x.npy', np.array(feat_data))
       
        
        
all_nodes = set()


print("Read data")
train_edges = read_edges(train_path)
test_edges = read_edges(test_path)
val_edges = read_edges(val_path)
all_edges = np.array(train_edges + test_edges + val_edges)

print("Making adj matrix")

adjacency_matrix = fill_adj(all_edges, feat_data)


# np.save('feat.npy', np.array(feat_data))
# np.save( 'nell_adj.npy', np.array(adjacency_matrix))


print("Extracting transductive and inductive neighbourhoods")
ind_neighbours, trans_neighbours, ind_nodes, trans_nodes = get_nodes_neighbourhood(adjacency_matrix, feat_data)


print("Getting positive edges")
transductive_train_edges, transductive_valid_edges, transductive_test_edges = get_pos_edges_from_edges(trans_neighbours)
inductive_train_edges, inductive_valid_edges, inductive_test_edges = get_pos_edges_from_edges(ind_neighbours)



# get false edges
print("Getting false edges")
transductive_train_edges_false, transductive_test_edges_false, transductive_val_edges_false = get_false_new(transductive_test_edges, transductive_train_edges, transductive_valid_edges, ind_nodes, feat_data)
inductive_train_edges_false, inductive_test_edges_false, inductive_val_edges_false = get_false_new(inductive_test_edges, inductive_train_edges, inductive_valid_edges, trans_nodes , feat_data )

# transductive_train_edges_false, transductive_test_edges_false, transductive_val_edges_false = get_false_edges(transductive_test_edges, transductive_train_edges, transductive_valid_edges, trans_nodes )
# inductive_train_edges_false, inductive_test_edges_false, inductive_val_edges_false = get_false_edges(inductive_test_edges, inductive_train_edges, inductive_valid_edges, ind_nodes )


# save files

print("Saving to files")

np.save(save_path + 'LLGF_' + output_dataset + '_x.npy', np.array(feat_data))
np.save(save_path + 'LLGF_' + output_dataset + 'ind_x.npy', np.array(feat_data))

np.save(save_path + 'LLGF_' + output_dataset + '_train_pos.npy', np.array(transductive_train_edges))

np.save(save_path + 'LLGF_' + output_dataset + '_test_pos.npy', np.array(transductive_test_edges))
np.save(save_path + 'LLGF_' + output_dataset + '_test_neg.npy', np.array(transductive_test_edges_false))

np.save(save_path + 'LLGF_' + output_dataset +  '_val_pos.npy', np.array(transductive_valid_edges))
np.save(save_path + 'LLGF_' + output_dataset + '_val_neg.npy', np.array(transductive_val_edges_false))



np.save(save_path + 'LLGF_' + output_dataset + 'ind_train_pos.npy', np.array(inductive_train_edges))

np.save(save_path + 'LLGF_' + output_dataset + 'ind_test_pos.npy', np.array(inductive_test_edges))
np.save(save_path + 'LLGF_' + output_dataset + 'ind_test_neg.npy', np.array(inductive_test_edges_false))

np.save(save_path + 'LLGF_' + output_dataset +  'ind_val_pos.npy', np.array(inductive_valid_edges))
np.save(save_path + 'LLGF_' + output_dataset + 'ind_val_neg.npy', np.array(inductive_val_edges_false))








# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5/test.txt', make_compatible_to_grail(transductive_test_edges), delimiter='\t' , fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5/train.txt',  make_compatible_to_grail(transductive_train_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5/valid.txt',  make_compatible_to_grail(transductive_valid_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5_ind/valid.txt',  make_compatible_to_grail(inductive_valid_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5_ind/test.txt', make_compatible_to_grail(inductive_test_edges),  delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/fb_v5_ind/train.txt', make_compatible_to_grail(inductive_train_edges),  delimiter='\t', fmt='%s')

