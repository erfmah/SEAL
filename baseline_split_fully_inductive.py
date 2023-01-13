#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import argparse

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--dataset', type=str, default='photos')
parser.add_argument('--semi_inductive', type=str, default=False)
args = parser.parse_args()
save_path = '/localhome/pnaddaf/Desktop/parmis/SEAl_miror/datasets_LLGF/'
input_dataset = args.dataset
output_dataset = input_dataset+ '_new'
ind = ''
semi_inductive = args.semi_inductive
if args.semi_inductive=="True":
    semi_inductive  = True
elif semi_inductive=="False":
    semi_inductive = False
    
train_path = save_path + input_dataset + "/train.txt"

def load_citeseer():
    dataSet = 'citeseer'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./datasets/citeseer/ind.{}.{}".format(dataSet, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./datasets/citeseer/ind.{}.test.index".format(dataSet))
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
    cora_content_file = './datasets/cora/cora.content'
    cora_cite_file = './datasets/cora/cora.cites'

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
    adj_file_name = "./datasets/ACM/edges.pkl"

    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj:
        adj +=matrix[0]


    obj = []
    with open("./datasets/ACM/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0]).todense()


    index = -1
    aa = adj[:index,:index].todense()
    at = aa.transpose((1, 0))
    adjacency_matrix = aa + at

    return np.asarray(feature[:index]) , sp.csr_matrix(adjacency_matrix)




def load_IMDB():
    adj_file_name = "./datasets/IMDB/edges.pkl"

    obj = []
    with open(adj_file_name, 'rb') as f:
            obj.append(pkl.load(f))

    # merging diffrent edge type into a single adj matrix
    adj = sp.csr_matrix(obj[0][0].shape)
    for matrix in obj[0]:
        adj +=matrix
    
    obj = []
    with open("./datasets/IMDB/node_features.pkl", 'rb') as f:
        obj.append(pkl.load(f))
    feature = sp.csr_matrix(obj[0])
    
    return feature.todense(), adj
    
def load_photos():
    path = os.getcwd()
    data = Amazon(path, 'photo')[0]
    features = data['x'].cpu().detach().numpy()
    adj = np.eye(data['x'].shape[0], dtype=int)
    adj[data.edge_index[0], data.edge_index[1]]= 1


    features = sp.csr_matrix(features)
    adj = sp.csr_matrix(adj)
    return features, adj




def load_computers():
    path = os.getcwd()
    data = Amazon(path, 'computers')[0]
    features = data['x'].cpu().detach().numpy()
    adj = np.eye(data['x'].shape[0], dtype=int)
    adj[data.edge_index[0], data.edge_index[1]]= 1


    features = sp.csr_matrix(features)
    adj = sp.csr_matrix(adj)
    return features, adj
        
                

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index       
        
def get_subgraph(x, adjacency_matrix):
    num_hops = 1
    node_label = 'drnl'
    ratio_per_hop = 1
    directed = False
    max_nodes_per_hop = None 
    A_csc = None

    #all edges in graph
    edges_index = torch.tensor([adjacency_matrix.nonzero()[0], adjacency_matrix.nonzero()[1]])
    
    x = torch.from_numpy(x)
    data = Data(x, edges_index)
   
    edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
    shape=(data.num_nodes, data.num_nodes))
    subgraphs = extract_enclosing_subgraphs( 
            edges_index, A, data.x, 1, num_hops, node_label, 
            ratio_per_hop, max_nodes_per_hop, directed, A_csc)
    return subgraphs
    

def vis_subgraph(x, adjacency_matri):
    f = plt.figure(figsize=(20, 20))
    limits = plt.axis('off')
    subgraph = get_subgraph(x, adjacency_matrix)
    g = torch_geometric.utils.to_networkx(subgraph[0], to_undirected=True)
    nx.draw(g)
    f.savefig('temp1.png')

def get_nodes_neighbourhood(adjacency_matrix, x):
    
    adjacency_matrix = adjacency_matrix.todense()
    root_list = random.sample(range(0, adjacency_matrix.shape[0]), math.floor(1/10 *  adjacency_matrix.shape[0]))
    trans_neighbours = []
    nodes_to_be_removed = set()
    links = len(adjacency_matrix.nonzero()[0])
    
    for root in root_list:
        if len(trans_neighbours) < (7/10)*links:
            nodes_to_be_removed.add(root)
            neighbour_list = adjacency_matrix[root].nonzero()[1]
            for neighbour in neighbour_list:
                trans_neighbours.append([root, neighbour ])
                trans_neighbours.append([neighbour, root ])
                nodes_to_be_removed.add(neighbour)
                neighbour_of_neighbour = adjacency_matrix[neighbour].nonzero()[1]
                for nn in neighbour_of_neighbour :
                    if nn in neighbour_list and nn != root:
                        trans_neighbours.append([neighbour, nn ])
                        trans_neighbours.append([nn, neighbour ])
                        nodes_to_be_removed.add(nn)
        else:
            continue
                        
   
    adjacency_matrix[: , np.array(list(nodes_to_be_removed))] = 0
    adjacency_matrix[np.array(list(nodes_to_be_removed)), :] = 0 
    
    # vis_subgraph(x, adjacency_matrix)
    all_nodes = [i for i in range(adjacency_matrix.shape[0])]
    trans_nodes =  np.array(list(nodes_to_be_removed))
    
    ind_nodes = []
    for node in all_nodes:
        if node not in trans_nodes:
            ind_nodes.append(node)

    ind_neighbours = []        
    for node in ind_nodes:
        for j in adjacency_matrix[node].nonzero()[1]:
            ind_neighbours.append([node, j ])

    
    return ind_neighbours, trans_neighbours, ind_nodes, trans_nodes
    



def make_semi_inductive_edges(adjacency_matrix, inductive_test_edges, inductive_val_edges):
    if "cora" in input_dataset:
        _ , adj = load_cora()
    elif "ACM" in input_dataset:
        _ , adj = load_ACM()
    elif "IMDB" in input_dataset:
        _ , adj = load_IMDB()
    elif "citeseer" in input_dataset:
        _, adj = load_citeseer()
    elif "photos" in input_dataset:
        _, adj = load_photos()
    elif "computers" in input_dataset:
        _, adj = load_computers()   
    i_list = adj.nonzero()[0]
    j_list = adj.nonzero()[1]
    edge_list = np.column_stack((i_list, j_list))
    # indutive_train_edges = []
    # for edge in edge_list:
    #     if edge not in inductive_test_edges and edge not in inductive_val_edges:
    #         indutive_train_edges.append(edge)
    res = np.array(list(set(map(tuple, edge_list)) - set(map(tuple, inductive_test_edges))))

    return np.array(list(set(map(tuple, res)) - set(map(tuple, inductive_val_edges))))
    
    
    
            
    
    


        
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
    
    
    



def extract_transductive_subgraph(subgraphs, adjacency_matrix ):
    random_list = random.sample(range(0, len(subgraphs)), math.floor(0.03 * len(subgraphs)))
    nodes_to_be_removed = set()
    transductive_subgraphs = []
    for i in random_list:
        transductive_subgraphs.append(subgraphs[i]) 
        nodes_to_be_removed = nodes_to_be_removed.union(set((subgraphs[i].node_id).numpy()))
    return transductive_subgraphs, nodes_to_be_removed
   
 
def extract_inductive_subgraph(adjacency_matrix, nodes_to_be_removed):
    A = adjacency_matrix.toarray()
    A[: , np.array(list(nodes_to_be_removed))] = 0
    A[np.array(list(nodes_to_be_removed)), :] = 0 
    return get_subgraph(feat_data, ssp.csr_matrix(A))
    
 

def split_subgraph_to_train_test(subgraphs, test_ratio= 0.10, valid_ratio=0.05):
    test_subgraphs = subgraphs[: int(test_ratio * len(subgraphs))]
    valid_subgraphs = subgraphs[int(test_ratio * len(subgraphs)): int((test_ratio * len(subgraphs)) +  (valid_ratio * len(subgraphs)))]
    train_subgraphs = subgraphs[ int((test_ratio * len(subgraphs)) +  (valid_ratio * len(subgraphs))):]
    return train_subgraphs, valid_subgraphs, test_subgraphs
    
        
def get_edges_from_subgraphs(subgraphs):
    all_edges = []
    for subgraph in subgraphs:
        for i in range(len(subgraph.edge_index[0])):
            src = subgraph.node_id[subgraph.edge_index[0][i]]
            trg = subgraph.node_id[subgraph.edge_index[1][i]]   
            all_edges.append([src.item(), trg.item()])
    return all_edges
        



def get_pos_edges_from_subgraph(subgraphs):
    edges = np.array(get_edges_from_subgraphs(subgraphs))
    
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



def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)


def get_false_edges(test_edges, train_edges, val_edges, nodes ):
    
    edges_all = np.asarray( test_edges.tolist() + train_edges.tolist() + val_edges.tolist(),  dtype=np.float32)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
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
    print("find tests")

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
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
    print("find vals")
    train_edges_false = []
    # while len(train_edges_false) < len(train_edges):
    #     idx_i = random.choice(nodes)
    #     idx_j = random.choice(nodes)
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

    print("find train")
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    
    
    print("we be done")
    return train_edges_false, test_edges_false, val_edges_false



def make_compatible_to_grail(edge_list):
    list_new = []
    for edge in edge_list:
        list_new.append([str(edge[0]), 0, str(edge[1])])
    return np.array(list_new)


     
    all_edges =  adjacency_matrix.todense().nonzero()
    all_edges = np.vstack((all_edges[0],all_edges[1])).T
    node_ids = [i for i in range(len(feat_data))]
    
    print("Getting positive edges")
    all_train_edges, all_val_edges, all_test_edges = get_pos_edges_from_edges(all_edges)
    
    print("Getting false edges")
    all_train_edges_false, all_test_edges_false, all_val_edges_false = get_false_edges(all_test_edges, all_train_edges, all_val_edges, node_ids )
        
    print("save to files")
    
    dataset = output_dataset+'_all'
    np.save(save_path + 'LLGF_' + dataset + '_x.npy', np.array(feat_data))
    np.save(save_path + 'LLGF_' + dataset + 'ind_x.npy', np.array(feat_data))
    
    np.save(save_path + 'LLGF_' + dataset + '_train_pos.npy', np.array(all_train_edges))
    
    np.save(save_path + 'LLGF_' + dataset + '_test_pos.npy', np.array(all_test_edges))
    np.save(save_path + 'LLGF_' + dataset + '_test_neg.npy', np.array(all_test_edges_false))
    
    np.save(save_path + 'LLGF_' + dataset +  '_val_pos.npy', np.array(all_val_edges))
    np.save(save_path + 'LLGF_' + dataset + '_val_neg.npy', np.array(all_val_edges_false))
    
def return_trans_ind(feat_data, adjacency_matrix, output_dataset):
    print("Extracting transductive and inductive neighbourhoods")
    ind_neighbours, trans_neighbours, ind_nodes, trans_nodes = get_nodes_neighbourhood(adjacency_matrix, feat_data)
    
    
    print("Getting positive edges")
    transductive_train_edges, transductive_valid_edges, transductive_test_edges = get_pos_edges_from_edges(trans_neighbours)
    inductive_train_edges, inductive_valid_edges, inductive_test_edges = get_pos_edges_from_edges(ind_neighbours)
    
    if semi_inductive:
        inductive_train_edges = make_semi_inductive_edges(adjacency_matrix, inductive_test_edges, inductive_valid_edges)
        output_dataset = output_dataset + "_semi"
    print("Getting false edges ind")
    inductive_train_edges_false, inductive_test_edges_false, inductive_val_edges_false = get_false_edges(inductive_test_edges, inductive_train_edges, inductive_valid_edges, ind_nodes )

    print("Getting false edges trans")
    transductive_train_edges_false, transductive_test_edges_false, transductive_val_edges_false = get_false_edges(transductive_test_edges, transductive_train_edges, transductive_valid_edges, trans_nodes )

    
    print("Saving to files")
    np.save(save_path + 'LLGF_' + output_dataset + '_x.npy', np.array(feat_data))
    np.save(save_path + 'LLGF_' + output_dataset + '_ind_x.npy', np.array(feat_data))
    
    np.save(save_path + 'LLGF_' + output_dataset + '_train_pos.npy', np.array(transductive_train_edges))
    
    np.save(save_path + 'LLGF_' + output_dataset + '_test_pos.npy', np.array(transductive_test_edges))
    np.save(save_path + 'LLGF_' + output_dataset + '_test_neg.npy', np.array(transductive_test_edges_false))
    
    np.save(save_path + 'LLGF_' + output_dataset +  '_val_pos.npy', np.array(transductive_valid_edges))
    np.save(save_path + 'LLGF_' + output_dataset + '_val_neg.npy', np.array(transductive_val_edges_false))
    
    
    
    np.save(save_path + 'LLGF_' + output_dataset + '_ind_train_pos.npy', np.array(inductive_train_edges))
    
    np.save(save_path + 'LLGF_' + output_dataset + '_ind_test_pos.npy', np.array(inductive_test_edges))
    np.save(save_path + 'LLGF_' + output_dataset + '_ind_test_neg.npy', np.array(inductive_test_edges_false))
    
    np.save(save_path + 'LLGF_' + output_dataset +  '_ind_val_pos.npy', np.array(inductive_valid_edges))
    np.save(save_path + 'LLGF_' + output_dataset + '_ind_val_neg.npy', np.array(inductive_val_edges_false))



print("Loading Data")
# read full data


if "cora" in input_dataset:
    feat_data , adjacency_matrix = load_cora()
elif "ACM" in input_dataset:
    feat_data , adjacency_matrix = load_ACM()       
elif "IMDB" in input_dataset:
    feat_data , adjacency_matrix = load_IMDB()      
elif "citeseer" in input_dataset:
    feat_data , adjacency_matrix = load_citeseer() 
elif "photos" in input_dataset:
    feat_data, adjacency_matrix = load_photos()
    feat_data = feat_data.todense()
elif "computers" in input_dataset:
    feat_data, adjacency_matrix = load_computers()
    feat_data = feat_data.todense()        

#return_all(feat_data, adjacency_matrix)
return_trans_ind(feat_data, adjacency_matrix, output_dataset)
# read full subgraph
#print("Loading full subgragh")
# subgraphs = torch.load('subgraphs.pt')
# subgraphs = get_subgraph(feat_data, adjacency_matrix)
# torch.save(subgraphs, 'subgraphs.pt')

# divide into trandusctive and inductive
# print("Extracting transductive and inductive subgraphs")
# transductive_subgraphs, nodes_to_be_removed = extract_transductive_subgraph(subgraphs, adjacency_matrix)
# inductive_subgraphs = extract_inductive_subgraph(adjacency_matrix, nodes_to_be_removed)





# transductive_train_edges, transductive_valid_edges, transductive_test_edges = get_pos_edges_from_subraph(transductive_subgraphs)
# inductive_train_edges, inductive_valid_edges, inductive_test_edges = get_pos_edges_from_subraph(inductive_subgraphs)


#all_train_edges, all_val_edges, all_test_edges = get_pos_edges_from_edges(all_edges)

# # split subgraphs into test/train/valid
# transductive_train_subgraphs, transductive_valid_subgraphs, transductive_test_subgraphs = split_subgraph_to_train_test(transductive_subgraphs)
# inductive_train_subgraphs, inductive_valid_subgraphs, inductive_test_subgraphs = split_subgraph_to_train_test(inductive_subgraphs)




# # # extract the edges of each subraph and save them to .npy file
# transductive_train_edges = get_edges_from_subgraphs(transductive_train_subgraphs)
# transductive_valid_edges = get_edges_from_subgraphs(transductive_valid_subgraphs)
# transductive_test_edges = get_edges_from_subgraphs(transductive_test_subgraphs)

# inductive_train_edges = get_edges_from_subgraphs(inductive_train_subgraphs)
# inductive_valid_edges = get_edges_from_subgraphs(inductive_valid_subgraphs)
# inductive_test_edges = get_edges_from_subgraphs(inductive_test_subgraphs)



# get false edges


#all_train_edges_false, all_test_edges_false, all_val_edges_false = get_false_edges(all_test_edges, all_train_edges, all_val_edges, node_ids )


# save files
print("save files")



# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'/test.txt', make_compatible_to_grail(transductive_test_edges), delimiter='\t' , fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'/train.txt',  make_compatible_to_grail(transductive_train_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'/valid.txt',  make_compatible_to_grail(transductive_valid_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'_ind'+'/valid.txt',  make_compatible_to_grail(inductive_valid_edges), delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'_ind'+'/test.txt', make_compatible_to_grail(inductive_test_edges),  delimiter='\t', fmt='%s')

# np.savetxt('/localhome/pnaddaf/Desktop/parmis/grail-master/data/'+input_dataset+'_v1'+'_ind'+'/train.txt', make_compatible_to_grail(inductive_train_edges),  delimiter='\t', fmt='%s')



