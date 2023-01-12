# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb
import csv

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score, recall_score, precision_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *


class SEALDataset(InMemoryDataset):
    def __init__(self, sub_list,  root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.sub_list = sub_list
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs_sublist(
            self.sub_list, pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs_sublist(
            self.sub_list, neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None
        
    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop, 
                             self.max_nodes_per_hop, node_features=self.data.x, 
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results



def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results
        

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)
    
    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    
    prediction_test = test_pred
    pred = np.array(torch.sigmoid(test_pred))
    pred[pred>.5] = 1
    pred[pred < .5] = 0
    pred_test = pred.astype(int)
    
    
    prediction_val = val_pred
    pred = np.array(torch.sigmoid(val_pred))
    pred[pred>.5] = 1
    pred[pred < .5] = 0
    pred_val = pred.astype(int)
    
    
    precision_val = precision_score(y_pred= pred_val,  y_true= val_true)
    precision_test = precision_score(y_pred=pred_test, y_true= test_true)
    print("\nVAL precision:",str(precision_val),"TEST precision:",str(precision_test))
    
    recall_val = recall_score(y_pred= pred_val, y_true= val_true)
    recall_test = recall_score(y_pred= pred_test, y_true= test_true)
    print("VAL recall:",str(recall_val),"TEST recall:",str(recall_test))
    
    
    acc_test = accuracy_score(y_pred= pred_test, y_true= test_true, normalize= True)
    acc_val = accuracy_score(y_pred= pred_val, y_true= val_true, normalize= True)
    print("VAL acc:",str(acc_val),"TEST acc:",str(acc_test))
    
    ap_val = average_precision_score(val_true, val_pred)
    ap_test = average_precision_score(test_true, test_pred)
    print("VAL AP:",str(ap_val),"TEST AP:",str(ap_test))

    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    print("VAL AUC:", str(valid_auc), "TEST AUC:", str(test_auc))
    
    test_pred_sig = torch.sigmoid(test_pred)
    
    if "_ind_" in args.dataset:
        #cll_test = (torch.log (torch.sum(torch.log(np.concatenate((test_pred_sig[np.array(test_true) == 1], 1-test_pred_sig[np.array(test_true) == 0])))))).item()
        cll_test = np.sum(np.log(np.concatenate((test_pred_sig[np.array(test_true) == 1], 1-test_pred_sig[np.array(test_true) == 0])))) #log e^a = a
        
    else:

        cll_test = np.mean(np.log((np.concatenate((test_pred_sig[np.array(test_true) == 1], 1-test_pred_sig[np.array(test_true) == 0])))))

    print("Conditional Likelihood:", str(cll_test))
    
    #hr_ind = np.argpartition(np.array(prediction_test), -1*len(pred_test)//2)[-1*len(pred_test)//2:]
    hr_ind = np.argpartition(np.array(prediction_test), -1*len(pred_test)//5)[-1*len(pred_test)//5:]
    HR_test = precision_score(y_pred=np.array(pred_test)[hr_ind], y_true=np.array(test_true)[hr_ind])
    print("HR:" ,str(HR_test))
    
    results = {}
    results['AUC'] = (valid_auc, test_auc)
    
    with open('results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset[5:]])
        writer.writerow([precision_val, recall_val, acc_val, ap_val, valid_auc, cll_test, HR_test])
    # results['Precision'] = (precision_val, precision_test)
    # results['Recall'] = (recall_val, recall_test)
    # results['AP'] = (ap_val, ap_test)
    # results['ACC'] = (acc_val, acc_test)

    return results
        
 

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--dataset', type=str, default='LLGF_photos_new_ind')
parser.add_argument('--fast_split', action='store_true', 
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=1)
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--num_workers', type=int, default=16, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', default= True, 
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true', 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None, 
                    help="test a link prediction heuristic (CN or AA)")
args = parser.parse_args()
print(args)


if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    f_name = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    args.data_appendix = f_name + args.dataset
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # Backup python files.
    copy('seal_link_pred.py', args.res_dir)
    copy('utils.py', args.res_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
elif args.dataset.startswith('LLGF'):
    def datasetsSnapShot(dataset_name):
        train_pos = np.load(dataset_name + "_train_pos.npy", allow_pickle=True)
        val_pos = np.load(dataset_name + "_val_pos.npy", allow_pickle=True)
        test_pos = np.load(dataset_name + "_test_pos.npy", allow_pickle=True)
        val_neg = np.load(dataset_name + "_val_neg.npy", allow_pickle=True)
        test_neg = np.load(dataset_name + "_test_neg.npy", allow_pickle=True)
        x = np.load(dataset_name + "_x.npy", allow_pickle=True)
        return torch.tensor(train_pos.transpose(),dtype=torch.int64), \
               torch.tensor(val_pos.transpose(),dtype=torch.int64),\
               torch.tensor(test_pos.transpose(),dtype=torch.int64),\
               torch.tensor(val_neg.transpose(),dtype=torch.int64),\
               torch.tensor(test_neg.transpose(),dtype= torch.int64)\
            ,torch.tensor(x)


    def do_edge_split_LLGF(data, train_pos, val_pos, test_pos, val_neg,test_neg):

        random.seed(234)
        torch.manual_seed(234)

        data.edge_index = None
        data.val_pos_edge_index= val_pos
        data.test_pos_edge_index = test_pos

        noneNegative = torch.cat((train_pos,val_pos,test_pos, to_undirected(val_neg), to_undirected(test_neg)),dim=1)
        data.train_pos_edge_index = to_undirected(train_pos)
        data.val_neg_edge_index = val_neg
        data.test_neg_edge_index = test_neg

        # Negative edges.
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[noneNegative[0],noneNegative[1]] = 0

        data.train_neg_adj_mask =neg_adj_mask

        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))



        split_edge = {'train': {}, 'valid': {}, 'test': {}}
        split_edge['train']['edge'] = data.train_pos_edge_index.t()
        split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
        split_edge['valid']['edge'] = data.val_pos_edge_index.t()
        split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
        split_edge['test']['edge'] = data.test_pos_edge_index.t()
        split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
        return split_edge

    path  = osp.join('/localhome/pnaddaf/Desktop/parmis/SEAl_miror/datasets_LLGF', args.dataset)
    # read the data with same split of LLFG
    train_pos, val_pos,test_pos,val_neg,test_neg,x = datasetsSnapShot(path)
    #all edges in graph
    edges_index = torch.cat([train_pos, val_pos, test_pos], dim=1)
    edges_index = to_undirected(edges_index)
    data = Data(x, edges_index)

    split_edge = do_edge_split_LLGF(data, train_pos, val_pos, test_pos, val_neg,test_neg)
    data.edge_index = split_edge['train']['edge'].t()

else:
    path = osp.join('dataset', args.dataset)
    dataset = Planetoid(path, args.dataset)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()

if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:  # assume other datasets are undirected
    args.eval_metric = 'auc'
    directed = False

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    if not directed:
        val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
if args.eval_metric == 'hits':
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
        # 'Precision': Logger(args.runs, args),
        # 'Recall': Logger(args.runs, args),
        # 'AP': Logger(args.runs, args),
        # 'ACC': Logger(args.runs, args),
    }
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = torch.cat([pos_val_pred, neg_val_pred])
        val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                              torch.zeros(neg_val_pred.size(0), dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                              torch.zeros(neg_test_pred.size(0), dtype=int)])
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    for key, result in results.items():
        loggers[key].add_result(0, result)
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    pdb.set_trace()
    exit()


# SEAL.
if args.dataset.startswith('LLGF'):
    path = "datasets_LLGF_r/" + '_seal{}'.format(args.data_appendix)
else:
    path = dataset.root + '_seal{}'.format(args.data_appendix)
use_coalesce = True if args.dataset == 'ogbl-collab' else False
if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
    args.num_workers = 0

dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
train_dataset = eval(dataset_class)(
    0,
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.train_percent, 
    split='train', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
) 
if False:  # visualize some graphs
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    i = 0
    save_folder = '/localhome/pnaddaf/Desktop/parmis/SEAl_miror/visualizations/'
    for g in loader:
        
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        g = g.to(device)
        node_size = 100
        with_labels = True
        G = to_networkx(g, node_attrs=['z'])
        labels = {i: G.nodes[i]['z'] for i in range(len(G))}
        nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                labels=labels)
        f.savefig( save_folder +  args.dataset + '/tmp_vis' +  str(i) +'.png')
        i += 1

dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
val_dataset = eval(dataset_class)(
    0,
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.val_percent, 
    split='valid', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
)
dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
test_dataset = eval(dataset_class)(
    0,
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.test_percent, 
    split='test', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
)

max_z = 1000  # set a large max_z so that every z has embeddings to look up

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                         num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

for run in range(args.runs):
    if args.model == 'DGCNN':
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, args.dynamic_train, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,  
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GIN':
        model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}', file=f)

    start_epoch = 1
    if args.only_test:
        print("Loading saved model")
        if "cora" in args.dataset:
            checkpoint = torch.load('modelLLGF_cora_new.pth')
        elif "ACM" in args.dataset:
            checkpoint = torch.load('modelLLGF_ACM_new.pth')       
        elif "IMDB" in args.dataset:
            checkpoint = torch.load('modelLLGF_IMDB_new.pth')       
        elif "citeseer" in args.dataset:
            checkpoint = torch.load('modelLLGF_citeseer_new.pth')
        elif "photos" in args.dataset:
            checkpoint = torch.load('modelLLGF_photos_new.pth')
        elif "computers" in args.dataset:
            checkpoint = torch.load('modelLLGF_computers_new.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for sub_list in range(100):
            if not sub_list == 0:
                dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
                train_dataset = eval(dataset_class)(
                    sub_list,
                    path, 
                    data, 
                    split_edge, 
                    num_hops=args.num_hops, 
                    percent=args.train_percent, 
                    split='train', 
                    use_coalesce=use_coalesce, 
                    node_label=args.node_label, 
                    ratio_per_hop=args.ratio_per_hop, 
                    max_nodes_per_hop=args.max_nodes_per_hop, 
                    directed=directed, 
                ) 
                
                dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
                val_dataset = eval(dataset_class)(
                    sub_list,
                    path, 
                    data, 
                    split_edge, 
                    num_hops=args.num_hops, 
                    percent=args.val_percent, 
                    split='valid', 
                    use_coalesce=use_coalesce, 
                    node_label=args.node_label, 
                    ratio_per_hop=args.ratio_per_hop, 
                    max_nodes_per_hop=args.max_nodes_per_hop, 
                    directed=directed, 
                )
                dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
                test_dataset = eval(dataset_class)(
                    sub_list,
                    path, 
                    data, 
                    split_edge, 
                    num_hops=args.num_hops, 
                    percent=args.test_percent, 
                    split='test', 
                    use_coalesce=use_coalesce, 
                    node_label=args.node_label, 
                    ratio_per_hop=args.ratio_per_hop, 
                    max_nodes_per_hop=args.max_nodes_per_hop, 
                    directed=directed, 
                )
                
                max_z = 1000  # set a large max_z so that every z has embeddings to look up
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                          shuffle=True, num_workers=args.num_workers)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers)

            print("Testing on the inductive dataset")
            results = test()
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, test_res = result
                print(key)
                print(f'Run: {run + 1:02d}, '
                      f'Valid: {100 * valid_res:.2f}%, '
                      f'Test: {100 * test_res:.2f}%')


#     for key in loggers.keys():
#         print(key)
#         loggers[key].print_statistics(run)
#         with open(log_file, 'a') as f:
#             print(key, file=f)
#             loggers[key].print_statistics(run, f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
print(f'Total number of parameters is {total_params}')
print(f'Results are saved in {args.res_dir}')


