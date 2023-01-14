#!/bin/bash

for j in "cora" "ACM" "IMDB" "citeseer"
do
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind"
    python baseline_split_fully_inductive.py --dataset "${j}" --semi_inductive "False"
    python baseline_inductive.py --dataset "LLGF_${j}_new_ind"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_x.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_test_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_test_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_val_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_val_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_train_pos.npy"
    
    
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind"
    python baseline_split_fully_inductive.py --dataset "${j}" --semi_inductive "True"
    python baseline_inductive.py --dataset "LLGF_${j}_new_semi_ind"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_x.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_test_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_test_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_val_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_val_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_train_pos.npy"

done
