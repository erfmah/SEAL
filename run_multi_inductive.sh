
#!/bin/bash

for j in "photos"

do

python baseline_split_fully_inductive.py --dataset "${j}" --semi_inductive "True"
python seal_link_pred.py --dataset "LLGF_${j}_new_ind"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_x.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_test_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_test_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_val_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_val_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_ind_train_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new"
rm -r "./datasets_LLGF/LLGF_${j}_new_x.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_test_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_test_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_val_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_val_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_train_pos.npy"

python baseline_multi_link_preprocess.py --dataset "${j}" --semi_inductive "True"


for i in {1..100}
do
python baseline_inductive.py --dataset "LLGF_${j}_new_semi_ind_${i}"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_x.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_test_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_test_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_val_pos.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_val_neg.npy"
rm -r "./datasets_LLGF/LLGF_${j}_new_semi_ind_${i}_train_pos.npy"

done
