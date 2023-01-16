for j in "photos"
do
#rm -r "./datasets_LLGF/LLGF_${j}_new_ind"
#rm -r "./datasets_LLGF/LLGF_${j}_new"
#python baseline_split_fully_inductive.py --dataset "${j}" --semi_inductive "False"
#python seal_splitting_data.py --dataset "LLGF_${j}_new"
python baseline_multi_link_preprocess.py --dataset "${j}" --semi_inductive "False"

for i in {1..50}
do
    
    python baseline_inductive.py --dataset "LLGF_${j}_new_ind_${i}"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_x.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_test_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_test_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_val_pos.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_val_neg.npy"
    rm -r "./datasets_LLGF/LLGF_${j}_new_ind_${i}_train_pos.npy"
done
done