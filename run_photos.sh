for i in {1..10}
do
    python baseline_inductive.py --dataset "LLGF_photos_new_semi_ind_${i}"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_x.npy"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_test_pos.npy"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_test_neg.npy"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_val_pos.npy"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_val_neg.npy"
    rm -r "./datasets_LLGF/LLGF_photos_new_semi_ind_${i}_train_pos.npy"

done