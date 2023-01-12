#!/bin/bash

for j in "cora" "ACM" "IMDB" "citeseer"
do
    python baseline_split_fully_inductive.py --dataset "${j}" --semi_inductive "False"
    python seal_link_pred.py --dataset "LLGF_${j}_new"

done
