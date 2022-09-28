#!/bin/bash

start_time=$SECONDS

echo "running pre-trained model......"

python main.py --dataroot ../../input_data/ \
--outroot results/fp/ \
--hyproot best_hyp_fp/KEGG/model_hyp_LCO.txt \
--pathway KEGG \
--foldtype cl \
--drug_feature_type fp \
--run_pretrained \
--modelroot ../../../../thesis_v2/results/KEGG/HiDRA/ic50_nonorm/model_weights/model_weights_LCO_4_fp.pt

echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
