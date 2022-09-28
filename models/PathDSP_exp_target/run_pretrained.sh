#!/bin/bash

start_time=$SECONDS

echo "running pre-trained model......"

python main.py --dataroot ../../input_data/ \
--outroot results/ \
--hyproot best_hyp/KEGG/model_hyp_LCO.txt \
--pathway KEGG \
--foldtype cl \
--run_pretrained \
--modelroot ../../../../thesis_v2/results/KEGG/PathDSP_exp_fp/ic50_nonorm/model_weights/model_weights_LCO_4.pt

echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
