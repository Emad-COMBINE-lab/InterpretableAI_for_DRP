#!/bin/bash

start_time=$SECONDS

echo "running pre-trained model......"

python main.py --dataroot ../../input_data/ \
--outroot results/original_CDS/ \
--hyproot best_hyp_original/KEGG/model_hyp_LCO.txt \
--pathway KEGG \
--foldtype cl \
--modeltype original \
--run_pretrained \
--modelroot ../../../../thesis_v2/results_CC/KEGG/ConsDeepSignaling/LCO/model_weights_LCO_4_epoch22.pt

echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
