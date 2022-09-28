#!/bin/bash

start_time=$SECONDS

echo "running model from scratch with specified hyperparameters......"

python main.py --dataroot ../../input_data/ \
--outroot results/ \
--hyproot best_hyp/KEGG/model_hyp_LDO.txt \
--pathway KEGG \
--foldtype drug \


echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"