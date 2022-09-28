#!/bin/bash

start_time=$SECONDS

echo "running model from scratch with specified hyperparameters......"

python main.py --dataroot ../../input_data/ \
--outroot results/target/ \
--hyproot best_hyp_target/PID/model_hyp_LCO.txt \
--pathway PID \
--foldtype cl \
--drug_feature_type target \


echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"