#!/bin/bash
PYTHON=/anaconda/envs/py37_pytorch/bin/python3
for i in {0..9}
do
   python run_RG_explainer.py --dataset syn1 --seed $i --policy REINFORCE --n_epochs 10 --size_reg 0.01 --sim_reg 1.0 --radius_penalty 0.1 > syn1_reinforce_seed_$i.txt 2>&1
done
