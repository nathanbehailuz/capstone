#!/bin/bash

# BA2 Dataset - Optimized PPO Configuration
# g_lr = 3e-4, ppo_epochs = 3, minibatch_size ≈ 1024, clip_eps = 0.1
# entropy_coef = 1e-3 → decay to 1e-4 → 0
# GAE(λ=0.95), global advantage normalization
# target_kl = 0.03 early stop, grad clip 1.0
# Potential-based frontier shaping (β = 0.05)
# Remove rollout-based step rewards (n_rollouts = 0)

PYTHON=/anaconda/envs/py37_pytorch/bin/python3

for i in {0..9}
do
   echo "Starting BA2 PPO optimized run with seed $i"
   python run_RG_explainer.py \
     --dataset ba2 \
     --seed $i \
     --n_epochs 10 \
     --policy PPO \
     --g_lr 3e-4 \
     --n_rollouts 0 \
     --entropy_coef 1e-3 \
     --ppo_epochs 3 \
     --minibatch_size 1024 \
     --clip_eps 0.1 \
     --target_kl 0.03 \
     --grad_clip 1.0 \
     --gae_lambda 0.95 \
     --frontier_shaping_beta 0.05 \
     --entropy_decay_start 1e-3 \
     --entropy_decay_end 1e-4 \
     --entropy_decay_final 0.0 \
     --size_reg 0.01 \
     --sim_reg 0.01 \
     --radius_penalty 0.01 \
     > ba2_ppo_optimized_seed_${i}.txt 2>&1
   
   echo "Completed BA2 PPO optimized run with seed $i"
done

echo "All BA2 PPO optimized runs completed!"
