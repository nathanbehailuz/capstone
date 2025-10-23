for dataset in syn1 syn2 syn3 syn4 mutag ba2; do
    sbatch slurm/${dataset}_ppo.slurm
done