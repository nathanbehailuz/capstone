
# Detailed Explanation of the Capstone Repository

Based on my analysis of the codebase, here's a comprehensive explanation of the repository structure, PPO implementation, and the SLURM workflow:

## 1. Overall File and Folder Structure

### **Repository Overview**
This is a **Reinforcement Learning-based Graph Neural Network Explainer (RC-Explainer)** that uses PPO to generate interpretable subgraph explanations for GNN predictions, enhanced with causal reasoning capabilities.

### **Core Directory Structure**

```
capstone/
├── components/                 # Core RL and GNN components
│   ├── generator_copy.py      # Main PPO generator with causal integration
│   ├── generator_core.py      # Base Agent and GraphConv classes
│   ├── causal_explainer.py    # Causal mechanism implementation
│   ├── graph.py              # Graph utilities and operations
│   ├── layers.py             # Neural network layers
│   └── locator_seed.py       # Seed node locator for graph tasks
├── datasets/                  # Dataset loaders and utilities
│   ├── dataset_loaders.py    # Load synthetic/molecular datasets
│   ├── ground_truth_loaders.py # Ground truth explanations
│   └── Mutagenicity/         # Molecular dataset
├── models/                    # GNN model implementations
│   ├── GNN_paper.py          # Graph Neural Network models
│   ├── PG_paper.py           # Policy Gradient models
│   └── model_selector.py     # Model selection utilities
├── evaluation/                # Evaluation metrics
│   ├── AUCEvaluation.py      # AUC-based evaluation
│   ├── BaseEvaluation.py     # Base evaluation class
│   └── EfficiencyEvaluation.py # Efficiency metrics
├── slurm/                     # SLURM job scripts for HPC
├── sh/                       # Shell scripts for batch runs
├── analysis/                  # Log analysis toolkit
├── results/                  # Experimental results (text files)
├── slurm_out/               # SLURM job output logs
├── ckpts/                   # Model checkpoints
└── run_RG_explainer.py      # Main execution script
```

### **Key Datasets Supported**
- **Synthetic**: `syn1`, `syn2`, `syn3`, `syn4` (with known motif structures)
- **Real-world**: `ba2` (Barabási-Albert), `mutag` (molecular graphs)


## 2. PPO Implementation

### **Reinforcement Learning Framework**

The system uses **Proximal Policy Optimization (PPO)** for sequential subgraph generation. Here's how it works:

### **Core Components**

#### **A. Agent Architecture** (`components/generator_core.py`)
```python
class Agent(nn.Module):
    def __init__(self, hidden_size, with_attr=False, attr_size=-1, stop_bias=0.0):
        # Neural networks for policy and value function
        self.seed_embedding = nn.Embedding(1, hidden_size)
        self.node_embedding = nn.Embedding(max_nodes, hidden_size)
        self.input_mapping = nn.Linear(hidden_size, hidden_size)
        self.node_score_layer = nn.Linear(hidden_size, 1)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.stopping_score_layer = nn.Linear(hidden_size, 2)  # STOP action
```

#### **B. Environment** (`components/generator_copy.py`)
```python
class ExpansionEnv:
    def __init__(self, graph, selected_nodes, max_size):
        # Sequential subgraph expansion environment
        # Maintains trajectories and termination conditions
```

### **PPO Training Process**

#### **1. Trajectory Generation**
```python
def _sample_trajectories(self, env):
    # Sequential node selection:
    # 1. Start with seed node
    # 2. At each step, find boundary nodes (candidates)
    # 3. Agent selects next node or STOP action
    # 4. Continue until max_size or no more candidates
```

#### **2. PPO Loss Computation**
```python
def train_from_rewards(self, seeds, fn, policy="PPO"):
    # GAE (Generalized Advantage Estimation)
    gamma = 0.99
    gae_lambda = 0.95
    
    # Compute advantages and returns
    advantages = compute_gae_advantages(rewards, values, gamma, gae_lambda)
    
    # PPO surrogate loss with clipping
    ratios = torch.exp(current_logps - old_logps)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value function loss with clipping
    value_loss = compute_clipped_value_loss(current_values, old_values, returns)
```

#### **3. Key PPO Parameters**
- **Clip Range**: `0.1` (prevents large policy updates)
- **GAE Lambda**: `0.95` (bias-variance tradeoff)
- **Target KL**: `0.0075` (early stopping threshold)
- **Minibatch Size**: `256` (for stability)
- **PPO Epochs**: `4` (multiple updates per batch)

### **Action Space and Rewards**

#### **Action Space**
- **Node Selection**: Choose from boundary nodes (neighbors of current subgraph)
- **STOP Action**: Terminate subgraph generation
- **Dynamic**: Action space changes as subgraph grows

#### **Reward Function**
```python
def score_fn(self, selected_nodes):
    # Multi-component reward:
    reward = -prediction_loss    # Fidelity to original prediction
    reward -= size_penalty       # Size regularization
    reward -= similarity_loss    # Similarity to original embeddings
    reward -= radius_penalty     # Distance from seed node
    reward -= step_cost          # Efficiency penalty
    
    # Causal reward integration (if enabled)
    if use_causal:
        causal_reward = compute_causal_reward(selected_nodes)
        reward = combine_rewards(reward, causal_reward, causal_weight)
```


## 3. Causal Mechanism Integration

### **Counterfactual Reasoning Approach**

The system implements a sophisticated causal mechanism that uses **counterfactual reasoning** to identify truly causal edges:

#### **A. Core Concept**
```python
# Factual: GNN prediction with full graph
full_prediction = gnn_model(full_graph_features, full_edge_index)

# Counterfactual: GNN prediction with selected edges only  
subgraph_prediction = gnn_model(full_graph_features, selected_edge_index)

# Causal Effect: Difference represents causal contribution
causal_effect = full_prediction - subgraph_prediction
```

#### **B. Implementation Classes**

##### **NodeLevelCausalExplainer** (`components/causal_explainer.py`)
```python
class NodeLevelCausalExplainer(nn.Module):
    def forward(self, graph, state, target_y):
        # 1. Compute factual representations (all edges)
        factual_reps = self.model.get_node_reps(graph.x, graph.edge_index)
        
        # 2. Compute counterfactual representations (selected edges only)
        selected_edges = graph.edge_index[:, state]
        counterfactual_reps = self.model.get_node_reps(graph.x, selected_edges)
        
        # 3. Causal effect = difference
        causal_reps = factual_reps - counterfactual_reps
        
        # 4. Generate edge importance scores
        return self.edge_action_prob_generator(causal_reps)
```

##### **GraphLevelCausalExplainer**
- Similar approach but uses graph-level representations
- More efficient for graph classification tasks

#### **C. Causal Reward Integration**
```python
class CausalRewardCalculator:
    def compute_causal_reward(self, graph, selected_nodes, target_y):
        # Compute true causal effect using CE-difference method
        # Control: subgraph prediction
        # Treatment: baseline (no edges) prediction
        # ICE = CE(control) - CE(treatment)
        
        ctrl_logp = gnn_model(subgraph_features, subgraph_edges)
        baseline_logp = gnn_model(features, empty_edges)
        
        causal_delta = ctrl_logp[target_y] - baseline_logp[target_y]
        return causal_delta + correctness_bonus
```

#### **D. Integration with PPO**
```python
def _compute_causal_bias(self, valid_index, trajectories, batch_candidates):
    # During action selection, compute causal importance for each candidate
    for candidate in candidates:
        # Compute ICE score for adding this candidate
        ice_score = compute_ice_score(current_nodes, candidate)
        causal_bias.append(ice_score)
    
    # Apply bias to action selection
    logits = logits + causal_weight * causal_bias
```


## 4. SLURM → Code → Text Files Full Lifecycle

### **Complete Workflow from Job Submission to Results**

#### **A. SLURM Job Submission**

##### **Job Script Structure** (`slurm/syn3_ppo.slurm`)
```bash
#!/bin/bash
#SBATCH -J rgx_syn3_ppo          # Job name
#SBATCH -p nvidia                 # Partition (GPU nodes)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -N 1 -n 1 -c 4           # 1 node, 1 task, 4 CPUs
#SBATCH --mem=24G                # 24GB RAM
#SBATCH -t 7:00:00               # 7-hour time limit
#SBATCH --array=6-7              # Array job (seeds 6-7)
#SBATCH -o slurm_out/%x_%A_%a.out # Output log
#SBATCH -e slurm_out/%x_%A_%a.err # Error log
```

##### **Environment Setup**
```bash
# Module loading
module purge
module load miniconda/3-4.11.0

# Conda environment activation
conda activate /home/nz2212/.conda/envs/rlgn

# GPU and debugging flags
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

##### **Parameter Configuration**
```bash
SEED="${SLURM_ARRAY_TASK_ID}"    # Array task ID becomes seed
DATASET="syn3"
POLICY="PPO"
N_EPOCHS=5
SIZE_REG=0.01
SIM_REG=0.01
RADIUS_PENALTY=0.1
CAUSAL_WEIGHT=0.3
```

#### **B. Code Execution Flow**

##### **1. Job Submission**
```bash
# Submit job array
sbatch slurm/syn3_ppo.slurm

# This creates multiple jobs:
# Job 12338466_6 (seed=6)
# Job 12338466_7 (seed=7)
```

##### **2. SLURM Resource Allocation**
- Each job gets: 1 GPU, 4 CPUs, 24GB RAM
- Jobs run on nvidia partition nodes
- Output directed to `slurm_out/` directory

##### **3. Python Execution**
```bash
# Main command executed by SLURM
srun --export=ALL --gres=gpu:1 -c 4 \
  "${PYBIN}" run_RG_explainer.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --policy "${POLICY}" \
    --n_epochs "${N_EPOCHS}" \
    [additional parameters...] \
  >> "$RESULT_FILE" 2>&1
```

#### **C. File Generation and Output Flow**

##### **1. SLURM Output Files** (`slurm_out/`)
```
slurm_out/
├── rgx_syn3_ppo_12338466_6.out    # Job 6 stdout
├── rgx_syn3_ppo_12338466_6.err    # Job 6 stderr
├── rgx_syn3_ppo_12338466_7.out    # Job 7 stdout
└── rgx_syn3_ppo_12338466_7.err    # Job 7 stderr
```

##### **2. Results Text Files** (`results/`)
```
results/
├── syn3_ppo_seed_0.txt    # Training logs and metrics
├── syn3_ppo_seed_1.txt    # Each seed gets separate file
├── syn3_ppo_seed_2.txt
└── ...
```

##### **3. Model Checkpoints** (`ckpts/`)
```
ckpts/syn3/
├── 20251014194533/         # Timestamp-based directory
│   ├── settings.json       # Configuration
│   ├── 00001_0.85.pth      # Epoch checkpoints
│   └── 00002_0.87.pth
```

#### **D. Detailed Text File Contents**

##### **Training Log Structure** (`results/syn3_ppo_seed_0.txt`)
```text
[Tue Oct 14 19:45:26 +04 2025] Starting seed=0
CUDA_VISIBLE_DEVICES=0
NVIDIA-SMI output...
= = = = = = = = = = = = = = = = = = = 
Namespace(dataset='syn3', model='GNN', seed=0, ...)
##  Starting Time: 2025-10-14 19:45:33
[DEBUG] Current working directory: /scratch/nz2212/capstone
[DEBUG] SLURM_JOB_ID: 12338466
Loading syn3 dataset
This model obtained: Train Acc: 0.9425, Val Acc: 0.9770, Test Acc: 0.9432.
Eval Seeds range(511, 871, 6)
Prediction Error for Whole Graph: 0.0539609644087256
Prediction Error for Eval Seeds: 0.1
[CAUSAL] Causal mechanism check passed: use_causal=True
[CAUSAL] Initialized NodeLevelCausalExplainer with 2 classes, weight 0.3
CODE: 58D7D6119138022A52ABDA51D7577C6E
Load the pre-trained model!
====================
[Epoch    1]
Entropy coefficient: 0.001000
Effective causal weight: 0.3000
Update G
[CAUSAL] Original scores: -6.2365, Causal scores: 0.0000, Combined: -4.3656
Reward=-6.24  PLoss= 0.56  Length=12.1  EarlyStopRate=0.000  AvgSize=12.1
Elapsed Time: 45.2s
[EVAL-Epoch    1] AUC=0.8543
[Epoch    2]
...
```

##### **Key Metrics Logged**
- **Training Metrics**: Reward, Policy Loss, Episode Length, Early Stop Rate
- **PPO Diagnostics**: KL Divergence, Clip Fraction, Entropy, Explained Variance
- **Causal Metrics**: Original scores, Causal scores, Combined scores
- **Evaluation Metrics**: AUC scores, Size efficiency
- **System Info**: GPU usage, memory, timing

#### **E. Complete Data Flow**

```
1. SLURM Job Submission
   ↓
2. Resource Allocation (GPU + CPU + RAM)
   ↓
3. Environment Setup (Conda + CUDA)
   ↓
4. Python Execution (run_RG_explainer.py)
   ↓
5. Data Loading (Dataset + Pretrained GNN)
   ↓
6. Model Initialization (PPO Agent + Causal Explainer)
   ↓
7. Training Loop (PPO + Causal Rewards)
   ↓
8. Evaluation (AUC Computation)
   ↓
9. Output Generation:
   - Training logs → results/syn3_ppo_seed_X.txt
   - SLURM logs → slurm_out/rgx_syn3_ppo_XXXXX_X.out/err
   - Model checkpoints → ckpts/syn3/TIMESTAMP/
   - Analysis data → analysis/ directory
```



## 5. Complete Training Pipeline

### **End-to-End Training Flow**

#### **A. Data Loading and Preprocessing** (`run_RG_explainer.py`)

##### **1. Dataset Loading**
```python
def load_data(self):
    # Load complete dataset
    graphs, features, ground_truth_labels, _, _, test_mask = dataset_loaders.load_dataset(args.dataset)
    
    # Determine task type
    if isinstance(graphs, list):
        task = "graph_task"  # Graph classification
    else:
        task = "node_task"   # Node classification
    
    return graphs, features, ground_truth_labels, task
```

##### **2. Model Loading**
```python
# Load pretrained GNN model
self.trained_model, checkpoint = model_selector.model_selector(
    args.model, args.dataset, pretrained=True, return_checkpoint=True
)

# Load ground truth explanations for evaluation
self.explanation_labels, explain_eval_seeds = ground_truth_loaders.load_dataset_ground_truth(args.dataset)
```

#### **B. Model Initialization**

##### **1. PPO Agent Setup**
```python
def init_g(self):
    # Create PPO agent
    g_model = Agent(args.hidden_size, args.with_attr, self.nodefeats_size)
    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr)
    
    # Initialize Generator with PPO parameters
    g = Generator(
        self.graphs, g_model, g_optimizer, device,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda,
        use_causal=args.use_causal,
        causal_weight=args.causal_weight
    )
```

##### **2. Causal Mechanism Integration**
```python
def init_causal_mechanism(self):
    if use_causal:
        # Create causal explainer based on task type
        if self.task == "node_task":
            causal_explainer = NodeLevelCausalExplainer(
                gnn_model=self.trained_model,
                num_labels=num_classes,
                hidden_size=args.hidden_size
            )
        else:
            causal_explainer = GraphLevelCausalExplainer(...)
        
        # Initialize causal reward calculator
        g.init_causal_reward_calculator(self.trained_model)
```

#### **C. Training Loop**

##### **1. Pretraining Phase**
```python
def pretrain(self):
    # Pretrain generator with supervised examples
    pretrain_g(self.g, self.pretrain_nodes, self.pretrain_eval_nodes, 
               args.max_size, args.pretrain_g_batch_size, 
               args.pretrain_list, self.score_fn, self.args.dataset)
```

##### **2. Main Training Loop**
```python
def run(self):
    for i_epoch in range(self.args.n_epochs):
        # Update entropy coefficient (decay schedule)
        self.g.update_entropy_coef(i_epoch, self.args.n_epochs)
        
        # Update causal weight (warmup schedule)
        effective_causal_w = self.g.get_effective_causal_weight(i_epoch)
        
        # PPO training step
        self.train_g_step(g_it)
        
        # Evaluation
        if (i_epoch + 1) % self.args.eval_every == 0:
            auc = self.evaluate_and_print(f'Epoch {i_epoch+1:4d}')
            self.save(self.savedir / f'{i_epoch + 1:0>5d}_{metrics_string}.pth')
```

##### **3. PPO Training Step**
```python
def train_g_step(self, g_it):
    # Sample trajectories with current policy
    selected_nodes, r, policy_loss, length = self.g.train_from_rewards(
        np.array(self.eval_seeds)[shuffle_index], 
        self.score_fn, 
        policy=self.args.policy
    )
    
    # Log metrics
    early_stop_rate = self.g.get_early_stop_rate(selected_nodes)
    avg_size = np.mean([len([x for x in traj if x != 'EOS']) for traj in selected_nodes])
```

#### **D. Evaluation Process**

##### **1. Subgraph Generation**
```python
def evaluate_and_print(self, prefix=''):
    # Generate explanations for evaluation seeds
    pred_sgs = self.g.generate(self.eval_seeds)
    
    # Compute loss components
    p_loss, s_loss, sim_loss, r_p = self.eval_loss(pred_sgs)
```

##### **2. AUC Evaluation**
```python
def eval_loss(self, cs):
    # Prepare explanation masks
    explanations = [self.prepare_mask(self.subgraphs[i], x) for i, x in enumerate(pred_sgs)]
    
    # Compute AUC score against ground truth
    auc_score = self.auc_eval.get_score(explanations)
    return auc_score
```

#### **E. Key Training Features**

##### **1. Adaptive Scheduling**
- **Entropy Decay**: Starts at `1e-3`, decays to `1e-4`, then to `0`
- **Causal Weight Warmup**: Gradually increases from 0 to target weight
- **Learning Rate Scheduling**: Optional learning rate decay

##### **2. Reward Engineering**
```python
def score_fn(self, cs):
    # Multi-objective reward function
    v = -prediction_loss                    # Fidelity
    v += args.size_reg * size_penalty       # Size control
    v += args.sim_reg * similarity_loss     # Similarity
    v += args.radius_penalty * radius_penalty  # Locality
    
    # Causal integration
    if use_causal:
        causal_rewards = compute_causal_rewards(cs)
        combined_scores = (1 - causal_weight) * original_scores + causal_weight * causal_scores
    
    return combined_scores
```

##### **3. Early Stopping and Monitoring**
- **KL-based Early Stopping**: Stop PPO updates if KL divergence > target
- **Gradient Clipping**: Prevent exploding gradients
- **Value Function Clipping**: Stabilize value learning
- **Comprehensive Logging**: Training diagnostics, metrics, timing


## Summary

This capstone repository implements a sophisticated **Reinforcement Learning-based Graph Neural Network Explainer** that combines:

### **Key Innovations**
1. **PPO for Subgraph Generation**: Uses Proximal Policy Optimization to sequentially select nodes for explanations
2. **Causal Mechanism Integration**: Implements counterfactual reasoning to identify truly causal edges
3. **Multi-Dataset Support**: Works with synthetic and real-world graph datasets
4. **HPC Integration**: Comprehensive SLURM workflow for large-scale experiments

### **Technical Architecture**
- **RL Framework**: PPO with GAE, entropy scheduling, and gradient clipping
- **Causal Reasoning**: Factual vs counterfactual GNN representations
- **Evaluation**: AUC-based metrics against ground truth explanations
- **Scalability**: Array jobs, checkpointing, and comprehensive logging

### **Research Impact**
This system addresses the critical need for **interpretable AI** in graph neural networks by providing:
- **Faithful explanations** that preserve GNN predictions
- **Causal insights** through counterfactual analysis  
- **Scalable training** on HPC infrastructure
- **Reproducible results** with comprehensive logging

The repository represents a complete research pipeline from experimental design through SLURM job submission to detailed result analysis, making it a valuable contribution to the field of explainable AI for graph neural networks.