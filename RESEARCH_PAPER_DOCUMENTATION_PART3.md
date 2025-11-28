# Research Documentation Part 3: Hyperparameters, Datasets, Logging, and Technical Specifications

*Continuation of RESEARCH_PAPER_DOCUMENTATION_PART2.md*

---

## 12. HYPERPARAMETERS & TRAINING STRATEGY

### 12.1 Complete Hyperparameter Table

**PPO Core Parameters**:

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| `gamma` | \( \gamma \) | 0.99 | [0.9, 0.999] | Discount factor for returns |
| `gae_lambda` | \( \lambda \) | 0.95 | [0.9, 0.99] | GAE lambda parameter |
| `clip_range_policy` | \( \epsilon_{\pi} \) | 0.2 | [0.1, 0.3] | PPO policy clip range |
| `clip_range_value` | \( \epsilon_V \) | 0.25 | [0.1, 0.5] | Value function clip range |
| `ppo_epochs` | \( E \) | 4 | [2, 10] | Number of PPO update epochs |
| `minibatch_size` | \( M \) | 256 | [64, 512] | Samples per minibatch |
| `grad_clip` | \( c_{\text{grad}} \) | 1.0 | [0.5, 5.0] | Gradient clipping threshold |
| `lr` | \( \eta \) | 5e-5 | [1e-5, 1e-3] | Learning rate (Adam) |
| `kl_target` | \( \text{KL}_{\text{target}} \) | 0.0075 | [0.005, 0.02] | Target KL divergence |

**Entropy Scheduling**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `entropy_coeff_start` | 0.01 | [0.001, 0.1] | Initial entropy coefficient |
| `entropy_coeff_end` | 0.001 | [0.0001, 0.01] | Middle entropy coefficient |
| `entropy_decay_final` | 0.0 | [0.0, 0.001] | Final entropy coefficient |
| `entropy_decay_fraction` | 0.7 | [0.5, 0.9] | Fraction of epochs for first decay |

**Reward Shaping**:

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| `reward_scale` | \( \alpha \) | 10.0 | [1.0, 20.0] | Global reward scaling |
| `lambda_size` | \( \lambda_s \) | 0.02 | [0.0, 0.1] | Size penalty weight |
| `lambda_radius` | \( \lambda_r \) | 0.10 | [0.0, 0.5] | Radius penalty weight |
| `lambda_sim` | \( \lambda_{\text{sim}} \) | 0.005 | [0.0, 0.1] | Similarity penalty weight |
| `lambda_conf` | \( \lambda_c \) | 0.1 | [0.0, 0.5] | Confidence penalty weight |
| `target_size_k` | \( k \) | 8 | [5, 15] | Target subgraph size |
| `stop_bonus_ok` | \( b_+ \) | 0.10 | [0.0, 0.5] | Bonus for correct STOP |
| `stop_penalty_bad` | \( b_- \) | 0.40 | [0.0, 1.0] | Penalty for wrong STOP |
| `fidelity_thresh` | \( \tau_f \) | 0.05 | [0.0, 0.2] | Confidence tolerance for STOP |

**Causal Mechanism**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_causal` | True | {True, False} | Enable causal rewards |
| `causal_weight` | 0.5 | [0.0, 1.0] | Weight for causal term |
| `causal_warmup_epochs` | 5 | [0, 10] | Epochs to warmup causal weight |
| `causal_gamma` | 0.5 | [0.1, 0.9] | Discount for causal computation |

**Network Architecture**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hidden_size` | 64 | [32, 128] | Hidden dimension for Agent |
| `with_attr` | True | {True, False} | Use node attributes |
| `stop_bias` | 0.5 | [-1.0, 1.0] | Bias for STOP action |
| `n_rollouts` | 5 | [1, 10] | Rollouts per training step |
| `max_size` | 20 | [10, 30] | Max trajectory length |

**Pretraining**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pretrain_list` | 10 | [5, 20] | List-based iterations |
| `pretrain_set` | 25 | [10, 50] | Set-based iterations |
| `pretrain_g_batch_size` | 32 | [16, 64] | Batch size for pretraining |
| `random_edge_drop` | 0.0 | [0.0, 0.3] | Edge dropout probability |

**Training Schedule**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_epochs` | 30 | [10, 100] | Total training epochs |
| `g_batch_size` | 128 | [64, 256] | Batch size for PPO updates |
| `eval_every` | 1 | [1, 5] | Evaluation frequency (epochs) |
| `warmup_epochs` | 3 | [0, 10] | Warmup before regularization |
| `schedule_epochs` | 12 | [5, 20] | Epochs to ramp regularizers |

### 12.2 Lambda Schedule

**Purpose**: Gradually increase regularization penalties during training

**Formula** (`run_RG_explainer.py:93-100`):

```python
def linear_schedule(epoch, warmup, span, start, end):
    """
    Linear schedule helper for λ parameters.
    
    Phase 1 (0..warmup): λ = start
    Phase 2 (warmup..warmup+span): λ linearly increases from start to end
    Phase 3 (warmup+span..∞): λ = end
    """
    if epoch < warmup:
        return start
    t = min(max(epoch - warmup, 0), span)
    if span <= 0:
        return end
    return start + (end - start) * (t / span)
```

**Application** (`run_RG_explainer.py:103-118`):

```python
def get_lambda_schedule(epoch, args):
    # Size penalty: 0.0 → 0.02 (after warmup)
    lam_size = 0.0 if epoch < args.warmup_epochs else linear_schedule(
        epoch, args.warmup_epochs, args.schedule_epochs, 0.0, args.lambda_size_final
    )
    
    # Radius penalty: 0.0 → 0.10 (after warmup)
    lam_radius = 0.0 if epoch < args.warmup_epochs else linear_schedule(
        epoch, args.warmup_epochs, args.schedule_epochs, 0.0, args.lambda_radius_final
    )
    
    # Similarity penalty: 0.0 → 0.005 (gradual)
    lam_sim = linear_schedule(
        epoch, args.warmup_epochs, args.schedule_epochs, 
        args.lambda_sim_start, args.lambda_sim_final
    )
    
    # Entropy: 0.01 → 0.001 (gradual)
    entropy_c = linear_schedule(
        epoch, 0, args.warmup_epochs + args.schedule_epochs, 
        args.entropy_coeff_start, args.entropy_coeff_end
    )
    
    return lam_size, lam_radius, lam_sim, entropy_c
```

**Example Schedule** (warmup=3, schedule=12):

| Epoch | λ_size | λ_radius | λ_sim | entropy |
|-------|--------|----------|-------|---------|
| 0 | 0.00 | 0.00 | 0.000 | 0.0100 |
| 1 | 0.00 | 0.00 | 0.000 | 0.0093 |
| 2 | 0.00 | 0.00 | 0.000 | 0.0087 |
| 3 | 0.00 | 0.00 | 0.001 | 0.0080 |
| 6 | 0.01 | 0.03 | 0.002 | 0.0060 |
| 9 | 0.01 | 0.06 | 0.003 | 0.0040 |
| 12 | 0.02 | 0.08 | 0.004 | 0.0020 |
| 15 | 0.02 | 0.10 | 0.005 | 0.0010 |
| 30+ | 0.02 | 0.10 | 0.005 | 0.0010 |

**Rationale**:
- **Warmup** (epochs 0-2): Focus on prediction fidelity (no regularization)
- **Ramp** (epochs 3-15): Gradually add size/radius/similarity penalties
- **Stable** (epochs 15+): Full regularization for compact, interpretable explanations

### 12.3 Dataset-Specific Configurations

**Syn1 (node classification, house motifs)**:

```bash
--dataset syn1
--n_hop 3
--max_size 20
--target_size_k 8
--pretrain_list 10
--pretrain_set 25
--random_edge_drop 0.1
--g_batch_size 128
--n_epochs 30
```

**Syn2 (node classification, house motifs, more complex)**:

```bash
--dataset syn2
--n_hop 3
--max_size 20
--target_size_k 8
--pretrain_list 10
--pretrain_set 25
--random_edge_drop 0.1
--g_batch_size 128
--n_epochs 30
```

**BA2 (graph classification, 5-node motifs)**:

```bash
--dataset ba2
--n_hop 3
--max_size 20
--target_size_k 10  # Larger motifs
--pretrain_list 10
--pretrain_set 25
--random_edge_drop 0.1
--g_batch_size 128
--pretrain_l_iter 200
--n_epochs 30
```

**Mutag (graph classification, chemical motifs)**:

```bash
--dataset mutag
--n_hop 3
--max_size 20
--target_size_k 6  # Smaller motifs
--pretrain_list 5  # Reduced (large dataset)
--pretrain_set 15  # Reduced (large dataset)
--random_edge_drop 0.0  # Disabled (connectivity issues)
--g_batch_size 64  # Smaller batches
--pretrain_l_iter 50  # Reduced (expensive)
--pretrain_l_sample_rate 0.3  # Sample subset
--n_epochs 30
```

### 12.4 Recommended Tuning Guidelines

**If AUC is low**:
1. Increase `entropy_coeff_start` (more exploration)
2. Decrease `lambda_size` (allow larger explanations)
3. Increase `causal_weight` (stronger causal signal)
4. Increase `random_edge_drop` (better robustness)

**If subgraphs are too large**:
1. Increase `lambda_size` (stronger size penalty)
2. Decrease `target_size_k` (smaller target)
3. Increase `stop_bonus_ok` (encourage early stopping)
4. Increase `stop_bias` (bias toward stopping)

**If training is unstable**:
1. Decrease `lr` (slower updates)
2. Decrease `clip_range_policy` (more conservative)
3. Increase `grad_clip` (stronger gradient clipping)
4. Increase `value_loss_coef` (better value estimates)

**If training is too slow**:
1. Increase `ppo_epochs` (more updates per batch)
2. Increase `g_batch_size` (larger batches)
3. Decrease `n_rollouts` (fewer rollouts per step)
4. Reduce `pretrain_list` and `pretrain_set`

---

## 13. DATASET-SPECIFIC LOGIC

### 13.1 Supported Datasets

| Dataset | Type | Task | Nodes | Edges | Graphs | Motif Size |
|---------|------|------|-------|-------|--------|------------|
| Syn1 | Synthetic | Node | 700 | ~4100 | 1 | 8 nodes |
| Syn2 | Synthetic | Node | 700 | ~4100 | 1 | 8 nodes |
| Syn3 | Synthetic | Node | 871 | ~5200 | 1 | Varies |
| Syn4 | Synthetic | Node | 800 | ~4800 | 1 | Varies |
| BA2 | Synthetic | Graph | 25 | ~50 | 1000 | 5 nodes |
| Mutag | Real | Graph | ~30 | ~60 | 4337 | 4-6 nodes |

### 13.2 Dataset Loading

**File**: `datasets/dataset_loaders.py`

**High-Level API** (`dataset_loaders.py:82-103`):

```python
def load_dataset(_dataset, skip_preproccessing=False, shuffle=True):
    """
    Load any supported dataset.
    
    Returns:
        graph: Edge index [2, num_edges] or list of edge indices
        features: Node features [num_nodes, feat_dim]
        labels: Ground truth labels
        train_mask, val_mask, test_mask: Boolean masks
    """
    if _dataset[:3] == "syn":  # Syn1-4
        return _load_node_dataset(_dataset)
    else:  # BA2, Mutag
        return load_graph_dataset(_dataset, shuffle)
```

**Node Dataset Loading** (`dataset_loaders.py:65-79`):

```python
def _load_node_dataset(_dataset):
    path = 'datasets/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)
    
    # Merge labels
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]
    
    # Preprocess
    preprocessed_features = preprocess_features(features)
    graph = preprocess_adj(adj)[0].astype('int64').T
    labels = np.argmax(labels, axis=1)
    
    return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
```

**Graph Dataset Loading** (`dataset_loaders.py:9-62`):

```python
def load_graph_dataset(_dataset, shuffle=True):
    if _dataset == "ba2":
        path = 'datasets/pkls/BA-2motif.pkl'
    elif _dataset == "mutag":
        path = 'datasets/pkls/Mutagenicity.pkl'
    
    with open(path, 'rb') as f:
        adjs, features, labels = pkl.load(f)
    
    # Shuffle with fixed seed (reproducibility)
    if shuffle:
        prng = RandomState(42)
        indices = prng.permutation(len(adjs))
        adjs = adjs[indices]
        features = features[indices]
        labels = labels[indices]
    
    # Create train/val/test splits (80/10/10)
    n = len(adjs)
    train_mask = np.full(n, False, dtype=bool)
    train_mask[:int(n*0.8)] = True
    val_mask = np.full(n, False, dtype=bool)
    val_mask[int(n*0.8):int(n*0.9)] = True
    test_mask = np.full(n, False, dtype=bool)
    test_mask[int(n*0.9):] = True
    
    # Convert adjacency matrices to edge indices
    edge_index = adj_to_edge_index(adjs)
    
    return edge_index, features, labels, train_mask, val_mask, test_mask
```

### 13.3 Ground Truth Loading

**File**: `datasets/ground_truth_loaders.py`

**Unified API** (`ground_truth_loaders.py:113-172`):

```python
def load_dataset_ground_truth(_dataset, shuffle=True, test_indices=None):
    """
    Load ground truth edge labels for any dataset.
    
    Returns:
        explanation_labels: (edge_index, edge_labels) for each graph
        eval_seeds: List of seed nodes/graphs to evaluate
    """
    if _dataset == "syn1" or _dataset == "syn2":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        eval_seeds = range(400, 700, 5)  # Every 5th node in test set
        return (graph, labels), eval_seeds
    
    elif _dataset == "syn3":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        eval_seeds = range(511, 871, 6)
        return (graph, labels), eval_seeds
    
    elif _dataset == "syn4":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        eval_seeds = range(511, 800, 1)
        return (graph, labels), eval_seeds
    
    elif _dataset == "ba2":
        edge_index, labels = load_ba2_ground_truth(shuffle)
        # First 100 test graphs + last 100 test graphs
        eval_seeds = list(range(0, 100)) + list(range(500, 600))
        return (edge_index, labels), eval_seeds
    
    elif _dataset == "mutag":
        edge_index, labels, edge_list, edge_labels = load_mutag_ground_truth(shuffle)
        # Select graphs with non-mutagenic label and exactly 4 motif edges
        eval_seeds = [i for i in range(len(edge_index)) 
                      if np.argmax(labels[i]) == 0 and np.sum(edge_labels[i]) == 4]
        return (edge_list, edge_labels), eval_seeds
```

**BA2 Ground Truth** (`ground_truth_loaders.py:11-51`):

```python
def load_ba2_ground_truth(shuffle=True):
    """
    BA2 ground truth: 5-node motif is nodes [20, 21, 22, 23, 24].
    Edge is motif edge if both endpoints in [20, 24].
    """
    path = 'datasets/pkls/BA-2motif.pkl'
    with open(path, 'rb') as f:
        adjs, features, labels = pkl.load(f)
    
    # Shuffle
    if shuffle:
        prng = RandomState(42)
        indices = prng.permutation(len(adjs))
        adjs = adjs[indices]
    
    # Convert to edge indices
    edge_index = adj_to_edge_index(adjs)
    
    # Create edge labels
    np_edge_labels = []
    insert = 20  # Motif starts at node 20
    skip = 5     # Motif has 5 nodes
    
    for ei in edge_index:
        labels = []
        for (r, c) in ei.T:
            # Both endpoints in motif range → motif edge
            if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                labels.append(1)
            else:
                labels.append(0)
        np_edge_labels.append(np.array(labels))
    
    return edge_index, np_edge_labels
```

**Mutag Ground Truth** (`ground_truth_loaders.py:54-91`):

```python
def load_mutag_ground_truth(shuffle=True):
    """
    Mutag ground truth: Chemical motifs (NO₂, NH₂, etc.).
    Loaded from raw graph files with edge labels.
    """
    path = 'datasets/pkls/Mutagenicity.pkl'
    with open(path, 'rb') as f:
        original_adjs, original_features, original_labels = pkl.load(f)
    
    # Load edge labels from raw files
    path = 'datasets/Mutagenicity/Mutagenicity_'
    edge_lists, _, edge_label_lists, _ = get_graph_data(path)
    
    # Shuffle
    if shuffle:
        prng = RandomState(42)
        indices = prng.permutation(len(original_adjs))
        edge_list = [edge_lists[i] for i in indices]
        edge_label_lists = [edge_label_lists[i] for i in indices]
        original_labels = original_labels[indices]
    
    # Convert to edge indices
    edge_index = adj_to_edge_index(original_adjs[indices])
    
    return edge_index, original_labels, edge_list, edge_label_lists
```

### 13.4 Dataset-Specific Preprocessing

**Node Features**:

| Dataset | Original | Preprocessed | Description |
|---------|----------|--------------|-------------|
| Syn1-4 | One-hot | One-hot | Node types (10 types) |
| BA2 | One-hot | One-hot | Node types (10 types) |
| Mutag | One-hot | One-hot | Atom types (14 types: C, O, Cl, ...) |

**Edge Features**:

| Dataset | Edge Features | Usage |
|---------|---------------|-------|
| Syn1-4 | None | N/A |
| BA2 | None | N/A |
| Mutag | Bond type (optional) | Not used (with_attr=False) |

**Label Encoding**:

| Dataset | Labels | Encoding |
|---------|--------|----------|
| Syn1-4 | Node classes | One-hot → Argmax |
| BA2 | Graph classes (2) | One-hot |
| Mutag | Graph classes (2) | One-hot |

### 13.5 Evaluation Seed Selection

**Why Not All Nodes/Graphs?**

Evaluation is expensive (generate explanation + compute AUC). We select representative subsets:

**Syn1/Syn2** (`ground_truth_loaders.py:122`):
```python
eval_seeds = range(400, 700, 5)  # 60 seeds
# Rationale: Test set is [400, 700), sample every 5th node
```

**Syn3** (`ground_truth_loaders.py:129`):
```python
eval_seeds = range(511, 871, 6)  # 60 seeds
# Rationale: Different test set range, sample every 6th node
```

**Syn4** (`ground_truth_loaders.py:136`):
```python
eval_seeds = range(511, 800, 1)  # 289 seeds
# Rationale: All nodes in test set (smaller dataset)
```

**BA2** (`ground_truth_loaders.py:146`):
```python
eval_seeds = list(range(0, 100)) + list(range(500, 600))  # 200 graphs
# Rationale: First 100 + middle 100 from test set
```

**Mutag** (`ground_truth_loaders.py:162`):
```python
eval_seeds = [i for i in range(len(edge_index)) 
              if np.argmax(labels[i]) == 0 and np.sum(edge_labels[i]) == 4]
# Rationale: Only non-mutagenic graphs with exactly 4 motif edges
#            (standard benchmark filtering)
```

---

## 14. LOGGING, METRICS, OUTPUT FORMATS

### 14.1 Console Logging

**Training Loop Output** (per epoch):

```
====================
[Epoch    1]
λ Schedule: size=0.0000, radius=0.0000, sim=0.0000, entropy=0.0100
Effective causal weight: 0.1000
Update G
[PPO Stats] Policy=0.0234, Value=0.1234, Entropy=0.0098, KL=0.0045, 
            ClipFrac=0.123, EV=0.456, GradNorm=0.789, Updates=128, 
            EpsPolicy=0.20, EpsValue=0.25
Reward=2.34 PLoss= 0.02 Length=11.2 EarlyStopRate=0.123 AvgSize=9.7
Elapsed Time: 45.3s
[Epoch    1] Schedule: λ_size=0.0000, λ_radius=0.0000, λ_sim=0.0000, entropy=0.0100
[Epoch    1] Stats: Reward=2.340, PLoss=0.0234, Length=11.2
[Epoch    1] Metrics: EarlyStop=0.123, AvgSize=9.7
```

**Evaluation Output** (every `eval_every` epochs):

```
[EVAL-Epoch    1] AUC=0.8234
n_id 405 len 9 p_loss=0.32 size_loss=0.18 sim_loss=0.05 r_p=0.10 sgs [0, 5, 12, 18, 23, 29, 34, 41, 47]
n_id 410 len 8 p_loss=0.28 size_loss=0.16 sim_loss=0.04 r_p=0.09 sgs [0, 3, 7, 11, 16, 21, 27, 32]
...
avg_p_loss=0.35 avg_size_loss=0.20 avg_sim_loss=0.06 r_p=0.12
```

**Pretraining Output**:

```
[Pretrain-List   1] Loss = 2.3456 | Batches = 32 | Samples = 1024
[Pretrain-List   2] Loss = 2.1234 | Batches = 32 | Samples = 1024
...
[Pretrain-Set    1] Loss = 1.9876 | Batches = 32 | Samples = 1024
[Pretrain-Set    2] Loss = 1.8765 | Batches = 32 | Samples = 1024
...
```

### 14.2 Metrics Tracked

**PPO Training Metrics** (per epoch):

| Metric | Symbol | Description | Good Range |
|--------|--------|-------------|------------|
| Policy Loss | \( L^{\text{CLIP}} \) | PPO surrogate loss | [0.0, 0.5] |
| Value Loss | \( L^{\text{VF}} \) | MSE of value predictions | [0.0, 1.0] |
| Entropy | \( H(\pi) \) | Policy entropy | [0.001, 0.1] |
| Approx KL | \( \text{KL}_{\text{approx}} \) | KL divergence (old vs new) | [0.0, 0.02] |
| Clip Fraction | \( f_{\text{clip}} \) | Fraction of clipped ratios | [0.0, 0.5] |
| Explained Variance | EV | Critic quality | [0.0, 1.0] |
| Gradient Norm | \( \|\nabla\| \) | Total gradient magnitude | [0.0, 10.0] |
| Reward | \( \bar{r} \) | Mean reward per trajectory | [-5, 5] |
| Early Stop Rate | \( f_{\text{stop}} \) | Fraction of STOP actions | [0.0, 1.0] |
| Avg Size | \( \bar{|G|} \) | Mean subgraph size | [5, 15] |

**Evaluation Metrics** (per eval epoch):

| Metric | Symbol | Description | Good Range |
|--------|--------|-------------|------------|
| AUC | \( \text{AUC}_{\text{ROC}} \) | Area under ROC curve | [0.5, 1.0] |
| Prediction Loss | \( L_{\text{CE}} \) | Cross-entropy loss | [0.0, 2.0] |
| Size Loss | \( L_s \) | Size penalty | [0.0, 1.0] |
| Similarity Loss | \( L_{\text{sim}} \) | Embedding distance | [0.0, 1.0] |
| Radius Penalty | \( L_r \) | Subgraph depth | [0.0, 0.5] |

### 14.3 TensorBoard Logging (if enabled)

**Scalars** (tracked per iteration):

```python
# Training
writer.add_scalar('G/Reward', reward, iteration)
writer.add_scalar('G/PolicyLoss', policy_loss, iteration)
writer.add_scalar('G/ValueLoss', value_loss, iteration)
writer.add_scalar('G/Entropy', entropy, iteration)
writer.add_scalar('G/ApproxKL', approx_kl, iteration)
writer.add_scalar('G/ClipFrac', clip_frac, iteration)
writer.add_scalar('G/ExplainedVar', explained_var, iteration)
writer.add_scalar('G/GradNorm', grad_norm, iteration)
writer.add_scalar('G/Length', avg_length, iteration)
writer.add_scalar('G/EarlyStopRate', early_stop_rate, iteration)

# Evaluation
writer.add_scalar('Eval/Auc', auc, epoch)
writer.add_scalar('Eval/PredLoss', avg_pred_loss, epoch)
writer.add_scalar('Eval/SizeLoss', avg_size_loss, epoch)
writer.add_scalar('Eval/SimLoss', avg_sim_loss, epoch)

# Schedules
writer.add_scalar('Schedule/LambdaSize', lambda_size, epoch)
writer.add_scalar('Schedule/LambdaRadius', lambda_radius, epoch)
writer.add_scalar('Schedule/LambdaSim', lambda_sim, epoch)
writer.add_scalar('Schedule/Entropy', entropy_coef, epoch)
writer.add_scalar('Schedule/CausalWeight', causal_weight, epoch)
```

### 14.4 Output File Formats

**Results File** (`results/{dataset}_ppo_seed_{seed}.txt`):

```
# Pretraining Performance Log
# Generated: 2025-11-28 10:23:45
# Dataset: syn1 | Seed: 0
# Hash Code: 5A3F8B2C...
# Task: node_task
# Model: GNN
# Max Size: 20 | N Hop: 3
# Pretrain Samples: 60
#
# Hyperparameters:
# pretrain_list=10 pretrain_set=25
# hidden_size=64 g_lr=5e-05 l_lr=0.01
# gamma=0.99 clip_eps=0.2 ppo_epochs=4
# minibatch_size=256 use_gae=True gae_lambda=0.95
#
# Pretraining Performance Metrics:
# List-based Pretraining:
# Iteration | Avg Loss | Batch Loss | Batches | Samples
LIST   1 |   2.3456 |    75.0592 |      32 |    1024
LIST   2 |   2.1234 |    67.9488 |      32 |    1024
...
# Set-based Pretraining:
SET    1 |   1.9876 |    63.6032 |      32 |    1024
SET    2 |   1.8765 |    60.0480 |      32 |    1024
...
====================
[Epoch    1]
λ Schedule: size=0.0000, radius=0.0000, sim=0.0000, entropy=0.0100
...
[EVAL-Epoch    1] AUC=0.8234
...
```

**Checkpoint Files** (`ckpts/{dataset}/{timestamp}/{epoch}_{auc}.pth`):

```python
# Saved state dict
{
    'g': generator.model.state_dict(),  # Policy and value networks
    'l': locator.model.state_dict()     # Locator (graph tasks only)
}
```

**Pretrained Model Cache** (`pretrained/{hash}.pth`):

```python
# Same format as checkpoint
{
    'g': generator.model.state_dict(),
    'l': locator.model.state_dict()  # If applicable
}
```

### 14.5 Analysis Toolkit

**Location**: `analysis/`

**Key Scripts**:

1. `visualize_explanations.py`: Generate side-by-side plots (predictions vs ground truth)
2. `extract_ppo_performance.py`: Parse training logs and extract metrics
3. `plot_rl.py`: Plot reward, loss, AUC curves over epochs
4. `analyze_ppo_dataset.py`: Analyze explanation quality statistics

**Example Usage**:

```bash
# Visualize worst AUC explanation
python analysis/visualize_explanations.py --dataset ba2 --seed 4

# Extract metrics to CSV
python analysis/extract_ppo_performance.py --input results/syn1_ppo_seed_0.txt --output metrics.csv

# Plot training curves
python analysis/plot_rl.py --input results/syn1_ppo_seed_0.txt
```

---

## 15. PSEUDOCODE & FORMULAS

### 15.1 Main Training Loop

```python
Algorithm: RL-Based GNN Explainer with PPO and Causal Mechanisms

Input:
    G = (V, E): Input graph
    M: Pretrained GNN model
    S: Set of seed nodes (or graphs)
    θ: Hyperparameters

Output:
    π*: Trained explanation policy

# ===== PHASE 1: PRETRAINING =====
1. Initialize policy network π_θ (Actor-Critic)
2. Initialize Locator L (for graph tasks)

3. Pretrain Locator L (graph tasks only):
   For epoch = 1 to pretrain_l_iter:
       Sample seeds s ~ L(G)
       Generate explanations G_sub = π_θ(s, G)
       Compute scores r = Score(G_sub, M)
       Update L via REINFORCE: ∇_L log p_L(s|G) * r

4. Pretrain Generator π_θ (List-Based):
   For epoch = 1 to pretrain_list:
       For each ground-truth subgraph G_gt in training set:
           Extract sequence [s, n₁, n₂, ..., nₖ]
           For t = 1 to k:
               Predict action: logits = π_θ(state, candidates)
               Supervise: loss -= log π_θ(n_t | state)
           Update θ via gradient descent

5. Pretrain Generator π_θ (Set-Based):
   For epoch = 1 to pretrain_set:
       For each ground-truth subgraph G_gt in training set:
           Extract node set {s, n₁, n₂, ..., nₖ}
           remaining = {n₁, n₂, ..., nₖ}
           While remaining ≠ ∅:
               candidates = boundary(state) ∩ remaining
               Choose n* = argmax π_θ(n | state) for n ∈ candidates
               loss -= log π_θ(n* | state)
               remaining = remaining \ {n*}
           Update θ via gradient descent

# ===== PHASE 2: MAIN TRAINING (PPO) =====
6. For epoch = 1 to n_epochs:
   
   # Update schedules
   λ_size, λ_radius, λ_sim, entropy_coef = Schedule(epoch)
   causal_weight = CausalWarmup(epoch)
   
   # Sample rollouts
   For i = 1 to g_batch_size:
       s_i ~ S  # Sample seed
       τ_i = []  # Trajectory
       state = {s_i}
       
       For t = 1 to max_size:
           # Get action
           candidates = OuterBoundary(state)
           logits, value = π_θ(state, candidates)
           action ~ Categorical(softmax(logits))
           
           # Store transition
           Store (state, action, log π_θ(action|state), value)
           
           # Update state
           If action == EOS:
               Break
           Else:
               state = state ∪ {action}
               τ_i.append(action)
   
   # Compute rewards
   For i = 1 to g_batch_size:
       For t = 1 to len(τ_i):
           G_t = state at step t
           
           # Composite loss J(G_t)
           masked_pred = M(X, E[G_t])
           J_t = CE(masked_pred, original_pred) + 
                 λ_size * |G_t| +
                 λ_radius * Radius(G_t) +
                 λ_sim * ||Embed(G_t) - Embed(G)||
           
           # Delta reward
           r_t = reward_scale * -(J_t - J_{t-1})
           
           # Causal reward (optional)
           If use_causal:
               r_causal = CausalReward(G_t, target_y)
               r_t = (1 - causal_weight) * r_t + causal_weight * r_causal
           
           # STOP reward (terminal step)
           If action == EOS:
               If ClassPreserved(G_t) AND |G_t| ≤ target_size_k:
                   r_t += stop_bonus_ok
               Else:
                   r_t -= stop_penalty_bad
   
   # Compute advantages (GAE)
   For i = 1 to g_batch_size:
       # TD residuals
       For t = 1 to len(τ_i):
           δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
       
       # GAE
       A_t = 0
       For t = len(τ_i) downto 1:
           A_t = δ_t + γ * λ * A_t
           advantages[t] = A_t
   
   # Normalize advantages
   advantages = (advantages - mean(advantages)) / (std(advantages) + ε)
   advantages = clamp(advantages, -10, 10)
   
   # PPO updates
   For ppo_epoch = 1 to ppo_epochs:
       Shuffle rollout buffer
       For minibatch in MiniBatches(rollout_buffer, size=minibatch_size):
           # Recompute policy
           For (state, action) in minibatch:
               logits_new, value_new = π_θ(state)
               logp_new = log π_θ(action | state)
           
           # Ratio
           ratio = exp(logp_new - logp_old)
           
           # PPO loss
           surr1 = ratio * advantages
           surr2 = clamp(ratio, 1-ε_policy, 1+ε_policy) * advantages
           policy_loss = -mean(min(surr1, surr2))
           
           # Value loss
           value_loss = MSE(value_new, returns)
           
           # Entropy bonus
           entropy = -sum(p * log p)
           
           # Total loss
           loss = policy_loss + c_1 * value_loss - entropy_coef * entropy
           
           # Update
           loss.backward()
           clip_grad_norm_(θ, max_norm=1.0)
           optimizer.step()
           
           # Monitor KL
           approx_kl = mean(logp_old - logp_new)
           If approx_kl > 0.2:
               lr = lr * 0.5  # Adaptive LR
   
   # Evaluation
   If epoch % eval_every == 0:
       For s in eval_seeds:
           G_sub = π_θ(s, G)  # Greedy generation
           mask = EdgeScore(G_sub)
           AUC_score = AUC(mask, ground_truth)
       Print mean AUC_score

7. Return π*
```

### 15.2 Key Formulas

**Composite Loss**:

\[
J(G_k) = \mathcal{L}_{\text{CE}}(M(G_k), M(G)) + \lambda_s |G_k| + \lambda_r \text{radius}(G_k) + \lambda_{\text{sim}} \|h(G_k) - h(G)\|_2
\]

**Delta Reward**:

\[
r_k = \alpha \cdot \left[ -(J(G_k) - J(G_{k-1})) \right]
\]

**Causal Reward**:

\[
r_{\text{causal}} = \text{clamp}\left( \log P_M(y^* | G_k) - \log P_M(y^* | G_{\emptyset}) + b_{\text{correct}}, -1, 1 \right)
\]

Where:
- \( G_{\emptyset} \): Empty graph (no edges, only node features)
- \( b_{\text{correct}} = 0.5 \) if prediction is correct, else \( -0.5 \)

**Combined Reward**:

\[
r_{\text{total}} = (1 - w_{\text{causal}}) \cdot r_k + w_{\text{causal}} \cdot r_{\text{causal}}
\]

**GAE Advantage**:

\[
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\end{aligned}
\]

**Recursive Form**:

\[
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
\]

**PPO Clipped Objective**:

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\]

Where:

\[
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
\]

**Total PPO Loss**:

\[
L(\theta) = L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 H(\pi_\theta)
\]

Where:
- \( L^{\text{VF}}(\theta) = \mathbb{E}_t[(V_\theta(s_t) - \hat{R}_t)^2] \)
- \( H(\pi_\theta) = -\mathbb{E}_t[\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)] \)

**STOP Reward**:

\[
r_{\text{stop}} = \begin{cases}
r_k + b_+ & \text{if } \arg\max M(G_k) = \arg\max M(G) \text{ AND } |G_k| \leq k \text{ AND } \text{conf. ok} \\
r_k - b_- & \text{otherwise}
\end{cases}
\]

**Entropy Schedule**:

\[
c_2(t) = \begin{cases}
c_{\text{start}} + \frac{t - t_w}{t_s} (c_{\text{end}} - c_{\text{start}}) & t_w \leq t < t_w + t_s \\
c_{\text{end}} + \frac{t - (t_w + t_s)}{T - (t_w + t_s)} (c_{\text{final}} - c_{\text{end}}) & t \geq t_w + t_s
\end{cases}
\]

Where:
- \( t_w \): Warmup epochs
- \( t_s \): Schedule epochs
- \( T \): Total epochs

---

## 16. FINAL SUMMARY & CONTRIBUTIONS

### 16.1 Core Technical Contributions

1. **First PPO-Based GNN Explainer**: Replaces REINFORCE with PPO for stable, sample-efficient training

2. **Hybrid RL + Causal Framework**: Integrates counterfactual reasoning into PPO rewards

3. **Delta Reward Shaping**: Dense per-step feedback via incremental loss changes

4. **Connectivity-Preserving Edge Dropout**: Robust pretraining with guaranteed graph connectivity

5. **Dual Pretraining Strategy**: Sequential list-based → set-based for better initialization

6. **Comprehensive Stability Fixes**: 11 major numerical stability improvements over vanilla PPO

7. **Adaptive Training Dynamics**: KL-based learning rate adaptation, scheduled entropy decay

8. **STOP Action Modeling**: Explicit stop action with tunable bias and reward shaping

### 16.2 Experimental Results (Summary)

| Dataset | Baseline (RG-Explainer) | This Work (PPO+Causal) | Improvement |
|---------|------------------------|------------------------|-------------|
| Syn1 | 0.823 ± 0.03 | 0.891 ± 0.02 | +8.3% |
| Syn2 | 0.801 ± 0.04 | 0.867 ± 0.03 | +8.2% |
| BA2 | 0.756 ± 0.05 | 0.834 ± 0.03 | +10.3% |
| Mutag | 0.712 ± 0.06 | 0.789 ± 0.04 | +10.8% |

*(Note: Actual experimental results should be filled in from real runs)*

### 16.3 Code Statistics

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Main Script | `run_RG_explainer.py` | 1504 | Training loop, data loading |
| Generator | `components/generator_copy.py` | 2435 | PPO, GAE, reward computation |
| Agent | `components/generator_core.py` | 132 | Policy/value network |
| Causal | `components/causal_explainer.py` | 584 | Counterfactual reasoning |
| Graph | `components/graph.py` | 187 | Graph utilities |
| Locator | `components/locator_seed.py` | 371 | Seed selection (graph tasks) |
| Datasets | `datasets/*.py` | ~400 | Data loading, ground truth |
| Evaluation | `evaluation/*.py` | ~300 | AUC computation |
| **Total** | | **~6000** | Full implementation |

### 16.4 Dependencies

**Core**:
- PyTorch 1.13+
- PyTorch Geometric 2.0+
- NumPy, SciPy
- NetworkX (for edge dropout)

**Optional**:
- torch-scatter (for graph-level causal explainer)
- matplotlib (for visualization)
- pandas (for analysis)

### 16.5 Recommended Citation Format

```bibtex
@misc{capstone2025rl,
  title={Reinforcement Learning-Based GNN Explainer with PPO and Causal Mechanisms},
  author={[Your Name]},
  year={2025},
  note={Capstone Research Project}
}
```

---

## END OF DOCUMENTATION

**Document Statistics**:
- Total Sections: 16
- Total Pages: ~50 (estimated)
- Total Words: ~25,000
- Total Code Examples: 100+
- Total Tables: 30+
- Total Formulas: 20+

**Usage**:
This documentation is designed to be self-contained and sufficient to:
1. Understand the entire system architecture
2. Reproduce all implementation details
3. Write a comprehensive research paper
4. Extend the codebase with new features

**Next Steps for Paper Writing**:
1. Extract relevant sections for Introduction, Related Work, Methods
2. Condense technical details into paper-appropriate descriptions
3. Add experimental results and comparisons
4. Create visualizations (architecture diagrams, training curves, example explanations)
5. Write Discussion and Conclusion based on contributions

**Contact**: For questions about this documentation, refer to the source code files cited throughout.

---

*End of Part 3*

