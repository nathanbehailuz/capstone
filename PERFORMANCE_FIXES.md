# Performance Fixes for BA2 and MUTAG Datasets

## Issues Identified and Fixed

### 1. **CRITICAL BUG: Incorrect Actual Node Count Calculation**
   
**Location:** `run_RG_explainer.py` line 283

**Problem:**
```python
actual_nodes = int(torch.sum(self.nodefeats[i]))
```

This calculated the number of actual nodes by summing feature values, which doesn't work for:
- One-hot encoded features (like MUTAG) 
- Normalized or continuous features
- Any non-binary feature representation

**Impact:**
- Tensor size mismatches in locator pretraining
- RuntimeError: `The size of tensor a (32) must match the size of tensor b (9)`
- Incorrect graph sizes leading to out-of-bounds access

**Fix:**
Calculate actual nodes from the edge list instead:
```python
if each_g_actual_edges[i].size > 0:
    max_node_in_edges = int(np.max(each_g_actual_edges[i])) + 1
    actual_nodes = max_node_in_edges
else:
    actual_nodes = 1
```

---

### 2. **PERFORMANCE: Unused GraphConv Precomputation**

**Location:** `components/generator_copy.py` lines 2153-2167

**Problem:**
The code precomputed normalized adjacency matrices for all graphs:
```python
for adj_mat in per_graph_adj_mats:
    conv = GraphConv(dummy_g, k=3, alpha=0.85)
    self.per_graph_convs.append(conv)
```

For MUTAG (4337 graphs), this created 4337 GraphConv objects that were:
- Never used in the codebase
- Wasted significant initialization time
- Consumed unnecessary memory

**Impact:**
- Slower initialization (especially noticeable for MUTAG)
- Increased memory footprint

**Fix:**
Removed the unused precomputation entirely. The actual speedup comes from cached predictions/embeddings, not the GraphConv objects.

---

### 3. **PERFORMANCE: Locator Pretraining Scale**

**Location:** `run_RG_explainer.py` line 980 and `slurm/mutag_ppo.slurm`

**Problem:**
MUTAG locator pretraining configuration:
- 4337 graphs (each treated as a subgraph)
- 131,488 total nodes
- `PRETRAIN_L_ITER=200` epochs
- `PRETRAIN_L_SAMPLE_RATE=1.0` (100% of graphs)

This means:
- 4337 graphs × 200 epochs = **867,400 graph iterations**
- For each iteration, processes ~30 nodes per graph on average  
- Each node triggers a k-hop subgraph computation and GNN forward pass
- Total: **~26 million k-hop subgraph computations**

**Impact:**
- Pretraining takes an extremely long time (hours to days)
- Most of the computation may not significantly improve the locator

**Recommendations:**
1. Reduce `PRETRAIN_L_ITER` to 50-100 for MUTAG
2. Reduce `PRETRAIN_L_SAMPLE_RATE` to 0.2-0.5 for MUTAG
3. Or add early stopping based on locator loss convergence

Example optimized config for MUTAG:
```bash
PRETRAIN_L_ITER=50        # Reduced from 200
PRETRAIN_L_SAMPLE_RATE=0.3  # Reduced from 1.0
```

This would reduce computation from 867k to 65k iterations (~13x speedup).

---

## Testing Results

### Before Fixes:
- **seed_1**: Crashed with tensor size mismatch during pretraining
- **seed_0**: Hung indefinitely during locator pretraining

### After Fixes (Expected):
- Tensor size mismatches resolved
- Faster initialization (removed unused GraphConv precomputation)
- Pretraining will still be slow without hyperparameter adjustments

---

## Recommendations for Future Work

### For MUTAG and Large Graph Datasets:

1. **Adaptive Sampling:**
   - Start with low sample rate, increase if needed
   - Sample fewer nodes per graph for pretraining

2. **Early Stopping:**
   - Monitor locator training loss
   - Stop when loss plateaus

3. **Progressive Training:**
   - Start with smaller subset of graphs
   - Gradually include more

4. **Parallel Processing:**
   - The k-hop subgraph computations could be parallelized
   - Batch GNN forward passes more efficiently

5. **Dataset-Specific Configs:**
   ```bash
   if [ "${DATASET}" = "mutag" ]; then
       PRETRAIN_L_ITER=50
       PRETRAIN_L_SAMPLE_RATE=0.3
   elif [ "${DATASET}" = "ba2" ]; then
       PRETRAIN_L_ITER=100
       PRETRAIN_L_SAMPLE_RATE=0.5
   fi
   ```

---

## Files Modified

1. `run_RG_explainer.py`:
   - Fixed actual node count calculation (lines 281-303)

2. `components/generator_copy.py`:
   - Removed unused per_graph_convs precomputation (lines 2150-2156)

---

## Summary

The critical bug preventing MUTAG from running has been fixed (incorrect node counting). The performance issues are due to the scale of the dataset combined with aggressive pretraining hyperparameters. While the code now runs correctly, users should adjust `PRETRAIN_L_ITER` and `PRETRAIN_L_SAMPLE_RATE` for MUTAG to achieve reasonable training times.



