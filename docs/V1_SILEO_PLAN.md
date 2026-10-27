# Sileo v1: correctness work plan

Working branch: `sileo`, starting at `58d6079` (copied from `edge-drop`).
Date: 2026-10-27. Status: planning notes; implementation has not started.

## Objective

Make the current explainer implement a clearly specified algorithm, train on the intended predictions, and produce verifiable metrics. V1 is complete when the known correctness defects are repaired and tested, and representative real-data runs pass. Tests establish those guarantees; they cannot prove that arbitrary code has no possible errors.

Baseline reproduction, method comparisons, hyperparameter searches, and publication dashboards are deferred. Keep any existing synthetic baseline figures outside v1 validation.

## 1. Write down the algorithm we intend to implement

Before changing loss functions, give the current method a short executable specification:

- Frozen pretrained GNN: its original prediction defines the target class for each explained instance.
- State: seed, selected nodes, boundary candidates, and the graph used for the episode.
- Actions: select a valid boundary node or STOP.
- Base objective: prediction cross-entropy plus explicit node-count, radius, and embedding-similarity terms.
- Base transition reward: improvement in that objective from the previous state to the next state.
- Causal component: the current branch uses reward shaping based on target log-probability improvement over an edgeless reference, with a correctness bonus and clipping. Candidate-level causal action guidance is currently disabled.
- Dropout: training augmentation of the episode graph. If rewards intentionally use the original graph, state that explicitly. Evaluation uses the original graph.
- Termination: distinguish chosen STOP, exhausted boundary, size limit, and any external truncation.

Record discounting, reward normalization, STOP minimum, STOP reward, causal mixture weight/schedule, and graph intervention semantics. Decide whether graph explanations retain unselected node features while masking edges or actually remove nodes; use the same convention throughout training and evaluation.

Do not silently re-enable causal action guidance or redesign the causal reward during a bug fix. If the intended method includes either change, record it as an explicit algorithm change with separate tests. Use one canonical definition of the objective and its components; document any deliberate differences for pretraining.

## 2. Repair PPO and transition bookkeeping first

Primary file: `components/generator_copy.py`.

- [ ] Store one record per action, including the exact observation, candidates, mask, action, old normalized log probability, old value, reward, and terminal status.
- [ ] Sample from a categorical distribution and store its actual `log_prob(action)`, not a max-shifted logit.
- [ ] Reconstruct the same distribution during updates, including the STOP mask and any enabled guidance. Keep invalid actions at zero probability; epsilon smoothing must not make them selectable.
- [ ] Detach old probabilities/values; retain gradients through the current policy/value calculation.
- [ ] Align rewards with transitions. Do not allocate an action reward for the initial seed. Give STOP its own transition rather than overwriting the previous node's reward.
- [ ] Check GAE and returns against explicit terminal/truncation handling. Exclude padding from normalization and losses.
- [ ] Handle one-step, mixed-length, exhausted, and empty-update cases deliberately, without NaNs or silent skipped learning.
- [ ] Wire the configured PPO epochs/minibatch size. Either implement advertised KL/value-clipping controls or reject unsupported settings.

Acceptance tests:

- Before any optimizer step: probability ratio ≈ 1, KL ≈ 0, clip fraction ≈ 0.
- State costs 10 → 8 → 5 yield expansion rewards 2 and 3 when scale=1 and extra terms/normalization are disabled; STOP adds only its own specified reward.
- Hand-calculated GAE matches implementation for episodes with different lengths and termination reasons.
- A real PyTorch update has finite gradients and changes policy/value parameters as expected.

## 3. Make graph identity and predictor calls correct

Primary files: `run_RG_explainer.py`, `models/*`, `components/generator_copy.py`, `components/causal_explainer.py`.

- [ ] Introduce a small predictor interface for node and graph tasks: original/subgraph logits, correct target, and the relevant embedding.
- [ ] Use named arguments for `batch` and `edge_weights`; never infer task type from output shape or a seed ID.
- [ ] Preserve explicit graph IDs and local/global node mappings. Never pool unrelated graphs into one prediction.
- [ ] Use graph embeddings for graph-task similarity and target-node embeddings for node tasks.
- [ ] Run the real frozen GNN on valid edgeless states. Remove fixed two-class logits and shape truncation/target clamping used to hide mismatches.
- [ ] Fail clearly on incompatible predictor checkpoints; record predictor identity and accuracy. Freeze it and keep it in eval mode in every path.

Acceptance tests:

- Explaining graph A gives the same prediction/reward when unrelated graph B is also in the batch.
- Batched results match separate per-instance calls.
- Two-, four-, and eight-class cases, edgeless states, and single-item batches retain correct dimensions.
- Predictor parameters and buffers remain unchanged by explainer training.

## 4. Make dropout, STOP, and causal rewards behave as configured

Primary files: `components/generator_copy.py`, `components/graph.py`, `utils/edge_dropout.py`.

- [ ] Candidate construction must read the episode graph. Define which graph supplies policy message propagation and apply that choice consistently.
- [ ] Environment steps must reject nodes outside the allowed boundary.
- [ ] Drop undirected edges as pairs. Preserve the full node universe, including isolated nodes and highest-index nodes.
- [ ] Handle seeds that become terminal after dropout without crashing the whole batch.
- [ ] Explicitly choose one dropout mask per episode or per batch; log the actual behavior and realized drop rate. Evaluation must not apply dropout.
- [ ] Include STOP in list/set pretraining targets and finish exhausted target episodes.
- [ ] Align minimum-STOP-length masks with active episode IDs, not positions in the full batch.
- [ ] Resolve causal targets by instance/target-node identity after locator seed selection.
- [ ] Apply the actual causal weight schedule, not only log it. Record raw/clipped causal reward, weight, and any skipped/failed calculation.
- [ ] Unexpected causal failures must fail validation rather than silently return a neutral value.

Acceptance tests:

- Removing edge 0–2 from a triangle prevents selecting node 2 directly from seed 0.
- Undirected edge symmetry and node IDs survive dropout; connectivity is preserved when requested.
- Drop probability zero reproduces the no-drop path; evaluation remains unchanged by the training dropout setting.
- STOP receives supervised gradients; mixed-length batches use the right masks.
- Causal weight zero removes its reward contribution; a known nonzero causal fixture changes the reward by the specified amount.

## 5. Correct the locator and experiment state

Primary files: `components/locator_seed.py`, `run_RG_explainer.py`, `models/model_selector.py`.

- [ ] Feed log-softmax predictions to KLDivLoss, with a deliberate per-graph reduction.
- [ ] Gather locator embeddings using actual node IDs; skipping isolated nodes must not shift another graph's embeddings.
- [ ] Make locator-update behavior explicit: fixed after pretraining versus periodically updated. Logs must describe what executes.
- [ ] Replace module-global `args` references with instance configuration so the runner can be imported safely.
- [ ] Use one validated configuration. Remove duplicate implementation paths from active execution, and fail explicitly for unsupported policy modes rather than silently running PPO.
- [ ] Preserve separate predictor/explainer identifiers in metadata; do not mutate configuration while saving it.
- [ ] Make pretraining cache keys include actual initialization dependencies, source/data/predictor identities, and effective parameters. Deliberately shared initialization should be explicit.
- [ ] Save optimizer, schedule, RNG, progress, and configuration for exact continuation. Use unique output directories.
- [ ] Give evaluation its own RNG stream; running evaluation must not change subsequent training samples.

Acceptance tests: equal locator distributions produce KL≈0; seed/graph mappings are stable with isolated nodes; effective CLI settings match constructed objects; resumed training matches uninterrupted training under a deterministic test setup; evaluation does not perturb training RNG.

## 6. Build only the diagnostics needed to validate v1

Create a lightweight validation runner and structured JSON/CSV outputs, rather than a comparison dashboard.

Log raw episode returns and each reward term separately from normalized learning signals. Include policy/value loss, KL, clip fraction, entropy, explained variance, gradient norm, actual update count, STOP/exhaustion/size-limit rates, causal failure/saturation rates, and realized dropout rate.

For explanations, record instance/target IDs, selected nodes and edges, edge scores, original/selected/deleted-edge confidence, class agreement, size, connectivity, and ROC-AUC/AP where ground truth is defined. Verify edge alignment, reverse-edge/self-loop conventions, and undefined single-class metrics. Keep raw per-instance records; rounded terminal output is not the source of truth.

Use declared training/validation/evaluation instance sets. Transductive evaluation is allowed if labeled; it is not held-out generalization. Keep ground-truth explanation labels out of the training reward.

Minimal plots: reward decomposition and PPO diagnostics over updates; explanation quality/fidelity versus size; length/termination distributions; a few actual graph explanations with stable layouts and visible instance IDs.

## Implementation order and release gate

1. Pin the environment, restore required datasets, and create small graph fixtures that reproduce the confirmed failures.
2. Fix PPO and transitions.
3. Fix predictor/graph identity and edgeless states.
4. Fix dropout, STOP, causal rewards, and locator behavior.
5. Fix configuration/cache/resume/evaluation state and add structured diagnostics.
6. Run real-data smoke checks on one node dataset and one graph dataset, then exercise all six supported datasets. Validate on CPU and the intended CUDA environment where available.

V1 is ready when the targeted regression tests pass, real-data runs complete pretraining/training/evaluation/save/reload with finite outputs, predictor freezing is verified, explanation validity holds, and every logged setting corresponds to actual execution. Record any remaining limitations and measured results. Higher AUC alone does not satisfy this gate.

## Deferred

Original-paper baseline, comparative performance claims, broad ablation sweeps, extensive hyperparameter tuning, and publication/reporting polish. Revisit these after v1 correctness is established.
