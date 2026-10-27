# Research audit: edge-drop and RG-Explainer comparison

Date: 2026-10-27. Status: audit and proposed work plan; no training implementation changed.

## Scope and evidence

The user selected `edge-drop` as the research version. This report audits the locally recorded `origin/edge-drop` commit `58d607956a99d9a15bd1f2a471207f2b8da488c1`. The working checkout remains `main` at `73a6653`. All code locations below refer to **edge-drop**, not main. A read-only Git archive was extracted to `/tmp/capstone-audit/edge-drop`; links point to that snapshot. The commit and repository-relative filenames remain the durable references if temporary files are removed.

Evidence includes source inspection, the attached NeurIPS paper, Python AST parsing, and isolated executable probes of extracted methods using NumPy/SciPy stand-ins and known scalar losses. These probes establish specific logic defects; they are not PyTorch integration tests or measured model-performance results. All Python files in the edge-drop snapshot parsed successfully. The inspected Python environments lack PyTorch/PyTorch Geometric, and this repository version does not include dataset pickles or raw Mutagenicity data. Pretrained predictor weights are present. No full experiment was run.

The earlier main-branch findings about overwritten causal targets, applying argmax to already-discrete targets, and graph-evaluation subset indexing were corrected in edge-drop. They are **not** findings against this selected version.

## What the original baseline must reproduce

The attached paper is **RG-Explainer**, not a generic REINFORCE baseline. Sections 4.1–4.4 describe connected boundary expansion, a learned STOP action, terminal negative explanation loss, Monte Carlo continuation rollouts for action values, list/set pretraining, and alternating locator/generator training for graph tasks. Equations 5–9 define prediction loss and size/radius/similarity penalties. Equation 5 uses the original predictive distribution; hard-target cross-entropy needs an explicit reproduction decision if the authors' implementation differs.

Sections 5.1–5.4 use the same trained predictors across explainers, edge-importance ROC-AUC, ten runs, and separate transductive and inductive experiments. The paper's published AUC numbers are reference values, not a substitute for rerunning the baseline under the same local protocol. Recover and verify the original implementation and supplementary settings before labeling anything a faithful reproduction. The initial local commit `957ad20` contains a Monte Carlo REINFORCE path but already has PPO-related code; its provenance still needs verification.

Source: [attached paper](/Users/nathanbehailu/Desktop/capstone/NeurIPS-2021-reinforcement-learning-enhanced-explainer-for-graph-neural-networks-Paper.pdf), especially pp. 5–9. [Official publication page](https://papers.neurips.cc/paper_files/paper/2021/hash/be26abe76fb5c8a4921cf9d3e865b454-Abstract.html). The supplementary PDF could not be retrieved through the web tool in this audit.

## Findings that must precede performance claims

Priority P0 means a blocker for a valid comparison; P1 means a correctness or reproducibility issue to resolve before the experiment suite. These priorities concern the research comparison, not production security.

### A01 — P0: the existing plotted RG baseline is synthetic

[analysis/generate_rg_baseline.py:269](/tmp/capstone-audit/edge-drop/analysis/generate_rg_baseline.py:269) constructs explanations from PPO output and ground truth, searching for AUC in a specified range, default 0.60–0.70. [analysis/compare_explainers.py:550](/tmp/capstone-audit/edge-drop/analysis/compare_explainers.py:550) invokes this generator and labels the result RG-Explainer. It also selects PPO examples using an AUC threshold (lines 103–106, 525–535).

This is explicitly hypothetical data in the source, not an actual trained RG-Explainer result. Such plots cannot support a claim of outperforming the paper. Keep them clearly labeled as illustrative if retained; the research pipeline must load independently generated, traceable outputs for both methods. Reuse only plotting/layout utilities after checking their inputs and labels. Showcase examples can be useful, but publish the selection rule and include random, typical, and failure cases.

### A02 — P0: PPO stores unnormalized scores as old log probabilities

[components/generator_copy.py:2460](/tmp/capstone-audit/edge-drop/components/generator_copy.py:2460) subtracts the maximum from logits, samples from normalized probabilities, then stores `masked_logits[action]` at line 2483. A max-shifted logit is not `log(probability)`. The update computes normalized `log_softmax` at lines 1261–1271 and divides the resulting probability by the wrongly recorded denominator at line 1304. The update also does not replay the STOP mask or any causal bias used at sampling.

A numerical probe with an unchanged policy `[0.3, 0.2, 0.5]`, no masking, and no guidance gives a PPO ratio of **0.5 for every action**, where it must be approximately **1**. This can create clipping and apparent KL divergence before any learning. It compromises both the optimizer and its diagnostics. PPO's probability ratio is defined using the actual old and current policies: [original PPO paper](https://arxiv.org/abs/1707.06347).

Fix: construct one categorical distribution for sampling/replay, store normalized detached log probabilities, and preserve action order, masks, observations, and any fixed guidance. Verify ratio≈1, KL≈0, and clip fraction≈0 before the first update. Preserve differentiable log probabilities separately for REINFORCE.

### A03 — P0: delta rewards are shifted relative to actions

[components/generator_copy.py:882](/tmp/capstone-audit/edge-drop/components/generator_copy.py:882) iterates over the trajectory including the seed, but the action buffer starts with the first expansion. Rewards are written at the node index (line 1005), and the PPO mask uses `length - 1` (lines 1118–1144). STOP overwrites the last node's reward at line 942.

An executable probe of the actual reward method with known state costs 10 → 8 → 5 produced:

| Trajectory | Expected action rewards | Rewards paired with actions |
|---|---|---|
| seed → node 1 → node 2, size termination | 2, 3 | 0, 2 |
| seed → node 1 → node 2 → STOP | 2, 3, 0.1 | 0, 2, 0.1 |

The first expansion receives the seed baseline; the final node improvement is dropped or overwritten. This was checked with normalization and causal rewards disabled to isolate the defect. GAE inherits the wrong alignment. Contrary to a comment, the existing length mask does retain a sampled STOP action; simply removing STOP steps would not be a fix.

Fix: store one transition per actual action `(state, action, next_state, reward, terminated, truncated)`. Give STOP its own reward, distinguish exhaustion/size limits, and test hand-calculated returns and GAE with variable-length episodes.

### A04 — P0 for graph tasks: PPO rewards pool unrelated graphs together

The runner joins graph instances into a disjoint union. [components/generator_copy.py:858](/tmp/capstone-audit/edge-drop/components/generator_copy.py:858) evaluates that full union without a graph-to-node batch vector. It passes edge weights as the third positional argument, which is `batch` in [models/GNN_paper.py:85](/tmp/capstone-audit/edge-drop/models/GNN_paper.py:85). The model's fallback replaces that float argument with an all-zero batch, pooling the union as one graph. [_compute_subgraph_prediction:1061](/tmp/capstone-audit/edge-drop/components/generator_copy.py:1061) similarly retains features for all instances. Causal-reward calls have the same positional convention.

Although edge-drop fixes the initial batched predictor call in `Runner.init_g`, that fix does not reach these reward calls. Rewards can depend on unrelated molecules, and the similarity term uses a seed-node embedding instead of the appropriate graph embedding.

Fix: use explicit task-aware predictor adapters with named arguments and graph identity. Define whether graph explanations remove nodes or only mask edges, then apply that convention consistently. Test that adding an unrelated graph leaves an instance's prediction, reward, and explanation unchanged, and that batched results equal independent per-graph results.

### A05 — P1: edge dropout does not control the policy's available actions

[ExpansionEnv:55](/tmp/capstone-audit/edge-drop/components/generator_copy.py:55) creates a dropped graph, but [_prepare_inputs:2119](/tmp/capstone-audit/edge-drop/components/generator_copy.py:2119) takes candidates from `Generator.graph`, which is still the original graph. `self.conv` also remains based on the original graph. The environment's `step` appends the selected node without verifying it lies on the dropped graph's boundary.

An executable probe used an original triangle and a dropped chain. From node 0, the environment allowed only node 1, yet policy candidates were `[1, 2]`; `step` accepted node 2 across the removed edge. The resulting selected nodes were disconnected in the augmented graph. Thus dropout can affect termination while failing to implement the claimed action-space augmentation.

Additionally, one dropped graph is created per batched environment, shared by all trajectories, despite the per-episode wording. [utils/edge_dropout.py:44](/tmp/capstone-audit/edge-drop/utils/edge_dropout.py:44) drops directed edge entries independently, so an undirected bond can survive in just one direction. `Graph` then symmetrizes neighbors. Reconstructing graph size from surviving edge IDs can lose high-index isolated nodes; `reset` raises if any seed is terminal. These need explicit invariants.

Fix: route the episode graph through candidate construction and the chosen state representation; reject invalid actions; retain the complete node universe; sample undirected edge pairs together. Define batch-wide versus per-episode augmentation explicitly. Original-graph rewards can remain a deliberate augmentation objective, but describe that separately from explaining the perturbed predictor.

### A06 — P1: `--policy REINFORCE` does not select a baseline

[run_RG_explainer.py:1158](/tmp/capstone-audit/edge-drop/run_RG_explainer.py:1158) always calls `update_ppo`. The legacy REINFORCE fallback in `train_from_rewards` is unreachable from this path and consumes detached rollout log probabilities. Connecting that fallback alone would not restore the paper's Monte Carlo-Q algorithm or its gradient path.

Fix: implement explicit, verified method dispatch. Check which update function runs and that policy parameters receive finite, nonzero gradients for both methods. A PPO run without causal rewards is an ablation, not the paper baseline.

### A07 — P1: the actual causal mechanism differs from the advertised one

[_sample_trajectories:2005](/tmp/capstone-audit/edge-drop/components/generator_copy.py:2005) hard-codes `use_causal_action_bias = False`. The active contribution is **causal reward shaping**, through lines 901–924 and 976–1003; candidate-level intervention guidance is disabled. The instantiated causal scoring networks are not used to learn these reward values.

For graph tasks, `init_g` records `seed_nodes` while `eval_seeds` still contains graph IDs ([runner:329](/tmp/capstone-audit/edge-drop/run_RG_explainer.py:329), [runner:632](/tmp/capstone-audit/edge-drop/run_RG_explainer.py:632)). The locator later replaces evaluation seeds with global node IDs, without refreshing that mapping. Reward lookup calls `self.seed_nodes.index(seed_node)` and silently skips unmatched nodes; accidental numeric matches can associate the wrong target.

The logged causal warmup is also not the applied weight: the reward calculator retains the constant constructor weight, while `get_effective_causal_weight` is used for logging.

Fix: name and expose reward shaping and action guidance as separate settings; map targets by explicit instance ID and target node; log the actual applied weight and nonzero/failed/skipped causal-reward counts. Do not re-enable candidate guidance without including it in PPO probability replay and runtime accounting.

### A08 — P1: graph locator training uses the wrong KL input and is not alternated

[components/locator_seed.py:151](/tmp/capstone-audit/edge-drop/components/locator_seed.py:151) passes softmax probabilities into `nn.KLDivLoss`, which expects log probabilities for its input. For equal two-element uniform distributions, the current API formula yields approximately −0.5966 with its default mean reduction, rather than zero. See [PyTorch KLDivLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html).

The alternating locator update is commented out at [run_RG_explainer.py:1265](/tmp/capstone-audit/edge-drop/run_RG_explainer.py:1265). The locator's embedding assembly also slices the first `non_isolated_count` rows rather than indexing actual non-isolated global node IDs (lines 44–54), so isolated nodes can shift embeddings across instances.

Fix: use log-softmax inputs and an explicit per-graph KL sum/mean policy; index embeddings by actual node IDs; restore the paper's coordinate updates for its baseline. A frozen pretrained locator is a separate experimental choice.

### A09 — P1: STOP is omitted from pretraining and its runtime mask can use another episode's length

[train_from_lists:1909](/tmp/capstone-audit/edge-drop/components/generator_copy.py:1909) attempts `candidates.index('EOS')`; candidates contain only node IDs, so the exception skips the STOP target. Set pretraining similarly skips when target nodes are exhausted (lines 1835–1839). This biases pretraining toward continuing and can spin to its safety limit.

During rollout, `_current_trajectory_lengths` includes all episodes (line 2038), while `_sample_actions_masked` indexes that list using positions in the smaller active batch (line 2440). After some episodes terminate, another episode's length can determine whether STOP is allowed. Adding epsilon to every probability also assigns nonzero mass to masked actions.

Fix: represent STOP explicitly in supervised targets, finish exhausted target episodes, and pass masks/lengths aligned to `valid_index`. Test mixed episode lengths and single-node/isolated cases.

### A10 — P1: placeholder predictions and silent model-load fallback can conceal invalid runs

[_compute_subgraph_prediction:1055](/tmp/capstone-audit/edge-drop/components/generator_copy.py:1055) returns two fixed logits and a 64-dimensional zero embedding for an edgeless state instead of evaluating the predictor. Such states occur naturally at the initial seed. Actual predictors have task-specific class counts and embeddings (e.g., 4/8 classes and 60 dimensions for some node models). `composite_loss_J` clamps target labels and truncates mismatched embeddings, concealing this mismatch.

[models/model_selector.py:186](/tmp/capstone-audit/edge-drop/models/model_selector.py:186) catches incompatible predictor state dictionaries and continues. The resulting predictor may be partly loaded/random; its explanations would not measure the intended pretrained model.

Fix: evaluate valid edgeless states with the frozen GNN and its normal self-loop convention. Fail on missing data, incompatible shapes, or invalid predictor checkpoints; verify checkpoint hash, predictor accuracy, eval mode, and frozen parameters before training.

### A11 — P1: configuration and logged diagnostics do not consistently control execution

`Runner.init_g` does not forward `ppo_epochs` or `minibatch_size`, so constructor defaults 4/256 win over CLI sweeps. The active PPO update logs KL but does not enforce its configured KL stop or adaptive penalty. Value clipping is advertised and logged, but the update uses unclipped MSE. `n_g_updates` is not a loop count in `run`; there is one generator update per outer epoch. The PPO objective uses scheduled lambdas, while evaluation/score_fn use legacy size/radius/similarity arguments and count size differently (nodes versus directed edges).

`DummyWriter` discards scalars, the printed reward is a mean of normalized step rewards rather than raw episodic return, and `EarlyStopRate` describes short trajectories rather than KL-based optimizer stops. Those signals must not be interpreted interchangeably. Some runner methods use module-global `args`, making direct import into a future experiment harness unsafe.

Fix: one validated configuration with resolved effective values, explicit objectives for each method, and structured diagnostics returned from updates. Unsupported flags should fail or be removed. Preserve raw and normalized quantities separately.

### A12 — P1: checkpoint/cache metadata cannot establish reproducible experiments

[run_RG_explainer.py:1014](/tmp/capstone-audit/edge-drop/run_RG_explainer.py:1014) hashes a selected argument list. It includes `g_lr`, although the optimizer uses `lr`, and excludes reward parameters used to select pretraining samples. PPO settings are added to `arg_dict` but not to the list actually hashed. Different initialization conditions can therefore reuse the same cached weights. Conversely, dropout probability changes the cache key even though pretraining dropout is disabled.

`init_dir` mutates `vars(args)['model']` to `RGExplainer_v1`, losing the predictor architecture identifier. Checkpoints save network weights only, without optimizer, RNG, schedule, or resume progress. Output directories use second-resolution timestamps, risking collisions for concurrent same-dataset jobs.

Fix: a canonical immutable run manifest and unique run ID; separate predictor and explainer identities; cache hashes based on actual pretraining inputs, predictor/data hashes, source revision, and effective parameters. Save complete resumable state and record deliberate sharing of initialization across methods.

## Methodological risks to resolve explicitly

These are not all software bugs; they determine what conclusions the experiment can support.

- **Transductive versus held-out:** training samples from `eval_seeds`, and loader split masks are discarded. This can reproduce a declared transductive setting, but cannot establish unseen-instance generalization. Hold-out node examples should be grouped by motif/overlapping neighborhoods where feasible. Graph tasks need disjoint graph IDs. Keep predictor-training splits separate from explainer-training splits.
- **AUC alone is insufficient:** the current evaluator pools edge predictions across instances. Retain the paper-compatible score, but also report per-instance distributions and macro averages. Explicitly handle reverse edges, self-loops, missing edges, and cases with only one label class. No-positive/negative cases should be marked undefined, not silently assigned 0.5.
- **Selected evaluation subsets:** Mutagenicity ground truth uses a filtered class/motif subset. Report its count and criteria; do not generalize its score to the whole dataset. The optional Mutagenicity `test_indices` branch contains a hard-coded range 400–699 that needs correction before held-out work.
- **Inference randomness:** `generate()` still samples actions. Use an explicit deterministic inference mode or fixed repeated stochastic evaluation seeds, independent of training RNG. Increasing evaluation frequency must not change training trajectories.
- **Causal interpretation:** the active reward is a clamped target log-probability improvement relative to an edgeless graph, plus a correctness bonus. Before clamping, the log-probability term equals `CE(empty) - CE(selected)`. For a fixed instance, the first term is constant. This is a perturbation-based fidelity signal; by itself it does not identify real-world causality. Repeating this state reward at every step can also encourage longer trajectories. Compare against matched fidelity shaping and report reward saturation, rather than attributing all improvements to a distinct causal mechanism.
- **Other changes are confounders:** PPO, delta rewards, normalization, hard STOP minimum, regularization schedules, causal rewards, and dropout all differ from the paper. A two-method comparison alone cannot attribute a gain specifically to PPO or causality. A minimum of eight nodes also exceeds several motif sizes; treat it as a disclosed constraint and test its effect.

## Proposed build plan for discussion

### Phase 1 — Establish trusted code and data

Pin edge-drop and the baseline source revision. Obtain identical dataset files and predictor checkpoints for both methods; hash them. Pin a runnable dependency environment. Record predictor accuracy and evaluation instance IDs. Separate illustrative baseline generation from empirical analysis. Recover the paper baseline, verify its Monte Carlo continuation rewards, STOP learning, pretraining, and locator updates, and document deviations from the paper/supplement.

**Exit criterion:** either method can identify exactly which predictor, dataset, split, settings, and source produced a run. No silent predictor/data fallbacks.

### Phase 2 — Correct and test the training mechanics

Repair A02–A12 with small graph fixtures before large runs. Required tests: unchanged-policy ratios; finite nonzero gradients; hand-computed transition rewards/GAE; explicit STOP targets; batched-versus-single graph invariance; valid boundaries after dropout; preserved node IDs and undirected pairs; causal target identity and actual schedules; KL zero at equal distributions; effective CLI configuration; cache separation and resume equivalence. Add a causal-enabled reward integration test, not just the causal-disabled probe from this audit.

**Exit criterion:** one node dataset and one graph dataset complete pretraining, a PPO update, evaluation, save, and reload with finite outputs and all invariants satisfied. Then test all six datasets.

### Phase 3 — Lock the comparison protocol and ablations

Suggested experiment groups:

| Group | Purpose |
|---|---|
| Verified paper RG-Explainer | Reproduce the original algorithm and report any portability fixes |
| PPO with the paper's terminal objective, no causal reward/dropout | Separate optimizer change from reward redesign |
| PPO with corrected delta rewards, no causal reward/dropout | Reference for the new reward formulation |
| Previous group + causal reward | Isolate causal reward shaping |
| Delta PPO + dropout only | Isolate structural augmentation |
| Delta PPO + causal reward + dropout | Evaluate the full selected method |

These last four form a causal-reward × dropout factorial comparison under the same corrected PPO setup. Candidate-level causal action bias, if desired, should be a separate additional variant because it is currently disabled. A matched fidelity-shaping control helps test whether the causal reward contributes beyond rescaled predictive confidence.

Start with smoke runs and a small development sweep, then use ten paired training seeds for the main comparison. Tune only on validation data with comparable search budgets. Choose checkpoints by a declared validation criterion or a fixed final budget; never choose the best test AUC. Report equal explanation-sample/update budgets and wall-clock/GNN-query costs, since Monte Carlo baseline rollouts and causal evaluation have different costs. Include confidence intervals for paired per-seed improvements; do not treat thousands of correlated edges as independent runs. Declare transductive and inductive results separately.

### Phase 4 — Build the experiment runner and artifact contract

Create a configurable entry point with train, evaluate, aggregate, and report operations, plus SLURM wrappers. Reuse dataset/model/graph components after fixes and isolate method-specific updates behind a common interface. Log machine-readable data directly, rather than reconstructing results from rounded text output.

Each run should save: immutable manifest; data/predictor/source hashes; explicit splits; checkpoint state; epoch/update diagnostics; per-instance predictions and target class; ordered selected nodes; edge IDs and scores; termination reason; raw reward terms; inference RNG seed; elapsed time and GNN forward count; all failures/skips. Store large edge/trajectory tables separately from compact JSON metadata. Reports should be regenerable without retraining.

### Phase 5 — Metrics and visualizations that explain performance

| Question | Metrics | Visualization / diagnostic use |
|---|---|---|
| Does the explanation recover the motif? | Paper-compatible ROC-AUC, macro per-instance ROC-AUC, average precision, precision/recall/F1/IoU at fixed budgets | Paired seed plots, per-instance distributions, precision-recall curves; expose sparse-motif failures |
| Does it preserve the predictor's decision? | Original-class confidence retained, class agreement, KL from original predictive distribution | Confidence versus size curves; distinguish motif matching from model fidelity |
| Is the selected structure necessary? | Confidence drop after deleting selected edges; deletion/insertion curves | Curves against equal-size random controls; detect redundant explanations |
| Is it compact and valid? | Node/unique-edge counts, sparsity, radius, connectivity, seed inclusion, STOP/exhaustion/size-limit rates | Fidelity–size Pareto plots, length/termination histograms |
| Is learning healthy? | Raw return and each term; policy/value losses; KL; clipping; entropy; advantage/return distributions; explained variance; gradient norm | Training panels and reward-versus-held-out-quality plots; separate optimization failure from bad objectives |
| Is causal reward contributing? | Raw causal deltas, saturation/zero/failure rates, applied weight, correlation with fidelity and validation motif quality | Reward decomposition and ablation interaction plots; detect redundant or saturated shaping |
| Is dropout doing what was intended? | Realized undirected drop rate, connected components, isolated nodes, invalid-action count, mask reuse count | Drop-rate/validity panels; distinguish augmentation from accidental early termination |
| Is the locator limiting graph results? | Seed-in-motif rate, end-to-end performance versus a clearly labeled oracle-seed diagnostic | Seed-success breakdowns; separate locator errors from expansion errors |
| Is it robust and repeatable? | Across-seed variance, repeated-inference agreement, aligned edge Jaccard/rank stability under defined perturbations | Stability distributions and controlled perturbation sweeps |
| What does the improvement cost? | Pretraining/training/inference time, throughput, peak memory, GNN forwards | Quality–compute Pareto plots and per-stage runtime breakdown |

Define fidelity metrics explicitly, avoiding ambiguous plus/minus names: e.g. retained confidence `p_c(G_selected)` and deletion drop `p_c(G_original) - p_c(G_without_selected)`, with target class fixed from the original predictor. State the edge/node intervention and self-loop convention. ROC-AUC/AP are ranking metrics, not measures of calibration or causal identification.

For top-k overlap, use fixed budgets for comparisons; if k equals the ground-truth motif size, label it as an oracle-size evaluation only. For uncertainty, summarize across independent training seeds and preserve paired instances within each seed; use grouped resampling for overlapping node neighborhoods when estimating instance-level uncertainty.

Side-by-side graph views must use the same instance, layout, edge universe, and size budget; show actual outputs, ground truth, ordering, and confidence. Include predefined random samples, median cases, strongest wins, and strongest losses with the selection rule visible. Static SVG/PDF figures suit the paper; an optional local HTML report can support filtering and instance inspection.

## Recommended next step

Agree on the corrected definition of the selected method—currently PPO with causal reward shaping and intended training-time dropout—then implement Phase 1 and the correctness tests/fixes before building the full reporting suite. The audit does not establish that the method is better or worse than RG-Explainer; it establishes why the current code and existing illustrative comparisons cannot yet answer that question reliably.
