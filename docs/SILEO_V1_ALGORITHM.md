# Sileo v1 algorithm

This document is the executable contract for Sileo v1. A run is invalid when
its effective configuration or behavior differs from this contract without an
explicit version change.

## Predictor and target

The pretrained GNN is frozen and kept in evaluation mode. Its prediction on
the original instance defines the target class. Node tasks use the target
node's logits and embedding. Graph tasks pool only nodes belonging to the
target graph and use its graph embedding. Predictor calls use named `batch`
and `edge_weights` arguments.

An explanation is the induced subgraph on selected nodes. Unselected node
features are removed from graph-task inputs. Valid edgeless states are passed
through the real predictor; no placeholder logits or embeddings are allowed.

## Episode

State consists of the instance id, seed node, selected nodes, current boundary,
and the episode graph. An action selects a valid boundary node or STOP.
Selecting any other node is an error.

Training edge dropout creates one episode graph per episode. Undirected edge
pairs are dropped together and the complete node universe is retained.
Candidate construction and policy message propagation use the episode graph.
Rewards intentionally use the original graph. Evaluation never applies
dropout.

Termination reasons are:

- `stop`: the policy selected STOP;
- `exhausted`: the episode graph has no boundary action;
- `size_limit`: the configured maximum number of selected nodes was reached;
- `truncated`: an external step or time limit interrupted the episode.

STOP is masked until `min_steps_before_stop` selected nodes are present. Masked
actions have exactly zero probability. STOP has its own transition and reward.

## Objective and rewards

The state cost is

`J = prediction_cross_entropy + lambda_size * node_count
     + lambda_radius * radius + lambda_similarity * embedding_distance`.

For an expansion from state `s` to `s'`, base reward is
`reward_scale * (J(s) - J(s'))`. The initial seed is a state and receives no
action reward. STOP does not overwrite an expansion reward; its transition
receives only its configured confidence/size bonus or penalty unless stopping
changes the defined state cost.

Causal action guidance is disabled in v1. Optional causal reward shaping uses
the configured warm-up weight actually applied at the current epoch. Raw,
clipped, weighted, skipped, and failed causal calculations are recorded.
Unexpected causal calculation failures fail validation.

Learning uses `gamma` and `gae_lambda` from the resolved run configuration.
Terminal transitions do not bootstrap; externally truncated transitions do.
Padding is excluded from reward/advantage normalization and every loss.
`traj_reward_norm` and `std_advantage` are recorded as effective booleans.

## PPO

Each action stores one transition containing the exact model observation,
ordered candidates, validity mask, selected action, detached old normalized
log probability, detached old value, reward, and termination reason. Sampling
and replay construct the same categorical distribution. Before an optimizer
step an unchanged policy therefore has ratio 1, KL 0, and clip fraction 0
within floating-point tolerance.

Policy clipping is supported. Value clipping and adaptive KL penalties are not
implemented in v1 and requesting either is an error. `ppo_epochs` and
`minibatch_size` are effective configuration values. Paper-style REINFORCE is
not implemented; selecting it is an error.

## Locator and reproducibility

The graph locator is pretrained with per-graph KL divergence using log
probabilities, then fixed during generator training. Node embeddings are
gathered by actual node id.

Every run records source revision, immutable resolved configuration, predictor
identity and accuracy, data identity, RNG states, optimizer and schedule state,
progress, GPU identity, and structured training/explanation metrics.
Evaluation has an independent RNG stream and cannot change future training
samples.
