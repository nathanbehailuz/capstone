from typing import Union, Optional, List, Set, Dict, Tuple
import numpy as np
from scipy import sparse as sp
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data

from .graph import Graph
from .generator_core import GraphConv, Agent, masked_categorical_logits
from .predictor import PredictorAdapter
from .ppo import (
    Termination,
    Transition,
    generalized_advantage_estimate,
    masked_categorical,
)


@dataclass
class TrainingStats:
    """Training statistics for PPO updates."""
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    adv_mean: float
    adv_std: float
    ret_mean: float
    ret_std: float
    explained_variance: float
    grad_norm: float
    update_count: int


class ExpansionEnv:

    def __init__(self, graph: Graph, selected_nodes: List[List[int]], max_size: int, debug_mode: bool = False,
                 training: bool = False, edge_dropout_prob: float = 0.0, preserve_connectivity: bool = False):
        """
        Graph expansion environment for RL.
        
        Args:
            graph: Graph object (original graph)
            selected_nodes: List of initial node selections per episode
            max_size: Maximum trajectory length
            debug_mode: Enable debug logging
            training: If True, apply edge dropout. If False, use original graph.
            edge_dropout_prob: Probability of dropping edges (only if training=True)
            preserve_connectivity: Whether to preserve connectivity during dropout
        
        Note:
            Edge dropout lives in the Environment (world dynamics), not the Generator (policy).
            When training=True, each episode sees a stochastically dropped version of the graph.
            Rewards are always computed using the original graph (not the dropped graph).
        """
        self.max_size = max_size
        
        self.original_graph = graph
        if training and edge_dropout_prob > 0.0:
            self.graphs = [
                Graph.from_dropped_edges(
                    graph, edge_dropout_prob, preserve_connectivity
                )
                for _ in selected_nodes
            ]
        else:
            self.graphs = [graph for _ in selected_nodes]
        self.graph = self.graphs[0] if self.graphs else graph
        
        self.n_nodes = self.graph.n_nodes
        self.data = selected_nodes
        self.bs = len(self.data)
        self.trajectories = None
        self.dones = None
        self.termination_reasons = None
        self.debug_mode = debug_mode
        original_pairs = max(1, graph.edge_index.size(1))
        retained = sum(g.edge_index.size(1) for g in self.graphs)
        self.realized_drop_rate = 1.0 - retained / (
            original_pairs * max(1, len(self.graphs))
        )

    @property
    def lengths(self):
        return [len(x) - (x[-1] == 'EOS') for x in self.trajectories]

    @property
    def done(self):
        return all(self.dones)

    @property
    def valid_index(self) -> List[int]:
        return [i for i, d in enumerate(self.dones) if not d]

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.trajectories = [x.copy() for x in self.data]
        self.dones = []
        self.termination_reasons = []
        for index, trajectory in enumerate(self.trajectories):
            if trajectory[-1] == "EOS":
                reason = Termination.STOP
            elif len(trajectory) >= self.max_size:
                reason = Termination.SIZE_LIMIT
            elif not self.graphs[index].outer_boundary(trajectory):
                reason = Termination.EXHAUSTED
            else:
                reason = None
            self.dones.append(reason is not None)
            self.termination_reasons.append(reason)
        seeds = [self.data[i][0] for i in range(self.bs)]
        nodes = [self.data[i] for i in range(self.bs)]
        x_seeds = self.make_single_node_encoding(seeds)
        x_nodes = self.make_nodes_encoding(nodes)
        return x_seeds, x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int]):
        assert len(new_nodes) == len(index)
        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            if v != "EOS" and v not in self.graphs[i].outer_boundary(
                self.trajectories[i]
            ):
                raise ValueError(
                    f"node {v} is outside episode {i}'s current boundary"
                )
            self.trajectories[i].append(v)
            
            # DEBUG: Track termination conditions
            if self.debug_mode:
                print(f"[DEBUG] Step - Trajectory {i}: added {v}, new length: {len(self.trajectories[i])}")
            
            if v == 'EOS':
                self.dones[i] = True
                self.termination_reasons[i] = Termination.STOP
                if self.debug_mode:
                    print(f"[DEBUG] Trajectory {i} terminated by EOS")
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
                self.termination_reasons[i] = Termination.SIZE_LIMIT
                if self.debug_mode:
                    print(f"[DEBUG] Trajectory {i} terminated by max_size ({self.max_size})")
            elif len(self.graphs[i].outer_boundary(self.trajectories[i])) == 0:
                self.dones[i] = True
                self.termination_reasons[i] = Termination.EXHAUSTED
                if self.debug_mode:
                    print(f"[DEBUG] Trajectory {i} terminated by empty boundary")
            else:
                full_new_nodes[i] = v
                if self.debug_mode:
                    print(f"[DEBUG] Trajectory {i} continues with node {v}")
        delta_x_nodes = self.make_single_node_encoding(full_new_nodes)
        return delta_x_nodes

    def make_single_node_encoding(self, nodes: List[Optional[int]]):
        bs = len(nodes)
        # assert bs == self.bs
        ind = np.array([[v, i] for i, v in enumerate(nodes) if v is not None], dtype=np.int64).T
        if len(ind):
            data = np.ones(ind.shape[1], dtype=np.float32)
            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)

    def make_nodes_encoding(self, nodes: List[List[int]]):
        bs = len(nodes)
        assert bs == self.bs
        ind = [[v, i] for i, vs in enumerate(nodes) for v in vs]
        ind = np.asarray(ind, dtype=np.int64).T
        if len(ind):
            data = np.ones(ind.shape[1], dtype=np.float32)
            return sp.csc_matrix((data, ind), shape=[self.n_nodes, bs])
        else:
            return sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)


class Generator:

    def __init__(self, graph: Graph, model: Agent, optimizer,
                 device: Optional[torch.device] = None,
                 entropy_coef: float = 0.01,  # STABILITY FIX 6: Non-zero entropy for exploration
                 n_rollouts: int = 10,
                 max_size: int = 25,
                 k: int = 3,
                 alpha: float = 0.85,
                 max_reward: float = 1.,
                 ppo_epochs: int = 4,
                 minibatch_size: int = 256,
                 target_kl: float = 0.03,
                 grad_clip: float = 1.0,
                 gae_lambda: float = 0.95,
                 frontier_shaping_beta: float = 0.0,
                 entropy_decay_start: float = 1e-3,
                 entropy_decay_end: float = 1e-4,
                 entropy_decay_final: float = 0.0,
                 # New parameters for STOP action and penalties
                 target_size_k: int = 8,
                 lambda_size: float = 0.5,
                 lambda_radius: float = 1.0,
                 lambda_radius_min: float = 0.2,
                 step_cost: float = 0.001,
                 stop_bias: float = 0.5,
                 fidelity_thresh: float = 0.9,
                 # Reward scaling and STOP parameters
                 reward_scale: float = 1.0,
                 stop_bonus_ok: float = 0.5,
                 stop_penalty_bad: float = 0.2,
                 # PPO stability parameters
                 ppo_clip: float = 0.1,
                 clip_range_policy: float = 0.2,
                 clip_range_value: float = 0.5,
                 kl_target: float = 0.015,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 max_grad_norm: float = 0.5,
                 use_value_clip: bool = False,
                 adaptive_kl_penalty: bool = False,
                 total_epochs: int = 10,
                 # Entropy scheduling (STABILITY FIX 6: ensuring healthy exploration)
                 entropy_coeff_start: float = 0.01,
                 entropy_coeff_end: float = 0.001,
                 # Causal reward scheduling
                 causal_warmup_epochs: int = 5,
                 # Causal mechanism parameters
                 use_causal: bool = False,
                 causal_explainer: Optional[Union['NodeLevelCausalExplainer', 'GraphLevelCausalExplainer']] = None,
                 causal_weight: float = 0.5,
                 # Debug parameters
                 debug_mode: bool = False,
                 # Args for reward normalization
                 args=None,
                 # Additional hyperparameters (previously hard-coded)
                 value_loss_coef: float = 0.3,
                 value_loss_scale: float = 0.5,
                 confidence_threshold: float = 0.5,
                 confidence_tolerance: float = 0.05,
                 causal_action_weight: float = 0.2,
                 causal_gamma: float = 0.5,
                 lambda_conf: float = 0.1,
                 ice_score_default: float = -5.0,
                 stop_mask_value: float = -1e9,
                 min_steps_before_stop: int = 8,
                 kl_adapt_high_multiplier: float = 2.0,
                 kl_adapt_low_multiplier: float = 0.5,
                 kl_adapt_high_threshold: float = 2.0,
                 kl_adapt_low_threshold: float = 0.5,
                 entropy_decay_fraction: float = 0.7):
        self.graph = graph
        if use_value_clip:
            raise ValueError("Sileo v1 does not implement PPO value clipping")
        if adaptive_kl_penalty:
            raise ValueError(
                "Sileo v1 does not implement an adaptive KL penalty"
            )
        self.model = model
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef
        self.max_reward = max_reward
        self.n_nodes = self.graph.n_nodes
        self.max_size = max_size
        self.n_rollouts = n_rollouts
        self.conv = GraphConv(graph, k, alpha)
        self.nodefeats = None
        self.old_logps = None
        
        # PPO parameters
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.grad_clip = grad_clip
        self.gae_lambda = gae_lambda
        self.frontier_shaping_beta = frontier_shaping_beta
        
        # Entropy decay parameters
        self.entropy_decay_start = entropy_decay_start
        self.entropy_decay_end = entropy_decay_end
        self.entropy_decay_final = entropy_decay_final
        self.current_entropy_coef = entropy_decay_start

        # New parameters for STOP action and penalties
        self.target_size_k = target_size_k
        self.lambda_size = lambda_size
        self.lambda_radius = lambda_radius
        self.lambda_radius_min = lambda_radius_min
        self.step_cost = step_cost
        self.stop_bias = stop_bias
        self.fidelity_thresh = fidelity_thresh
        
        # Reward scaling and STOP parameters
        self.reward_scale = reward_scale
        self.stop_bonus_ok = stop_bonus_ok
        self.stop_penalty_bad = stop_penalty_bad
        
        # PPO stability parameters
        self.clip_range_policy = clip_range_policy
        self.clip_range_value = clip_range_value
        self.vf_clip = clip_range_value if use_value_clip else None
        self.gamma = gamma
        self.use_gae = True
        self.reward_norm = False
        self.kl_target = kl_target
        self.max_grad_norm = max_grad_norm
        self.adaptive_kl_penalty = adaptive_kl_penalty
        self.total_epochs = total_epochs
        
        # Adaptive KL penalty parameters
        self.kl_penalty_coef = 0.0
        self.kl_penalty_target = kl_target
        
        # KL early stopping parameters
        self.kl_consecutive_threshold = 3  # Stop after N consecutive high-KL minibatches
        self.kl_consecutive_count = 0
        
        # Entropy scheduling
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end = entropy_coeff_end
        
        # Causal reward scheduling
        self.causal_warmup_epochs = causal_warmup_epochs
        self.current_epoch = 0
        
        # Monitoring
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0
        self.last_training_stats = {}
        self.last_rollout_metrics = {}

        # Causal mechanism integration
        self.use_causal = use_causal
        self.causal_explainer = causal_explainer
        self.causal_weight = causal_weight
        self.causal_reward_calculator = None
        if self.use_causal and self.causal_explainer is not None:
            # This will be initialized when we have access to the GNN model
            pass
        
        # Debug mode
        self.debug_mode = debug_mode
        
        # Edge dropout configuration (from args)
        # IMPORTANT: Generator does NOT apply dropout, only passes config to Environment
        # Edge dropout lives in Environment (world dynamics), not Generator (policy)
        self.edge_dropout_prob = getattr(args, 'random_edge_drop', 0.0) if args else 0.0
        self.preserve_connectivity = getattr(args, 'preserve_connectivity', False) if args else False
        
        # Store args for reward normalization
        self.args = args
        
        # Store additional hyperparameters (previously hard-coded)
        self.value_loss_coef = value_loss_coef
        self.value_loss_scale = value_loss_scale
        self.confidence_threshold = confidence_threshold
        self.confidence_tolerance = confidence_tolerance
        self.causal_action_weight = causal_action_weight
        self.causal_gamma = causal_gamma
        self.lambda_conf = lambda_conf
        self.ice_score_default = ice_score_default
        self.stop_mask_value = stop_mask_value
        self.min_steps_before_stop = min_steps_before_stop
        self.kl_adapt_high_multiplier = kl_adapt_high_multiplier
        self.kl_adapt_low_multiplier = kl_adapt_low_multiplier
        self.kl_adapt_high_threshold = kl_adapt_high_threshold
        self.kl_adapt_low_threshold = kl_adapt_low_threshold
        self.entropy_decay_fraction = entropy_decay_fraction

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Causal mechanism data (will be set later)
        self.gnn_model = None
        self.full_edge_index = None
        self.target_classes = None
        self.seed_nodes = None
        self.predictor = None
        self.instance_ids = None

    def load_nodefeats(self, x):
        # Accept torch.Tensor or np.ndarray; store as np.ndarray on CPU
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.nodefeats = x

    def set_predictor(
        self,
        predictor: PredictorAdapter,
        instance_ids: List[int],
        target_nodes: List[int],
    ):
        self.predictor = predictor
        self.instance_ids = list(instance_ids)
        self.seed_nodes = list(target_nodes)

    def _identity_for_seed(self, seed: int) -> int:
        if self.predictor is None:
            raise RuntimeError("predictor context is not configured")
        if self.predictor.task == "graph_task":
            return int(self.predictor.batch[seed].item())
        return int(seed)

    def _convolve_episode_graphs(self, values, graphs):
        if not graphs or all(graph is self.graph for graph in graphs):
            return self.conv(values)
        columns = []
        for episode, graph in enumerate(graphs):
            column = values[:, episode]
            columns.append(
                GraphConv(
                    graph, self.conv.k, self.conv.alpha
                )(column)
            )
        return sp.hstack(columns, format="csc")

    def _stack_episode_sequences(self, episode_xs):
        if not episode_xs:
            return torch.zeros((0, 0), device=self.device)
        stacked = []
        for xs in episode_xs:
            if xs:
                stacked.append(torch.stack(xs))
            else:
                stacked.append(torch.zeros(0, device=self.device))
        return nn.utils.rnn.pad_sequence(stacked, batch_first=True)
    
    def update_entropy_coef(self, epoch, total_epochs):
        """Update entropy coefficient based on decay schedule"""
        if total_epochs <= 1:
            self.current_entropy_coef = self.entropy_decay_start
            return
        
        # Linear decay from start to end over first 70% of epochs
        decay_epochs = int(self.entropy_decay_fraction * total_epochs)
        if epoch < decay_epochs:
            progress = epoch / decay_epochs
            self.current_entropy_coef = self.entropy_decay_start + progress * (self.entropy_decay_end - self.entropy_decay_start)
        else:
            # Linear decay from end to final over remaining 30% of epochs
            remaining_epochs = total_epochs - decay_epochs
            if remaining_epochs > 0:
                progress = (epoch - decay_epochs) / remaining_epochs
                self.current_entropy_coef = self.entropy_decay_end + progress * (self.entropy_decay_final - self.entropy_decay_end)
            else:
                self.current_entropy_coef = self.entropy_decay_final
        
        # Update entropy coefficient used in training
        self.entropy_coef = self.current_entropy_coef

    def update_epoch(self, epoch: int):
        """
        Update current epoch for scheduling.
        
        Args:
            epoch: Current epoch (0-indexed)
        """
        self.current_epoch = epoch

    def update_adaptive_kl_penalty(self, approx_kl: float):
        """
        Update adaptive KL penalty coefficient based on current KL divergence.
        
        Args:
            approx_kl: Current approximate KL divergence
        """
        if not self.adaptive_kl_penalty:
            return
            
        # Update KL penalty coefficient based on target
        if approx_kl > self.kl_adapt_high_threshold * self.kl_penalty_target:
            self.kl_penalty_coef *= self.kl_adapt_high_multiplier
        elif approx_kl < self.kl_adapt_low_threshold * self.kl_penalty_target:
            self.kl_penalty_coef *= self.kl_adapt_low_multiplier
            
        # Clamp to reasonable range
        self.kl_penalty_coef = max(0.0, min(1.0, self.kl_penalty_coef))

    def get_effective_causal_weight(self, epoch: int) -> float:
        """
        Get effective causal weight with warmup scheduling.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Effective causal weight considering warmup
        """
        if epoch < self.causal_warmup_epochs:
            return self.causal_weight * (epoch / self.causal_warmup_epochs)
        else:
            return self.causal_weight

    def get_early_stop_rate(self, selected_nodes: List[List[int]]) -> float:
        """
        Calculate the rate of early stopping (STOP action chosen before max steps).
        
        Args:
            selected_nodes: List of trajectories
            
        Returns:
            Early stop rate (0.0 to 1.0)
        """
        early_stops = 0
        total_trajectories = len(selected_nodes)
        
        if total_trajectories == 0:
            return 0.0
            
        for traj in selected_nodes:
            # Check if trajectory ended early (before max_size)
            if len(traj) < self.max_size and traj[-1] != 'EOS':
                early_stops += 1
            elif traj[-1] == 'EOS' and len(traj) < self.max_size + 1:
                early_stops += 1
                
        return early_stops / total_trajectories

    def get_effective_lambda_radius(self, epoch: int, total_epochs: int) -> float:
        """
        Get effective lambda_radius with linear decay schedule.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Effective lambda_radius considering decay
        """
        if total_epochs > 1:
            progress = epoch / (total_epochs - 1)
            return self.lambda_radius + progress * (self.lambda_radius_min - self.lambda_radius)
        else:
            return self.lambda_radius

    def composite_loss_J(self, masked_pred, original_pred, subgraph, lam_size, lam_radius, lam_sim, masked_embed=None, original_embed=None):
        """
        Compute composite objective J(G) for a specific graph state.
        
        Args:
            masked_pred: Prediction logits for masked subgraph
            original_pred: Original prediction logits (target)
            subgraph: Current subgraph (nodes or edge_index)
            lam_size: Size penalty weight
            lam_radius: Radius/depth penalty weight  
            lam_sim: Similarity penalty weight
            masked_embed: Embedding for masked subgraph (optional)
            original_embed: Original embedding (optional)
            
        Returns:
            J: Composite loss value
            parts: Dictionary with individual loss components
        """
        # 1) prediction loss (CE between masked and original predictions)
        # Convert original_pred logits to class index for cross-entropy target
        if original_pred.dim() > 1:
            original_target = torch.argmax(original_pred, dim=-1)
        else:
            original_target = torch.argmax(original_pred)
        
        if masked_pred.dim() == 1:
            masked_pred = masked_pred.unsqueeze(0)
            if original_target.dim() == 0:
                original_target = original_target.unsqueeze(0)
        elif masked_pred.dim() > 1 and original_target.dim() == 0:
            original_target = original_target.unsqueeze(0)

        num_classes = masked_pred.size(-1)
        if original_pred.size(-1) != num_classes:
            raise ValueError(
                f"predictor class mismatch: {original_pred.size(-1)} != "
                f"{num_classes}"
            )
        if bool(((original_target < 0) | (original_target >= num_classes)).any()):
            raise ValueError("target class is outside predictor output dimensions")
        original_target = original_target.long()
        
        ce = torch.nn.functional.cross_entropy(masked_pred, original_target, reduction="none")
        ce = ce.mean()  # scalar

        # 2) size penalty (|G|)
        if hasattr(subgraph, "nodes"):
            size_term = float(len(subgraph.nodes))
        elif hasattr(subgraph, "size"):
            size_term = float(subgraph.size()[1])  # edge_index size
        elif isinstance(subgraph, (list, tuple)):
            size_term = float(len(subgraph))
        else:
            size_term = 0.0

        # 3) radius/depth penalty
        if isinstance(subgraph, (list, tuple)):
            depth_term = self.graph.subgraph_depth(subgraph)
        else:
            depth_term = 0.0

        # 4) similarity loss
        sim_term = 0.0
        if masked_embed is not None and original_embed is not None:
            if masked_embed.shape != original_embed.shape:
                raise ValueError(
                    f"predictor embedding mismatch: {masked_embed.shape} != "
                    f"{original_embed.shape}"
                )
            sim_term = torch.norm(masked_embed - original_embed, p=2)

        J = ce + lam_size * size_term + lam_radius * depth_term + lam_sim * sim_term
        return J, dict(ce=float(ce.item()), size=float(size_term), depth=float(depth_term), sim=float(sim_term))

    def class_preserved(self, masked_pred, original_pred):
        """
        Check if class prediction is preserved between masked and original predictions.
        
        FIXED: Use strict argmax match to avoid softmax ties causing false positives.
        This ensures STOP bonus only applies when prediction is truly preserved.
        
        Args:
            masked_pred: Prediction logits for masked subgraph
            original_pred: Original prediction logits (target)
            
        Returns:
            bool: True if argmax(masked_pred) == argmax(original_pred) with strict equality
        """
        # FIXED: Use strict argmax comparison to avoid softmax ties
        masked_class = torch.argmax(masked_pred, dim=-1).item()
        original_class = torch.argmax(original_pred, dim=-1).item()
        
        # Additional confidence check: ensure the winning class has reasonable confidence
        masked_confidence = torch.softmax(masked_pred, dim=-1).max().item()
        original_confidence = torch.softmax(original_pred, dim=-1).max().item()
        
        # Require both class match AND reasonable confidence to avoid ties
        return (masked_class == original_class and 
                masked_confidence > self.confidence_threshold and 
                original_confidence > self.confidence_threshold)

    def _strip_eos(self, traj):
        """Strip EOS tokens from trajectory"""
        return [v for v in traj if v != 'EOS']
    
    def apply_frontier_shaping(self, selected_nodes, rewards):
        """Apply potential-based frontier shaping to rewards"""
        if self.frontier_shaping_beta <= 0:
            return rewards
        
        shaped_rewards = rewards.copy()
        for i, nodes in enumerate(selected_nodes):
            base = self._strip_eos(nodes)
            if len(base) < 2:
                continue
            
            # Apply shaping: β * (Φ(s') - Φ(s))
            # For terminal rewards, we shape based on the final frontier size
            if i < len(rewards) and rewards[i] != 0:  # Non-zero reward (terminal)
                s_prev, s_curr = base[:-1], base
                phi_prev = len(self.graph.outer_boundary(s_prev))
                phi_curr = len(self.graph.outer_boundary(s_curr))
                shaping_bonus = self.frontier_shaping_beta * (phi_curr - phi_prev)
                shaped_rewards[i] += shaping_bonus
        
        return shaped_rewards

    def generate(self, seeds: List[int], max_size: Optional[int] = None):
        """
        Generate explanations (inference mode, no dropout).
        
        Args:
            seeds: List of seed nodes
            max_size: Maximum trajectory length (default: self.max_size)
        
        Returns:
            List of generated trajectories
        
        Note:
            Inference always uses the original graph with NO edge dropout.
            This ensures explanations are generated on the true world dynamics.
        """
        max_size = self.max_size if max_size is None else max_size
        
        # Inference: always use original graph (training=False, no dropout)
        env = ExpansionEnv(
            self.graph,
            [[s] for s in seeds],
            max_size,
            self.debug_mode,
            training=False,  # No dropout during inference
            edge_dropout_prob=0.0,
            preserve_connectivity=False
        )
        
        self.model.eval()
        with torch.no_grad():
            episodes, *_ = self._sample_trajectories(env)
        return episodes


    def sample_episodes(self, seeds: List[int], max_size: Optional[int] = None, training: bool = True):
        """
        Sample episodes for training or inference.
        
        Args:
            seeds: List of seed nodes
            max_size: Maximum trajectory length (default: self.max_size)
            training: If True, apply edge dropout. If False, use original graph.
        
        Returns:
            Sampled trajectories and associated data
        
        Note:
            When training=True, Environment handles edge dropout (world dynamics).
            Generator only passes configuration, never applies dropout itself.
        """
        max_size = self.max_size if max_size is None else max_size
        
        # Create environment with explicit training flag
        # Environment handles edge dropout, NOT Generator
        env = ExpansionEnv(
            self.graph,
            [[s] for s in seeds],
            max_size,
            self.debug_mode,
            training=training,  # Explicit: Environment decides dropout
            edge_dropout_prob=self.edge_dropout_prob,
            preserve_connectivity=self.preserve_connectivity
        )
        result = self._sample_trajectories(env)
        reasons = env.termination_reasons or []
        denominator = max(1, len(reasons))
        self.last_rollout_metrics = {
            "realized_dropout_rate": env.realized_drop_rate,
            "stop_rate": reasons.count(Termination.STOP) / denominator,
            "exhaustion_rate": reasons.count(Termination.EXHAUSTED) / denominator,
            "size_limit_rate": reasons.count(Termination.SIZE_LIMIT) / denominator,
            "truncation_rate": reasons.count(Termination.TRUNCATED) / denominator,
        }
        return result

    def sample_rollouts(self, prefix: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        bs = len(prefix)
        if bs * self.n_rollouts > 10000:
            return self._sample_rollouts_loop(prefix, max_size)
        else:
            return self._sample_rollouts_batch(prefix, max_size)

    def _compute_delta_rewards(self, selected_nodes, lam_size, lam_radius, lam_sim):
        raise RuntimeError(
            "placeholder delta rewards are disabled; PPO uses "
            "_compute_action_rewards_real_gnn"
        )
        """
        Compute incremental (delta) rewards per step that reflect causal screening.
        
        Args:
            selected_nodes: List of trajectories (each trajectory is a list of nodes)
            lam_size: Size penalty weight
            lam_radius: Radius penalty weight
            lam_sim: Similarity penalty weight
            
        Returns:
            rewards: numpy array of shape [batch_size, max_steps] with delta rewards
        """
        batch_size = len(selected_nodes)
        max_steps = max(len(traj) for traj in selected_nodes) if selected_nodes else 0
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        
        # Cache for empty subgraph J (baseline)
        cached_J_empty = None
        
        for i, trajectory in enumerate(selected_nodes):
            if not trajectory:
                continue
                
            # Precompute original prediction and embedding for this seed
            seed_node = trajectory[0]
            
            # Get original prediction and embedding (this would need access to the GNN model)
            # For now, we'll use placeholder values - this should be passed from the caller
            original_pred = torch.tensor([0.0, 1.0])  # Placeholder - should be actual prediction
            original_embed = torch.zeros(64)  # Placeholder - should be actual embedding
            
            J_prev = None
            aux_logs = []
            
            for t, node in enumerate(trajectory):
                if node == 'EOS':
                    break
                    
                # Current subgraph up to step t
                current_subgraph = trajectory[:t+1]
                
                # Compute masked prediction and embedding for current subgraph
                # This would need access to the GNN model - placeholder for now
                masked_pred = torch.tensor([0.1, 0.9])  # Placeholder
                masked_embed = torch.zeros(64)  # Placeholder
                
                # Compute J(G_k) using composite loss
                J_k, parts = self.composite_loss_J(
                    masked_pred, original_pred, current_subgraph,
                    lam_size, lam_radius, lam_sim,
                    masked_embed=masked_embed, original_embed=original_embed
                )
                
                if J_prev is None:
                    # FIXED: First step baseline is J(empty) for the same graph, not 0.0
                    # This ensures r₁ is properly scaled relative to the empty subgraph
                    if cached_J_empty is None:
                        # Compute J for empty subgraph (just the seed node)
                        empty_subgraph = [seed_node]
                        # FIXED: Use actual empty subgraph prediction instead of placeholder
                        # This should be computed using the actual GNN model with empty subgraph
                        # For now, we'll use a more realistic baseline
                        empty_masked_pred = torch.tensor([0.5, 0.5])  # Placeholder - should be actual empty prediction
                        empty_masked_embed = torch.zeros(64)  # Placeholder - should be actual empty embedding
                        cached_J_empty, _ = self.composite_loss_J(
                            empty_masked_pred, original_pred, empty_subgraph,
                            lam_size, lam_radius, lam_sim,
                            masked_embed=empty_masked_embed, original_embed=original_embed
                        )
                    J_prev = cached_J_empty
                
                # FIXED: Apply reward scaling to un-choke learning
                # Incremental (delta) reward: improvement yields positive reward
                r_k = self.reward_scale * -(J_k - J_prev)
                J_prev = J_k
                
                # STOP handling - check if this is a STOP action
                if t == len(trajectory) - 1 and trajectory[-1] == 'EOS':
                    # FIXED: Strict STOP condition check with confidence gate
                    # 1. Class must be preserved (with confidence check)
                    class_ok = self.class_preserved(masked_pred, original_pred)
                    # 2. Size must be <= target_size_k (strict inequality)
                    size_ok = len(current_subgraph) <= self.target_size_k
                    
                    # FIXED: Confidence gate - only grant bonus if prediction confidence is maintained
                    confidence_ok = False
                    if class_ok:
                        masked_probs = torch.softmax(masked_pred, dim=-1)
                        original_probs = torch.softmax(original_pred, dim=-1)
                        original_class = torch.argmax(original_pred, dim=-1).item()
                        
                        # Check if masked prediction confidence is within tolerance of original
                        masked_confidence = masked_probs[original_class].item()
                        original_confidence = original_probs[original_class].item()
                        confidence_ok = masked_confidence >= (original_confidence - self.confidence_tolerance)
                    
                    # FIXED: Only apply STOP bonus if ALL conditions are met
                    if class_ok and size_ok and confidence_ok:
                        r_k = r_k + self.stop_bonus_ok
                    else:
                        # Apply penalty for premature or incorrect STOP
                        r_k = r_k - self.stop_penalty_bad
                
                rewards[i, t] = float(r_k)
                aux_logs.append({"t": t, **parts, "r": float(r_k)})
        
        return rewards

    def _compute_delta_rewards_real_gnn(self, selected_nodes, lam_size, lam_radius, lam_sim):
        raise RuntimeError(
            "legacy aligned-by-node rewards are disabled; PPO uses "
            "_compute_action_rewards_real_gnn"
        )
        """
        Compute per-step incremental rewards using ORIGINAL graph predictor.
        
        CRITICAL: Rewards must ALWAYS use full_edge_index (original graph),
                  never the dropped graph from training environment.
        
        Args:
            selected_nodes: List of trajectories
            lam_size: Size penalty weight
            lam_radius: Radius penalty weight
            lam_sim: Similarity penalty weight
        
        Returns:
            numpy array of rewards
        """
        # BRUTAL ASSERTION: Ensure we're using original graph for rewards
        assert hasattr(self, 'full_edge_index'), \
            "full_edge_index not set - cannot compute rewards on original graph"
        assert self.full_edge_index is not None, \
            "full_edge_index is None - rewards need original graph, not dropped graph"
        
        # Log on first call to verify original graph usage
        if not hasattr(self, '_reward_graph_logged'):
            print(f"[Reward] Using original graph: {self.full_edge_index.size(1)} edges", flush=True)
            print(f"[Reward] Predictor GNN always sees full graph, never dropped edges", flush=True)
            self._reward_graph_logged = True
        """
        Compute incremental (delta) rewards per step using real GNN calls.
        
        Args:
            selected_nodes: List of trajectories (each trajectory is a list of nodes)
            lam_size: Size penalty weight
            lam_radius: Radius penalty weight
            lam_sim: Similarity penalty weight
            
        Returns:
            rewards: numpy array of shape [batch_size, max_steps] with delta rewards
        """
        batch_size = len(selected_nodes)
        max_steps = max(len(traj) for traj in selected_nodes) if selected_nodes else 0
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        
        # Cache for empty subgraph J (baseline) per episode
        cached_J_empty = {}
        
        # Debug episode tracking
        if getattr(self.args, 'dry_run_debug', False):
            self._last_episode_debug = []
        
        for i, trajectory in enumerate(selected_nodes):
            if not trajectory:
                continue
                
            # Precompute original prediction and embedding for this seed
            seed_node = trajectory[0]
            
            # Get original prediction and embedding using the frozen GNN
            if self.gnn_model is not None and self.nodefeats is not None and self.full_edge_index is not None:
                # Convert nodefeats to tensor if needed
                if torch.is_tensor(self.nodefeats):
                    nodefeats_tensor = self.nodefeats.to(self.device)
                else:
                    nodefeats_tensor = torch.from_numpy(self.nodefeats).float().to(self.device)
                
                # Compute original prediction for full graph
                edge_weights = torch.ones(self.full_edge_index.size(1), device=self.device)
                original_pred = self.gnn_model(nodefeats_tensor, self.full_edge_index, edge_weights)
                
                # For node classification, get prediction for seed node only
                if original_pred.dim() > 1 and original_pred.size(0) > seed_node:
                    original_pred = original_pred[seed_node].unsqueeze(0)
                elif original_pred.dim() == 1:
                    original_pred = original_pred.unsqueeze(0)
                
                # Compute original embedding
                original_embed = self.gnn_model.embedding(nodefeats_tensor, self.full_edge_index, edge_weights)
                if original_embed.dim() > 1 and original_embed.size(0) > seed_node:
                    original_embed = original_embed[seed_node]
                elif original_embed.dim() == 1:
                    original_embed = original_embed
            else:
                # Fallback to placeholder values if GNN not available
                original_pred = torch.tensor([0.0, 1.0], device=self.device)
                original_embed = torch.zeros(64, device=self.device)
            
            J_prev = None
            aux_logs = []
            
            for t, node in enumerate(trajectory):
                if node == 'EOS':
                    # Apply STOP reward before terminating
                    current_subgraph = trajectory[:t]  # Exclude EOS
                    if len(current_subgraph) > 0:
                        # Compute final J_k for STOP reward
                        masked_pred, masked_embed = self._compute_subgraph_prediction(current_subgraph)
                        J_k, parts = self.composite_loss_J(
                            masked_pred, original_pred, current_subgraph,
                            lam_size, lam_radius, lam_sim,
                            masked_embed=masked_embed, original_embed=original_embed
                        )
                        
                        # Apply delta reward
                        if J_prev is not None:
                            r_k = self.reward_scale * -(J_k - J_prev)
                        else:
                            r_k = 0.0
                        
                        # CAUSAL REWARD INTEGRATION: Add causal reward if enabled (STOP case)
                        if self.use_causal and self.causal_reward_calculator is not None:
                            # Get target class for this seed node
                            if hasattr(self, 'target_classes') and self.target_classes is not None:
                                if hasattr(self, 'seed_nodes') and self.seed_nodes is not None:
                                    # Find the index of current seed in seed_nodes list
                                    try:
                                        seed_idx = self.seed_nodes.index(seed_node)
                                        target_y = self.target_classes[seed_idx].item() if torch.is_tensor(self.target_classes[seed_idx]) else self.target_classes[seed_idx]
                                        
                                        # Create a simple graph object for causal reward computation
                                        graph = Data(x=nodefeats_tensor, edge_index=self.full_edge_index)
                                        
                                        # Compute causal reward for final subgraph at STOP
                                        causal_r = self.causal_reward_calculator.compute_causal_reward(
                                            graph, current_subgraph, target_y
                                        )
                                        
                                        # Combine base reward with causal reward
                                        r_k = self.causal_reward_calculator.compute_combined_reward(r_k, causal_r)
                                        
                                    except (ValueError, IndexError) as e:
                                        # Seed node not in evaluation set, skip causal reward
                                        pass
                        
                        # Apply STOP bonus/penalty with confidence and size gates
                        class_ok = torch.argmax(masked_pred) == torch.argmax(original_pred)
                        size_ok = len(current_subgraph) <= self.target_size_k
                        
                        # Confidence gate
                        conf_ok = True
                        if self.fidelity_thresh > 0:
                            po = torch.softmax(original_pred, -1)[..., torch.argmax(original_pred)]
                            pm = torch.softmax(masked_pred, -1)[..., torch.argmax(original_pred)]
                            conf_ok = (pm >= po - self.fidelity_thresh)
                        
                        if class_ok and conf_ok and size_ok:
                            r_k += self.stop_bonus_ok
                        else:
                            r_k -= self.stop_penalty_bad
                        
                        rewards[i, t-1] = float(r_k)  # Apply to last real step
                    break
                    
                # Current subgraph up to step t
                current_subgraph = trajectory[:t+1]
                
                # Compute masked prediction and embedding for current subgraph
                masked_pred, masked_embed = self._compute_subgraph_prediction(current_subgraph)
                
                # Compute J(G_k) using composite loss
                J_k, parts = self.composite_loss_J(
                    masked_pred, original_pred, current_subgraph,
                    lam_size, lam_radius, lam_sim,
                    masked_embed=masked_embed, original_embed=original_embed
                )
                
                if J_prev is None:
                    # First step baseline is J(empty) for the same graph
                    if i not in cached_J_empty:
                        # Compute J for empty subgraph (just the seed node)
                        empty_subgraph = [seed_node]
                        empty_masked_pred, empty_masked_embed = self._compute_subgraph_prediction(empty_subgraph)
                        cached_J_empty[i], _ = self.composite_loss_J(
                            empty_masked_pred, original_pred, empty_subgraph,
                            lam_size, lam_radius, lam_sim,
                            masked_embed=empty_masked_embed, original_embed=original_embed
                        )
                    J_prev = cached_J_empty[i]
                
                # Apply reward scaling to un-choke learning
                # Incremental (delta) reward: improvement yields positive reward
                r_k = self.reward_scale * -(J_k - J_prev)
                J_prev = J_k
                
                # CAUSAL REWARD INTEGRATION: Add causal reward if enabled
                if self.use_causal and self.causal_reward_calculator is not None:
                    # Get target class for this seed node
                    if hasattr(self, 'target_classes') and self.target_classes is not None:
                        if hasattr(self, 'seed_nodes') and self.seed_nodes is not None:
                            # Find the index of current seed in seed_nodes list
                            try:
                                seed_idx = self.seed_nodes.index(seed_node)
                                target_y = self.target_classes[seed_idx].item() if torch.is_tensor(self.target_classes[seed_idx]) else self.target_classes[seed_idx]
                                
                                # Create a simple graph object for causal reward computation
                                from torch_geometric.data import Data
                                graph = Data(x=nodefeats_tensor, edge_index=self.full_edge_index)
                                
                                # Compute causal reward for current subgraph
                                causal_r = self.causal_reward_calculator.compute_causal_reward(
                                    graph, current_subgraph, target_y
                                )
                                
                                # REVERT: Don't scale causal reward - keeps it conservative
                                # Causal rewards stay in [-1, 1] range while base rewards are scaled by reward_scale
                                # This gives causal ~6-15% effective weight, preventing it from dominating
                                # Combine base reward with causal reward (unscaled)
                                r_k = self.causal_reward_calculator.compute_combined_reward(r_k, causal_r)
                                
                            except (ValueError, IndexError) as e:
                                # Seed node not in evaluation set, skip causal reward
                                pass
                
                rewards[i, t] = float(r_k)
                aux_logs.append({"t": t, **parts, "r": float(r_k)})
                
                # Debug tracking for dry run
                if getattr(self.args, 'dry_run_debug', False) and i == 0:  # Track first episode only
                    self._last_episode_debug.append({
                        "t": t,
                        "J_prev": float(J_prev) if J_prev is not None else 0.0,
                        "J_k": float(J_k),
                        "delta": float(J_k - J_prev) if J_prev is not None else 0.0,
                        "r_k": float(r_k),
                        "action": node,
                        "size": len(current_subgraph)
                    })
        
        return rewards

    def _compute_action_rewards_real_gnn(
        self, trajectories, rollout_buffer, lam_size, lam_radius, lam_sim
    ):
        """Return one reward for every collected action, never for the seed."""
        if self.predictor is None:
            raise RuntimeError("PPO rewards require a configured PredictorAdapter")
        rewards_by_step = {}
        transition_by_step = {
            (episode, step): transition
            for episode, step, transition in zip(
                rollout_buffer["episode_indices"],
                rollout_buffer["step_indices"],
                rollout_buffer["transitions"],
            )
        }

        for episode_index, trajectory in enumerate(trajectories):
            seed = int(trajectory[0])
            instance_id = self._identity_for_seed(seed)
            original = self.predictor.original(
                instance_id=instance_id,
                target_node=seed,
            )
            previous_pred = self.predictor.predict(
                [seed],
                instance_id=instance_id,
                target_node=seed,
                target_class=original.target_class,
            )
            previous_cost, _ = self.composite_loss_J(
                previous_pred.logits,
                original.logits,
                [seed],
                lam_size,
                lam_radius,
                lam_sim,
                previous_pred.embedding,
                original.embedding,
            )
            selected = [seed]

            for action_step, action in enumerate(trajectory[1:]):
                transition = transition_by_step.get((episode_index, action_step))
                if transition is None:
                    continue
                if action == "EOS":
                    class_ok = previous_pred.logits.argmax() == original.logits.argmax()
                    conf_ok = True
                    if self.fidelity_thresh > 0:
                        target = original.target_class
                        original_conf = torch.softmax(original.logits, -1)[target]
                        selected_conf = torch.softmax(previous_pred.logits, -1)[target]
                        conf_ok = bool(
                            selected_conf >= original_conf - self.fidelity_thresh
                        )
                    reward = (
                        self.stop_bonus_ok
                        if class_ok and conf_ok and len(selected) <= self.target_size_k
                        else -self.stop_penalty_bad
                    )
                    reward_terms = {
                        "base": 0.0,
                        "stop": float(reward),
                        "causal_raw": 0.0,
                    }
                    transition.termination = Termination.STOP
                else:
                    selected.append(int(action))
                    current_pred = self.predictor.predict(
                        selected,
                        instance_id=instance_id,
                        target_node=seed,
                        target_class=original.target_class,
                    )
                    current_cost, _ = self.composite_loss_J(
                        current_pred.logits,
                        original.logits,
                        selected,
                        lam_size,
                        lam_radius,
                        lam_sim,
                        current_pred.embedding,
                        original.embedding,
                    )
                    reward = self.reward_scale * float(
                        (previous_cost - current_cost).item()
                    )
                    reward_terms = {
                        "base": float(reward),
                        "stop": 0.0,
                        "causal_raw": 0.0,
                    }
                    previous_cost = current_cost
                    previous_pred = current_pred
                    if len(selected) >= self.max_size:
                        transition.termination = Termination.SIZE_LIMIT

                if self.use_causal and self.causal_reward_calculator is not None:
                    causal = self.causal_reward_calculator.compute_causal_reward(
                        Data(
                            x=self.predictor.x,
                            edge_index=self.predictor.edge_index,
                        ),
                        selected,
                        original.target_class,
                    )
                    reward = self.causal_reward_calculator.compute_combined_reward(
                        reward, causal
                    )
                    reward_terms["causal_raw"] = float(causal)
                    reward_terms["causal_weight"] = float(
                        self.causal_reward_calculator.causal_weight
                    )
                transition.reward = float(reward)
                transition.reward_terms = reward_terms
                rewards_by_step[(episode_index, action_step)] = float(reward)

        rewards = torch.tensor(
            [
                rewards_by_step[(episode, step)]
                for episode, step in zip(
                    rollout_buffer["episode_indices"],
                    rollout_buffer["step_indices"],
                )
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return rewards

    def _compute_subgraph_prediction(
        self, subgraph_nodes, instance_id=None, target_node=None
    ):
        """
        Compute prediction and embedding for a subgraph using the frozen GNN.
        
        Args:
            subgraph_nodes: List of nodes in the subgraph
            
        Returns:
            masked_pred: Prediction logits for the subgraph
            masked_embed: Embedding for the subgraph
        """
        if self.predictor is not None:
            target_node = subgraph_nodes[0] if target_node is None else target_node
            if instance_id is None:
                instance_id = self._identity_for_seed(int(target_node))
            prediction = self.predictor.predict(
                subgraph_nodes,
                instance_id=int(instance_id),
                target_node=int(target_node),
            )
            return prediction.logits.unsqueeze(0), prediction.embedding

        if self.gnn_model is None or self.nodefeats is None or self.full_edge_index is None:
            raise RuntimeError("frozen predictor context has not been configured")
        
        # Convert nodefeats to tensor if needed
        if torch.is_tensor(self.nodefeats):
            nodefeats_tensor = self.nodefeats.to(self.device)
        else:
            nodefeats_tensor = torch.from_numpy(self.nodefeats).float().to(self.device)
        
        # Create subgraph edge index
        node_set = set(subgraph_nodes)
        src, dst = self.full_edge_index[0], self.full_edge_index[1]
        edge_mask = torch.tensor([
            (src[i].item() in node_set and dst[i].item() in node_set)
            for i in range(self.full_edge_index.size(1))
        ], dtype=torch.bool, device=self.device)
        
        subgraph_edge_index = self.full_edge_index[:, edge_mask]
        
        # Forward pass for subgraph
        edge_weights = torch.ones(subgraph_edge_index.size(1), device=self.device)
        masked_pred = self.gnn_model(
            nodefeats_tensor,
            subgraph_edge_index,
            edge_weights=edge_weights,
        )
        
        # For node classification, get prediction for seed node only
        seed_node = subgraph_nodes[0]
        if masked_pred.dim() > 1 and masked_pred.size(0) > seed_node:
            masked_pred = masked_pred[seed_node].unsqueeze(0)
        elif masked_pred.dim() == 1:
            masked_pred = masked_pred.unsqueeze(0)
        
        # Compute subgraph embedding
        masked_embed = self.gnn_model.embedding(
            nodefeats_tensor, subgraph_edge_index, edge_weights=edge_weights
        )
        if masked_embed.dim() > 1 and masked_embed.size(0) > seed_node:
            masked_embed = masked_embed[seed_node]
        elif masked_embed.dim() == 1:
            masked_embed = masked_embed
        
        return masked_pred, masked_embed

    def update_ppo(self, seeds: List[int], fn, lam_size=0.0, lam_radius=0.0, lam_sim=0.0, entropy_coef=1e-3):
        """
        PPO update with delta rewards from real GNN calls and proper old log-prob storage.
        
        Args:
            seeds: List of seed nodes for rollout
            fn: Reward function (not used in delta reward mode)
            lam_size: Size penalty weight
            lam_radius: Radius penalty weight  
            lam_sim: Similarity penalty weight
            entropy_coef: Entropy coefficient for this epoch
            
        Returns:
            selected_nodes: List of trajectories
            mean_reward: Average reward across trajectories
            policy_loss: Average policy loss
            mean_length: Average trajectory length
        """
        bs = len(seeds)
        self.model.train()
        self.last_training_stats = {}
        
        # Rollout trajectories with buffered data
        selected_nodes, logps, values, entropys, rollout_buffer = self.sample_episodes(seeds)
        if self.causal_reward_calculator is not None:
            self.causal_reward_calculator.set_weight(
                self.get_effective_causal_weight(self.current_epoch)
            )
        lengths = torch.LongTensor(
            [max(0, len(x) - 1) for x in selected_nodes]
        ).to(self.device)
        if not rollout_buffer["transitions"]:
            return selected_nodes, 0.0, 0.0, 0.0

        rewards_flat = self._compute_action_rewards_real_gnn(
            selected_nodes, rollout_buffer, lam_size, lam_radius, lam_sim
        )
        reward_term_names = {
            key
            for transition in rollout_buffer["transitions"]
            for key in (transition.reward_terms or {})
        }
        reward_term_metrics = {
            f"reward_{key}": float(
                np.mean([
                    (transition.reward_terms or {}).get(key, 0.0)
                    for transition in rollout_buffer["transitions"]
                ])
            )
            for key in reward_term_names
        }
        old_logps = torch.stack(rollout_buffer['old_logps']).to(self.device)
        old_values = torch.stack(rollout_buffer['old_values']).to(self.device)
        max_steps = max(
            1,
            max(rollout_buffer["step_indices"]) + 1,
        )
        rewards = torch.zeros((bs, max_steps), device=self.device)
        values_matrix = torch.zeros_like(rewards)
        next_values = torch.zeros_like(rewards)
        valid = torch.zeros_like(rewards, dtype=torch.bool)
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
        step_indices = torch.tensor(
            list(
                zip(
                    rollout_buffer["episode_indices"],
                    rollout_buffer["step_indices"],
                )
            ),
            device=self.device,
        )
        for index, (episode, step) in enumerate(step_indices.tolist()):
            rewards[episode, step] = rewards_flat[index]
            values_matrix[episode, step] = old_values[index]
            valid[episode, step] = True
            terminated[episode, step] = rollout_buffer["transitions"][
                index
            ].terminated
        next_values[:, :-1] = values_matrix[:, 1:]

        if getattr(self.args, 'traj_reward_norm', False):
            for episode in range(bs):
                episode_mask = valid[episode]
                if int(episode_mask.sum()) > 1:
                    values = rewards[episode, episode_mask]
                    rewards[episode, episode_mask] = (
                        values - values.mean()
                    ) / values.std().clamp_min(1e-6)

        advantages, returns = generalized_advantage_estimate(
            rewards,
            values_matrix,
            next_values,
            terminated,
            self.gamma,
            self.gae_lambda,
            valid,
        )
        adv_flat = torch.stack(
            [advantages[episode, step] for episode, step in step_indices.tolist()]
        )
        ret_flat = torch.stack(
            [returns[episode, step] for episode, step in step_indices.tolist()]
        )
        indices = torch.arange(len(step_indices), device=self.device)
        filtered_step_indices = step_indices
        filtered_old_logps = old_logps
        filtered_old_values = old_values
        mask = valid.float()
        
        # Global advantage normalization
        if getattr(self.args, 'std_advantage', False):
            adv_flat = (
                adv_flat - adv_flat.mean()
            ) / adv_flat.std(unbiased=False).clamp_min(1e-8)
        
        # PPO training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_explained_var = 0.0
        total_grad_norm = 0.0
        update_count = 0
        
        # Global advantage normalization (done once per update)
        adv_mean = adv_flat.mean().item()
        adv_std = adv_flat.std(unbiased=False).item()
        ret_mean = ret_flat.mean().item()
        ret_std = ret_flat.std(unbiased=False).item()
        
        for epoch in range(self.ppo_epochs):
            # Create minibatches using filtered indices
            num_steps = len(indices)
            if num_steps == 0:
                print("WARNING: No valid steps for PPO training")
                break
            
            # Ensure minibatch size doesn't exceed available steps
            effective_minibatch_size = min(self.minibatch_size, num_steps)
            batch_indices = torch.randperm(num_steps, device=self.device)
            
            for start_idx in range(0, num_steps, effective_minibatch_size):
                end_idx = min(start_idx + effective_minibatch_size, num_steps)
                batch_indices_subset = batch_indices[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Recompute policy for this minibatch
                batch_current_logps = []
                batch_values = []
                batch_entropys = []
                batch_old_logps = []
                batch_old_values = []
                batch_advantages = []
                batch_returns = []
                
                for idx in batch_indices_subset:
                    # Map back to original indices
                    original_idx = indices[idx].item()
                    epi_idx, step_idx = filtered_step_indices[idx].tolist()
                    model_inputs = rollout_buffer['model_inputs'][original_idx]
                    action_idx = rollout_buffer['action_indices'][original_idx]
                    valid_mask = rollout_buffer['valid_masks'][original_idx].to(
                        self.device
                    )
                    
                    # Use stored per-example tensors and local indptr directly
                    vals_attr_single, vals_seed_single, vals_node_single, local_indptr, candidates = model_inputs
                    
                    # Create indptr tensor for this single example
                    indptr_single = torch.from_numpy(local_indptr).unsqueeze(0).to(self.device)  # [1, 3]
                    
                    model_output = self.model(
                        vals_attr_single,
                        vals_seed_single,
                        vals_node_single,
                        indptr_single
                    )
                    
                    # Unpack model output
                    current_logits, current_value = model_output
                    
                    # Remove batch dimension
                    current_logits = current_logits[0]
                    current_value = current_value[0]
                    
                    # Check bounds and skip if out of bounds
                    if action_idx >= current_logits.size(0):
                        print(f"WARNING: action_idx {action_idx} >= logits size {current_logits.size(0)} - skipping")
                        continue
                    
                    distribution = masked_categorical(current_logits, valid_mask)
                    action_tensor = torch.as_tensor(
                        action_idx, device=self.device
                    )
                    current_logp = distribution.log_prob(action_tensor)
                    entropy = distribution.entropy()
                    
                    # Get old values from filtered data
                    old_logp = filtered_old_logps[idx]
                    old_value = filtered_old_values[idx]
                    
                    # Get pre-computed advantage and return for this step
                    advantage = adv_flat[idx]
                    return_val = ret_flat[idx]
                    
                    batch_current_logps.append(current_logp)
                    batch_values.append(current_value)
                    batch_entropys.append(entropy)
                    batch_old_logps.append(old_logp)
                    batch_old_values.append(old_value)
                    batch_advantages.append(advantage)
                    batch_returns.append(return_val)
                
                # Batch shape validation
                if len(batch_current_logps) == 0:
                    print("WARNING: All samples in minibatch were skipped")
                    continue
                
                # Stack all batch data
                current_logps = torch.stack(batch_current_logps)
                current_values = torch.stack(batch_values)
                current_entropys = torch.stack(batch_entropys)
                old_logps_batch = torch.stack(batch_old_logps)
                old_values_batch = torch.stack(batch_old_values)
                advantages = torch.stack(batch_advantages)
                returns_batch = torch.stack(batch_returns)
                
                # Compute ratios for policy loss
                ratios = torch.exp(current_logps - old_logps_batch)
                
                # PPO surrogate loss with separate policy clipping
                eps_policy = self.clip_range_policy
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_policy, 1 + eps_policy) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute approximate KL divergence
                approx_kl = (old_logps_batch - current_logps).mean().item()
                
                # Compute clip fraction
                clip_frac = ((ratios > 1.0 + eps_policy) | (ratios < 1.0 - eps_policy)).float().mean().item()
                
                # Value loss without clipping (Issue 1 fix: removing redundant clipping)
                # Keep clipping only for policy; let critic adapt freely
                value_loss = self.value_loss_scale * torch.nn.functional.mse_loss(current_values, returns_batch.detach())
                
                # Entropy bonus
                entropy_bonus = current_entropys.mean()
                
                # Total loss with scheduled entropy coefficient
                loss = policy_loss + self.value_loss_coef * value_loss - entropy_coef * entropy_bonus
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                update_count += 1
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_bonus.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_grad_norm += grad_norm
                
                # Explained variance calculation
                y_true = returns_batch.detach().cpu().numpy()
                y_pred = current_values.detach().cpu().numpy()
                var_y = np.var(y_true)
                ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0
                total_explained_var += ev
        
        # Create comprehensive training statistics
        if update_count > 0:
            avg_stats = TrainingStats(
                policy_loss=total_policy_loss / update_count,
                value_loss=total_value_loss / update_count,
                entropy=total_entropy_loss / update_count,
                approx_kl=total_approx_kl / update_count,
                clip_frac=total_clip_frac / update_count,
                adv_mean=adv_mean,
                adv_std=adv_std,
                ret_mean=ret_mean,
                ret_std=ret_std,
                explained_variance=total_explained_var / update_count,
                grad_norm=total_grad_norm / update_count,
                update_count=update_count
            )
            
            # Pretty logging of training statistics
            print(f"[PPO Stats] Policy={avg_stats.policy_loss:.4f}, "
                  f"Value={avg_stats.value_loss:.4f}, "
                  f"Entropy={avg_stats.entropy:.4f}, "
                  f"KL={avg_stats.approx_kl:.4f}, "
                  f"ClipFrac={avg_stats.clip_frac:.3f}, "
                  f"EV={avg_stats.explained_variance:.4f}, "
                  f"GradNorm={avg_stats.grad_norm:.4f}, "
                  f"Updates={avg_stats.update_count}, "
                  f"EpsPolicy={self.clip_range_policy:.2f}, "
                  f"EpsValue={self.clip_range_value:.2f}", flush=True)
            
            # Additional delta reward diagnostics
            if getattr(self.args, 'ppo_debug', False):
                mean_delta_reward = rewards[mask.bool()].mean().item()
                std_delta_reward = rewards[mask.bool()].std(
                    unbiased=False
                ).item()
                avg_subgraph_size = np.mean([len([x for x in traj if x != 'EOS']) for traj in selected_nodes])
                stop_rate = self.get_early_stop_rate(selected_nodes)
                
                print(f"[PPO Debug] MeanDeltaReward={mean_delta_reward:.4f}, "
                      f"StdDeltaReward={std_delta_reward:.4f}, "
                      f"AvgSubgraphSize={avg_subgraph_size:.1f}, "
                      f"StopRate={stop_rate:.3f}", flush=True)
            
            # Store last KL for monitoring
            self.last_approx_kl = avg_stats.approx_kl
            self.last_clip_frac = avg_stats.clip_frac
            self.last_training_stats = dict(avg_stats.__dict__)
            self.last_training_stats.update(reward_term_metrics)

        if update_count == 0:
            print("WARNING: No updates performed - all minibatches were empty")
            mean_reward = (
                float(rewards[mask.bool()].mean().item())
                if bool(mask.bool().any()) else 0.0
            )
            self.last_training_stats = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "update_count": 0,
                **reward_term_metrics,
            }
            return (
                selected_nodes,
                mean_reward,
                0.0,
                lengths.float().mean().item()
            )
        
        avg_policy_loss = total_policy_loss / update_count
        
        return (
            selected_nodes,
            rewards[mask.bool()].mean().item(),
            avg_policy_loss,
            lengths.float().mean().item()
        )

    def train_from_rewards(self, seeds: List[int], fn, policy="PPO", ppo_epochs=None, minibatch_size=None,
                          lam_size=0.0, lam_radius=0.0, lam_sim=0.0, entropy_coef=1e-3):
        if policy != "PPO":
            raise ValueError(
                "Sileo v1 supports PPO only; paper-style REINFORCE is not implemented"
            )
        return self.update_ppo(
            seeds,
            fn,
            lam_size=lam_size,
            lam_radius=lam_radius,
            lam_sim=lam_sim,
            entropy_coef=entropy_coef,
        )


    def train_from_sets(self, episodes: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size, self.debug_mode)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self._convolve_episode_graphs(x_seeds, env.graphs)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        max_steps = max_size * 2  # Safety limit to prevent infinite loops
        step_count = 0
        while not env.done and step_count < max_steps:
            k += 1
            step_count += 1
            z_nodes += self._convolve_episode_graphs(delta_x_nodes, env.graphs)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(
                valid_index,
                env.trajectories,
                z_nodes,
                z_seeds,
                debug=True,
                episode_graphs=env.graphs,
            )
            batch_logits, values = self.model(*model_inputs)
            actions = []
            logps_sel = []
            vals_sel = []
            proc_idx = []
            new_nodes = []
            
            for j, (logits, candidates, i) in enumerate(zip(batch_logits, batch_candidates, valid_index)):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    action = len(candidates)
                    actions.append(action)
                    logps_sel.append(logits[action])
                    vals_sel.append(values[j])
                    proc_idx.append(i)
                    new_nodes.append('EOS')
                else:
                    sub_idx = [idx for idx, v in enumerate(candidates) if v in valid_candidates]
                    action = sub_idx[logits[sub_idx].argmax().item()]
                    actions.append(action)
                    logps_sel.append(logits[action])
                    vals_sel.append(values[j])  # Take the matching value
                    proc_idx.append(i)
                    # Map action to node or EOS right here so order stays aligned
                    if action < len(candidates):
                        new_nodes.append(candidates[action])
                    else:
                        new_nodes.append('EOS')
            
            if len(actions) == 0:
                # No valid actions, skip this step
                continue
                
            delta_x_nodes = env.step(new_nodes, proc_idx)
            for i, v1, v2 in zip(proc_idx, logps_sel, vals_sel):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
        logps = self._stack_episode_sequences(episode_logps)
        values = self._stack_episode_sequences(episode_values)
        lengths = torch.LongTensor([len(x) for x in env.trajectories]).to(self.device)
        if logps.numel() == 0:
            return 0.0
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum().clamp_min(1)
        policy_loss = -(1 * logps * mask).sum() / n
        # loss = policy_loss + .5 * td_loss
        # loss.backward()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def train_from_lists(self, episodes: List[List[int]], max_size: Optional[int] = None):
        episodes = [(x + ['EOS']) if x[-1] != 'EOS' else x for x in episodes]
        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size, self.debug_mode)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self._convolve_episode_graphs(x_seeds, env.graphs)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        max_steps = max_size * 2  # Safety limit to prevent infinite loops
        step_count = 0
        while not env.done and step_count < max_steps:
            k += 1
            step_count += 1
            z_nodes += self._convolve_episode_graphs(delta_x_nodes, env.graphs)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(
                valid_index,
                env.trajectories,
                z_nodes,
                z_seeds,
                debug=True,
                episode_graphs=env.graphs,
            )
            batch_logits, values = self.model(*model_inputs)
            actions = []
            logps_sel = []
            vals_sel = []
            proc_idx = []
            new_nodes = []
            
            for j, (logits, candidates, i) in enumerate(zip(batch_logits, batch_candidates, valid_index)):
                if k > len(episodes[i]) - 1:
                    v = 'EOS'
                else:
                    v = episodes[i][k]
                try:
                    action = len(candidates) if v == 'EOS' else candidates.index(v)
                    actions.append(action)
                    logps_sel.append(logits[action])
                    vals_sel.append(values[j])
                    proc_idx.append(i)
                    if action < len(candidates):
                        new_nodes.append(candidates[action])
                    else:
                        new_nodes.append('EOS')
                except ValueError:
                    continue
            
            if len(actions) == 0:
                # No valid actions, check if all episodes are done
                all_done = all(k > len(episodes[i]) - 1 for i in valid_index)
                if all_done:
                    break
                continue
                
            delta_x_nodes = env.step(new_nodes, proc_idx)
            for i, v1, v2 in zip(proc_idx, logps_sel, vals_sel):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
        logps = self._stack_episode_sequences(episode_logps)
        values = self._stack_episode_sequences(episode_values)
        lengths = torch.LongTensor([len(x) for x in episodes]).to(self.device)
        if logps.numel() == 0:
            return 0.0
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum().clamp_min(1)
        policy_loss = -(1. * logps * mask).sum() / n
        # loss = policy_loss + .5 * td_loss
        # loss.backward()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def _sample_rollouts_loop(self, prefix, max_size=None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, prefix, max_size, self.debug_mode)
        rollouts = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.n_rollouts):
                trajectories, *_ = self._sample_trajectories(env)
                rollouts.append(trajectories)
        return list(zip(*rollouts))

    def _sample_rollouts_batch(self, prefix, max_size=None):
        max_size = self.max_size if max_size is None else max_size
        bs = len(prefix)
        env = ExpansionEnv(self.graph, prefix * self.n_rollouts, max_size, self.debug_mode)
        self.model.eval()
        with torch.no_grad():
            trajectories, *_ = self._sample_trajectories(env)
        rollouts = []
        for i in range(self.n_rollouts):
            rollouts.append(trajectories[i*bs:(i+1)*bs])
        return list(zip(*rollouts))

    def _sample_trajectories(self, env: ExpansionEnv):
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self._convolve_episode_graphs(x_seeds, env.graphs)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        episode_entropys = [[] for _ in range(bs)]
        
        # Buffer for PPO: store all inputs needed to recompute policy
        rollout_buffer = {
            'model_inputs': [],  # List of (vals_attr, vals_seed, vals_node, indptr, batch_candidates)
            'action_indices': [],  # List of chosen action indices
            'episode_indices': [],  # List of episode indices for each step
            'step_indices': [],  # List of step indices within each episode
            'old_logps': [],  # List of old log probabilities (detached)
            'old_values': [],  # List of old values (detached)
            'old_entropys': [],  # List of old entropies (detached)
            'valid_masks': [],
            'transitions': [],
        }
        if env.done:
            empty = torch.empty((bs, 0), device=self.device)
            return env.trajectories, empty, empty, empty, rollout_buffer
        
        step_count = 0
        max_steps = env.max_size * 2  # Safety limit to prevent infinite loops
        while not env.done and step_count < max_steps:
            z_nodes += self._convolve_episode_graphs(delta_x_nodes, env.graphs)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(
                valid_index,
                env.trajectories,
                z_nodes,
                z_seeds,
                debug=True,
                episode_graphs=env.graphs,
            )
            batch_logits, values = self.model(*model_inputs)
            
            # ========== CAUSAL MECHANISM INTEGRATION ==========
            causal_bias = None
            # PERFORMANCE FIX: Disable causal action bias - too computationally expensive
            # (Causal rewards are still computed in _compute_delta_rewards_real_gnn)
            use_causal_action_bias = False  # Set to True to enable (WARNING: 100x slower!)
            if use_causal_action_bias and self.use_causal and self.causal_explainer is not None and self.gnn_model is not None:
                # Get node features as tensor
                # Ensure node features are on correct device
                if torch.is_tensor(self.nodefeats):
                    node_feats_tensor = self.nodefeats.to(self.device)
                else:
                    node_feats_tensor = torch.from_numpy(self.nodefeats).float().to(self.device)

                # Add device assertions for safety (after node_feats_tensor creation)
                assert node_feats_tensor.device == self.device, \
                    f"Node features on wrong device: {node_feats_tensor.device} vs {self.device}"
                assert self.full_edge_index.device == self.device, \
                    f"Edge index on wrong device: {self.full_edge_index.device} vs {self.device}"
                
                # Compute causal importance for current state
                with torch.no_grad():
                    causal_bias = self._compute_causal_bias(
                        valid_index=valid_index,
                        trajectories=env.trajectories,
                        batch_candidates=batch_candidates,
                        gnn_model=self.gnn_model,
                        node_features=node_feats_tensor,
                        full_edge_index=self.full_edge_index,
                        target_classes=self.target_classes,
                        seed_nodes=self.seed_nodes,
                        ce_mode=True,
                        gamma=self.causal_gamma
                    )
            
            # Track current trajectory lengths for minimum steps guard
            active_lengths = [len(env.trajectories[i]) for i in valid_index]
            
            # Action sampling with optional causal bias and minimum steps guard
            # Use min_steps for minimum trajectory length before allowing STOP
            actions, logps, entropys, valid_masks = self._sample_actions_masked(
                batch_logits,
                causal_bias,
                min_steps_before_stop=self.min_steps_before_stop,
                trajectory_lengths=active_lengths,
            )
            
            # DEBUG: Track action selection and boundary violations
            if self.debug_mode:
                print(f"[DEBUG] Actions: {actions}")
                print(f"[DEBUG] Batch candidates: {batch_candidates}")
            
            # Handle EOS head model - action < len(candidates) selects candidate, action == len(candidates) is EOS
            new_nodes = []
            for action, candidates in zip(actions, batch_candidates):
                if action < len(candidates):
                    # Select the actual candidate
                    selected_node = candidates[action]
                    new_nodes.append(selected_node)
                    if self.debug_mode:
                        print(f"[DEBUG] Selected node: {selected_node} from candidates: {candidates}")
                else:
                    # EOS action (action == len(candidates))
                    new_nodes.append('EOS')
                    if self.debug_mode:
                        print(f"[DEBUG] Selected EOS action")
            
            delta_x_nodes = env.step(new_nodes, valid_index)
            
            # Store all necessary data for PPO recomputation
            for j, i in enumerate(valid_index):
                # Extract per-example data from batched tensors
                global_indptr_j = model_inputs[3][j]  # (start, end, cut) in the *batched* tensors
                s, e, c = global_indptr_j
                
                # Extract per-example tensors with detach/clone for safety
                if model_inputs[0] is not None:
                    vals_attr_single = model_inputs[0][s:e].detach().clone()  # Per-example slice
                else:
                    vals_attr_single = None
                vals_seed_single = model_inputs[1][s:e].detach().clone()  # Per-example slice
                vals_node_single = model_inputs[2][s:e].detach().clone()  # Per-example slice
                
                # Create local indptr for this single example
                local_indptr = np.array([0, e - s, c - s], dtype=np.int64)
                
                # Store per-example tensors and local indptr
                rollout_buffer['model_inputs'].append((
                    vals_attr_single,
                    vals_seed_single,
                    vals_node_single,
                    local_indptr,
                    batch_candidates[j]  # candidates for this step (j is the index in valid_index)
                ))
                rollout_buffer['action_indices'].append(actions[j])
                rollout_buffer['episode_indices'].append(i)
                rollout_buffer['step_indices'].append(step_count)
                rollout_buffer['old_logps'].append(logps[j].detach())
                rollout_buffer['old_values'].append(values[j].detach())
                rollout_buffer['old_entropys'].append(entropys[j].detach())
                rollout_buffer['valid_masks'].append(valid_masks[j].detach().clone())
                rollout_buffer['transitions'].append(Transition(
                    observation=rollout_buffer['model_inputs'][-1],
                    candidates=tuple(batch_candidates[j]),
                    valid_mask=valid_masks[j].detach().clone(),
                    action=int(actions[j]),
                    old_log_prob=logps[j].detach(),
                    old_value=values[j].detach(),
                ))
            
            for i, v1, v2, v3 in zip(valid_index, logps, values, entropys):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
                episode_entropys[i].append(v3)
            
            step_count += 1

        if not env.done:
            for episode in env.valid_index:
                env.dones[episode] = True
                env.termination_reasons[episode] = Termination.TRUNCATED
        last_transition = {}
        for transition, episode in zip(
            rollout_buffer["transitions"], rollout_buffer["episode_indices"]
        ):
            last_transition[episode] = transition
        for episode, transition in last_transition.items():
            if transition.termination is None:
                transition.termination = env.termination_reasons[episode]
            
        # Stack and Padding
        logps = self._stack_episode_sequences(episode_logps)
        values = self._stack_episode_sequences(episode_values)
        entropys = self._stack_episode_sequences(episode_entropys)
        return env.trajectories, logps, values, entropys, rollout_buffer

    def _prepare_inputs(
        self,
        valid_index: List[int],
        trajectories: List[List[int]],
        z_nodes: sp.csc_matrix,
        z_seeds: sp.csc_matrix,
        debug=False,
        episode_graphs=None,
    ):
        vals_attr = [] if self.nodefeats is not None else None
        vals_seed = []
        vals_node = []
        indptr = []
        offset = 0
        batch_candidates = []
        for i in valid_index:
            episode_graph = (
                episode_graphs[i] if episode_graphs is not None else self.graph
            )
            boundary_nodes = episode_graph.outer_boundary(trajectories[i])
            # Make candidate ordering deterministic
            candidate_nodes = sorted(boundary_nodes)
            
            # DEBUG: Track boundary violations and candidate selection
            if debug and self.debug_mode:
                print(f"[DEBUG] Trajectory {i}: {trajectories[i]}")
                print(f"[DEBUG] Boundary nodes: {sorted(boundary_nodes)}")
                print(f"[DEBUG] Candidate nodes: {candidate_nodes}")
                print(f"[DEBUG] Boundary size: {len(boundary_nodes)}, Candidates: {len(candidate_nodes)}")
                if len(candidate_nodes) == 0:
                    print(f"[DEBUG] ⚠️  NO CANDIDATES AVAILABLE for trajectory {i}")
                print(f"[DEBUG] ---")
            
            involved_nodes = candidate_nodes + trajectories[i]  # 1-ego net
            batch_candidates.append(candidate_nodes)  # candidates
            if self.nodefeats is not None:
                vals_attr.append(self.nodefeats[involved_nodes])
            vals_seed.append(z_seeds.T[i, involved_nodes].todense())
            vals_node.append(z_nodes.T[i, involved_nodes].todense())
            indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))
            offset += len(involved_nodes)
        if self.nodefeats is not None:
            # vals_attr = torch.cat(vals_attr, 0)
            vals_attr = np.concatenate(vals_attr, 0)
            vals_attr = torch.from_numpy(vals_attr).float().to(self.device)
        vals_seed = np.array(np.concatenate(vals_seed, 1))[0]
        vals_node = np.array(np.concatenate(vals_node, 1))[0]
        vals_seed = torch.from_numpy(vals_seed).float().to(self.device)
        vals_node = torch.from_numpy(vals_node).float().to(self.device)
        indptr = np.array(indptr, dtype=np.int64)
        return vals_attr, vals_seed, vals_node, indptr, batch_candidates


    def init_causal_reward_calculator(self, gnn_model):
        """
        Initialize the causal reward calculator with the GNN model.
        This should be called after the GNN model is available.
        
        Args:
            gnn_model: The trained GNN model for causal analysis
        """
        if self.use_causal and self.causal_explainer is not None:
            from .causal_explainer import CausalRewardCalculator
            self.causal_reward_calculator = CausalRewardCalculator(
                gnn_model=gnn_model,
                causal_explainer=self.causal_explainer,
                causal_weight=self.causal_weight,
                predictor=self.predictor,
            )
            print(f"[CAUSAL] Initialized causal reward calculator with weight {self.causal_weight}")
        else:
            print("[CAUSAL] Causal mechanism not enabled")

    def set_causal_data(self, gnn_model, full_edge_index, target_classes, seed_nodes):
        """
        Set the data needed for causal bias computation.
        
        Args:
            gnn_model: Frozen target GNN model
            full_edge_index: Full graph edge index [2, num_edges]
            target_classes: Target class for each seed node
            seed_nodes: Seed nodes for evaluation
        """
        self.gnn_model = gnn_model
        self.full_edge_index = full_edge_index
        self.target_classes = target_classes
        self.seed_nodes = seed_nodes
        
        # Freeze the GNN model
        if self.gnn_model is not None:
            self.gnn_model.eval()
            for param in self.gnn_model.parameters():
                param.requires_grad = False
            print(f"[CAUSAL] GNN model frozen for causal interventions")
    
    def _check_subgraph_connected(self, nodes: List[int], seed_node: int) -> bool:
        """
        Check if a subgraph is connected to the seed node.
        
        Args:
            nodes: List of nodes in the subgraph
            seed_node: The seed/target node
            
        Returns:
            True if all nodes are reachable from seed_node, False otherwise
        """
        if seed_node not in nodes:
            return False
        
        # Use BFS to check connectivity from seed_node
        visited = {seed_node}
        queue = [seed_node]
        node_set = set(nodes)
        
        while queue:
            current = queue.pop(0)
            for neighbor in self.graph.neighbors.get(current, set()):
                if neighbor in node_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all nodes are visited
        return len(visited) == len(nodes)

    def _make_subgraph_mask(self, nodes: List[int], full_edge_index: torch.Tensor):
        """
        Create subgraph edge mask for nodes (treatment/control intervention).
        """
        node_set = set(nodes)
        
        # Validate nodes are within bounds
        max_node_id = full_edge_index.max().item() if full_edge_index.size(1) > 0 else -1
        invalid_nodes = [n for n in nodes if n > max_node_id or n < 0]
        if invalid_nodes:
            print(f"[CAUSAL] Warning: Invalid node IDs {invalid_nodes}, max allowed: {max_node_id}", flush=True)
            # Filter out invalid nodes
            node_set = set(n for n in nodes if 0 <= n <= max_node_id)
        
        # Find edges where both endpoints are in the node set
        src, dst = full_edge_index[0], full_edge_index[1]
        edge_mask = torch.tensor([
            (src[i].item() in node_set and dst[i].item() in node_set)
            for i in range(full_edge_index.size(1))
        ], dtype=torch.bool, device=full_edge_index.device)
        
        subgraph_edge_index = full_edge_index[:, edge_mask]
        
        return edge_mask, subgraph_edge_index
    
    def _compute_causal_bias(self, valid_index: List[int], trajectories: List[List[int]], 
                            batch_candidates: List[List[int]], gnn_model, 
                            node_features: torch.Tensor, full_edge_index: torch.Tensor,
                            target_classes: torch.Tensor, seed_nodes: List[int],
                            ce_mode: bool = True, gamma: float = 0.5):
        """
        Compute causal bias for action selection using CE-difference method.
        
        This implements the full causal mechanism as described:
        1. Control: G_{k-1} (current trajectory)
        2. Treatment: G_{k-1} ∪ {a_k} (current + candidate)
        3. ICE = CE(control) - CE(treatment) [positive is good]
        4. Add correctness bonus/penalty
        
        Args:
            valid_index: List of valid trajectory indices
            trajectories: Current trajectories (nodes selected so far)
            batch_candidates: Available candidates for each trajectory
            gnn_model: Frozen target GNN model
            node_features: Full graph node features [num_nodes, feat_dim]
            full_edge_index: Full graph edge index [2, num_edges]
            target_classes: Target class for each seed node
            seed_nodes: Original seed nodes for each trajectory
            ce_mode: If True, use CE-difference; else use log-prob delta
            gamma: Correctness bonus/penalty weight
            
        Returns:
            causal_bias: List of causal importance scores for each valid trajectory [N_candidates+1]
        """
        if not self.use_causal or self.causal_explainer is None or gnn_model is None:
            return None
        
        # Ensure GNN is frozen and in eval mode
        gnn_model.eval()
        
        causal_bias = []

        # WRAP EVERYTHING in no_grad to prevent memory issues
        with torch.no_grad():
            for idx, (i, candidates) in enumerate(zip(valid_index, batch_candidates)):
                try:
                    current_nodes = trajectories[i]
                    # Replace the seed node lookup with:
                    seed_node = trajectories[i][0]  # First node in trajectory is always the seed

                    # FIX: Map seed_node ID to its index in seed_nodes list
                    try:
                        seed_idx = self.seed_nodes.index(seed_node)
                    except (ValueError, AttributeError):
                        print(f"[CAUSAL] Warning: seed_node {seed_node} not found in seed_nodes list", flush=True)
                        causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
                        continue
                    
                    # Then validate the index is in the expected range
                    if seed_idx >= len(target_classes):
                        print(f"[CAUSAL] Warning: seed_idx {seed_idx} out of bounds (target_classes len={len(target_classes)})", flush=True)
                        causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
                        continue

                    target_class = target_classes[seed_idx].item() if torch.is_tensor(target_classes[seed_idx]) else target_classes[seed_idx]
                    
                    if len(candidates) == 0:
                        # No candidates - only STOP action available
                        causal_bias.append(torch.ones(1, dtype=torch.float32, device=self.device))
                        continue
                    
                    # Validate current_nodes is not empty
                    if len(current_nodes) == 0:
                        # Empty trajectory - return uniform bias
                        causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
                        continue
                    
                    # ========== CONTROL: G_{k-1} ==========
                    _, ctrl_edge_index = self._make_subgraph_mask(current_nodes, full_edge_index)
                    
                    # Skip if seed node is not in current nodes (shouldn't happen but be safe)
                    if seed_node not in current_nodes:
                        causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
                        continue
                    
                    # After creating ctrl_edge_index, add this check:
                    if ctrl_edge_index.size(1) == 0:
                        print(f"[CAUSAL] Warning: Empty control subgraph for trajectory {i}, seed {seed_node}", flush=True)
                        print(f"[CAUSAL]   Current nodes: {current_nodes}", flush=True)
                        print(f"[CAUSAL]   Falling back to uniform bias", flush=True)
                        causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
                        continue
                    
                    # Forward pass for control - use FULL node features, restricted edges
                    edge_weights = torch.ones(ctrl_edge_index.size(1), device=ctrl_edge_index.device)
                    ctrl_logits = gnn_model(node_features, ctrl_edge_index, edge_weights)
                    
                    # For node classification, get prediction for seed node only
                    if ctrl_logits.dim() > 1 and ctrl_logits.size(0) > seed_node:
                        ctrl_logits = ctrl_logits[seed_node].unsqueeze(0)
                    elif ctrl_logits.dim() == 1:
                        ctrl_logits = ctrl_logits.unsqueeze(0)
                    
                    ctrl_logp = torch.log_softmax(ctrl_logits, dim=-1).squeeze()
                    ctrl_ce = -ctrl_logp[target_class]
                    
                    # ========== TREATMENT: G_{k-1} ∪ {a_k} for each candidate ==========
                    ice_scores = []
                    
                    for cand in candidates:
                        treatment_nodes = current_nodes + [cand]
                        _, treat_edge_index = self._make_subgraph_mask(treatment_nodes, full_edge_index)
                        
                        # Skip treatment if it has no edges (same issue as control)
                        if treat_edge_index.size(1) == 0:
                            # No edges means adding this candidate doesn't create connectivity
                            # Penalize this action (isolated node)
                            ice_scores.append(self.ice_score_default)
                            continue
                        
                        # Forward pass for treatment - use FULL node features, restricted edges
                        edge_weights = torch.ones(treat_edge_index.size(1), device=treat_edge_index.device)
                        treat_logits = gnn_model(node_features, treat_edge_index, edge_weights)
                        
                        # For node classification, get prediction for seed node only
                        if treat_logits.dim() > 1 and treat_logits.size(0) > seed_node:
                            treat_logits = treat_logits[seed_node].unsqueeze(0)
                        elif treat_logits.dim() == 1:
                            treat_logits = treat_logits.unsqueeze(0)
                        
                        treat_logp = torch.log_softmax(treat_logits, dim=-1).squeeze()
                        treat_ce = -treat_logp[target_class]
                        
                        # Compute ICE score
                        if ce_mode:
                            ice = ctrl_ce - treat_ce  # Positive means treatment improves prediction
                        else:
                            ice = treat_logp[target_class] - ctrl_logp[target_class]
                        
                        # Add correctness bonus/penalty
                        pred_class = treat_logits.argmax(dim=-1).item() if treat_logits.dim() > 1 else treat_logits.argmax().item()
                        bonus = gamma if pred_class == target_class else -gamma
                        
                        total_score = ice + bonus
                        ice_scores.append(total_score.item())
                
                    # Causal Confidence Regularizer: penalize high variance in ICE scores
                    # This encourages consistent causal effects across actions
                    ice_conf_penalty = 0.0
                    if len(ice_scores) > 1:
                        ice_var = torch.var(torch.tensor(ice_scores, dtype=torch.float32))
                        lambda_conf = getattr(self, 'lambda_conf', 0.1)  # Default confidence penalty weight
                        ice_conf_penalty = lambda_conf * ice_var.item()
                    
                    # Convert to tensor and add neutral score for STOP action
                    ice_tensor = torch.tensor(ice_scores + [0.0], dtype=torch.float32, device=self.device)
                    
                    # Apply confidence penalty by subtracting from all scores uniformly
                    # This encourages consistent causal effects
                    ice_tensor = ice_tensor - ice_conf_penalty
                    
                    # Normalize to avoid extreme values
                    ice_tensor = torch.tanh(ice_tensor)  # Squash to [-1, 1]
                    
                    causal_bias.append(ice_tensor)
                    
                except Exception as e:
                    # Enhanced error handling
                    if self.debug_mode:
                        import traceback
                        print(f"[CAUSAL] Full error traceback:", flush=True)
                        traceback.print_exc()
                    # Fallback to uniform bias
                    causal_bias.append(torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device))
        
        return causal_bias

    def _sample_actions_masked(
        self,
        batch_logits: List,
        causal_bias: Optional[List] = None,
        min_steps_before_stop: int = 8,
        trajectory_lengths: Optional[List[int]] = None,
    ):
        """
        Sample actions with proper masking and old log-prob storage.
        
        Args:
            batch_logits: List of logits for each batch element
            causal_bias: Optional list of causal importance scores to bias action selection
            min_steps_before_stop: Minimum number of steps before STOP action is allowed
            
        Returns:
            actions: Selected action indices
            logps: Log probabilities of selected actions (detached for PPO)
            entropys: Entropy values (detached for PPO)
        """
        batch = []
        masks = []
        for i, logits in enumerate(batch_logits):
            # Apply causal bias if available
            if causal_bias is not None and i < len(causal_bias):
                logits = logits + self.causal_action_weight * causal_bias[i].detach()

            valid_mask = torch.ones_like(logits, dtype=torch.bool)
            current_length = (
                trajectory_lengths[i] if trajectory_lengths is not None else 1
            )
            if current_length < min_steps_before_stop:
                valid_mask[-1] = False

            distribution = masked_categorical(logits, valid_mask)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            logp = distribution.log_prob(action_tensor)
            entropy = distribution.entropy()
            batch.append([action, logp.detach(), entropy.detach()])
            masks.append(valid_mask)
        actions, logps, entropys = zip(*batch)
        actions = np.array(actions)
        return actions, logps, entropys, masks

    def _sample_actions(self, batch_logits: List, causal_bias: Optional[List] = None, min_steps_before_stop: int = 8) -> (List, List, List):
        """
        Sample actions with optional causal bias and minimum steps before STOP.
        
        Args:
            batch_logits: List of logits for each batch element
            causal_bias: Optional list of causal importance scores to bias action selection
            min_steps_before_stop: Minimum number of steps before STOP action is allowed (8-10 for syn3)
            
        Returns:
            actions: Selected action indices
            logps: Log probabilities of selected actions
            entropys: Entropy values
        """
        batch = []
        for i, logits in enumerate(batch_logits):
            # logits are expected to be log-probabilities (log-softmax output)
            
            # MINIMUM STEPS GUARD: Mask STOP action for first few steps
            # Get current trajectory length
            current_length = getattr(self, '_current_trajectory_lengths', [1] * len(batch_logits))[i]
            if current_length < min_steps_before_stop:
                # Set STOP action logit to very negative value (effectively mask it)
                logits = logits.clone()  # Don't modify original
                logits[-1] = self.stop_mask_value
            
            # Apply causal bias if available
            if causal_bias is not None and i < len(causal_bias):
                # Causal bias is already in [-1, 1] range (tanh applied)
                # Scale it and add to logits (weight controlled by causal_weight during reward computation)
                # Here we use a moderate weight for exploration
                # Issue 2 fix: Explicitly detach causal bias to prevent gradient leakage
                logits = logits + self.causal_action_weight * causal_bias[i].detach()
            
            # STABILITY FIX 1: Safe softmax with logit stabilization
            # Subtract max for numerical stability
            logits = logits - logits.max()
            ps = torch.softmax(logits, dim=0)
            ps = ps + 1e-8
            ps = ps / ps.sum()
            
            # Check for invalid probabilities
            if not torch.isfinite(ps).all() or (ps < 0).any():
                epoch_num = getattr(self, 'current_epoch', -1)
                # print(f"[WARN] Invalid probs at epoch {epoch_num} step {i}: logits range [{logits.min():.2f}, {logits.max():.2f}]", flush=True)
                ps = torch.nan_to_num(ps, nan=0.0, posinf=0.0, neginf=0.0)
                ps = torch.clamp(ps, min=0.0)
                
                ps_sum = ps.sum()
                if ps_sum.item() <= 0:
                    ps = torch.ones_like(ps) / ps.size(0)
                else:
                    ps = ps / ps_sum
            
            entropy = -(ps * torch.log(ps + 1e-8)).sum()
            action = torch.multinomial(ps, 1).item()
            logp = logits[action]  # Use biased logits for policy gradient
            batch.append([action, logp, entropy])
        actions, logps, entropys = zip(*batch)
        actions = np.array(actions)
        return actions, logps, entropys
