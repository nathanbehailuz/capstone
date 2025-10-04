from typing import Union, Optional, List, Set, Dict
import numpy as np
from scipy import sparse as sp

import torch
from torch import nn

from .graph import Graph
from .generator_core import GraphConv, Agent
from .causal_explainer import NodeLevelCausalExplainer, GraphLevelCausalExplainer, CausalRewardCalculator


class ExpansionEnv:

    def __init__(self, graph: Graph, selected_nodes: List[List[int]], max_size: int):
        self.max_size = max_size
        self.graph = graph
        self.n_nodes = self.graph.n_nodes
        self.data = selected_nodes
        self.bs = len(self.data)
        self.trajectories = None
        self.dones = None

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
        self.dones = [x[-1] == 'EOS' or len(x) >= self.max_size or len(self.graph.outer_boundary(x)) == 0
                      for x in self.trajectories]
        assert not any(self.dones)
        seeds = [self.data[i][0] for i in range(self.bs)]
        # Construct initial neighborhood around seeds
        data = np.hstack([self.graph.seeds_attr[seeds], self.graph.nodes_attr])
        rows = np.hstack([np.arange(self.bs), np.arange(self.n_nodes)])
        cols = np.hstack([np.arange(self.bs), np.zeros(self.n_nodes, dtype=np.int32)])
        x_seeds = sp.csc_matrix((data, (rows, cols)), shape=(self.bs + self.n_nodes, self.bs))
        # Construct initial neighbors mask
        rows = np.hstack([np.arange(self.n_nodes), self.graph.seeds_nbrs[seeds]])
        cols = np.hstack([np.zeros(self.n_nodes, dtype=np.int32), np.repeat(np.arange(self.bs), np.diff(self.graph.seeds_indptr)[seeds])])
        delta_x_nodes = sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_nodes + self.bs, self.bs))
        return x_seeds, delta_x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int]):
        assert len(new_nodes) == len(index)
        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            self.trajectories[i].append(v)
            if v == 'EOS':
                self.dones[i] = True
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif self.graph.outer_boundary(self.trajectories[i]) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v
        if self.done:
            return None
        
        # Update neighborhood after adding nodes
        data = []
        rows = []
        cols = []
        for i in index:
            v = full_new_nodes[i]
            if v is not None:
                nbrs: Set[int] = self.graph.outer_boundary(self.trajectories[i])
                data.append(np.ones(len(nbrs)))
                rows.append(np.array(list(nbrs)))
                cols.append(np.full(len(nbrs), i))
        if len(data) == 0:
            return sp.csc_matrix((self.n_nodes + self.bs, self.bs), dtype=np.float32)
        data = np.hstack(data)
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        return sp.csc_matrix((data, (rows, cols)), shape=(self.n_nodes + self.bs, self.bs))


class Generator:

    def __init__(self, graph: Graph, model: Agent, optimizer,
                 device: Optional[torch.device] = None,
                 entropy_coef: float = 1e-6,
                 n_rollouts: int = 10,
                 max_size: int = 25,
                 k: int = 3,
                 alpha: float = 0.85,
                 max_reward: float = 1.,
                 # Causal mechanism parameters
                 use_causal: bool = False,
                 causal_explainer: Optional[Union[NodeLevelCausalExplainer, GraphLevelCausalExplainer]] = None,
                 causal_weight: float = 0.5,
                 # New parameters for STOP action and penalties
                 target_size_k: int = 8,
                 lambda_size: float = 0.5,
                 lambda_radius: float = 1.0,
                 lambda_radius_min: float = 0.2,
                 step_cost: float = 0.001,
                 stop_bias: float = 0.5,
                 fidelity_thresh: float = 0.9,
                 # PPO stability parameters
                 ppo_clip: float = 0.1,
                 kl_target: float = 0.0075,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_grad_norm: float = 0.5,
                 use_value_clip: bool = True,
                 # Entropy scheduling
                 entropy_coeff_start: float = 1e-3,
                 entropy_coeff_end: float = 1e-4,
                 # Causal reward scheduling
                 causal_warmup_epochs: int = 5):
        self.graph = graph
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

        # Causal mechanism integration
        self.use_causal = use_causal
        self.causal_explainer = causal_explainer
        self.causal_weight = causal_weight
        self.causal_reward_calculator = None
        if self.use_causal and self.causal_explainer is not None:
            # This will be initialized when we have access to the GNN model
            pass

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        
        # New parameters for STOP action and penalties
        self.target_size_k = target_size_k
        self.lambda_size = lambda_size
        self.lambda_radius = lambda_radius
        self.lambda_radius_min = lambda_radius_min
        self.step_cost = step_cost
        self.stop_bias = stop_bias
        self.fidelity_thresh = fidelity_thresh
        
        # PPO stability parameters
        self.clip_range = ppo_clip
        self.vf_clip = ppo_clip if use_value_clip else None
        self.ppo_epochs = 4
        self.minibatch_size = 256
        self.gamma = gamma
        self.use_gae = True
        self.gae_lambda = gae_lambda
        self.reward_norm = False
        self.kl_target = kl_target
        self.max_grad_norm = max_grad_norm
        
        # Entropy scheduling
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end = entropy_coeff_end
        self.current_entropy_coef = entropy_coeff_start
        
        # Causal reward scheduling
        self.causal_warmup_epochs = causal_warmup_epochs
        self.current_epoch = 0
        
        # Monitoring
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0

    def load_nodefeats(self, x):
        #self.nodefeats = torch.from_numpy(x).float().to(self.device) 
        if isinstance(x, torch.Tensor):
            self.nodefeats = x.cpu().numpy()
        else:
            self.nodefeats = x
        #self.nodefeats = x.astype(np.float32)

    def init_causal_reward_calculator(self, gnn_model):
        """
        Initialize the causal reward calculator with the GNN model.
        This should be called after the GNN model is available.
        
        Args:
            gnn_model: The trained GNN model for causal analysis
        """
        if self.use_causal and self.causal_explainer is not None:
            self.causal_reward_calculator = CausalRewardCalculator(
                gnn_model=gnn_model,
                causal_explainer=self.causal_explainer,
                causal_weight=self.causal_weight
            )
            print(f"[CAUSAL] Initialized causal reward calculator with weight {self.causal_weight}")
        else:
            print("[CAUSAL] Causal mechanism not enabled")

    def update_entropy_coef(self, epoch: int, total_epochs: int):
        """
        Update entropy coefficient using linear decay schedule.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
        """
        if total_epochs > 1:
            progress = epoch / (total_epochs - 1)
            self.current_entropy_coef = self.entropy_coeff_start + progress * (self.entropy_coeff_end - self.entropy_coeff_start)
        else:
            self.current_entropy_coef = self.entropy_coeff_start

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

    def update_epoch(self, epoch: int):
        """
        Update current epoch for scheduling.
        
        Args:
            epoch: Current epoch (0-indexed)
        """
        self.current_epoch = epoch

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
        
        for trajectory in selected_nodes:
            # Check if trajectory ended with EOS before reaching max_size
            if trajectory[-1] == 'EOS' and len(trajectory) < self.max_size:
                early_stops += 1
        
        return early_stops / total_trajectories if total_trajectories > 0 else 0.0

    def generate(self, seeds: List[int], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)
        return self._sample_trajectories(env)

    def _sample_trajectories(self, env: ExpansionEnv):
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        episode_entropys = [[] for _ in range(bs)]
        while not env.done:
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            
            # ========== CAUSAL MECHANISM INTEGRATION ==========
            causal_bias = None
            if self.use_causal and self.causal_explainer is not None:
                # Compute causal importance for current state
                causal_bias = self._compute_causal_bias(valid_index, env.trajectories, batch_candidates)
            
            # Action masking: ensure only valid candidates are sampled
            # Now with optional causal bias
            actions, logps, entropys = self._sample_actions(batch_logits, causal_bias)
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1, v2, h in zip(valid_index, logps, values, entropys):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
                episode_entropys[i].append(h)
        # Stack and pad
        logps, values, entropys = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                                   for episode_xs in (episode_logps, episode_values, episode_entropys)]
        return env.trajectories, logps.to(self.device), values.to(self.device), entropys.to(self.device)

    def _compute_causal_bias(self, valid_index: List[int], trajectories: List[List[int]], batch_candidates: List[List[int]]):
        """
        Compute causal bias for action selection.
        
        Args:
            valid_index: List of valid trajectory indices
            trajectories: Current trajectories
            batch_candidates: Available candidates for each trajectory
            
        Returns:
            causal_bias: List of causal importance scores for each valid trajectory
        """
        if not self.use_causal or self.causal_explainer is None:
            return None
        
        causal_bias = []
        for i, candidates in zip(valid_index, batch_candidates):
            # For each candidate, we need to compute its causal importance
            # This is a simplified version - in practice, you'd need to:
            # 1. Create a temporary graph state with the candidate added
            # 2. Compute causal importance using the explainer
            # 3. Return importance scores for each candidate
            
            # For now, return uniform bias (no causal information)
            # TODO: Implement full causal bias computation
            # Note: Add +1 for EOS action (model has len(candidates)+1 logits)
            # Important: Create tensor on the correct device (GPU)
            uniform_bias = torch.ones(len(candidates) + 1, dtype=torch.float32, device=self.device)
            causal_bias.append(uniform_bias)
        
        return causal_bias

    def _sample_rollouts_loop(self, prefix: List[List[int]], max_size: int):
        max_size = max([len(x) for x in prefix]) if max_size is None else max_size
        episodes = []
        for x in prefix:
            env = ExpansionEnv(self.graph, [x], max_size)
            episode, *_ = self._sample_trajectories(env)
            episodes.append(episode[0])
        return episodes

    def _sample_rollouts_batch(self, prefix: List[List[int]], max_size: int):
        max_size = max([len(x) for x in prefix]) if max_size is None else max_size
        env = ExpansionEnv(self.graph, prefix, max_size)
        episodes, *_ = self._sample_trajectories(env)
        return episodes

    def sample_episodes(self, seeds: List[int], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)
        return self._sample_trajectories(env)

    def sample_rollouts(self, prefix: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        bs = len(prefix)
        if bs * self.n_rollouts > 10000:
            return self._sample_rollouts_loop(prefix, max_size)
        else:
            return self._sample_rollouts_batch(prefix, max_size)

    def train_from_rewards(self, seeds: List[int], fn, policy="REINFORCE"):
        print(f"[DEBUG] train_from_rewards called with policy: {policy}", flush=True)
        print(f"[DEBUG] seeds length: {len(seeds)}", flush=True)
        bs = len(seeds)
        self.model.train()
        # rollout trajectories
        selected_nodes, logps, values, entropys = self.sample_episodes(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # compute reward matrix [B, T]
        rewards = np.zeros(logps.shape, dtype=np.float32)
        final_scores = fn(selected_nodes)
        
        # Add step cost penalty for each step
        for i in range(logps.size(0)):
            for t in range(lengths[i].item() - 1):  # -1 to exclude EOS step
                rewards[i, t] -= self.step_cost
        
        # Add final rewards
        rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2] += final_scores
        
        # Add size penalty and radius penalty
        for i, trajectory in enumerate(selected_nodes):
            # Remove EOS if present
            clean_trajectory = [x for x in trajectory if x != 'EOS']
            size = len(clean_trajectory)
            
            # Size penalty: hinge loss on target size
            if size > self.target_size_k:
                size_penalty = self.lambda_size * (size - self.target_size_k)
                rewards[i, lengths[i].item() - 2] -= size_penalty
            
            # Radius penalty: graph radius of explanation
            if len(clean_trajectory) > 1:
                radius = self.graph.subgraph_depth(clean_trajectory)
                effective_lambda_radius = self.get_effective_lambda_radius(self.current_epoch, 100)  # Assume 100 total epochs
                radius_penalty = effective_lambda_radius * radius
                rewards[i, lengths[i].item() - 2] -= radius_penalty
            
            # STOP bonus for fidelity preservation
            if hasattr(self, 'fidelity_thresh') and final_scores[i] >= self.fidelity_thresh:
                stop_bonus = self.lambda_size * self.target_size_k
                rewards[i, lengths[i].item() - 2] += stop_bonus
        if self.n_rollouts:
            for k in range(2, logps.shape[1] + 1):
                valid_idx = [i for i, x in enumerate(selected_nodes) if len(x) >= k
                             and x[k-1] != 'EOS' and len(self.graph.outer_boundary(x[:k]))]
                if len(valid_idx):
                    rollouts = self.sample_rollouts([selected_nodes[i][:k] for i in valid_idx])
                    for i, samples in zip(valid_idx, rollouts):
                        rewards[i, k-2] = np.mean(fn(samples))
        else:
            rewards[...] = final_scores[:, None]

        rewards = torch.from_numpy(rewards).float().to(self.device)

        # mask valid timesteps (exclude last EOS/terminal slot)
        T = rewards.size(1)
        mask = torch.arange(T, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum().clamp_min(1.0)

        if policy == "PPO":
            print(f"[DEBUG] PPO training path taken", flush=True)
            # hyper-params (use new parameters)
            gamma = self.gamma
            use_gae = self.use_gae
            lam = self.gae_lambda
            clip_range = self.clip_range
            vf_clip = self.vf_clip
            ppo_epochs = self.ppo_epochs
            mb_size = self.minibatch_size

            # reward normalization (masked, per rollout batch)
            if getattr(self, "reward_norm", False):
                r_mean = (rewards * mask).sum() / n
                r_var = (((rewards - r_mean) * mask).pow(2)).sum() / n
                rewards = (rewards - r_mean) / (r_var.sqrt() + 1e-8)

            # compute returns or GAE advantages
            with torch.no_grad():
                if use_gae:
                    # values: [B, T], mask: [B, T], rewards: [B, T]
                    adv = torch.zeros_like(rewards)
                    last_adv = torch.zeros(bs, device=self.device)
                    for t in reversed(range(T - 1)):  # last slot is EOS; we used < lengths-1
                        # mask for step t and t+1
                        m_t = mask[:, t]
                        m_tp1 = mask[:, t+1]
                        # one-step TD error (δ)
                        v_t = values[:, t]
                        v_tp1 = values[:, t+1]
                        delta = (rewards[:, t] + gamma * v_tp1 * m_tp1 - v_t)
                        delta = delta * m_t
                        last_adv = delta + gamma * lam * last_adv
                        adv[:, t] = last_adv
                    returns = adv + values.detach()
                    advantages = adv
                else:
                    # Discounted returns with masking (no leakage past terminal)
                    returns = torch.zeros_like(rewards)
                    for t in reversed(range(T)):
                        if t == T - 1:
                            returns[:, t] = rewards[:, t] * mask[:, t]
                        else:
                            returns[:, t] = (rewards[:, t] * mask[:, t] +
                                             gamma * returns[:, t+1] * mask[:, t+1])
                    advantages = (returns - values).detach()

            # mask advantages and normalize (masked)
            advantages = advantages * mask
            adv_n = mask.sum().clamp_min(1.0)
            adv_mean = (advantages.sum() / adv_n).detach()
            adv_var = (((advantages - adv_mean).pow(2)) * mask).sum() / adv_n
            advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

            # value targets masked
            returns = returns * mask

            # Freeze old log-probs for all PPO epochs (shape-check)
            old_logps = logps.detach()
            if getattr(self, "old_logps", None) is None or self.old_logps.shape != logps.shape:
                self.old_logps = old_logps
            else:
                self.old_logps = old_logps  # refresh after new rollout

            # Flatten valid timesteps
            valid = mask.bool().view(-1)
            flat = lambda x: x.view(-1)[valid]
            flat_logps = flat(logps)
            flat_old = flat(self.old_logps)
            flat_advs = flat(advantages)
            flat_returns = flat(returns)
            flat_values = flat(values)
            # For entropy, we have per-step entropys already
            flat_ent = flat(entropys)

            N = flat_logps.numel()
            idx = torch.arange(N, device=self.device)

            # Train for K epochs over minibatches
            for _ in range(ppo_epochs):
                perm = idx[torch.randperm(N)]
                for start in range(0, N, mb_size):
                    mb = perm[start:start+mb_size]

                    mb_logps = flat_logps[mb]
                    mb_old = flat_old[mb]
                    mb_advs = flat_advs[mb]
                    mb_returns = flat_returns[mb]
                    mb_values = flat_values[mb]
                    mb_ent = flat_ent[mb]

                    ratios = torch.exp(mb_logps - mb_old)
                    surr1 = ratios * mb_advs
                    surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss with clipping
                    v_pred = mb_values
                    if vf_clip is not None:
                        v_clipped = mb_values.detach() + (v_pred - mb_values.detach()).clamp(-vf_clip, vf_clip)
                        v_loss_unclipped = (v_pred - mb_returns).pow(2)
                        v_loss_clipped = (v_clipped - mb_returns).pow(2)
                        value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        value_loss = (v_pred - mb_returns).pow(2).mean()

                    entropy_loss = mb_ent.mean()

                    # KL penalty if KL divergence exceeds target
                    with torch.no_grad():
                        approx_kl = (mb_old - mb_logps).mean()
                    
                    kl_penalty = 0.0
                    if approx_kl > self.kl_target:
                        kl_penalty = 10.0 * (approx_kl - self.kl_target)

                    loss = policy_loss + 0.5 * value_loss - self.current_entropy_coef * entropy_loss + kl_penalty

                    self.optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping with new parameter
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()

            # For monitoring (approx KL and clip fraction on the last minibatch)
            with torch.no_grad():
                approx_kl = (mb_old - mb_logps).mean().item()
                clip_frac = ((ratios > 1.0 + clip_range) | (ratios < 1.0 - clip_range)).float().mean().item()
                self.last_approx_kl = approx_kl
                self.last_clip_frac = clip_frac

                # Critic explained variance
                y_true = mb_returns.detach().cpu().numpy()
                y_pred = mb_values.detach().cpu().numpy()
                var_y = np.var(y_true)
                ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

                # Print diagnostics
                print(f"[Diagnostics] KL={approx_kl:.4f}, ClipFrac={clip_frac:.3f}, "
                      f"Entropy={mb_ent.mean().item():.4f}, ExplainedVar={ev:.4f}, "
                      f"PolicyLoss={policy_loss.item():.4f}, ValueLoss={value_loss.item():.4f}", flush=True)

            # cache current logps for next rollout
            self.old_logps = old_logps

            return (
                selected_nodes,
                float(np.mean(final_scores)),
                float(policy_loss.item()),
                lengths.float().mean().item()
            )

        else:
            # REINFORCE baseline
            # vanilla REINFORCE over valid steps
            loss = - (rewards * logps * mask).sum() / n
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            return (
                selected_nodes,
                float(np.mean(final_scores)),
                float(loss.item()),
                lengths.float().mean().item()
            )

    def train_from_sets(self, episodes: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        while not env.done:
            k += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            logps = []
            actions = []
            for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    action = len(candidates)
                else:
                    sub_idx = [idx for idx, v in enumerate(candidates) if v in valid_candidates]
                    action = sub_idx[logits[sub_idx].argmax().item()]
                actions.append(action)
                logps.append(logits[action])
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1, v2 in zip(valid_index, logps, values):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
        # Stack and Padding
        logps, values = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                         for episode_xs in (episode_logps, episode_values)]
        lengths = torch.LongTensor(env.lengths).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device, dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        returns = mask
        advantages = returns - values.detach()
        advantages = advantages * mask
        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = - (torch.min(advantages, returns) * mask).sum() / mask.sum()
        loss.backward()
        self.optimizer.step()

    def _prepare_inputs(self, valid_index: List[int], episodes: List[List[int]], z_nodes: sp.csc_matrix,
                        z_seeds: sp.csc_matrix):
        # seed vectors
        n = z_seeds.shape[0]
        data = [np.asarray(z_seeds[:, i]).reshape(-1) for i in valid_index]
        vals_seed = np.vstack([x[:self.n_nodes] for x in data])
        # candidate node vectors
        batch_candidates = [list(self.graph.outer_boundary(episodes[i])) for i in valid_index]
        candidate_set = set.union(*map(set, batch_candidates)) if batch_candidates else set()
        candidate_set = np.array(list(candidate_set))
        node_index = np.zeros(self.n_nodes + 1, dtype=np.int32)
        node_index[candidate_set] = np.arange(len(candidate_set)) + 1
        data = []
        indptr = [0]
        for candidates in batch_candidates:
            data.append(np.ones(len(candidates)))
            indptr.append(indptr[-1] + len(candidates))
        vals_attr = sp.csc_matrix((np.hstack(data), (candidate_set, np.arange(indptr[-1]))), shape=(self.n_nodes, indptr[-1]))
        # dynamic node vectors (neighborhood)
        data = [np.asarray(z_nodes[:, i]).reshape(-1) for i in valid_index]
        vals_node = np.vstack([x[:self.n_nodes] for x in data])
        # attributes on candidates
        vals_attr = (vals_attr.T * self.graph.nodes_attr).toarray()
        # Re-index split
        vals_attr = torch.from_numpy(vals_attr).to(self.device)
        vals_seed = torch.from_numpy(vals_seed).to(self.device)
        vals_node = torch.from_numpy(vals_node).to(self.device)
        indptr = np.array(indptr)
        return vals_attr, vals_seed, vals_node, indptr, batch_candidates

    def _sample_actions(self, batch_logits: List, causal_bias: Optional[List] = None) -> (List, List, List):
        """
        Sample actions with optional causal bias and STOP bias.
        
        Args:
            batch_logits: List of logits for each batch element
            causal_bias: Optional list of causal importance scores to bias action selection
            
        Returns:
            actions: Selected action indices
            logps: Log probabilities of selected actions
            entropys: Entropy values
        """
        batch = []
        for i, logits in enumerate(batch_logits):
            # logits are expected to be log-probabilities (log-softmax output)
            ps = torch.exp(logits)
            
            # Apply STOP bias to the last logit (STOP action)
            if len(logits) > 0:
                # Add bias to STOP action (last index)
                stop_logit_idx = len(logits) - 1
                logits[stop_logit_idx] += self.stop_bias
                ps = torch.exp(logits)
                ps = ps / ps.sum()  # Renormalize
            
            # Apply causal bias if available
            if causal_bias is not None and i < len(causal_bias):
                # Add causal importance as bias to logits
                causal_logits = torch.log(causal_bias[i] + 1e-8)  # Convert to log space
                # Combine original logits with causal bias
                biased_logits = logits + 0.1 * causal_logits  # Weight the causal bias
                ps = torch.exp(biased_logits)
                ps = ps / ps.sum()  # Renormalize
            
            entropy = -(ps * torch.log(ps + 1e-8)).sum()
            action = torch.multinomial(ps, 1).item()
            logp = logits[action]  # Use original logits for policy gradient
            batch.append([action, logp, entropy])
        actions, logps, entropys = zip(*batch)
        actions = np.array(actions)
        return actions, logps, entropys
