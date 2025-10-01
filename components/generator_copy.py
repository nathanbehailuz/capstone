from typing import Union, Optional, List, Set, Dict, Tuple
import numpy as np
from scipy import sparse as sp

import torch
from torch import nn

from .graph import Graph
from .generator_core import GraphConv, Agent


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
        if any(self.dones):
            raise ValueError("One or more seeds are already terminal.")
        seeds = [self.data[i][0] for i in range(self.bs)]
        nodes = [self.data[i] for i in range(self.bs)]
        x_seeds = self.make_single_node_encoding(seeds)
        x_nodes = self.make_nodes_encoding(nodes)
        return x_seeds, x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int]):
        assert len(new_nodes) == len(index)
        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            self.trajectories[i].append(v)
            if v == 'EOS':
                self.dones[i] = True
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif len(self.graph.outer_boundary(self.trajectories[i])) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v
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
                 entropy_coef: float = 1e-6,
                 n_rollouts: int = 10,
                 max_size: int = 25,
                 k: int = 3,
                 alpha: float = 0.85,
                 max_reward: float = 1.,
                 ppo_epochs: int = 4,
                 minibatch_size: int = 256,
                 clip_eps: float = 0.2,
                 target_kl: float = 0.03,
                 grad_clip: float = 1.0,
                 gae_lambda: float = 0.95,
                 frontier_shaping_beta: float = 0.0,
                 entropy_decay_start: float = 1e-3,
                 entropy_decay_end: float = 1e-4,
                 entropy_decay_final: float = 0.0,
                 # Causal mechanism parameters
                 use_causal: bool = False,
                 causal_explainer: Optional[Union['NodeLevelCausalExplainer', 'GraphLevelCausalExplainer']] = None,
                 causal_weight: float = 0.5):
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
        
        # PPO parameters
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.grad_clip = grad_clip
        self.gae_lambda = gae_lambda
        self.frontier_shaping_beta = frontier_shaping_beta
        
        # Entropy decay parameters
        self.entropy_decay_start = entropy_decay_start
        self.entropy_decay_end = entropy_decay_end
        self.entropy_decay_final = entropy_decay_final
        self.current_entropy_coef = entropy_decay_start

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

    def load_nodefeats(self, x):
        # Accept torch.Tensor or np.ndarray; store as np.ndarray on CPU
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.nodefeats = x
    
    def update_entropy_coef(self, epoch, total_epochs):
        """Update entropy coefficient based on decay schedule"""
        if total_epochs <= 1:
            self.current_entropy_coef = self.entropy_decay_start
            return
        
        # Linear decay from start to end over first 70% of epochs
        decay_epochs = int(0.7 * total_epochs)
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
        
        # Update the entropy coefficient used in training
        self.entropy_coef = self.current_entropy_coef
    
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
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)
        self.model.eval()
        with torch.no_grad():
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

    def train_from_rewards(self, seeds: List[int], fn, policy="PPO", ppo_epochs=None, minibatch_size=None):
        bs = len(seeds)
        self.model.train()
        
        # Use instance parameters if not provided
        if ppo_epochs is None:
            ppo_epochs = self.ppo_epochs
        if minibatch_size is None:
            minibatch_size = self.minibatch_size

        # rollout trajectories with buffered data
        selected_nodes, logps, values, entropys, rollout_buffer = self.sample_episodes(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # compute reward matrix
        rewards = np.zeros(logps.shape, dtype=np.float32)
        final_scores = fn(selected_nodes)
        rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2] = final_scores
        
        # Apply frontier shaping to terminal rewards
        if self.frontier_shaping_beta > 0:
            terminal_rewards = rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2]
            shaped_terminal_rewards = self.apply_frontier_shaping(selected_nodes, terminal_rewards)
            rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2] = shaped_terminal_rewards

        if self.n_rollouts:
            for k in range(2, logps.shape[1] + 1):
                valid_idx = [i for i, x in enumerate(selected_nodes) if len(x) >= k
                            and x[k-1] != 'EOS' and len(self.graph.outer_boundary(x[:k]))]
                if len(valid_idx):
                    rollouts = self.sample_rollouts([selected_nodes[i][:k] for i in valid_idx])
                    for i, samples in zip(valid_idx, rollouts):
                        rewards[i, k-2] = np.mean(fn(samples))
        # Note: No else clause - keep rewards sparse (only terminal rewards)

        rewards = torch.from_numpy(rewards).float().to(self.device)

        # mask valid timesteps
        mask = torch.arange(rewards.size(1), device=self.device, dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()

        if policy == "PPO":
            # GAE(λ) parameters
            gamma = 0.99  # discount factor
            lam = self.gae_lambda    # GAE lambda parameter
            T = rewards.size(1)
            
            # Store old values for PPO
            old_logps = torch.stack(rollout_buffer['old_logps']).to(self.device)
            old_values = torch.stack(rollout_buffer['old_values']).to(self.device)
            
            # Create step indices for proper alignment
            step_indices = []
            for i, epi_idx in enumerate(rollout_buffer['episode_indices']):
                step_idx = rollout_buffer['step_indices'][i]
                step_indices.append((epi_idx, step_idx))
            
            # Convert to tensor for easier indexing
            step_indices = torch.tensor(step_indices, device=self.device)
            
            # Filter out EOS/terminal steps from PPO minibatches
            epi = step_indices[:, 0].long()
            t = step_indices[:, 1].long()
            keep = t < (lengths[epi] - 1)  # keep only non-terminal steps
            indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
            
            # Filter all rollout buffer data
            filtered_step_indices = step_indices[indices]
            filtered_old_logps = old_logps[indices]
            filtered_old_values = old_values[indices]
            
            # Build V tensor aligned with rollout_buffer 'keep' mask
            V = torch.zeros_like(rewards)
            V[:] = float('nan')
            for k, (epi_idx, step_idx) in enumerate(step_indices.tolist()):
                V[epi_idx, step_idx] = old_values[k]  # old_values stacked in rollout order
            
            # Make V NaN-safe before GAE math
            V = torch.nan_to_num(V, nan=0.0)
            
            # Compute next values (bootstrap 0 at terminal)
            V_next = torch.zeros_like(V)
            V_next[:, :-1] = V[:, 1:].clone()
            V_next[:, -1] = 0.0
            
            # TD residuals (δ) - only compute where we have valid data
            delta = rewards + gamma * V_next - V
            delta = torch.where(mask.bool(), delta, torch.zeros_like(delta))
            
            # GAE advantages
            advantages = torch.zeros_like(delta)
            for t in reversed(range(T)):
                advantages[:, t] = delta[:, t] + gamma * lam * (advantages[:, t+1] if t+1 < T else 0.0)
            advantages = advantages * mask
            
            # Returns for value loss
            returns = (advantages + V) * mask
            
            # Flatten to the kept indices ONCE, and normalize advantages ONCE
            adv_flat = torch.stack([advantages[epi, t] for epi, t in filtered_step_indices.tolist()])
            ret_flat = torch.stack([returns[epi, t] for epi, t in filtered_step_indices.tolist()])
            val_old_flat = filtered_old_values
            
            # Global advantage normalization
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            
            # PPO training loop
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            update_count = 0
            stop_early = False
            
            for epoch in range(ppo_epochs):
                # Create minibatches using filtered indices
                num_steps = len(indices)
                if num_steps == 0:
                    print("WARNING: No valid steps for PPO training")
                    break
                
                # Ensure minibatch size doesn't exceed available steps
                effective_minibatch_size = min(minibatch_size, num_steps)
                batch_indices = torch.randperm(num_steps, device=self.device)
                
                for start_idx in range(0, num_steps, effective_minibatch_size):
                    end_idx = min(start_idx + effective_minibatch_size, num_steps)
                    batch_indices_subset = batch_indices[start_idx:end_idx]
                    
                    self.optimizer.zero_grad()
                    
                    # Recompute policy for this minibatch
                    batch_logits = []
                    batch_values = []
                    batch_entropys = []
                    batch_old_logps = []
                    batch_old_values = []
                    batch_advantages = []
                    batch_returns = []
                    batch_adjusted_action_indices = []
                    batch_current_logps = []  # Store log-probs, not logits
                    
                    for idx in batch_indices_subset:
                        # Map back to original indices
                        original_idx = indices[idx].item()
                        epi_idx, step_idx = filtered_step_indices[idx].tolist()
                        model_inputs = rollout_buffer['model_inputs'][original_idx]
                        action_idx = rollout_buffer['action_indices'][original_idx]
                        
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
                        
                        # Model has EOS head - outputs len(candidates) + 1 logits
                        candidates = model_inputs[4]  # batch_candidates for this step
                        expected = len(candidates) + 1  # EOS head - model outputs len(candidates) + 1 logits
                        if current_logits.size(0) != expected:
                            print(f"DEBUG: logits={current_logits.size(0)} vs expected={expected}, candidates={candidates}")
                        assert current_logits.size(0) == expected, f"logits={current_logits.size(0)} vs expected={expected}, candidates={len(candidates)}"
                        
                        # Check bounds and skip if out of bounds (shouldn't happen after EOS filter)
                        if action_idx >= current_logits.size(0):
                            print(f"WARNING: action_idx {action_idx} >= logits size {current_logits.size(0)} - skipping")
                            continue
                        adjusted_action_idx = action_idx
                        
                        # Compute entropy and log-probs from full distribution
                        probs = torch.softmax(current_logits, dim=0)
                        log_probs = torch.log_softmax(current_logits, dim=0)
                        entropy = -(probs * log_probs).sum()
                        
                        # Compute current log probability of taken action (use log-probs, not raw logits)
                        current_logp = log_probs[adjusted_action_idx]
                        
                        # Get old values from filtered data
                        old_logp = filtered_old_logps[idx]
                        old_value = filtered_old_values[idx]
                        
                        # Get pre-computed advantage and return for this step
                        advantage = adv_flat[idx]
                        return_val = ret_flat[idx]
                        
                        batch_logits.append(current_logits)
                        batch_values.append(current_value)
                        batch_entropys.append(entropy)
                        batch_old_logps.append(old_logp)
                        batch_old_values.append(old_value)
                        batch_advantages.append(advantage)
                        batch_returns.append(return_val)
                        batch_adjusted_action_indices.append(adjusted_action_idx)
                        batch_current_logps.append(current_logp)  # Store log-prob, not logit
                    
                    # Stack batch data - use stored log-probs, not raw logits
                    if len(batch_current_logps) == 0:
                        print("WARNING: All samples in minibatch were skipped due to logits length mismatch")
                        continue  # Skip this minibatch entirely
                    current_logps = torch.stack(batch_current_logps)
                    current_values = torch.stack(batch_values)
                    current_entropys = torch.stack(batch_entropys)
                    old_logps_batch = torch.stack(batch_old_logps)
                    old_values_batch = torch.stack(batch_old_values)
                    advantages = torch.stack(batch_advantages)
                    returns_batch = torch.stack(batch_returns)
                    
                    # Advantages are already globally normalized
                    
                    # Compute ratios
                    ratios = torch.exp(current_logps - old_logps_batch)
                    
                    # PPO surrogate loss
                    eps = self.clip_eps
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping
                    vf_clip = self.clip_eps
                    value_pred_clipped = old_values_batch + torch.clamp(
                        current_values - old_values_batch, -vf_clip, vf_clip
                    )
                    value_loss1 = (current_values - returns_batch) ** 2
                    value_loss2 = (value_pred_clipped - returns_batch) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                    
                    # Entropy bonus (positive entropy encourages exploration)
                    entropy_bonus = current_entropys.mean()
                    
                    # Total loss
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_bonus
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.optimizer.step()
                    update_count += 1
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_bonus.item()
                    
                    # Log diagnostics for this minibatch
                    with torch.no_grad():
                        approx_kl = (old_logps_batch - current_logps).mean().item()  # Non-negative KL
                        clip_frac = ((ratios > 1.0 + eps) | (ratios < 1.0 - eps)).float().mean().item()
                        
                        # Explained variance
                        y_true = returns_batch.detach().cpu().numpy()
                        y_pred = current_values.detach().cpu().numpy()
                        var_y = np.var(y_true)
                        ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0
                        
                        print(f"[Diagnostics]"
                              f"KL={approx_kl:.4f}, ClipFrac={clip_frac:.3f}, "
                              f"Entropy={current_entropys.mean().item():.4f}, "
                              f"ExplainedVar={ev:.4f}, "
                              f"PolicyLoss={policy_loss.item():.4f}, "
                              f"ValueLoss={value_loss.item():.4f}", flush=True)
                        
                        # KL early stopping
                        if approx_kl > self.target_kl:
                            print(f"Early stopping: KL divergence {approx_kl:.4f} > target {self.target_kl}")
                            stop_early = True
                            break
                
                if stop_early:
                    break

            # Guard against empty minibatches
            if update_count == 0:
                print("WARNING: No updates performed - all minibatches were empty due to logits length mismatch")
                return (
                    selected_nodes,
                    float(np.mean(final_scores)),
                    float('nan'),
                    lengths.float().mean().item()
                )
            
            avg_policy_loss = total_policy_loss / update_count
            
            return (
                selected_nodes,
                np.mean(final_scores),
                avg_policy_loss,
                lengths.float().mean().item()
            )

        else:
            # REINFORCE fallback
            self.optimizer.zero_grad()
            n = mask.sum()
            policy_loss = -(rewards * logps * mask).sum() / n
            loss = policy_loss
            loss.backward()
            self.optimizer.step()

            return (
                selected_nodes,
                np.mean(final_scores),
                policy_loss.item(),
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
        max_steps = max_size * 2  # Safety limit to prevent infinite loops
        step_count = 0
        while not env.done and step_count < max_steps:
            k += 1
            step_count += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            actions = []
            logps_sel = []
            vals_sel = []
            proc_idx = []
            new_nodes = []
            
            for j, (logits, candidates, i) in enumerate(zip(batch_logits, batch_candidates, valid_index)):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    # Skip: no valid supervised target for this step with a no-EOS head
                    continue
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
        # Stack and Padding
        logps, values = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                         for episode_xs in [episode_logps, episode_values]]
        lengths = torch.LongTensor([len(x) for x in env.trajectories]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        # td_loss = ((values - self.max_reward) ** 2 * mask).sum() / n
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
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        max_steps = max_size * 2  # Safety limit to prevent infinite loops
        step_count = 0
        while not env.done and step_count < max_steps:
            k += 1
            step_count += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            actions = []
            logps_sel = []
            vals_sel = []
            proc_idx = []
            new_nodes = []
            
            for j, (logits, candidates, i) in enumerate(zip(batch_logits, batch_candidates, valid_index)):
                # Check if episode has enough steps (k is 1-indexed for trajectory steps)
                if k > len(episodes[i]) - 1:  # k=1 corresponds to episodes[i][1], k=2 to episodes[i][2], etc.
                    continue
                v = episodes[i][k]
                try:
                    action = candidates.index(v)
                    actions.append(action)
                    logps_sel.append(logits[action])
                    vals_sel.append(values[j])  # Take the matching value
                    proc_idx.append(i)
                    # Map action to node or EOS right here so order stays aligned
                    if action < len(candidates):
                        new_nodes.append(candidates[action])
                    else:
                        new_nodes.append('EOS')
                except ValueError:
                    # Target not in candidates; skip this timestep
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
        # Stack and Padding
        logps, values = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                         for episode_xs in [episode_logps, episode_values]]
        lengths = torch.LongTensor([len(x) for x in episodes]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        # td_loss = ((values - self.max_reward) ** 2 * mask).sum() / n
        policy_loss = -(1. * logps * mask).sum() / n
        # loss = policy_loss + .5 * td_loss
        # loss.backward()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def _sample_rollouts_loop(self, prefix, max_size=None):
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, prefix, max_size)
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
        env = ExpansionEnv(self.graph, prefix * self.n_rollouts, max_size)
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
        z_seeds = self.conv(x_seeds)
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
            'old_entropys': []  # List of old entropies (detached)
        }
        
        step_count = 0
        max_steps = env.max_size * 2  # Safety limit to prevent infinite loops
        while not env.done and step_count < max_steps:
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits, values = self.model(*model_inputs)
            
            # ========== CAUSAL MECHANISM INTEGRATION ==========
            causal_bias = None
            if self.use_causal and self.causal_explainer is not None:
                # Compute causal importance for current state
                causal_bias = self._compute_causal_bias(valid_index, env.trajectories, batch_candidates)
            
            # Action sampling with optional causal bias
            actions, logps, entropys = self._sample_actions(batch_logits, causal_bias)
            
            # Handle EOS head model - action < len(candidates) selects candidate, action == len(candidates) is EOS
            new_nodes = []
            for action, candidates in zip(actions, batch_candidates):
                if action < len(candidates):
                    # Select the actual candidate
                    new_nodes.append(candidates[action])
                else:
                    # EOS action (action == len(candidates))
                    new_nodes.append('EOS')
            
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
            
            for i, v1, v2, v3 in zip(valid_index, logps, values, entropys):
                episode_logps[i].append(v1)
                episode_values[i].append(v2)
                episode_entropys[i].append(v3)
            
            step_count += 1
            
        # Stack and Padding
        logps, values, entropys = [nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_xs], True)
                                   for episode_xs in [episode_logps, episode_values, episode_entropys]]
        return env.trajectories, logps, values, entropys, rollout_buffer

    def _prepare_inputs(self, valid_index: List[int], trajectories: List[List[int]],
                        z_nodes: sp.csc_matrix, z_seeds: sp.csc_matrix):
        vals_attr = [] if self.nodefeats is not None else None
        vals_seed = []
        vals_node = []
        indptr = []
        offset = 0
        batch_candidates = []
        for i in valid_index:
            boundary_nodes = self.graph.outer_boundary(trajectories[i])
            # Make candidate ordering deterministic
            candidate_nodes = sorted(boundary_nodes)
            # assert len(candidate_nodes)
            
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
                causal_weight=self.causal_weight
            )
            print(f"[CAUSAL] Initialized causal reward calculator with weight {self.causal_weight}")
        else:
            print("[CAUSAL] Causal mechanism not enabled")

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
            uniform_bias = torch.ones(len(candidates), dtype=torch.float32)
            causal_bias.append(uniform_bias)
        
        return causal_bias

    def _sample_actions(self, batch_logits: List, causal_bias: Optional[List] = None) -> (List, List, List):
        """
        Sample actions with optional causal bias.
        
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
