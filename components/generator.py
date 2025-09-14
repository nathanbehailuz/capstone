from typing import Union, Optional, List, Set, Dict
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
                 max_reward: float = 1.):
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

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        # PPO hyperparameters (defaults)
        self.clip_range = 0.2
        self.vf_clip = 0.2
        self.ppo_epochs = 4
        self.minibatch_size = 256
        self.gamma = 0.99
        self.use_gae = True
        self.gae_lambda = 0.90
        self.reward_norm = False
        # Monitoring
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0

    def load_nodefeats(self, x):
        #self.nodefeats = torch.from_numpy(x).float().to(self.device) 
        self.nodefeats = x.numpy()
        #self.nodefeats = x.astype(np.float32)

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
            # Action masking: ensure only valid candidates are sampled
            actions, logps, entropys = self._sample_actions(batch_logits)
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
        rewards[np.arange(logps.size(0)), lengths.cpu().numpy() - 2] = final_scores
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
            # hyper-params (expose via __init__)
            gamma = getattr(self, "gamma", 0.99)
            use_gae = getattr(self, "use_gae", False)
            lam = getattr(self, "gae_lambda", 0.95)
            clip_range = getattr(self, "clip_range", 0.2)
            vf_clip = getattr(self, "vf_clip", 0.2)
            ppo_epochs = getattr(self, "ppo_epochs", 4)
            mb_size = getattr(self, "minibatch_size", 256)

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
                    v_clipped = mb_values.detach() + (v_pred - mb_values.detach()).clamp(-vf_clip, vf_clip)
                    v_loss_unclipped = (v_pred - mb_returns).pow(2)
                    v_loss_clipped = (v_clipped - mb_returns).pow(2)
                    value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = mb_ent.mean()

                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
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
                      f"Entropy={mb_ent.mean().item():.4f}, ExplainedVar={ev:.4f}", flush=True)

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

    def _sample_actions(self, batch_logits: List) -> (List, List, List):
        batch = []
        for logits in batch_logits:
            # logits are expected to be log-probabilities (log-softmax output)
            ps = torch.exp(logits)
            entropy = -(ps * logits).sum()
            action = torch.multinomial(ps, 1).item()
            logp = logits[action]
            batch.append([action, logp, entropy])
        actions, logps, entropys = zip(*batch)
        actions = np.array(actions)
        return actions, logps, entropys
