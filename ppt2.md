---
marp: true

---

# Weekly Update  
**RL-Based GNN Explainer**  
Nathan Behailu

---

# 1. RG-Explainer: Main Components

- **Seed Selection**  
  Pick an influential starting node.

- **Iterative Graph Generation**  
  Grow a connected subgraph step by step from the frontier.

- **Stopping Criteria Learning**  
  Learn when to halt to avoid oversized subgraphs.

👉 Our focus: **Iterative Graph Generation**  
Seed + Stop are unchanged for now.

---

# 2. Iterative Graph Generation (RG-Explainer)

- **State:** partial subgraph $S_t$ + frontier $\partial S_t$  
- **Action:** pick a node from frontier  
- **Reward:** negative of explanation loss  

**Loss function:**  
\[
L(S) = L_{\text{pred}} + \lambda_1 L_{\text{size}} + \lambda_2 L_{\text{radius}} + \lambda_3 L_{\text{sim}}
\]

- REINFORCE update:  
\[
\nabla J(\theta) = \mathbb{E}\!\left[\sum_t \nabla \log \pi_\theta(a_t|s_t)\, Q(s_t,a_t)\right]
\]

---

# 3. Monte Carlo Rollouts (RG-Explainer)

- **Why:** reward only available at the end of generation  
- **Fix:** approximate $Q(s_t, a_t)$ with rollouts

\[
Q(s_t, a_t) \approx \tfrac{1}{M}\sum_{i=1}^M -L(S^{(i)})
\]

where $S^{(i)}$ are complete sampled subgraphs.

---

# 4. Our Modification: PPO

RG used **vanilla REINFORCE**.  
We switch to **Proximal Policy Optimization (PPO):**

- **Policy ratio:**  
\[
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
\]

- **Clipped surrogate objective:**  
\[
L^{\text{PPO}}(\theta) = \mathbb{E}\!\left[\min(r_t A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon) A_t)\right]
\]

---

# 5. What is $A_t$? (Advantage)

- Computed with **Generalized Advantage Estimation (GAE):**

\[
\delta_t = r_t + \gamma V_{t+1} - V_t
\]

\[
A_t = \delta_t + \gamma \lambda \delta_{t+1} + \gamma^2 \lambda^2 \delta_{t+2} + \dots
\]

- In code:
  - Compute TD residuals $\delta_t$  
  - Backward recursion with $\gamma, \lambda$  
  - Normalize $A_t$ across batch

---

# 6. PPO Loss Function (in our code)

**Total loss:**

\[
\mathcal{L} = L_{\text{policy}} + 0.5 L_{\text{value}} - \beta H(\pi)
\]

- **Policy loss (clipped)**  
- **Value loss (clipped MSE)**  
- **Entropy bonus** $H(\pi)$ encourages exploration  
- **KL early stop** + **gradient clipping** for stability

---

# 7. Why PPO Helps vs REINFORCE

- **Stability:** Clipping prevents destructive updates  
- **Variance reduction:** Advantage estimates via value head  
- **Efficiency:** Multiple epochs + minibatches per rollout  
- **Exploration:** Entropy regularization  

---

# 8. Results (Placeholder)

**Previous (REINFORCE baseline)**

| Dataset | AUC | Notes |
|---------|-----|-------|
| BA-Shapes | … | vanilla REINFORCE |
| BA-Community | … | MC rollouts |

**Current (PPO-based)**

| Dataset | AUC | Notes |
|---------|-----|-------|
| BA-Shapes | … | PPO (clip=0.2) |
| BA-Community | … | GAE + entropy |

---

# 9. Next Steps

- Tune entropy decay schedule  
- Experiment with frontier shaping  
- Sweep target KL thresholds  
- Fill in full results table

---
