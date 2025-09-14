As we discussed, I focused this week on replacing the REINFORCE algorithm with a full implementation of PPO.

##### PPO Implementation

PPO extends policy gradient methods by adding a clipping mechanism to prevent overly large updates.

* Policy Ratio

For each action $a_t$ at state $s_t$:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp\!\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{\text{old}}}(a_t|s_t)\right)
$$

* Clipped Policy Loss

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t\big)\right]
$$

* Value Function Loss (with clipping)

$$
L^{\text{VF}}(\theta) = \mathbb{E}_t \Big[\max\Big((V_\theta(s_t) - R_t)^2,\ (V_{\theta_{\text{old}}}(s_t) + \text{clip}(V_\theta(s_t) - V_{\theta_{\text{old}}}(s_t), -\epsilon_v, \epsilon_v) - R_t)^2\Big)\Big]
$$

* Entropy Bonus

$$
L^{\text{ENTROPY}}(\theta) = \beta \, \mathbb{E}_t \big[- \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)\big]
$$

* Combined Objective

$$
L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 L^{\text{ENTROPY}}(\theta)
$$

##### For advantage estimation, I implemented two options:

1. Discounted Returns

$$
R_t = r_t + \gamma R_{t+1}
$$

2. Generalized Advantage Estimation (GAE) (controlled with a boolean flag)

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad 
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

#### Experimental Results

I tested PPO on three of the six benchmark datasets and compared it to both the paper’s reported results and REINFORCE (by running their code).

| Dataset         | Paper Results (AUC ± std) | REINFORCE     | PPO           |
| --------------- | ------------------------- | ------------- | ------------- |
| **BA-Shapes**   | 0.985 ± 0.013             | 0.951 ± 0.096 | 0.796 ± 0.067 |
| **BA-2 Motifs** | 0.657 ± 0.017             | 0.618 ± 0.041 | 0.588 ± 0.074 |
| **MUTAG**       | 0.873 ± 0.028             | 0.842 ± 0.080 | 0.829 ± 0.085 |

* PPO is currently underperforming REINFORCE on all datasets.
* The AUC values I obtained differ from those reported in the paper, though I was able to replicate one dataset last semester. I am currently trying to debug this issue. 
* To reduce instability, I added GAE (for reward) and entropy (for loss). These improved performance slightly, but it still needs some progress.

#### Next Steps

- I am planning to do  hyperparameter search (clip range, learning rate, entropy coefficient, $\gamma$, $\lambda$). Plus monitoring loss and reward values across iterations to check instability and identify the patterns.

