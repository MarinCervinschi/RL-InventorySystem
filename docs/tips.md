# Master Protocol: RL Implementation & Statistical Analysis 

## Part 1: RL Implementation & Engineering Guidelines
**Goal:** Build a robust, stable agent using "Software Engineering" principles.

1. **Environment Design**

    - Keep **episode length short** and **dense** in reward signal to accelerate learning.    
    - **Normalize observation/action spaces** with wrappers like
    **`NormalizeObservation`**, **`NormalizeReward`**.
    - For **stochastic environments**: separate train and test seeds.
        
        Use deterministic resets and seed control for debugging and
        reproducibility:
        
        ```cpp
        env = gym.make("YourCustomEnv-v0")
        env.reset(seed=42)
        ```
        
2. **Choose the Right Algorithm + Sensible Defaults**
    - PPO is a solid generalist — stable, good sample efficiency.
    - SAC is best for continuous actions + industrial control.
    - Avoid reinventing the wheel: use **`StableBaselines3`** or **`CleanRL`** baselines to start.
    
3.  **Debugging and Training Practices**
    
    Before running long trainings, perform these unit tests:
    - The $\gamma = 0$ Test (Crucial):
        - Set discount factor $\gamma = 0$.
        - **Expected Result**: The optimal policy should converge to "Never Order" ($a_t = 0$).
        - **Logic**: With $\gamma=0$, the agent ignores future stockouts and only sees the immediate penalty of ordering ($K + i \cdot q$). If it still orders, your reward calculation or code logic is buggy.
    - **The Overfit Test:**
        Train on a tiny, deterministic environment (e.g., constant demand, 0 lead time). The agent should quickly achieve perfect behavior.
    - **Log everything**: use **`TensorBoard`**, **`Weights`** & **`Biases`**.
    - Use curriculum learning for hard problems: increase complexity gradually.
    - Keep training deterministic when possible:
        
        ```cpp
        import torch, random
        torch.manual_seed(0)
        random.seed(0)
        ```
        
        Don’t trust the reward alone — visualize behaviors
        
4.  **Reward Shaping**
    - Sparse or delayed rewards? Shape the reward to guide early learning:
        - Add intermediate incentives (e.g., reducing delay, energy use, backlog).
        - Penalize unsafe/infeasible actions (e.g., overcapacity, invalid routes).
        
        Ensure shaped reward aligns with long-term business KPIs.
        
        Avoid giving away the optimal policy — don’t let the reward do the planning.
        
        ```cpp
        reward =-travel_time
        if item_picked:
        reward += 5
        if wrong_pick:
        reward-= 10
        ```
        
    - **Bonus**: log both shaped and original reward for analysis/debugging
5. **Hyperparameter Tuning**
    - Tune the few that matter most:
        - **`learning_rate`**, **`gamma`**, **`entropy_coef`**, **`batch_size`**
        - **`buffer_size`** and **`train_freq`** for off-policy (e.g., DQN, SAC)
    - Use logarithmic scales for **`learning_rate`**, **`entropy_coef`**.
    - Prefer **random search** or **Bayesian optimization** (e.g. Optuna) over grid search.
    
    **Practical ranges:**
    
    - **learning rate:** [1e−5,1e−3]
    - **gamma**: [0.95,0.9999]
    - **entropy_coef** (PPO/SAC): [0.0,0.01]
    
    **Pro Tip**: Fix env seed + log reward variance to avoid chasing noise.
    

**EXAMPLE: TUNING PPO WITH OPTUNA + STABLE-BASELINE3**

```cpp
def optimize_agent(trial):
env = gym.make("CartPole-v1")
env = make_vec_env(lambda: env, n_envs=4)
model = PPO(
"MlpPolicy", env,
learning_rate=trial.suggest_loguniform("lr", 1e-5, 1e-3),
gamma=trial.suggest_uniform("gamma", 0.95, 0.9999),
ent_coef=trial.suggest_loguniform("ent", 1e-5, 1e-2),
n_steps=2048,
verbose=0,
) model.learn(total_timesteps=20000)
return evaluate(model)
study = optuna.create_study(direction="maximize")
study.optimize(optimize_agent, n_trials=30)
```

**Additional Tips:**

- Use custom callbacks to track reward and early stop bad runs.
- Run multiple seeds per trial for stability
1. **Evaluation is not just Mean Reward**
    - Mean episodic reward hides variance and real-world risk.
    - Always evaluate across multiple seeds (e.g. 5–10).
    - Plot confidence intervals or reward distributions.
    - Track domain-specific KPIs:
        - Inventory levels, stockouts, late deliveries, utilization.
        - Cost breakdowns (e.g. transport vs. holding).
    - Use visualization to validate learned policy behavior.
2. **Hybridize with Rules or OR Models**
    - Don’t force DRL to do everything — mix and match:
        - DRL selects heuristic parameters (e.g., reorder point, route limit).
        - Use MILP or heuristics for local planning inside the env.
        - Hierarchical control: DRL for macro-decisions, rules for micro.
        
        This improves stability, interpretability, and data efficiency.
        
3. **Treat DRL as Software Engineering**
    - Use experiment trackers like **`wandb`**, **`mlflow`**, **`tensorboard`**.
    - Always log:
        - Code version (commit hash), gym/SB3 versions
        - Seed, env config, number of steps, training time
    - Automate evaluation after training — don’t trust last 100 episodes.
    - Use YAML/JSON configs for reproducibility.
    - Save best model + replay buffer + full metadata.
    - **Pro Tip:** Treat every DRL experiment like a mini software product.


## Part 2: Required Statistical Analysis
**Goal:** Prove your results are valid and not just random noise

1. **Identify the Simulation Type**
    
    - **Type:** Non-terminating (Infinite horizon).
    - **Reason:** Inventory management is a continuous process without a "natural" end event (like a bank closing at 5 PM).
    - **Implication:** We must analyze Steady-State Parameters (Long-run average cost)

2. **Warm-up Period Analysis**

    Since the simulation likely starts with empty inventory ($I=0$), the initial data is biased (Transient Phase). You must detect when the "Steady State" begins and delete the initial data.

    **Method: Welch’s Graphical Procedure**
    - Make $n$ replications (e.g., $n \ge 5$) of length $m$ (large).
    - Calculate the average $\bar{Y}_i$ across replications for each time step $i$. 
    - Apply a Moving Average window $w$ to smooth high-frequency oscillations6.
    - **Plot:** Plot the smoothed average over time.
    - **Visual Inspection:** Identify time step $l$ where the curve "flattens out"7
    - **Action:** Delete all data before step $l$ for your final analysis.
3. **Estimating Performance (Replication/Deletion Approach)**
    
    Once $l$ (warm-up length) is determined:
    1. Run $n'$ independent replications (different random seeds) of length $m' \gg l$.
    2. Compute the average metric (e.g., Daily Cost) for each replication $j$, using only data from $l+1$ to $m'$:
    $$X_j = \frac{\sum_{i=l+1}^{m'} Y_{ji}}{m' - l}$$
    3. Calculate the Point Estimate (Mean of means) and the Confidence Interval (CI):
        $$\bar{X}(n') \pm t_{n'-1, 1-\alpha/2} \sqrt{\frac{S^2(n')}{n'}}$$
        -  Use $\alpha = 0.05$ for a 95% CI.
        - $S^2$ is the sample variance of the replications.
    4. **Sample Size Determination**

        How many replications ($n'$) are enough?

        - You cannot just guess. You must ensure your error is within a specific limit (e.g., relative error $\gamma = 5\%$).
        - Use the Sequential Procedure:
            1. Start with $n_0 = 10$ replications.
            2. Calculate the CI half-length $\delta(n, \alpha)$.
            3. Check if $\frac{\delta(n, \alpha)}{|\bar{X}(n)|} \le \gamma$ (e.g., is the error $\le$ 5% of the mean?).
            4. If No: Add 1 replication and repeat.
            5. If Yes: Stop.
## Part 3: The Comparison (RL vs. (s, S) Policy)

The assignment requires comparing your RL agent against a baseline $(s, S)$ policy (reorder to $S$ when inventory drops below $s$).
Do not compare single numbers.
- **Wrong**: "RL cost is 100, (s,S) cost is 105. RL is better."
- **Right**: 
    - Calculate the 95% Confidence Interval for RL: $[98, 102]$. 
    - Calculate the 95% Confidence Interval for (s,S): $[103, 107]$. 
    - **Conclusion:** Since the intervals do not overlap, RL is statistically significantly better.

**Metrics to track (KPIs):**
Don't just track Reward (Negative Cost). As per the tips, track domain-specific KPIs:

- **Service Level:** % of demand satisfied immediately (inverse of stockouts).
- **Average Inventory Level:** (Drivers of holding cost).
- **Order Frequency:** How often orders are placed (Drivers of setup cost $K$).