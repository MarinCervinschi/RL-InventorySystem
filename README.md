# RL-InventorySystem

**Reinforcement Learning for Multi-Product Inventory Management**

This project implements and evaluates Deep Reinforcement Learning agents (DQN and PPO) for managing replenishment policies in a two-product warehouse system, comparing their performance against classical (s,S) baseline policies.

## 📋 Assignment Overview

The system manages inventory for **two products** with different demand patterns and lead times. At the beginning of each day, the agent must decide:
1. Whether to place a replenishment order for each product
2. How many units to order (if any)

**Objective:** Minimize total operational costs (ordering, holding, and shortage costs).

### Key Characteristics
- **Products:** 2 independent items with distinct suppliers
- **Demand:** Exponential inter-arrival times (λ=0.1) with discrete quantity distributions
- **Lead Times:** Stochastic and **unobservable** (POMDP setting)
  - Product 1: U(0.5, 1.0) months
  - Product 2: U(0.2, 0.7) months
- **Cost Structure:**
  - Setup cost (K): $10 per order
  - Incremental cost (i): $3 per unit
  - Holding cost (h): $1 per unit-day
  - Shortage cost (π): $7 per backlogged unit-day

For complete problem formulation, see [docs/assigment.md](docs/assigment.md) and [docs/mdp.md](docs/mdp.md).

## 🎯 Approach

### 1. Discrete Event Simulation
Built a custom inventory simulation using **SimPy** that models:
- Customer demand arrival processes
- Supplier lead time delays
- Inventory dynamics (on-hand, backorders, in-transit)
- Daily cost accumulation

### 2. MDP Formulation
Addressed the POMDP challenge using **frame stacking** to approximate Markov property.

**State:** `[Inventory_Level, Outstanding_Orders]` for each product, stacked over k+1 time steps.

**Action:** Discrete order quantities `[q₁, q₂]` for each product.

**Reward:** Negative total cost (ordering + holding + shortage)

See [docs/mdp.md](docs/mdp.md) for mathematical details.

### 3. RL Algorithms Implemented
- **DQN (Deep Q-Network):** Value-based method with experience replay
- **PPO (Proximal Policy Optimization):** Policy gradient method with clipped objective

Both implemented using **Stable-Baselines3** with custom Gymnasium environment wrappers.

### 4. Baseline Policy
**Classical (s,S) policy:** Order up to S when inventory falls below s
- Tuned empirically through grid search on steady-state costs

## 📊 Results

Performance evaluated using **Welch's procedure** with 1000 independent replications to ensure steady-state convergence.

### Key Findings
- ✅ Both RL agents successfully learned non-trivial inventory policies
- ✅ Policies account for lead time uncertainty through observation history
- ✅ Warmup period detection applied to exclude transient behavior
- 📈 Performance varies based on hyperparameters (Q_max, learning rate, network architecture)

> **Note:** Run [notebooks/welch_procedure.ipynb](notebooks/welch_procedure.ipynb) to generate detailed performance comparison and statistical analysis.

See [notebooks/](notebooks/) for complete experimental results and visualizations.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended for fast dependency management)

### Installation

Clone and set up the environment:

```bash
git clone https://github.com/MarinCervinschi/rl-inventorysystem.git
cd rl-inventorysystem
uv sync
```

That's it! `uv sync` creates a virtual environment and installs everything you need.


## 🛠️ Technologies

- **Simulation:** SimPy (Discrete Event Simulation)
- **RL Framework:** Stable-Baselines3 + Gymnasium
- **Algorithms:** DQN, PPO
- **Analysis:** NumPy, Pandas, Matplotlib, Seaborn

## 📚 Documentation

- **[Assignment Specification](docs/assigment.md)** - Original problem statement
- **[MDP Formulation](docs/mdp.md)** - Complete mathematical formulation
- **[Implementation Tips](docs/tips.md)** - Development guidelines

## 🔬 Experiments & Notebooks

Explore the experimental workflow:
1. **[MDP Exploration](notebooks/basic/01_mdp_exploration.ipynb)** - Understanding the state/action space
2. **[Simulation Basics](notebooks/basic/02_simulation_basics.ipynb)** - Testing the SimPy engine
3. **[Baseline Tuning](notebooks/basic/03_sS_policy_empirical.ipynb)** - Optimizing (s,S) parameters
4. **[DQN Training](notebooks/dqn_experiments.ipynb)** - Hyperparameter tuning & results
5. **[PPO Training](notebooks/ppo_experiments.ipynb)** - Policy gradient experiments
6. **[Welch Analysis](notebooks/welch_procedure.ipynb)** - Steady-state cost comparison

## 🎓 Course

**Supply Chain Management** - Master's Degree Program  
University Project - January 2026

## 📄 License

Academic project for educational purposes.

