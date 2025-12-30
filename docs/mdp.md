# Mathematical MDP Formulation - Inventory Management System

## 📐 Markov Decision Process Definition

A **Markov Decision Process (MDP)** is formally defined as a tuple:

**$MDP = (S, A, P, R, \gamma)$**

Where:

- **$S$**: State space
- **$A$**: Action space
- **$P$**: Transition probability function
- **$R$**: Reward function
- **$\gamma$**: Discount factor

Let's define each component for our inventory management problem.

---

## 🎯 1. State Space ($S$)

The inventory control problem is modeled as a Partially Observable Markov Decision Process (POMDP) due to the unobservable lead times. We utilize Frame Stacking to allow the agent to infer the latent state of in-transit orders.

### Mathematical Definition

At each decision epoch $t$ (beginning of day $t$), the environment emits an observation vector $o_t \in \mathbb{R}^6$:

**$o_t = \bigl(I_0^t, I_1^t, B_0^t, B_1^t, O_0^t, O_1^t\bigr)$**

Where for each product $i \in \{0, 1\}$:
- **$I_0^t, I_1^t \in \mathbb{Z}$**: On-hand inventory levels for products 0 and 1
- **$B_0^t, B_1^t \in \mathbb{Z}_+$**: Backlog (unsatisfied demand)
- **$O_0^t, O_1^t \in \mathbb{Z}_+$**: Outstanding (in-transit) orders

The **state** at time $t$ is defined as a sequence of the most recent $k+1$ observations:

**$s_t = \bigl[o_t, o_{t-1}, \dots, o_{t-k}\bigr]$**

with $k = 3$ in our implementation.

Thus, the state is a $(k+1) \times 6$-dimensional vector, flattened when used as input to the neural network.

### State Space Bounds

**$S = \{s \mid -100 \leq I_i \leq 200, 0 \leq B_i \leq 100, 0 \leq O_i \leq 150\}$** for $i \in \{0,1\}$.  
Note: Negative inventory means backorders ($I_i < 0 \rightarrow B_i = -I_i$)

### Inventory Position

For completeness, the inventory position for product $i$ at time $t$ is defined as:

**$IP_i(s) = I_i - B_i + O_i$**

This represents the effective inventory level accounting for backorders and incoming orders. While not explicitly included as a separate state variable, this quantity is implicitly available to the agent through the stacked observations.

### Continuous vs Discrete

- **Implementation**: Continuous (ℝ⁶) for neural networks
- **Conceptually**: Discrete (ℤ⁶) for units of inventory
- **Normalization**: For neural networks, we normalize:

  **$\tilde{s} = (s - \mu) / \sigma$**

  where μ = (50, 50, 5, 5, 20, 20) and σ = (40, 40, 15, 15, 30, 30)

---

## 🎬 2. Action Space (A)

### Mathematical Definition

The action at time _t_ is a 2-dimensional vector:

**$a_t = (q₀, q₁)$**

Where:

- **$q₀$** ∈ $Q₀$: Order quantity for product 0
- **$q₁$** ∈ $Q₁$: Order quantity for product 1

### Discretization

To make the problem tractable, we discretize the action space:

**$Q_i = \{0, \Delta, 2\Delta, ..., Q_{\max}\}$**

Where:

- **$\Delta$**: Order increment (e.g., 5, 10, or 20 units)
- **$Q_{\max}$**: Maximum order quantity (e.g., 100 units)

### Action Space Cardinality

**$|A| = |Q_0| \times |Q_1|$**

Examples:

- **Coarse** ($\Delta=20$, $Q_{\max}=100$): |A| = 6 × 6 = 36
- **Medium** ($\Delta=10$, $Q_{\max}=100$): |A| = 11 × 11 = 121
- **Fine** ($\Delta=5$, $Q_{\max}=100$): |A| = 21 × 21 = 441

### No-Order Action

The special case of no orders:

**$a_0 = (0, 0)$**

---

## 🔄 3. Transition Dynamics (P)

### Transition Function

The transition probability is:

**$P(s_{t+1} | s_t, a_t) = P(s' | s, a)$**

This represents the probability of reaching state $s'$ from state $s$ after taking action $a$.

### Components of Transition

The transition has **stochastic** and **deterministic** components:

#### 3.1 Deterministic: Order Placement

When action **$a = (q₀, q₁)$** is taken:

**$O'_i = O_i + q_i$** for i ∈ {0,1}

Orders are placed and will arrive after a stochastic lead time.

#### 3.2 Stochastic: Customer Demand

During period [t, t+1], customers arrive according to:

**Customer arrivals ~ Poisson($\lambda$ = 0.1)** (exponential inter-arrival)

Each customer demands:

**For Product 0:**

- $D_0 \in \{1, 2, 3, 4\}$
- $P(D_0 = k) = \{1/6, 1/3, 1/3, 1/6\}$

**For Product 1:**

- $D_1 \in \{5, 4, 3, 2\}$
- $P(D_1 = k) = \{1/8, 1/2, 1/4, 1/8\}$

**Total demand** in period [t, t+1]:

**$D_i^{total} = \sum_{j=1}^{N} D_i^{(j)}$**
where N ~ Poisson(0.1) is the number of arrivals.

#### 3.3 Demand Fulfillment

For each product i:

**If $I_i \geq D_i^{total}$:**

- $I'_i = I_i - D_i^{total}$
- $B'_i = B_i$ (unchanged)

**If $I_i < D_i^{total}$:**

- $I'_i = 0$
- $B'_i = B_i + (D_i^{total} - I_i)$

#### 3.4 Stochastic: Order Arrivals

Orders placed at time $\tau < t$ arrive at time $\tau + L$, where:

**For Product 0:**

- $L_0 \sim \text{Uniform}(0.5, 1.0)$

**For Product 1:**

- $L_1 \sim \text{Uniform}(0.2, 0.7)$

When order of quantity Q arrives:

**If $B_i > 0$:**

- Fill backorders first: $B'_i = \max(0, B_i - Q)$
- Remaining: $I'_i = I_i + \max(0, Q - B_i)$

**If $B_i = 0$:**

- $I'_i = I_i + Q$

And update outstanding:

- $O'_i = O_i - Q$

### Transition Equation Summary

**$s_{t+1} = T(s_t, a_t, \xi_t)$**
Where **$\xi_t$** represents all stochastic elements:

- $\xi_t = (N_t, \{D_0^{(j)}, D_1^{(j)}\}_{j=1}^{N_t}, \{L_0^{(k)}, L_1^{(k)}\}_{k \in \text{arrivals}})$

---

## 💰 4. Reward Function (R)

### Mathematical Definition

The reward at time t is:

**$R(s_t, a_t) = -C(s_t, a_t)$**

We use **negative cost** as reward (minimizing cost = maximizing reward).

### Cost Components

The total cost is:

**$C(s, a) = C_h(s) + C_b(s) + C_o(a) + C_p(a)$**

#### 4.1 Holding Cost

**$C_h(s) = h \cdot \sum_i \max(0, I_i)$**
Where:

- **$h = 1$**: Holding cost per unit per day

This penalizes keeping excess inventory.

#### 4.2 Backorder Cost

**$C_b(s) = \pi \cdot \sum_i B_i$**

Where:

- **$\pi = 7$**: Backorder penalty per unit per day

This heavily penalizes stockouts and unfulfilled demand.

#### 4.3 Ordering Cost

**$C_o(a) = K \cdot \sum_i \mathbb{1}\{q_i > 0\}$**

Where:

- **$K = 10$**: Fixed cost per order
- **$\mathbb{1}\{\cdot\}$**: Indicator function (1 if true, 0 if false)
  This is a fixed cost incurred when placing an order (regardless of quantity).

#### 4.4 Purchase Cost

**$C_p(a) = i \cdot \sum_i q_i$**

Where:

- **$i = 3$**: Unit purchase cost

This is the variable cost of ordering units.

### Total Reward

**$R(s, a) = -(h \cdot \sum_i \max(0, I_i) + \pi \cdot \sum_i B_i + K \cdot \sum_i \mathbb{1}\{q_i > 0\} + i \cdot \sum_i q_i)$**

### Example Calculation

Given:

- State: s = (I₀=40, I₁=45, B₀=0, B₁=0, O₀=0, O₁=0)
- Action: a = (20, 15)

**Costs:**

- C_h = 1 × (40 + 45) = 85
- C_b = 7 × (0 + 0) = 0
- C_o = 10 × (1 + 1) = 20 (both products ordered)
- C_p = 3 × (20 + 15) = 105

**Total cost:** C = 85 + 0 + 20 + 105 = 210

**Reward:** R = -210

---

## 🎲 5. Discount Factor ($\gamma$)

**$\gamma \in [0, 1]$**

Typical values:

- $\gamma = 0.95$ (balances short-term and long-term)
- $\gamma = 0.99$ (emphasizes long-term cumulative reward)

The discount factor determines how much we value future rewards:

## **$V(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) | s_0 = s]$**

## 🎯 Optimization Objective

### Value Function

The **state value function** under policy $\pi$ is:

**$V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) | s_0 = s]$**

### Action-Value Function (Q-Function)

**$Q^\pi(s, a) = \mathbb{E}_\pi[R(s, a) + \gamma \cdot \sum_{s'} P(s'|s,a) V^\pi(s')]$**

### Optimal Policy

Find policy $\pi^*$ that maximizes expected cumulative reward:

**$\pi^* = \arg\max_\pi \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)]$**

### Bellman Optimality Equation

**$V^*(s) = \max_a [R(s, a) + \gamma \cdot \sum_{s'} P(s'|s,a) V^*(s')]$**
**$Q^*(s, a) = R(s, a) + \gamma \cdot \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')$**

---

## 🔢 Problem Characteristics

### 1. State Space

- **Type**: Continuous (but discrete in units)
- **Dimensionality**: 6
- **Size**: Theoretically infinite, practically bounded

### 2. Action Space

- **Type**: Discrete
- **Dimensionality**: 2
- **Size**: 36 to 441 (depending on discretization)

### 3. Transition Dynamics

- **Type**: Stochastic
- **Markov Property**: ✅ Yes (next state depends only on current state and action)
- **Model-free or Model-based**: We have access to simulator (model-based possible)

### 4. Reward

- **Type**: Dense (received every step)
- **Structure**: Negative cost
- **Bounded**: No (costs can grow arbitrarily)

### 5. Horizon

- **Type**: Infinite (continuing task)
- **Episodes**: We simulate finite episodes for training

---

## 📊 MDP Properties

### Markov Property

**$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)$**

✅ **Holds**: The next state depends only on the current state and action, not on history.

### Stationary

**$P(s' | s, a)$** and **$R(s, a)$** do not change over time.

✅ **Holds**: The system dynamics are time-invariant.

### Episodic vs Continuing

- **Implementation**: Episodic (finite-length episodes for training)
- **Reality**: Continuing task (inventory management never "ends")

---

## 🧮 Special Cases and Simplifications

### 1. Deterministic Demand

If demand were deterministic ($D_i = \bar{d}_i$):

**$s_{t+1} = T_{\text{det}}(s_t, a_t)$**

This would make the problem much easier (Dynamic Programming).

### 2. No Lead Time

If $L_i = 0$ (orders arrive instantly):

**$O'_i = 0$** always (no outstanding orders)

State reduces to 4 dimensions.

### 3. Independent Products

If products were truly independent (no joint ordering decision):

MDP would decompose into two independent 3-dimensional MDPs.

### 4. Continuous Actions

If we didn't discretize actions:

Action space would be **$A = \mathbb{R}_+^2$** (continuous)

Would require different RL algorithms (Actor-Critic, DDPG, etc.)

---

## 📈 Curse of Dimensionality

### State Space Size

If we discretize each dimension into 10 bins:

**$|S| \approx 10^6 = 1,000,000$ states**

With 121 actions:

**$|S \times A| \approx 121,000,000$ state-action pairs**

### Implications

1. **Tabular Q-Learning**: Requires storing Q(s,a) for all pairs
2. **Function Approximation**: Neural networks can generalize across states
3. **State Abstraction**: Coarser discretization reduces complexity
