# Experiment Log — Per-product DQN vs (s,S)

Log of the experiments run on 2026-10-27/28 to improve the DQN agent: what was changed,
how it was evaluated, every result (also the ones that were later revised) and how to reproduce them.

## Map

| What | Where |
|---|---|
| Original Welch analysis (joint DQN + PPO, January) | [`notebooks/welch_procedure.ipynb`](../../notebooks/welch_procedure.ipynb) |
| Welch analysis v1 — per-product DQN vs **original** (s,S) | [`notebooks/welch_procedure_product_dqn_v1.ipynb`](../../notebooks/welch_procedure_product_dqn_v1.ipynb) |
| Welch analysis v2 — vs **re-tuned** (s,S), phase C, ablation | [`notebooks/welch_procedure_product_dqn_v2.ipynb`](../../notebooks/welch_procedure_product_dqn_v2.ipynb) |
| Per-product environment | [`src/environment/structured_env.py`](../../src/environment/structured_env.py), [`src/environment/product_control.py`](../../src/environment/product_control.py) |
| Simulation with common random numbers | [`src/simulation/streams.py`](../../src/simulation/streams.py) |
| Joint evaluation (CI, paired differences, service levels) | [`src/agents/metrics/joint_evaluation.py`](../../src/agents/metrics/joint_evaluation.py) |
| Scripts | [`scripts/`](../../scripts/) (training, evaluation, (s,S) tuning, ablation) |
| Raw results (tables, grid search, training curves) | [`data/`](data/) |

Trained models are in `models/` (gitignored, local only).

## 1. What changed with respect to the original DQN

| | Original (`InventoryEnvironment`) | New (`ProductInventoryEnvironment`) |
|---|---|---|
| Agent | one DQN for both products | one DQN per product, the other product run by a "partner" policy |
| Action | joint quantities (q₀, q₁): `Discrete(961)` | order-up-to level: `Discrete(52)` — 0 = no order, a ≥ 1 → raise IP to S = a − 1 |
| Observation | 31 stacked raw frames of (I, O) × 2 products = 124 values | [I, O, IP, own orders of the last ⌈max lead time⌉ days] = 33 (P0) / 24 (P1) values |
| Scaling | none | observation / 50, reward × 1/100 (fixed, saved with the model) |
| Reward | total cost of both products | cost of the controlled product (optionally without i·q, phase C) |

An (s,S) policy is a special case of the new action space (order up to S when IP ≤ s), so the
agent can represent at least the baseline.

## 2. Evaluation procedure

The same procedure is used by `scripts/evaluate_joint.py`, the training callback and the notebooks.

1. **Common random numbers (CRN).** `StreamedInventorySimulation` draws the initial state, the
   customers (arrivals + demand sizes) and the lead times of each product from **separate random
   streams** derived from the episode seed. With the same seed every policy faces exactly the same
   customers and the k-th order of each product gets the same lead time, whatever the decisions.
   Consequences:
   - differences between policies are measured **per seed** (paired), with much tighter CIs;
   - the two products are fully independent: the cost of P0 does not depend on the policy used for P1,
     so mixed configurations (DQN on one product, (s,S) on the other) can be read off the per-product breakdown.
2. **Disjoint seed sets** (indices in `src/seeds.py`):
   - 0–299: **test** (final evaluations, notebooks);
   - 300–499: (s,S) grid search (coarse 300–319, refinement 400–499);
   - 900–909: **validation** during training (checkpoint selection).
3. **Replications.** Each replication = one 1000-day episode. Steady-state cost of a replication =
   mean daily cost after the warmup.
4. **Warmup (Welch).** Average the daily cost across replications, smooth it with a centred moving
   average (half-width 25), take the first day after which a 50-day window stays within 5% of the
   second-half mean. Detected warmup ≈ 50 days for every policy (50 is the minimum the rule can return);
   results do not change with a fixed warmup of 200 or 500 days ([`data/03_*`](data/)).
5. **Confidence intervals.** Student-t 95% CI over replications; **Δ vs reference** = CI of the
   per-seed difference (paired, thanks to CRN).
6. **Service levels.** Fill rate = units served immediately from stock / units demanded;
   no-stockout days = fraction of days ending without backorders.

## 3. Chronology

| # | Experiment | Result | Data |
|---|---|---|---|
| 0 | Original joint DQN (`notebooks/dqn_agent_best.ipynb`, 10M steps, no scaling) | ≈ $92.6/day, 77–81% of the cost is shortage | — |
| 1 | Per-product DQN, reward × 1/20, lr 1e-4, 1M steps | stuck on a plateau: DQN+DQN $62.69 ± 2.73 vs baseline (s,S) $44.60 (20 test seeds) | [`01`](data/01_joint_eval_dqn_rs20_20seeds.md) |
| 2 | Same, lr 3e-5 | same plateau → the learning rate is not the issue | [`08`](data/08_training_curves.json) |
| 3 | Same, reward × **1/100** | DQN+DQN $43.20 vs $44.60 (Δ −1.40 ± 1.21, 20 seeds); same with warmup 500; `final` checkpoints worse than `best` | [`02`](data/02_joint_eval_dqn_rs100_20seeds.md), [`03`](data/03_joint_eval_dqn_rs100_warmup500.md), [`04`](data/04_joint_eval_dqn_rs100_final_checkpoints.md) |
| 4 | Welch v1, 300 test seeds, reference = original (s,S) | DQN+DQN $41.62 vs $43.70, Δ **−2.09 ± 0.32** | notebook v1 |
| 5 | (s,S) **re-tuned** per product with CRN (1116 pairs × 20 seeds, top 15 × 100 seeds) | P0 (25, 30): $17.83 ± 0.35; P1 (23, 29): $20.92 ± 0.38 — optimum inside the grid (also checked S − s from 1 to 12) | [`05`](data/05_ss_grid_search.txt), [`06`](data/06_ss_grid_search_small_gap.txt) |
| 6 | Phase C: reward without i·q (×1/100), 2 training seeds per product | −0.78 $/day vs the DQN with i·q, higher fill rate, somewhat less oscillation | notebook v2 |
| 7 | Second training seed (1234) for the i·q variant | best checkpoint per product chosen on the same 10 validation seeds | notebook v2 |
| 8 | **Ablation**: original joint env + same hyperparameters + reward × 1/100 (and × 1/20) | $5622.85 ± 221.57/day (30 seeds): inventories of thousands of units; stopped at ≈ 500k / 250k steps | notebook v2 §9, [`08`](data/08_training_curves.json) |

## 4. Final results (notebook v2, 300 test seeds, warmup 51 days)

| Policy | $/day ± CI95 | Δ vs (s,S) re-tuned | P0 $/day | P1 $/day | Fill rate P0 / P1 |
|---|---|---|---|---|---|
| (s,S) baseline (20, 38) / (19, 37) | 43.70 ± 0.36 | +4.85 ± 0.25 | 20.10 | 23.61 | 83.3% / 83.2% |
| **(s,S) re-tuned (25, 30) / (23, 29)** | **38.86 ± 0.34** | — | 17.91 | 20.94 | 84.0% / 83.2% |
| DQN, i·q in reward | 41.47 ± 0.51 | +2.61 ± 0.29 | 19.71 | 21.75 | 73.9% / 76.2% |
| DQN, no i·q (phase C) | 40.69 ± 0.44 | +1.84 ± 0.24 | 19.23 | 21.47 | 76.1% / 78.0% |

## 5. Findings

- **Problem structure** is the enabling factor: with the same hyperparameters and reward scale the
  original joint formulation does not learn (quantity actions make random exploration over-order by
  ~15 units/day against a demand of ~1 unit/day; with unscaled observations the network never recovers).
- **Reward scale** decides between a plateau at ~2× the (s,S) cost and the (s,S) level: SB3's DQN uses
  the Huber loss, which caps the gradient for TD errors > 1; with Q ≈ cost / (1 − γ) the useful
  differences between actions (~1% of Q) are lost at scale 1/20.
- The earlier "DQN beats (s,S)" result (step 4) was due to a **badly tuned baseline**: against the
  re-tuned (s,S) the best DQN is +4.7% more expensive.
- Why the DQN still loses: it learns an almost order-up-to rule but with a **target too low**
  (IP ≈ 24 for P0, ≈ 25 for P1 vs a band 25–30 / 23–29 for the optimal (s,S)), so it stocks out more
  often (π = 7 per unit-day). The optimal band is only 5 units wide; the DQN does not hit it reliably
  (validation cost oscillates by ±2–3 $/day between checkpoints).

## 6. Reproduce

```bash
uv sync
# (s,S) tuning with CRN (≈ 10 min on 4 workers)
uv run python -m scripts.tune_ss_policy --workers 4
# per-product DQN, phase C (≈ 45 min per run; run the two products in parallel)
uv run python -m scripts.train_product_dqn --product 0 --timesteps 1000000 --reward-scale 0.01 --no-purchase-cost --eval-seeds 10 --out models/dqn_p0_rs100_noiq_1M
uv run python -m scripts.train_product_dqn --product 1 --timesteps 1000000 --reward-scale 0.01 --no-purchase-cost --eval-seeds 10 --out models/dqn_p1_rs100_noiq_1M
#   i·q variant: drop --no-purchase-cost; second training seed: --seed 1234
# ablation on the original joint environment
uv run python -m scripts.train_joint_dqn --reward-scale 0.01 --out models/dqn_joint_rs0.01_1M
# quick joint evaluation (20 seeds) of two checkpoints vs the baseline (s,S)
uv run python -m scripts.evaluate_joint --p0 models/dqn_p0_rs100_noiq_1M/best --p1 models/dqn_p1_rs100_noiq_1M/best
```

Then run `notebooks/welch_procedure_product_dqn_v2.ipynb` (≈ 15 min). Note: the default
`--reward-scale` of `train_product_dqn` is still 1/20.

## 7. Open points

- **Phase D**: residual policy on top of the re-tuned (s,S) (the agent picks a ±k correction of s and S),
  starting from $38.86/day instead of learning the threshold from scratch.
- **Model tracking**: `models/` is local only; to be logged to MLflow.
