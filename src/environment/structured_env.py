from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.product_control import (
    OrderHistory,
    ProductControlSpec,
    ProductPolicy,
    SSProductPolicy,
    product_costs,
)
from src.mdp import Action, CostParameters, State
from src.simulation.streams import StreamedInventorySimulation

_COST_PARAMS = CostParameters()


class ProductInventoryEnvironment(gym.Env):
    """
    Single-product Gymnasium environment with order-up-to actions.

    The full two-product simulation runs underneath; the agent controls only
    `product_id`, the other product is managed by a `partner` policy
    (default: baseline (s,S)). Products are independent (separate suppliers,
    separate K), so learning one controller per product is sufficient.

    Observation: `ProductControlSpec.observation` -> Box(3 + H,)
        [net_inv, outstanding, IP, own orders of the last H days] / obs_scale
    Action: Discrete(len(levels) + 1), 0 = no order, a -> order up to levels[a-1]
    Reward: -(K·1{q>0} + i·q + h·I⁺ + π·I⁻) of the controlled product × reward_scale
        (without i·q when `purchase_cost_in_reward=False`: in steady state every
        demanded unit is bought once, so i·q only adds a constant on average,
        while per step it penalises ordering long before the benefit arrives)

    `info` exposes the unscaled cost of the controlled product, of the partner
    and the total.
    """

    def __init__(
        self,
        product_id: int = 0,
        partner: Optional[ProductPolicy] = None,
        spec: Optional[ProductControlSpec] = None,
        reward_scale: float = 1.0 / 20.0,
        purchase_cost_in_reward: bool = True,
        episode_length: int = 1000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize environment.

        Args:
            product_id: Product controlled by the agent (0 or 1)
            partner: Policy for the other product (default: baseline (s,S))
            spec: Observation/action encoding (default: `ProductControlSpec.for_product`)
            reward_scale: Multiplier applied to the negative cost
            purchase_cost_in_reward: Include the unit cost i·q in the reward
                (`info` always reports the full cost)
            episode_length: Steps per episode (default: 1000)
            random_seed: Seed of the first episode (later resets continue
                from an internal seed sequence unless a seed is passed)
        """
        if product_id not in (0, 1):
            raise ValueError(f"product_id must be 0 or 1, got {product_id}")

        self.product_id = product_id
        self.partner_id = 1 - product_id
        self.control_spec = spec or ProductControlSpec.for_product(product_id)
        if self.control_spec.product_id != product_id:
            raise ValueError("spec.product_id does not match product_id")
        self.partner = partner or SSProductPolicy.baseline(self.partner_id)
        if self.partner.product_id != self.partner_id:
            raise ValueError("partner must control the other product")

        self.reward_scale = reward_scale
        self.purchase_cost_in_reward = purchase_cost_in_reward
        self.episode_length = episode_length

        # Episode seeds are drawn from self.np_random (seeded on first reset)
        self._initial_seed = random_seed

        self.simulation = StreamedInventorySimulation()
        self.history: OrderHistory = self.control_spec.new_history()
        self.state: Optional[State] = None
        self.current_step = 0

        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.control_spec.obs_dim,),
            dtype=np.float32,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(
            self.control_spec.n_actions
        )

        self.episode_costs: list = []
        self.episode_partner_costs: list = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset to a new episode.

        Args:
            seed: Episode seed (same seed -> same demand / lead-time streams)
            options: Unused

        Returns:
            (observation, info)
        """
        if seed is None and self._initial_seed is not None:
            seed, self._initial_seed = self._initial_seed, None
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(2**31 - 1))

        self.simulation.seed(episode_seed)
        self.simulation.reset()
        self.state = self.simulation.get_current_state()

        self.history.reset()
        self.partner.reset()
        self.current_step = 0
        self.episode_costs = []
        self.episode_partner_costs = []

        info = {
            "step": 0,
            "episode_seed": episode_seed,
            "net_inventory": self.state.net_inventory,
            "outstanding": self.state.outstanding_orders,
        }
        return self.control_spec.observation(self.state, self.history), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one day.

        Args:
            action: Action index (see class docstring)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        q_own = self.control_spec.quantity(int(action), self.state)
        q_partner = self.partner.act(self.state)

        quantities = [0, 0]
        quantities[self.product_id] = q_own
        quantities[self.partner_id] = q_partner

        next_state, sim_info = self.simulation.execute_daily_decision(
            Action(tuple(quantities))
        )
        self.history.push(q_own)
        self.state = next_state

        own = product_costs(next_state, self.product_id, q_own)
        partner = product_costs(next_state, self.partner_id, q_partner)
        cost = own.total_cost
        reward_cost = cost
        if not self.purchase_cost_in_reward:
            reward_cost -= _COST_PARAMS.i * q_own
        reward = -reward_cost * self.reward_scale

        self.current_step += 1
        self.episode_costs.append(cost)
        self.episode_partner_costs.append(partner.total_cost)

        terminated = False
        truncated = self.current_step >= self.episode_length

        info = {
            "step": self.current_step,
            "order_quantity": q_own,
            "partner_order_quantity": q_partner,
            "cost": cost,
            "ordering_cost": own.ordering_cost,
            "holding_cost": own.holding_cost,
            "shortage_cost": own.shortage_cost,
            "partner_cost": partner.total_cost,
            "total_cost": cost + partner.total_cost,
            "net_inventory": next_state.net_inventory,
            "outstanding": next_state.outstanding_orders,
            "num_customers": sim_info["num_customers"],
            "total_demand": sim_info["total_demand"],
        }

        if truncated:
            info["episode_cost"] = {
                "avg_cost": float(np.mean(self.episode_costs)),
                "avg_partner_cost": float(np.mean(self.episode_partner_costs)),
                "avg_total_cost": float(
                    np.mean(self.episode_costs) + np.mean(self.episode_partner_costs)
                ),
            }

        return (
            self.control_spec.observation(next_state, self.history),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def __str__(self):
        return (
            f"Product Inventory Environment (product={self.product_id}, "
            f"actions={self.control_spec.n_actions}, obs_dim={self.control_spec.obs_dim}, "
            f"partner={self.partner.name})"
        )
