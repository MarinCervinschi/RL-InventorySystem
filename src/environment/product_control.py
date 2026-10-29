from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.mdp import (
    Action,
    ActionSpace,
    CostComponents,
    RewardFunction,
    State,
    StateHistory,
    update_history,
)
from src.simulation.product import Product, create_default_products

# (s,S) parameters from the baseline grid search (notebooks/baseline.ipynb)
BASELINE_SS_PARAMS: Tuple[Tuple[int, int], ...] = ((20, 38), (19, 37))

# (s,S) re-tuned per product with common random numbers on the streamed
# simulator (scripts/tune_ss_policy.py, results/ss_grid_search_*.json)
TUNED_SS_PARAMS: Tuple[Tuple[int, int], ...] = ((25, 30), (23, 29))


def default_history_length(product: Product) -> int:
    """Days after which an order has surely arrived: ceil(max lead time)."""
    return int(math.ceil(product.lead_time_max))


class OrderHistory:
    """
    Quantities ordered for one product in the last `length` days.

    Index 0 is the order placed yesterday, index `length-1` the oldest one.
    This is information the manager legitimately has (own past orders);
    lead times stay unobservable.
    """

    def __init__(self, length: int):
        self.length = length
        self._orders: Deque[int] = deque([0] * length, maxlen=length)

    def reset(self) -> None:
        """Clear history (no order placed before the episode start)."""
        self._orders = deque([0] * self.length, maxlen=self.length)

    def push(self, quantity: int) -> None:
        """Record today's order (0 if no order)."""
        self._orders.appendleft(int(quantity))

    def to_array(self) -> NDArray[np.float32]:
        """History as array, most recent first."""
        return np.array(self._orders, dtype=np.float32)


@dataclass(frozen=True)
class ProductControlSpec:
    """
    Observation and action encoding for a single-product controller.

    Shared by `ProductInventoryEnvironment` (training) and `ModelProductPolicy`
    (evaluation), so features are computed by exactly one piece of code.

    Observation (divided by `obs_scale`):
        [net_inventory, outstanding, inventory_position, q_{t-1}, ..., q_{t-H}]

    Action (order-up-to): Discrete(len(levels) + 1)
        0     -> do not order
        a >= 1 -> raise IP to S = levels[a-1], i.e. q = max(0, S - IP)

    An (s,S) policy is the special case "a = index(S)+1 if IP <= s else 0".

    Residual mode (`residual_base=(s, S)`, phase D): Discrete((2·max_delta+1)²)
        a -> (δs, δS) ∈ {-max_delta..max_delta}², then the (s+δs, S+δS) rule:
        q = S+δS - IP if IP <= s+δs else 0.
    The base (s,S) is the action (0, 0), and every action stays close to it.
    """

    product_id: int
    history_length: int
    levels: Tuple[int, ...] = tuple(range(51))
    obs_scale: float = 50.0
    residual_base: Optional[Tuple[int, int]] = None
    max_delta: int = 3

    @classmethod
    def for_product(
        cls,
        product_id: int,
        levels: Optional[Tuple[int, ...]] = None,
        obs_scale: float = 50.0,
    ) -> "ProductControlSpec":
        """Default spec: history window = ceil(max lead time) of the product."""
        product = create_default_products()[product_id]
        return cls(
            product_id=product_id,
            history_length=default_history_length(product),
            levels=tuple(levels) if levels is not None else tuple(range(51)),
            obs_scale=obs_scale,
        )

    @classmethod
    def residual_for_product(
        cls,
        product_id: int,
        base: Optional[Tuple[int, int]] = None,
        max_delta: int = 3,
        obs_scale: float = 50.0,
    ) -> "ProductControlSpec":
        """Residual spec around `base` (default: `TUNED_SS_PARAMS`)."""
        product = create_default_products()[product_id]
        return cls(
            product_id=product_id,
            history_length=default_history_length(product),
            obs_scale=obs_scale,
            residual_base=tuple(base or TUNED_SS_PARAMS[product_id]),
            max_delta=max_delta,
        )

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        if self.residual_base is not None:
            return (2 * self.max_delta + 1) ** 2
        return len(self.levels) + 1

    def residual(self, action: int) -> Tuple[int, int]:
        """(δs, δS) encoded by a residual action."""
        width = 2 * self.max_delta + 1
        return action // width - self.max_delta, action % width - self.max_delta

    @property
    def zero_residual_action(self) -> int:
        """Action index of (δs, δS) = (0, 0), i.e. the base (s,S)."""
        return self.max_delta * (2 * self.max_delta + 1) + self.max_delta

    @property
    def obs_dim(self) -> int:
        """Observation size."""
        return 3 + self.history_length

    def new_history(self) -> OrderHistory:
        """Empty order history of the right length."""
        return OrderHistory(self.history_length)

    def observation(self, state: State, history: OrderHistory) -> NDArray[np.float32]:
        """Build the (scaled) observation for this product."""
        net_inv = state.net_inventory[self.product_id]
        outstanding = state.outstanding_orders[self.product_id]
        ip = state.get_inventory_position(self.product_id)
        raw = np.concatenate(
            [np.array([net_inv, outstanding, ip], dtype=np.float32), history.to_array()]
        )
        return raw / np.float32(self.obs_scale)

    def quantity(self, action: int, state: State) -> int:
        """Decode an action into an order quantity given the current state."""
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")
        if self.residual_base is not None:
            ds, dS = self.residual(action)
            s, S = self.residual_base[0] + ds, self.residual_base[1] + dS
            ip = state.get_inventory_position(self.product_id)
            return max(0, S - ip) if ip <= s else 0
        if action == 0:
            return 0
        target = self.levels[action - 1]
        return max(0, target - state.get_inventory_position(self.product_id))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (saved with the model)."""
        d = asdict(self)
        d["levels"] = list(self.levels)
        if self.residual_base is not None:
            d["residual_base"] = list(self.residual_base)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductControlSpec":
        """Inverse of `to_dict`."""
        return cls(
            product_id=int(d["product_id"]),
            history_length=int(d["history_length"]),
            levels=tuple(int(x) for x in d["levels"]),
            obs_scale=float(d["obs_scale"]),
            residual_base=(
                tuple(int(x) for x in d["residual_base"])  # type: ignore[arg-type]
                if d.get("residual_base") is not None
                else None
            ),
            max_delta=int(d.get("max_delta", 3)),
        )


_SINGLE_PRODUCT_REWARD = RewardFunction()


def product_costs(state: State, product_id: int, quantity: int) -> CostComponents:
    """
    Daily cost of a single product (same formula as `RewardFunction`).

    Args:
        state: State at the end of the day
        product_id: Product to account for
        quantity: Quantity ordered for that product at the start of the day
    """
    single_state = State(
        net_inventory=(state.net_inventory[product_id],),
        outstanding_orders=(state.outstanding_orders[product_id],),
    )
    return _SINGLE_PRODUCT_REWARD.calculate_costs(single_state, Action((quantity,)))


class ProductPolicy(ABC):
    """
    Ordering policy for one product of the joint system.

    `act` is called exactly once per day, before the day is simulated.
    """

    product_id: int

    def reset(self) -> None:
        """Reset internal memory at the start of an episode."""

    @abstractmethod
    def act(self, state: State) -> int:
        """Return the order quantity for today."""

    @property
    def name(self) -> str:
        """Short label used in reports."""
        return self.__class__.__name__


@dataclass
class SSProductPolicy(ProductPolicy):
    """(s,S) rule for one product: if IP <= s order S - IP."""

    product_id: int
    s: int
    S: int

    def act(self, state: State) -> int:
        ip = state.get_inventory_position(self.product_id)
        return max(0, self.S - ip) if ip <= self.s else 0

    @property
    def name(self) -> str:
        return f"(s,S)=({self.s},{self.S})"

    @classmethod
    def baseline(cls, product_id: int) -> "SSProductPolicy":
        """(s,S) from the baseline grid search."""
        s, S = BASELINE_SS_PARAMS[product_id]
        return cls(product_id=product_id, s=s, S=S)

    @classmethod
    def tuned(cls, product_id: int) -> "SSProductPolicy":
        """(s,S) re-tuned with common random numbers (`TUNED_SS_PARAMS`)."""
        s, S = TUNED_SS_PARAMS[product_id]
        return cls(product_id=product_id, s=s, S=S)


@dataclass
class ModelProductPolicy(ProductPolicy):
    """
    Wraps a trained SB3 model (e.g. DQN) trained on `ProductInventoryEnvironment`.

    Keeps its own order history so it can run inside the joint system
    next to any other policy.
    """

    model: Any
    spec: ProductControlSpec
    label: str = "DQN"
    deterministic: bool = True
    history: OrderHistory = field(init=False)

    def __post_init__(self):
        self.history = self.spec.new_history()
        self.last_action: Optional[int] = None  # for analysis (e.g. residuals)

    @property
    def product_id(self) -> int:  # type: ignore[override]
        return self.spec.product_id

    def reset(self) -> None:
        self.history.reset()

    def act(self, state: State) -> int:
        obs = self.spec.observation(state, self.history)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        self.last_action = int(action)
        quantity = self.spec.quantity(self.last_action, state)
        self.history.push(quantity)
        return quantity

    @property
    def name(self) -> str:
        return self.label

    @classmethod
    def load(cls, path: Path, label: str = "DQN") -> "ModelProductPolicy":
        """
        Load a DQN saved by `scripts/train_product_dqn.py`.

        Args:
            path: Directory containing `dqn_model.zip` and `dqn_model_metadata.json`
        """
        from stable_baselines3 import DQN

        path = Path(path)
        with open(path / "dqn_model_metadata.json") as f:
            metadata = json.load(f)
        spec = ProductControlSpec.from_dict(metadata["spec"])
        model = DQN.load(path / "dqn_model", device="cpu")
        return cls(model=model, spec=spec, label=label)


class FrameStackJointPolicy:
    """
    Adapter for agents trained on the original `InventoryEnvironment`.

    Rebuilds the stacked observation exactly as the original environment
    (initial state repeated k+1 times, then one frame per day) and decodes the
    joint action. The two per-product views returned by `as_product_policies`
    let the agent run in the joint evaluation with common random numbers.
    Product 0 is queried first each day (as in `run_joint_episode`), which
    triggers the joint decision; product 1 returns the cached quantity.
    """

    def __init__(self, model: Any, k: int, Q_max: int, deterministic: bool = True):
        self.model = model
        self.k = k
        self.action_space = ActionSpace(Q_max=Q_max)
        self.deterministic = deterministic
        self._history: Optional[StateHistory] = None
        self._quantities: Tuple[int, ...] = (0, 0)

    def reset(self) -> None:
        self._history = None

    def decide(self, state: State) -> Tuple[int, ...]:
        if self._history is None:
            self._history = StateHistory(states=tuple(state for _ in range(self.k + 1)))
        else:
            self._history = update_history(self._history, state)
        action, _ = self.model.predict(
            self._history.to_array(), deterministic=self.deterministic
        )
        self._quantities = self.action_space.get_action(int(action)).order_quantities
        return self._quantities

    def as_product_policies(
        self, label: str = "Joint DQN"
    ) -> Tuple["ProductPolicy", "ProductPolicy"]:
        return (_JointView(self, 0, label), _JointView(self, 1, label))


@dataclass
class _JointView(ProductPolicy):
    """Per-product view of a `FrameStackJointPolicy`."""

    joint: FrameStackJointPolicy
    product_id: int
    label: str

    def reset(self) -> None:
        if self.product_id == 0:
            self.joint.reset()

    def act(self, state: State) -> int:
        if self.product_id == 0:
            self.joint.decide(state)
        return int(self.joint._quantities[self.product_id])

    @property
    def name(self) -> str:
        return self.label
