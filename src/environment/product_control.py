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

from src.mdp import Action, CostComponents, RewardFunction, State
from src.simulation.product import Product, create_default_products

# (s,S) parameters from the baseline grid search (notebooks/baseline.ipynb)
BASELINE_SS_PARAMS: Tuple[Tuple[int, int], ...] = ((20, 38), (19, 37))


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
    """

    product_id: int
    history_length: int
    levels: Tuple[int, ...] = tuple(range(51))
    obs_scale: float = 50.0

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

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (no-order + one per level)."""
        return len(self.levels) + 1

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
        if action == 0:
            return 0
        target = self.levels[action - 1]
        return max(0, target - state.get_inventory_position(self.product_id))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (saved with the model)."""
        d = asdict(self)
        d["levels"] = list(self.levels)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductControlSpec":
        """Inverse of `to_dict`."""
        return cls(
            product_id=int(d["product_id"]),
            history_length=int(d["history_length"]),
            levels=tuple(int(x) for x in d["levels"]),
            obs_scale=float(d["obs_scale"]),
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

    @property
    def product_id(self) -> int:  # type: ignore[override]
        return self.spec.product_id

    def reset(self) -> None:
        self.history.reset()

    def act(self, state: State) -> int:
        obs = self.spec.observation(state, self.history)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        quantity = self.spec.quantity(int(action), state)
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
