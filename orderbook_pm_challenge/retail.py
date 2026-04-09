from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import RetailFlowConfig


@dataclass(frozen=True)
class RetailOrder:
    side: str
    notional: float


def _sample_poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0.0:
        return 0

    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


class RetailFlow:
    def __init__(self, config: RetailFlowConfig, seed: int) -> None:
        self._config = config
        self._rng = random.Random(seed)

    def generate_orders(self) -> list[RetailOrder]:
        count = _sample_poisson(self._rng, self._config.arrival_rate)
        if count <= 0:
            return []

        sigma = max(0.01, self._config.size_sigma)
        mean = max(0.01, self._config.mean_notional)
        mu_ln = math.log(mean) - 0.5 * sigma * sigma

        orders: list[RetailOrder] = []
        for _ in range(count):
            side = "BUY" if self._rng.random() < self._config.buy_probability else "SELL"
            notional = self._rng.lognormvariate(mu_ln, sigma)
            orders.append(RetailOrder(side=side, notional=notional))
        return orders
