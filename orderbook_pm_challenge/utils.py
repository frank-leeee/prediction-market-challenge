from __future__ import annotations

import math


EPSILON = 1e-9


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def tick_to_price(price_ticks: int) -> float:
    return price_ticks / 100.0


def quantize_down(value: float, quantum: float) -> float:
    if value <= 0.0:
        return 0.0
    scaled = math.floor((value + EPSILON) / quantum)
    return round(scaled * quantum, 10)


def is_integer_tick(price_ticks: object) -> bool:
    return isinstance(price_ticks, int) and not isinstance(price_ticks, bool)


def largest_visible_tick_below(probability: float, *, min_tick: int, max_tick: int) -> int | None:
    for tick in range(max_tick, min_tick - 1, -1):
        if tick_to_price(tick) < probability - EPSILON:
            return tick
    return None


def smallest_visible_tick_above(probability: float, *, min_tick: int, max_tick: int) -> int | None:
    for tick in range(min_tick, max_tick + 1):
        if tick_to_price(tick) > probability + EPSILON:
            return tick
    return None


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
