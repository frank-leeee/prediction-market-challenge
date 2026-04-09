from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OwnOrderView:
    """Participant order state exposed on callback."""

    order_id: str
    side: Side
    price_ticks: int
    remaining_quantity: float
    submitted_step: int


@dataclass(frozen=True)
class StepState:
    """Single callback payload delivered to the strategy.

    `buy_filled_quantity` and `sell_filled_quantity` are the participant's
    aggregate filled quantities from the previous step.
    """

    step: int
    steps_remaining: int
    yes_inventory: float
    no_inventory: float
    cash: float
    reserved_cash: float
    free_cash: float
    competitor_best_bid_ticks: int | None
    competitor_best_ask_ticks: int | None
    buy_filled_quantity: float
    sell_filled_quantity: float
    own_orders: tuple[OwnOrderView, ...]


@dataclass(frozen=True)
class PlaceOrder:
    side: Side
    price_ticks: int
    quantity: float
    client_order_id: str | None = None


@dataclass(frozen=True)
class CancelOrder:
    order_id: str


@dataclass(frozen=True)
class CancelAll:
    pass


Action = Union[PlaceOrder, CancelOrder, CancelAll]
