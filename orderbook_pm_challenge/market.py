from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable, Sequence

from .config import ChallengeConfig
from .types import CancelAll, CancelOrder, OwnOrderView, PlaceOrder, Side, StepState
from .utils import (
    EPSILON,
    is_integer_tick,
    largest_visible_tick_below,
    quantize_down,
    smallest_visible_tick_above,
    tick_to_price,
)


PARTICIPANT = "participant"
COMPETITOR = "competitor"
ARB = "arb"
RETAIL = "retail"


@dataclass
class RestingOrder:
    order_id: str
    owner: str
    side: Side
    price_ticks: int
    remaining_quantity: float
    submitted_step: int
    sequence: int
    reserved_cash: float = 0.0
    reserved_yes: float = 0.0

    @property
    def price(self) -> float:
        return tick_to_price(self.price_ticks)


@dataclass(frozen=True)
class RecordedFill:
    order_id: str
    side: Side
    price_ticks: int
    quantity: float
    aggressor: str
    step: int


@dataclass(frozen=True)
class ParticipantStats:
    total_edge: float = 0.0
    retail_edge: float = 0.0
    arb_edge: float = 0.0
    traded_quantity: float = 0.0
    traded_notional: float = 0.0
    fill_count: int = 0


class OrderBookError(Exception):
    pass


class PredictionMarket:
    def __init__(self, config: ChallengeConfig) -> None:
        self._config = config
        self._cash = config.starting_cash
        self._yes_inventory = 0.0
        self._no_inventory = 0.0
        self._stats = ParticipantStats()
        self._orders: dict[str, RestingOrder] = {}
        self._participant_order_ids: set[str] = set()
        self._competitor_order_ids: set[str] = set()
        self._order_counter = itertools.count(1)
        self._sequence_counter = itertools.count(1)
        self._pending_competitor_replenishments: list[tuple[Side, int]] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def yes_inventory(self) -> float:
        return self._yes_inventory

    @property
    def no_inventory(self) -> float:
        return self._no_inventory

    @property
    def stats(self) -> ParticipantStats:
        return self._stats

    def net_inventory(self) -> float:
        return round(self._yes_inventory - self._no_inventory, 10)

    def reserved_cash(self) -> float:
        return round(
            sum(self._orders[order_id].reserved_cash for order_id in self._participant_order_ids),
            10,
        )

    def free_cash(self) -> float:
        return round(self._cash - self.reserved_cash(), 10)

    def _reserved_yes(self) -> float:
        return round(
            sum(self._orders[order_id].reserved_yes for order_id in self._participant_order_ids),
            10,
        )

    def _available_yes(self) -> float:
        return round(max(0.0, self._yes_inventory - self._reserved_yes()), 10)

    def initialize_competitor(self, starting_probability: float) -> None:
        best_bid, best_ask = self._initial_competitor_best_quotes(starting_probability)
        for tick in range(self._config.min_price_tick, (best_bid or 0) + 1):
            self._create_competitor_order(Side.BUY, tick, submitted_step=-1)
        if best_ask is not None:
            for tick in range(best_ask, self._config.max_price_tick + 1):
                self._create_competitor_order(Side.SELL, tick, submitted_step=-1)

    def _initial_competitor_best_quotes(self, probability: float) -> tuple[int | None, int | None]:
        lower = largest_visible_tick_below(
            probability,
            min_tick=self._config.min_price_tick,
            max_tick=self._config.max_price_tick,
        )
        upper = smallest_visible_tick_above(
            probability,
            min_tick=self._config.min_price_tick,
            max_tick=self._config.max_price_tick,
        )
        spread = self._config.competitor.spread_ticks
        best_bid = None if lower is None else lower - (spread - 1)
        best_ask = None if upper is None else upper + (spread - 1)

        if best_bid is not None and best_bid < self._config.min_price_tick:
            best_bid = None
        if best_ask is not None and best_ask > self._config.max_price_tick:
            best_ask = None
        return best_bid, best_ask

    def refresh_competitor(self, step: int) -> None:
        replenishments = self._pending_competitor_replenishments
        self._pending_competitor_replenishments = []
        for side, tick in replenishments:
            if self._config.min_price_tick <= tick <= self._config.max_price_tick:
                self._create_competitor_order(side, tick, submitted_step=step)

    def build_step_state(
        self,
        *,
        step: int,
        steps_remaining: int,
        buy_filled_quantity: float,
        sell_filled_quantity: float,
    ) -> StepState:
        best_bid, best_ask = self.competitor_best_quotes()
        own_orders = tuple(
            OwnOrderView(
                order_id=order.order_id,
                side=order.side,
                price_ticks=order.price_ticks,
                remaining_quantity=order.remaining_quantity,
                submitted_step=order.submitted_step,
            )
            for order in sorted(
                self._participant_orders(),
                key=lambda order: (
                    order.side.value,
                    order.price_ticks if order.side is Side.SELL else -order.price_ticks,
                    order.sequence,
                ),
            )
        )
        return StepState(
            step=step,
            steps_remaining=steps_remaining,
            yes_inventory=self._yes_inventory,
            no_inventory=self._no_inventory,
            cash=self._cash,
            reserved_cash=self.reserved_cash(),
            free_cash=self.free_cash(),
            competitor_best_bid_ticks=best_bid,
            competitor_best_ask_ticks=best_ask,
            buy_filled_quantity=buy_filled_quantity,
            sell_filled_quantity=sell_filled_quantity,
            own_orders=own_orders,
        )

    def apply_actions(self, actions: Sequence[object], *, step: int) -> None:
        for action in actions:
            if isinstance(action, CancelAll):
                self.cancel_all_orders()
            elif isinstance(action, CancelOrder):
                self.cancel_order(action.order_id)
            elif isinstance(action, PlaceOrder):
                self.place_order(action, step=step)
            else:
                raise OrderBookError(f"Unsupported action: {action!r}")

    def place_order(self, action: PlaceOrder, *, step: int) -> None:
        if action.side not in (Side.BUY, Side.SELL):
            raise OrderBookError(f"Invalid side: {action.side!r}")
        if not is_integer_tick(action.price_ticks):
            raise OrderBookError("price_ticks must be an integer")
        if not (self._config.min_price_tick <= action.price_ticks <= self._config.max_price_tick):
            raise OrderBookError(f"price_ticks out of range: {action.price_ticks}")

        quantity = quantize_down(float(action.quantity), self._config.share_quantum)
        if quantity <= 0.0:
            raise OrderBookError("quantity must be positive after rounding")

        order_id = action.client_order_id or f"order-{next(self._order_counter)}"
        if order_id in self._orders:
            raise OrderBookError(f"Order id already active: {order_id}")

        price = tick_to_price(action.price_ticks)
        reserved_cash = 0.0
        reserved_yes = 0.0
        if action.side is Side.BUY:
            reserved_cash = round(price * quantity, 10)
        else:
            available_yes = self._available_yes()
            reserved_yes = min(quantity, available_yes)
            uncovered = quantity - reserved_yes
            reserved_cash = round((1.0 - price) * uncovered, 10)

        if self.free_cash() + EPSILON < reserved_cash:
            raise OrderBookError("Insufficient free cash for order collateral")

        order = RestingOrder(
            order_id=order_id,
            owner=PARTICIPANT,
            side=action.side,
            price_ticks=action.price_ticks,
            remaining_quantity=quantity,
            submitted_step=step,
            sequence=next(self._sequence_counter),
            reserved_cash=reserved_cash,
            reserved_yes=reserved_yes,
        )
        self._orders[order_id] = order
        self._participant_order_ids.add(order_id)

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is None or order.owner != PARTICIPANT:
            raise OrderBookError(f"Unknown participant order: {order_id}")
        self._remove_order(order)

    def cancel_all_orders(self) -> None:
        for order_id in list(self._participant_order_ids):
            self._remove_order(self._orders[order_id])

    def competitor_best_quotes(self) -> tuple[int | None, int | None]:
        bids = [order.price_ticks for order in self._competitor_orders() if order.side is Side.BUY]
        asks = [order.price_ticks for order in self._competitor_orders() if order.side is Side.SELL]
        return (max(bids) if bids else None, min(asks) if asks else None)

    def execute_arbitrage(self, *, probability: float, step: int) -> list[RecordedFill]:
        fills: list[RecordedFill] = []
        while True:
            best_ask = self._best_order(Side.SELL)
            if best_ask is None or best_ask.price >= probability - EPSILON:
                break
            fills.extend(self._execute_buy_quantity(best_ask.remaining_quantity, aggressor=ARB, step=step))

        while True:
            best_bid = self._best_order(Side.BUY)
            if best_bid is None or best_bid.price <= probability + EPSILON:
                break
            fills.extend(self._execute_sell_quantity(best_bid.remaining_quantity, aggressor=ARB, step=step))

        return fills

    def execute_retail_buy(self, *, notional: float, step: int) -> list[RecordedFill]:
        return self._execute_buy_notional(notional, aggressor=RETAIL, step=step)

    def execute_retail_sell(self, *, quantity: float, step: int) -> list[RecordedFill]:
        return self._execute_sell_quantity(quantity, aggressor=RETAIL, step=step)

    def settle(self, *, outcome: float) -> float:
        return round(self._cash + self._yes_inventory * outcome + self._no_inventory * (1.0 - outcome), 10)

    def _participant_orders(self) -> Iterable[RestingOrder]:
        for order_id in self._participant_order_ids:
            yield self._orders[order_id]

    def _competitor_orders(self) -> Iterable[RestingOrder]:
        for order_id in self._competitor_order_ids:
            yield self._orders[order_id]

    def _best_order(self, side: Side) -> RestingOrder | None:
        candidates = [order for order in self._orders.values() if order.side is side and order.remaining_quantity > EPSILON]
        if not candidates:
            return None
        if side is Side.SELL:
            return min(candidates, key=lambda order: (order.price_ticks, order.sequence))
        return max(candidates, key=lambda order: (order.price_ticks, -order.sequence))

    def _execute_buy_notional(self, notional: float, *, aggressor: str, step: int) -> list[RecordedFill]:
        remaining_cash = max(0.0, notional)
        fills: list[RecordedFill] = []
        while remaining_cash > EPSILON:
            best_ask = self._best_order(Side.SELL)
            if best_ask is None:
                break
            max_quantity = quantize_down(remaining_cash / best_ask.price, self._config.share_quantum)
            if max_quantity <= 0.0:
                break
            fill_quantity = min(best_ask.remaining_quantity, max_quantity)
            fills.extend(self._fill_order(best_ask, fill_quantity, aggressor=aggressor, step=step))
            remaining_cash = round(remaining_cash - best_ask.price * fill_quantity, 10)
        return fills

    def _execute_buy_quantity(self, quantity: float, *, aggressor: str, step: int) -> list[RecordedFill]:
        remaining_quantity = quantize_down(quantity, self._config.share_quantum)
        fills: list[RecordedFill] = []
        while remaining_quantity > EPSILON:
            best_ask = self._best_order(Side.SELL)
            if best_ask is None:
                break
            fill_quantity = min(best_ask.remaining_quantity, remaining_quantity)
            fills.extend(self._fill_order(best_ask, fill_quantity, aggressor=aggressor, step=step))
            remaining_quantity = round(remaining_quantity - fill_quantity, 10)
        return fills

    def _execute_sell_quantity(self, quantity: float, *, aggressor: str, step: int) -> list[RecordedFill]:
        remaining_quantity = quantize_down(quantity, self._config.share_quantum)
        fills: list[RecordedFill] = []
        while remaining_quantity > EPSILON:
            best_bid = self._best_order(Side.BUY)
            if best_bid is None:
                break
            fill_quantity = min(best_bid.remaining_quantity, remaining_quantity)
            fills.extend(self._fill_order(best_bid, fill_quantity, aggressor=aggressor, step=step))
            remaining_quantity = round(remaining_quantity - fill_quantity, 10)
        return fills

    def _fill_order(
        self,
        order: RestingOrder,
        quantity: float,
        *,
        aggressor: str,
        step: int,
    ) -> list[RecordedFill]:
        quantity = quantize_down(quantity, self._config.share_quantum)
        if quantity <= 0.0:
            return []

        fills: list[RecordedFill] = []
        price = order.price

        if order.owner == PARTICIPANT:
            if order.side is Side.BUY:
                cash_spent = round(price * quantity, 10)
                self._cash = round(self._cash - cash_spent, 10)
                self._yes_inventory = round(self._yes_inventory + quantity, 10)
                order.reserved_cash = round(order.reserved_cash - cash_spent, 10)
            else:
                covered = min(order.reserved_yes, quantity)
                uncovered = round(quantity - covered, 10)
                if covered > 0.0:
                    self._yes_inventory = round(self._yes_inventory - covered, 10)
                    self._cash = round(self._cash + price * covered, 10)
                    order.reserved_yes = round(order.reserved_yes - covered, 10)
                if uncovered > 0.0:
                    collateral = round((1.0 - price) * uncovered, 10)
                    self._cash = round(self._cash - collateral, 10)
                    self._no_inventory = round(self._no_inventory + uncovered, 10)
                    order.reserved_cash = round(order.reserved_cash - collateral, 10)

            fills.append(
                RecordedFill(
                    order_id=order.order_id,
                    side=order.side,
                    price_ticks=order.price_ticks,
                    quantity=quantity,
                    aggressor=aggressor,
                    step=step,
                )
            )

        order.remaining_quantity = round(order.remaining_quantity - quantity, 10)
        if order.remaining_quantity <= EPSILON:
            if order.owner == COMPETITOR:
                self._schedule_competitor_replenishment(order)
            self._remove_order(order)
        return fills

    def _schedule_competitor_replenishment(self, order: RestingOrder) -> None:
        spread = self._config.competitor.spread_ticks
        if order.side is Side.SELL:
            tick = order.price_ticks - spread
            side = Side.BUY
        else:
            tick = order.price_ticks + spread
            side = Side.SELL
        self._pending_competitor_replenishments.append((side, tick))

    def _remove_order(self, order: RestingOrder) -> None:
        self._orders.pop(order.order_id, None)
        self._participant_order_ids.discard(order.order_id)
        self._competitor_order_ids.discard(order.order_id)

    def _create_competitor_order(self, side: Side, tick: int, *, submitted_step: int) -> None:
        price = tick_to_price(tick)
        quantity = quantize_down(
            self._config.competitor.quote_notional / price,
            self._config.share_quantum,
        )
        if quantity <= 0.0:
            return
        order_id = f"competitor-{next(self._order_counter)}"
        order = RestingOrder(
            order_id=order_id,
            owner=COMPETITOR,
            side=side,
            price_ticks=tick,
            remaining_quantity=quantity,
            submitted_step=submitted_step,
            sequence=next(self._sequence_counter),
        )
        self._orders[order_id] = order
        self._competitor_order_ids.add(order_id)

    def summarize_participant_fills(self, fills: Sequence[RecordedFill]) -> tuple[float, float]:
        buy_filled_quantity = 0.0
        sell_filled_quantity = 0.0

        for fill in fills:
            if fill.side is Side.BUY:
                buy_filled_quantity += fill.quantity
            else:
                sell_filled_quantity += fill.quantity

        return round(buy_filled_quantity, 10), round(sell_filled_quantity, 10)

    def record_participant_fills(self, fills: Sequence[RecordedFill], *, probability: float) -> None:
        total_edge = self._stats.total_edge
        retail_edge = self._stats.retail_edge
        arb_edge = self._stats.arb_edge
        traded_quantity = self._stats.traded_quantity
        traded_notional = self._stats.traded_notional
        fill_count = self._stats.fill_count

        for fill in fills:
            price = tick_to_price(fill.price_ticks)
            edge = fill.quantity * (probability - price) if fill.side is Side.BUY else fill.quantity * (price - probability)
            total_edge += edge
            traded_quantity += fill.quantity
            traded_notional += fill.quantity * price
            fill_count += 1
            if fill.aggressor == RETAIL:
                retail_edge += edge
            elif fill.aggressor == ARB:
                arb_edge += edge

        self._stats = ParticipantStats(
            total_edge=round(total_edge, 10),
            retail_edge=round(retail_edge, 10),
            arb_edge=round(arb_edge, 10),
            traded_quantity=round(traded_quantity, 10),
            traded_notional=round(traded_notional, 10),
            fill_count=fill_count,
        )
