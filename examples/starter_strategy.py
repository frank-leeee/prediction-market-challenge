from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """Minimal static ladder strategy.

    This example demonstrates the public participant API for the hidden-book
    version of the challenge.
    """

    quote_size = 5

    def on_step(self, state: StepState):
        competitor_bid = state.competitor_best_bid_ticks or 49
        competitor_ask = state.competitor_best_ask_ticks or 51
        midpoint = (competitor_bid + competitor_ask) // 2

        actions = [CancelAll()]

        if state.yes_inventory < 100:
            actions.append(
                PlaceOrder(
                    side=Side.BUY,
                    price_ticks=max(1, midpoint - 2),
                    quantity=self.quote_size,
                )
            )

        if state.free_cash > 0 and state.no_inventory < 100:
            actions.append(
                PlaceOrder(
                    side=Side.SELL,
                    price_ticks=min(99, midpoint + 2),
                    quantity=self.quote_size,
                )
            )

        return actions
