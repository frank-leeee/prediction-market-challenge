from __future__ import annotations

import unittest
from unittest import mock

from orderbook_pm_challenge.backend import resolve_engine_backend
from orderbook_pm_challenge.cli import resolve_worker_count
from orderbook_pm_challenge.config import ChallengeConfig, CompetitorConfig, JumpDiffusionConfig
from orderbook_pm_challenge.engine import SimulationEngine
from orderbook_pm_challenge.market import PredictionMarket
from orderbook_pm_challenge.types import PlaceOrder, Side


class BadStrategy:
    def on_step(self, _state):
        return [PlaceOrder(side=Side.BUY, price_ticks=0, quantity=1.0)]


class PassiveStrategy:
    def on_step(self, _state):
        return []


class OrderbookChallengeTests(unittest.TestCase):
    def test_resolve_worker_count_defaults_to_auto_for_unsandboxed_batches(self) -> None:
        self.assertEqual(resolve_worker_count(None, n_simulations=1, sandbox=False), 1)
        self.assertEqual(resolve_worker_count(None, n_simulations=4, sandbox=True), 1)
        self.assertEqual(resolve_worker_count(3, n_simulations=20, sandbox=False), 3)
        self.assertGreaterEqual(resolve_worker_count(None, n_simulations=20, sandbox=False), 1)

    def test_initial_competitor_quotes_follow_start_price_and_spread(self) -> None:
        config = ChallengeConfig(competitor=CompetitorConfig(spread_ticks=2, quote_notional=10.0))
        market = PredictionMarket(config)
        market.initialize_competitor(0.503)
        self.assertEqual(market.competitor_best_quotes(), (49, 52))

    def test_resolve_engine_backend_defaults_to_python_in_sandbox(self) -> None:
        self.assertEqual(resolve_engine_backend("auto", sandbox=True), "python")
        self.assertEqual(resolve_engine_backend("python", sandbox=True), "python")
        with self.assertRaises(ValueError):
            resolve_engine_backend("rust", sandbox=True)

    def test_resolve_engine_backend_uses_rust_when_available(self) -> None:
        with mock.patch("orderbook_pm_challenge.backend.rust_backend_available", return_value=True):
            self.assertEqual(resolve_engine_backend("auto", sandbox=False), "rust")
            self.assertEqual(resolve_engine_backend("python", sandbox=False), "python")

    def test_uncovered_sell_fill_creates_no_inventory(self) -> None:
        market = PredictionMarket(ChallengeConfig())
        market.place_order(
            PlaceOrder(side=Side.SELL, price_ticks=60, quantity=1.0, client_order_id="ask"),
            step=0,
        )
        fills = market.execute_retail_buy(notional=1.0, step=0)
        market.record_participant_fills(fills, probability=0.55)

        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(market.cash, 999.6, places=6)
        self.assertAlmostEqual(market.yes_inventory, 0.0, places=6)
        self.assertAlmostEqual(market.no_inventory, 1.0, places=6)
        self.assertAlmostEqual(market.stats.traded_quantity, 1.0, places=6)

    def test_bid_fill_spends_reserved_cash_and_receives_yes(self) -> None:
        market = PredictionMarket(ChallengeConfig())
        market.place_order(
            PlaceOrder(side=Side.BUY, price_ticks=40, quantity=2.5, client_order_id="bid"),
            step=0,
        )
        self.assertAlmostEqual(market.reserved_cash(), 1.0, places=6)
        fills = market.execute_retail_sell(quantity=2.5, step=0)
        market.record_participant_fills(fills, probability=0.45)

        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(market.cash, 999.0, places=6)
        self.assertAlmostEqual(market.yes_inventory, 2.5, places=6)
        self.assertAlmostEqual(market.reserved_cash(), 0.0, places=6)

    def test_step_state_only_exposes_aggregate_buy_and_sell_fill_quantities(self) -> None:
        market = PredictionMarket(ChallengeConfig())
        market.place_order(
            PlaceOrder(side=Side.BUY, price_ticks=40, quantity=1.25, client_order_id="bid-1"),
            step=0,
        )
        market.place_order(
            PlaceOrder(side=Side.BUY, price_ticks=40, quantity=0.75, client_order_id="bid-2"),
            step=0,
        )
        market.place_order(
            PlaceOrder(side=Side.SELL, price_ticks=60, quantity=1.5, client_order_id="ask-1"),
            step=0,
        )
        market.place_order(
            PlaceOrder(side=Side.SELL, price_ticks=60, quantity=0.5, client_order_id="ask-2"),
            step=0,
        )

        fills = []
        fills.extend(market.execute_retail_sell(quantity=2.0, step=0))
        fills.extend(market.execute_retail_buy(notional=1.2, step=0))
        buy_filled_quantity, sell_filled_quantity = market.summarize_participant_fills(fills)

        state = market.build_step_state(
            step=1,
            steps_remaining=10,
            buy_filled_quantity=buy_filled_quantity,
            sell_filled_quantity=sell_filled_quantity,
        )

        self.assertAlmostEqual(state.buy_filled_quantity, 2.0, places=6)
        self.assertAlmostEqual(state.sell_filled_quantity, 2.0, places=6)
        self.assertFalse(hasattr(state, "fills"))
        self.assertFalse(hasattr(state, "fill_count"))
        self.assertEqual(
            set(state.__dataclass_fields__),
            {
                "step",
                "steps_remaining",
                "yes_inventory",
                "no_inventory",
                "cash",
                "reserved_cash",
                "free_cash",
                "competitor_best_bid_ticks",
                "competitor_best_ask_ticks",
                "buy_filled_quantity",
                "sell_filled_quantity",
                "own_orders",
            },
        )

    def test_invalid_action_fails_simulation(self) -> None:
        config = ChallengeConfig(process=JumpDiffusionConfig(n_steps=5))
        result = SimulationEngine(config, lambda: BadStrategy(), seed=0).run()
        self.assertTrue(result.failed)
        self.assertIn("price_ticks out of range", result.error or "")

    def test_smoke_run_with_passive_strategy(self) -> None:
        config = ChallengeConfig(process=JumpDiffusionConfig(n_steps=25))
        result = SimulationEngine(config, lambda: PassiveStrategy(), seed=2).run()
        self.assertFalse(result.failed)
        self.assertEqual(result.fill_count, 0)


if __name__ == "__main__":
    unittest.main()
