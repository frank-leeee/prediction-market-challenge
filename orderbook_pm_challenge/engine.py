from __future__ import annotations

import traceback

from .config import ChallengeConfig
from .market import OrderBookError, PredictionMarket
from .process import JumpDiffusionScoreProcess
from .results import RegimeSummary, SimulationResult
from .retail import RetailFlow
from .utils import average, quantize_down


class SimulationEngine:
    def __init__(self, config: ChallengeConfig, strategy_factory, *, seed: int) -> None:
        self._config = config
        self._strategy_factory = strategy_factory
        self._seed = seed

    def run(self) -> SimulationResult:
        strategy = self._strategy_factory()
        process = JumpDiffusionScoreProcess(self._config.process, seed=self._seed)
        starting_probability = process.current_true_probability()

        market = PredictionMarket(self._config)
        market.initialize_competitor(starting_probability)
        retail = RetailFlow(self._config.retail, seed=self._seed + 1)

        pending_buy_filled_quantity = 0.0
        pending_sell_filled_quantity = 0.0
        inventory_path: list[float] = []

        try:
            for step in range(self._config.process.n_steps):
                market.refresh_competitor(step)
                state = market.build_step_state(
                    step=step,
                    steps_remaining=self._config.process.n_steps - step,
                    buy_filled_quantity=pending_buy_filled_quantity,
                    sell_filled_quantity=pending_sell_filled_quantity,
                )
                actions = strategy.on_step(state)
                market.apply_actions(actions, step=step)

                process.step()
                probability = process.current_true_probability()

                step_fills = []
                step_fills.extend(market.execute_arbitrage(probability=probability, step=step))

                for order in retail.generate_orders():
                    if order.side == "BUY":
                        step_fills.extend(market.execute_retail_buy(notional=order.notional, step=step))
                    else:
                        quantity = quantize_down(
                            order.notional / max(probability, self._config.retail.sell_quantity_price_floor),
                            self._config.share_quantum,
                        )
                        step_fills.extend(market.execute_retail_sell(quantity=quantity, step=step))

                market.record_participant_fills(step_fills, probability=probability)
                pending_buy_filled_quantity, pending_sell_filled_quantity = market.summarize_participant_fills(
                    step_fills
                )
                inventory_path.append(market.net_inventory())
        except Exception as exc:  # noqa: BLE001
            return SimulationResult(
                seed=self._seed,
                failed=True,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                regime=RegimeSummary.from_config(self._config, initial_probability=starting_probability),
                total_edge=market.stats.total_edge,
                retail_edge=market.stats.retail_edge,
                arb_edge=market.stats.arb_edge,
                traded_quantity=market.stats.traded_quantity,
                traded_notional=market.stats.traded_notional,
                fill_count=market.stats.fill_count,
                average_net_inventory=average(inventory_path),
                average_abs_inventory=average([abs(value) for value in inventory_path]),
                max_abs_inventory=max((abs(value) for value in inventory_path), default=0.0),
                final_cash=market.cash,
                final_yes_inventory=market.yes_inventory,
                final_no_inventory=market.no_inventory,
                settlement_outcome=0.0,
                final_wealth=market.cash,
            )

        outcome = 1.0 if process.current_score > self._config.process.terminal_threshold else 0.0
        final_wealth = market.settle(outcome=outcome)
        return SimulationResult(
            seed=self._seed,
            failed=False,
            error=None,
            regime=RegimeSummary.from_config(self._config, initial_probability=starting_probability),
            total_edge=market.stats.total_edge,
            retail_edge=market.stats.retail_edge,
            arb_edge=market.stats.arb_edge,
            traded_quantity=market.stats.traded_quantity,
            traded_notional=market.stats.traded_notional,
            fill_count=market.stats.fill_count,
            average_net_inventory=average(inventory_path),
            average_abs_inventory=average([abs(value) for value in inventory_path]),
            max_abs_inventory=max((abs(value) for value in inventory_path), default=0.0),
            final_cash=market.cash,
            final_yes_inventory=market.yes_inventory,
            final_no_inventory=market.no_inventory,
            settlement_outcome=outcome,
            final_wealth=final_wealth,
        )
