from __future__ import annotations

from dataclasses import dataclass

from .config import ChallengeConfig


@dataclass(frozen=True)
class RegimeSummary:
    initial_score: float
    initial_probability: float
    jump_intensity: float
    jump_mean: float
    jump_sigma: float
    retail_arrival_rate: float
    retail_mean_notional: float
    competitor_quote_notional: float
    competitor_spread_ticks: int

    @classmethod
    def from_config(cls, config: ChallengeConfig, *, initial_probability: float) -> "RegimeSummary":
        return cls(
            initial_score=config.process.initial_score,
            initial_probability=initial_probability,
            jump_intensity=config.process.jump_intensity,
            jump_mean=config.process.jump_mean,
            jump_sigma=config.process.jump_sigma,
            retail_arrival_rate=config.retail.arrival_rate,
            retail_mean_notional=config.retail.mean_notional,
            competitor_quote_notional=config.competitor.quote_notional,
            competitor_spread_ticks=config.competitor.spread_ticks,
        )


@dataclass(frozen=True)
class SimulationResult:
    seed: int
    failed: bool
    error: str | None
    regime: RegimeSummary
    total_edge: float
    retail_edge: float
    arb_edge: float
    traded_quantity: float
    traded_notional: float
    fill_count: int
    average_net_inventory: float
    average_abs_inventory: float
    max_abs_inventory: float
    final_cash: float
    final_yes_inventory: float
    final_no_inventory: float
    settlement_outcome: float
    final_wealth: float


@dataclass(frozen=True)
class BatchResult:
    simulation_results: tuple[SimulationResult, ...]

    @property
    def success_count(self) -> int:
        return sum(0 if result.failed else 1 for result in self.simulation_results)

    @property
    def failure_count(self) -> int:
        return sum(1 if result.failed else 0 for result in self.simulation_results)

    @property
    def mean_edge(self) -> float:
        successes = [result.total_edge for result in self.simulation_results if not result.failed]
        return sum(successes) / len(successes) if successes else 0.0

    @property
    def mean_retail_edge(self) -> float:
        successes = [result.retail_edge for result in self.simulation_results if not result.failed]
        return sum(successes) / len(successes) if successes else 0.0

    @property
    def mean_arb_edge(self) -> float:
        successes = [result.arb_edge for result in self.simulation_results if not result.failed]
        return sum(successes) / len(successes) if successes else 0.0

    @property
    def mean_final_wealth(self) -> float:
        successes = [result.final_wealth for result in self.simulation_results if not result.failed]
        return sum(successes) / len(successes) if successes else 0.0
