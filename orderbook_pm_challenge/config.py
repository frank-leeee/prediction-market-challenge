from __future__ import annotations

from dataclasses import dataclass, field


MIN_PRICE_TICK = 1
MAX_PRICE_TICK = 99
SHARE_QUANTUM = 0.01
DEFAULT_SIMULATIONS = 200


@dataclass(frozen=True)
class JumpDiffusionConfig:
    """Latent score-process parameters."""

    n_steps: int = 2_000
    initial_score: float = 0.0
    diffusion_sigma: float = 0.02
    jump_intensity: float = 0.001
    jump_mean: float = 0.0
    jump_sigma: float = 0.75
    terminal_threshold: float = 0.0
    poisson_tail_mass: float = 1e-12


@dataclass(frozen=True)
class RetailFlowConfig:
    """Retail market-order flow parameters."""

    arrival_rate: float = 0.8
    mean_notional: float = 12.0
    size_sigma: float = 1.2
    buy_probability: float = 0.5
    sell_quantity_price_floor: float = 0.05


@dataclass(frozen=True)
class CompetitorConfig:
    """Parameters for the static hidden competitor ladder."""

    quote_notional: float = 60.0
    spread_ticks: int = 2


@dataclass(frozen=True)
class ChallengeConfig:
    """Top-level challenge parameters."""

    process: JumpDiffusionConfig = field(default_factory=JumpDiffusionConfig)
    retail: RetailFlowConfig = field(default_factory=RetailFlowConfig)
    competitor: CompetitorConfig = field(default_factory=CompetitorConfig)
    min_price_tick: int = MIN_PRICE_TICK
    max_price_tick: int = MAX_PRICE_TICK
    share_quantum: float = SHARE_QUANTUM
    default_simulations: int = DEFAULT_SIMULATIONS
    starting_cash: float = 1_000.0


@dataclass(frozen=True)
class ParameterVariance:
    """Recommended simulation-randomization bands."""

    initial_score_min: float = -0.75
    initial_score_max: float = 0.75
    jump_intensity_min: float = 0.0008
    jump_intensity_max: float = 0.0030
    jump_sigma_min: float = 0.20
    jump_sigma_max: float = 0.60
    jump_mean_min: float = -0.04
    jump_mean_max: float = 0.04
    retail_arrival_rate_min: float = 0.154
    retail_arrival_rate_max: float = 0.352
    retail_mean_notional_min: float = 2.64
    retail_mean_notional_max: float = 6.336
    competitor_quote_notional_min: float = 24.0
    competitor_quote_notional_max: float = 72.0
    competitor_spread_ticks_min: int = 1
    competitor_spread_ticks_max: int = 4
