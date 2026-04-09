from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, replace
import random
import sys

from .config import ChallengeConfig, ParameterVariance
from .engine import SimulationEngine
from .results import BatchResult, RegimeSummary, SimulationResult


def sample_config(
    base_config: ChallengeConfig,
    variance: ParameterVariance,
    *,
    seed: int,
) -> ChallengeConfig:
    rng = random.Random(seed)
    process = replace(
        base_config.process,
        initial_score=rng.uniform(variance.initial_score_min, variance.initial_score_max),
        jump_intensity=rng.uniform(variance.jump_intensity_min, variance.jump_intensity_max),
        jump_sigma=rng.uniform(variance.jump_sigma_min, variance.jump_sigma_max),
        jump_mean=rng.uniform(variance.jump_mean_min, variance.jump_mean_max),
    )
    retail = replace(
        base_config.retail,
        arrival_rate=rng.uniform(variance.retail_arrival_rate_min, variance.retail_arrival_rate_max),
        mean_notional=rng.uniform(variance.retail_mean_notional_min, variance.retail_mean_notional_max),
    )
    competitor = replace(
        base_config.competitor,
        quote_notional=rng.uniform(
            variance.competitor_quote_notional_min,
            variance.competitor_quote_notional_max,
        ),
        spread_ticks=rng.randint(
            variance.competitor_spread_ticks_min,
            variance.competitor_spread_ticks_max,
        ),
    )
    return replace(base_config, process=process, retail=retail, competitor=competitor)


# ---------------------------------------------------------------------------
# Module-level helper for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------


def _run_single_simulation(
    strategy_path: str,
    base_config_dict: dict,
    variance_dict: dict,
    seed: int,
) -> dict:
    """Execute one simulation in a worker process and return the result dict."""
    from .config import (
        ChallengeConfig,
        CompetitorConfig,
        JumpDiffusionConfig,
        ParameterVariance,
        RetailFlowConfig,
    )
    from .loader import load_strategy_factory

    base_config = ChallengeConfig(
        process=JumpDiffusionConfig(**base_config_dict["process"]),
        retail=RetailFlowConfig(**base_config_dict["retail"]),
        competitor=CompetitorConfig(**base_config_dict["competitor"]),
        min_price_tick=base_config_dict["min_price_tick"],
        max_price_tick=base_config_dict["max_price_tick"],
        share_quantum=base_config_dict["share_quantum"],
        default_simulations=base_config_dict["default_simulations"],
        starting_cash=base_config_dict["starting_cash"],
    )
    variance = ParameterVariance(**variance_dict)

    config = sample_config(base_config, variance, seed=seed)
    factory = load_strategy_factory(strategy_path)
    engine = SimulationEngine(config, factory, seed=seed)
    return asdict(engine.run())


def _result_from_dict(d: dict) -> SimulationResult:
    regime = RegimeSummary(**d.pop("regime"))
    return SimulationResult(regime=regime, **d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_batch(
    strategy_factory=None,
    *,
    strategy_path: str | None = None,
    base_config: ChallengeConfig | None = None,
    variance: ParameterVariance | None = None,
    n_simulations: int | None = None,
    seed_start: int = 0,
    workers: int = 1,
    sandbox: bool = False,
) -> BatchResult:
    """Run a batch of simulations.

    Parameters
    ----------
    strategy_factory:
        Callable that returns a Strategy instance. Not needed when
        *strategy_path* is provided and *workers* > 1 or *sandbox* is True.
    strategy_path:
        Filesystem path to the strategy ``.py`` file. Required when using
        ``workers > 1`` or ``sandbox=True``.
    workers:
        Number of parallel workers.  ``1`` (default) runs serially in the
        current process.  Values > 1 distribute simulations across a
        process pool.
    sandbox:
        When True, each simulation runs in a sandboxed subprocess with
        restricted Python imports/builtins (and nsjail if available).
    """
    base = base_config or ChallengeConfig()
    var = variance or ParameterVariance()
    count = n_simulations or base.default_simulations

    if sandbox:
        if strategy_path is None:
            raise ValueError("strategy_path is required for sandbox mode")
        return _run_batch_sandboxed(strategy_path, base, var, count, seed_start, workers)

    if workers > 1:
        if strategy_path is None:
            raise ValueError("strategy_path is required when workers > 1")
        return _run_batch_parallel(strategy_path, base, var, count, seed_start, workers)

    # Serial in-process execution (original behaviour)
    if strategy_factory is None:
        if strategy_path is None:
            raise ValueError("strategy_factory or strategy_path is required")
        from .loader import load_strategy_factory

        strategy_factory = load_strategy_factory(strategy_path)

    results = []
    for offset in range(count):
        seed = seed_start + offset
        config = sample_config(base, var, seed=seed)
        engine = SimulationEngine(config, strategy_factory, seed=seed)
        results.append(engine.run())
    return BatchResult(simulation_results=tuple(results))


# ---------------------------------------------------------------------------
# Parallel (unsandboxed) execution via ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _run_batch_parallel(
    strategy_path: str,
    base: ChallengeConfig,
    var: ParameterVariance,
    count: int,
    seed_start: int,
    workers: int,
) -> BatchResult:
    base_dict = asdict(base)
    var_dict = asdict(var)
    seeds = [seed_start + offset for offset in range(count)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_single_simulation, strategy_path, base_dict, var_dict, seed)
            for seed in seeds
        ]
        result_dicts = [f.result() for f in futures]

    return BatchResult(simulation_results=tuple(_result_from_dict(d) for d in result_dicts))


# ---------------------------------------------------------------------------
# Sandboxed execution (subprocess per simulation)
# ---------------------------------------------------------------------------


def _run_batch_sandboxed(
    strategy_path: str,
    base: ChallengeConfig,
    var: ParameterVariance,
    count: int,
    seed_start: int,
    workers: int,
) -> BatchResult:
    from .sandbox import find_nsjail, run_sandboxed_simulation

    nsjail_path = find_nsjail()
    if nsjail_path:
        print(f"[sandbox] nsjail found at {nsjail_path} — using OS-level sandboxing", file=sys.stderr)
    else:
        print("[sandbox] nsjail not found — using Python-level sandboxing only", file=sys.stderr)

    seeds = [seed_start + offset for offset in range(count)]

    def _run_one(seed: int) -> SimulationResult:
        return run_sandboxed_simulation(
            strategy_path, base, var, seed, nsjail_path=nsjail_path,
        )

    if workers > 1:
        # Each simulation already runs in its own subprocess, so threads
        # are sufficient for concurrency management here.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, seeds))
    else:
        results = [_run_one(seed) for seed in seeds]

    return BatchResult(simulation_results=tuple(results))
