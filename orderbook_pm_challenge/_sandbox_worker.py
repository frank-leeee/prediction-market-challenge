"""Subprocess entry point for sandboxed strategy execution.

Runs inside a child process (optionally wrapped by nsjail).  Receives
simulation parameters as a JSON line on stdin, executes the simulation
with Python-level import/builtin restrictions, and writes the result as
a JSON line on stdout.

**All engine imports happen BEFORE ``install_restrictions()``** so that
the simulation infrastructure has unrestricted access to the standard
library while the strategy code is locked down.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict

# --- Engine imports (unrestricted) ----------------------------------------
from orderbook_pm_challenge.config import (
    ChallengeConfig,
    CompetitorConfig,
    JumpDiffusionConfig,
    ParameterVariance,
    RetailFlowConfig,
)
from orderbook_pm_challenge.engine import SimulationEngine
from orderbook_pm_challenge.runner import sample_config
from orderbook_pm_challenge.sandbox import (
    install_builtin_restrictions,
    install_import_restrictions,
    load_strategy_factory_in_sandbox,
)


def _config_from_dict(d: dict) -> ChallengeConfig:
    return ChallengeConfig(
        process=JumpDiffusionConfig(**d["process"]),
        retail=RetailFlowConfig(**d["retail"]),
        competitor=CompetitorConfig(**d["competitor"]),
        min_price_tick=d["min_price_tick"],
        max_price_tick=d["max_price_tick"],
        share_quantum=d["share_quantum"],
        default_simulations=d["default_simulations"],
        starting_cash=d["starting_cash"],
    )


def main() -> int:
    try:
        raw = sys.stdin.readline()
        if not raw.strip():
            _emit_error("Empty input on stdin")
            return 1
        task = json.loads(raw)
    except (json.JSONDecodeError, EOFError) as exc:
        _emit_error(f"Invalid input: {exc}")
        return 1

    strategy_path: str = task["strategy_path"]
    base_config = _config_from_dict(task["config"])
    variance = ParameterVariance(**task["variance"])
    seed: int = task["seed"]

    # Phase 1: lock down imports BEFORE loading untrusted strategy code
    install_import_restrictions()

    try:
        factory = load_strategy_factory_in_sandbox(strategy_path)

        # Phase 2: block dangerous builtins globally for runtime code.
        install_builtin_restrictions()

        sim_config = sample_config(base_config, variance, seed=seed)
        engine = SimulationEngine(sim_config, factory, seed=seed)
        result = engine.run()
        print(json.dumps({"success": True, "result": asdict(result)}), flush=True)
    except Exception:
        _emit_error(traceback.format_exc())
        return 1

    return 0


def _emit_error(message: str) -> None:
    print(json.dumps({"success": False, "error": message}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
