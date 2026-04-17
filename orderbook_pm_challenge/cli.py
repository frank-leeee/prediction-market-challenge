from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

from .backend import resolve_engine_backend
from .config import ChallengeConfig
from .loader import load_strategy_factory
from .runner import run_batch


def resolve_worker_count(
    requested_workers: int | None,
    *,
    n_simulations: int,
    sandbox: bool,
) -> int:
    if requested_workers is not None:
        return requested_workers
    if sandbox or n_simulations <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, n_simulations))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orderbook-pm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a strategy against the local challenge")
    run_parser.add_argument("strategy_path", help="Path to a Python strategy file")
    run_parser.add_argument("--simulations", type=int, default=None, help="Number of simulations")
    run_parser.add_argument("--steps", type=int, default=None, help="Steps per simulation")
    run_parser.add_argument("--seed-start", type=int, default=0, help="Starting simulation seed")
    run_parser.add_argument("--json", action="store_true", help="Print full JSON results")
    run_parser.add_argument(
        "--engine",
        choices=("auto", "python", "rust"),
        default="auto",
        help="Execution backend (default: auto; rust falls back to python if unavailable)",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto for unsandboxed multi-simulation runs)",
    )
    run_parser.add_argument(
        "--sandbox",
        action="store_true",
        help=(
            "Run strategy in a sandboxed subprocess with restricted imports/builtins. "
            "Uses nsjail for OS-level isolation when available."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"Unknown command: {args.command}")

    base_config = ChallengeConfig()
    if args.steps is not None:
        base_config = replace(base_config, process=replace(base_config.process, n_steps=args.steps))

    use_sandbox = args.sandbox
    engine_backend = resolve_engine_backend(args.engine, sandbox=use_sandbox)
    simulation_count = args.simulations or base_config.default_simulations
    num_workers = resolve_worker_count(
        args.workers,
        n_simulations=simulation_count,
        sandbox=use_sandbox,
    )

    # When running serial + unsandboxed, we can load the factory in-process
    strategy_factory = None
    if not use_sandbox and num_workers <= 1:
        strategy_factory = load_strategy_factory(args.strategy_path)

    batch = run_batch(
        strategy_factory,
        strategy_path=args.strategy_path,
        base_config=base_config,
        n_simulations=simulation_count,
        seed_start=args.seed_start,
        workers=num_workers,
        sandbox=use_sandbox,
        engine_backend=engine_backend,
    )

    if args.json:
        print(json.dumps(asdict(batch), indent=2))
        return 0

    print(f"Simulations: {len(batch.simulation_results)}")
    print(f"Successes: {batch.success_count}")
    print(f"Failures: {batch.failure_count}")
    print(f"Mean Edge: {batch.mean_edge:.6f}")
    print(f"Mean Retail Edge: {batch.mean_retail_edge:.6f}")
    print(f"Mean Arb Edge: {batch.mean_arb_edge:.6f}")
    print(f"Mean Final Wealth: {batch.mean_final_wealth:.6f}")

    failed = [result for result in batch.simulation_results if result.failed]
    if failed:
        print("Failed Seeds:")
        for result in failed[:10]:
            print(f"  seed={result.seed}: {result.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
