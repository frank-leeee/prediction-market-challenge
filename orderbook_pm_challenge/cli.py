from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from .config import ChallengeConfig
from .loader import load_strategy_factory
from .runner import run_batch


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
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, serial execution)",
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
    num_workers = args.workers

    # When running serial + unsandboxed, we can load the factory in-process
    strategy_factory = None
    if not use_sandbox and num_workers <= 1:
        strategy_factory = load_strategy_factory(args.strategy_path)

    batch = run_batch(
        strategy_factory,
        strategy_path=args.strategy_path,
        base_config=base_config,
        n_simulations=args.simulations,
        seed_start=args.seed_start,
        workers=num_workers,
        sandbox=use_sandbox,
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
