from __future__ import annotations

import os
import tempfile
import unittest

from orderbook_pm_challenge.config import ChallengeConfig, JumpDiffusionConfig, ParameterVariance
from orderbook_pm_challenge.runner import run_batch
from orderbook_pm_challenge.sandbox import _generate_nsjail_config, run_sandboxed_simulation


def _write_strategy(code: str) -> str:
    """Write strategy code to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="strat_")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    return path


SHORT_CONFIG = ChallengeConfig(process=JumpDiffusionConfig(n_steps=10))

GOOD_STRATEGY = """\
from orderbook_pm_challenge.strategy import BaseStrategy

class Strategy(BaseStrategy):
    pass
"""

EVIL_IMPORT_OS = """\
import os

class Strategy:
    def on_step(self, state):
        return []
"""

EVIL_IMPORT_SUBPROCESS = """\
import subprocess

class Strategy:
    def on_step(self, state):
        return []
"""

EVIL_IMPORT_SOCKET = """\
import socket

class Strategy:
    def on_step(self, state):
        return []
"""

EVIL_IMPORT_IO = """\
import io

class Strategy:
    def on_step(self, state):
        return []
"""

EVIL_IMPORT_SANDBOX_MODULE = """\
import orderbook_pm_challenge.sandbox

class Strategy:
    def on_step(self, state):
        return []
"""

EVIL_RUNTIME_IMPORT = """\
from orderbook_pm_challenge.strategy import BaseStrategy

class Strategy(BaseStrategy):
    def on_step(self, state):
        import os
        return []
"""

EVIL_OPEN_FILE = """\
from orderbook_pm_challenge.strategy import BaseStrategy

class Strategy(BaseStrategy):
    def on_step(self, state):
        open("/etc/passwd")
        return []
"""

EVIL_SAVED_OPEN = """\
from orderbook_pm_challenge.strategy import BaseStrategy

saved_open = open

class Strategy(BaseStrategy):
    def on_step(self, state):
        with saved_open("/tmp/should_not_exist", "w") as handle:
            handle.write("sandbox escape")
        return []
"""

SLOW_STRATEGY = """\
from orderbook_pm_challenge.strategy import BaseStrategy

class Strategy(BaseStrategy):
    def on_step(self, state):
        while True:
            pass
"""

NOISY_STRATEGY = """\
from orderbook_pm_challenge.strategy import BaseStrategy

class Strategy(BaseStrategy):
    def on_step(self, state):
        print("x" * (2**20 + 1024))
        return []
"""


class SandboxTests(unittest.TestCase):
    def test_good_strategy_succeeds_in_sandbox(self) -> None:
        path = _write_strategy(GOOD_STRATEGY)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.success_count, 1)
            self.assertEqual(batch.failure_count, 0)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_import_os(self) -> None:
        path = _write_strategy(EVIL_IMPORT_OS)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_import_subprocess(self) -> None:
        path = _write_strategy(EVIL_IMPORT_SUBPROCESS)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_import_socket(self) -> None:
        path = _write_strategy(EVIL_IMPORT_SOCKET)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_import_io(self) -> None:
        path = _write_strategy(EVIL_IMPORT_IO)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_import_of_internal_sandbox_module(self) -> None:
        path = _write_strategy(EVIL_IMPORT_SANDBOX_MODULE)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_runtime_import(self) -> None:
        path = _write_strategy(EVIL_RUNTIME_IMPORT)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_open(self) -> None:
        path = _write_strategy(EVIL_OPEN_FILE)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("not allowed in sandbox", batch.simulation_results[0].error)
        finally:
            os.unlink(path)

    def test_sandbox_blocks_saved_open_reference(self) -> None:
        path = _write_strategy(EVIL_SAVED_OPEN)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=1, sandbox=True,
            )
            self.assertEqual(batch.failure_count, 1)
            self.assertIn("builtin 'open' is not allowed", batch.simulation_results[0].error)
            self.assertFalse(os.path.exists("/tmp/should_not_exist"))
        finally:
            os.unlink(path)
            if os.path.exists("/tmp/should_not_exist"):
                os.unlink("/tmp/should_not_exist")

    def test_sandbox_timeout_returns_failed_result(self) -> None:
        path = _write_strategy(SLOW_STRATEGY)
        try:
            result = run_sandboxed_simulation(
                path,
                ChallengeConfig(process=JumpDiffusionConfig(n_steps=1)),
                ParameterVariance(),
                seed=0,
                timeout=1,
            )
            self.assertTrue(result.failed)
            self.assertIn("timed out", result.error or "")
        finally:
            os.unlink(path)

    def test_sandbox_output_limit_returns_failed_result(self) -> None:
        path = _write_strategy(NOISY_STRATEGY)
        try:
            result = run_sandboxed_simulation(
                path,
                ChallengeConfig(process=JumpDiffusionConfig(n_steps=1)),
                ParameterVariance(),
                seed=0,
                max_output_bytes=1024,
            )
            self.assertTrue(result.failed)
            self.assertIn("stdout limit", result.error or "")
        finally:
            os.unlink(path)

    def test_nsjail_config_includes_pid_and_seccomp_limits(self) -> None:
        config = _generate_nsjail_config(
            python_bin="/usr/bin/python3",
            strategy_path="/tmp/strategy.py",
            package_path="/tmp/package",
        )
        self.assertIn("keep_env: false", config)
        self.assertIn("cgroup_pids_max: 32", config)
        self.assertIn("cgroup_cpu_ms_per_sec: 1000", config)
        self.assertIn('seccomp_string: "ERRNO(1) {', config)


class ParallelTests(unittest.TestCase):
    def test_parallel_matches_serial(self) -> None:
        """Parallel execution produces the same results as serial."""
        path = _write_strategy(GOOD_STRATEGY)
        try:
            serial = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=4, seed_start=0, workers=1,
            )
            parallel = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=4, seed_start=0, workers=2,
            )
            for s, p in zip(serial.simulation_results, parallel.simulation_results):
                self.assertEqual(s.seed, p.seed)
                self.assertAlmostEqual(s.total_edge, p.total_edge, places=6)
                self.assertAlmostEqual(s.final_wealth, p.final_wealth, places=6)
        finally:
            os.unlink(path)

    def test_sandbox_parallel(self) -> None:
        """Sandbox + workers > 1 executes without errors."""
        path = _write_strategy(GOOD_STRATEGY)
        try:
            batch = run_batch(
                strategy_path=path, base_config=SHORT_CONFIG,
                n_simulations=4, sandbox=True, workers=2,
            )
            self.assertEqual(batch.success_count, 4)
            self.assertEqual(batch.failure_count, 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
