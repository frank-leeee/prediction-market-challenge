"""Sandbox execution for untrusted strategy code.

Provides two layers of isolation:

1. **Python-level**: A restricted import hook blocks strategies from importing
   dangerous modules (os, subprocess, socket, ctypes, etc.).  After the
   strategy module is loaded, dangerous builtins (open, breakpoint) are also
   replaced with stubs.  This is always active when sandbox mode is enabled.

2. **OS-level (nsjail)**: When nsjail is installed, the subprocess runs inside
   a Linux namespace jail with no network access, read-only filesystem mounts,
   memory/CPU limits, and PID isolation. nsjail is Linux-only; on other
   platforms only Python-level sandboxing is applied.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path

from .config import ChallengeConfig, ParameterVariance
from .results import RegimeSummary, SimulationResult

# ---------------------------------------------------------------------------
# Import allowlist - only these top-level modules may be imported by strategy
# code.  Transitive imports (e.g. numpy importing os internally) are allowed
# so that legitimate libraries work, but the strategy itself cannot directly
# ``import os``.
# ---------------------------------------------------------------------------

ALLOWED_MODULES: frozenset[str] = frozenset({
    # Math and numerics
    "math", "cmath", "statistics", "random",
    "decimal", "fractions", "numbers",
    # Data structures and algorithms
    "collections", "heapq", "bisect", "array", "queue",
    # Functional programming
    "itertools", "functools", "operator",
    # Type system and metaprogramming
    "dataclasses", "typing", "typing_extensions", "enum", "abc",
    # Utilities
    "copy", "string", "re", "json", "csv", "struct",
    "datetime", "time", "hashlib",
    # Scientific computing (commonly used in quant strategies)
    "numpy", "scipy", "pandas",
    # This package (for types, base classes, etc.)
    "orderbook_pm_challenge",
})

SAFE_ORDERBOOK_PM_IMPORTS: frozenset[str] = frozenset({
    "orderbook_pm_challenge",
    "orderbook_pm_challenge.config",
    "orderbook_pm_challenge.strategy",
    "orderbook_pm_challenge.types",
})

BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "open", "breakpoint",
})

MAX_OUTPUT_BYTES = 1 << 20
NSJAIL_PID_LIMIT = 32
NSJAIL_CPU_MS_PER_SEC = 1000

NSJAIL_SECCOMP_POLICY = """\
ERRNO(1) {
  execve, execveat,
  fork, vfork, clone, clone3,
  socket, socketpair, connect, accept, accept4, bind, listen,
  ptrace, process_vm_readv, process_vm_writev,
  bpf, userfaultfd, perf_event_open,
  mount, umount2, pivot_root, setns, unshare,
  init_module, finit_module, delete_module
}
DEFAULT ALLOW
"""

# ---------------------------------------------------------------------------
# Python-level sandboxing
# ---------------------------------------------------------------------------

_original_import = builtins.__import__
_import_nesting: int = 0


def _make_blocked(name: str):
    """Return a callable that raises RuntimeError when invoked."""

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(f"builtin '{name}' is not allowed in sandbox mode")

    _blocked.__name__ = name
    _blocked.__qualname__ = name
    return _blocked


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Import hook that enforces the module allowlist.

    Only *direct* (top-level) imports are checked.  Transitive imports
    triggered by an allowed module are permitted so that libraries like
    numpy (which internally imports os, etc.) keep working.
    """
    global _import_nesting

    # Relative imports are always fine (within the strategy's own package)
    if level > 0:
        return _original_import(name, globals, locals, fromlist, level)

    # Only enforce the allowlist on direct imports (nesting == 0)
    if _import_nesting == 0:
        base = name.split(".")[0]
        if base == "orderbook_pm_challenge" and name not in SAFE_ORDERBOOK_PM_IMPORTS:
            raise ImportError(
                f"Import of '{name}' is not allowed in sandbox mode. "
                f"Allowed orderbook_pm_challenge imports: {', '.join(sorted(SAFE_ORDERBOOK_PM_IMPORTS))}"
            )
        # Allow CPython internal/private C-extension modules (e.g. _io,
        # _collections_abc).  These are implementation details of the
        # stdlib needed by the import machinery itself; the nsjail layer
        # prevents any OS-level escape they might otherwise enable.
        if not base.startswith("_") and base not in ALLOWED_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed in sandbox mode. "
                f"Allowed top-level modules: {', '.join(sorted(ALLOWED_MODULES))}"
            )

    _import_nesting += 1
    try:
        return _original_import(name, globals, locals, fromlist, level)
    finally:
        _import_nesting -= 1


def install_import_restrictions() -> None:
    """Lock down the import system to the allowlist.

    Call AFTER all engine/framework imports are complete and BEFORE
    loading any untrusted strategy code.
    """
    builtins.__import__ = _restricted_import


def install_builtin_restrictions() -> None:
    """Block dangerous builtins (open, breakpoint).

    Call AFTER the strategy module has been loaded (since the import
    machinery needs ``open`` / ``exec`` internally to read ``.py`` files).
    The strategy's ``on_step()`` will execute with these blocked.
    """
    for name in BLOCKED_BUILTINS:
        setattr(builtins, name, _make_blocked(name))


def _sandbox_builtins_dict() -> dict[str, object]:
    """Return a builtins mapping with sandbox restrictions applied.

    This is used while executing the untrusted strategy module so that
    module-level code cannot cache references to dangerous builtins before
    the global restrictions are installed.
    """
    sandboxed = dict(vars(builtins))
    for name in BLOCKED_BUILTINS:
        sandboxed[name] = _make_blocked(name)
    return sandboxed


def load_strategy_factory_in_sandbox(strategy_path: str):
    """Load a strategy module with restricted builtins from first execution."""
    path = Path(strategy_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file does not exist: {path}")

    module_name = f"orderbook_pm_strategy_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load strategy module from {path}")

    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    module.__dict__["__builtins__"] = _sandbox_builtins_dict()
    spec.loader.exec_module(module)

    strategy_cls = getattr(module, "Strategy", None)
    if strategy_cls is None:
        raise AttributeError(f"{path} does not define a Strategy class")

    def factory():
        instance = strategy_cls()
        if not hasattr(instance, "on_step"):
            raise TypeError("Strategy instance must define on_step(state)")
        return instance

    return factory


# ---------------------------------------------------------------------------
# nsjail integration
# ---------------------------------------------------------------------------


def find_nsjail() -> str | None:
    """Return the path to the nsjail binary, or ``None`` if unavailable."""
    return shutil.which("nsjail")


def _generate_nsjail_config(
    python_bin: str,
    strategy_path: str,
    package_path: str,
    *,
    time_limit: int = 300,
    memory_limit_mb: int = 512,
) -> str:
    """Generate a protobuf-format nsjail configuration string."""
    python_prefix = os.path.dirname(os.path.dirname(os.path.realpath(python_bin)))
    strategy_dir = os.path.dirname(os.path.abspath(strategy_path))

    mounts = []

    def _add_mount(src: str, dst: str | None = None, *, rw: bool = False, mandatory: bool = True):
        dst = dst or src
        entry = f'mount {{ src: "{src}" dst: "{dst}" is_bind: true rw: {"true" if rw else "false"}'
        if not mandatory:
            entry += " mandatory: false"
        entry += " }"
        mounts.append(entry)

    def _add_tmpfs(dst: str, *, rw: bool = True, options: str = ""):
        entry = f'mount {{ dst: "{dst}" fstype: "tmpfs" rw: {"true" if rw else "false"}'
        if options:
            entry += f' options: "{options}"'
        entry += " }"
        mounts.append(entry)

    # Python installation
    _add_mount(python_prefix)

    # Common system library paths
    for lib_dir in ("/usr/lib", "/lib", "/lib64", "/usr/lib64"):
        if os.path.isdir(lib_dir):
            _add_mount(lib_dir, mandatory=False)

    # Package and strategy
    _add_mount(package_path)
    _add_mount(strategy_dir)

    # Site-packages directories (for numpy, scipy, etc.)
    for path in sys.path:
        if "site-packages" in path and os.path.isdir(path):
            _add_mount(path)

    # Writable tmpfs and required devices
    _add_tmpfs("/tmp", options="size=16m")
    _add_tmpfs("/dev", rw=False)
    mounts.append('mount { src: "/dev/null" dst: "/dev/null" is_bind: true rw: true }')
    mounts.append('mount { src: "/dev/urandom" dst: "/dev/urandom" is_bind: true rw: false }')
    mounts.append('mount { dst: "/proc" fstype: "proc" rw: false }')

    mount_block = "\n".join(mounts)
    return f"""\
name: "pm-challenge-sandbox"
mode: ONCE
hostname: "sandbox"
time_limit: {time_limit}
max_cpus: 1
keep_env: false

rlimit_as_type: SOFT
rlimit_as: {memory_limit_mb}
rlimit_cpu_type: SOFT
rlimit_cpu: {time_limit}
rlimit_fsize: 1
rlimit_nofile: 64
cgroup_pids_max: {NSJAIL_PID_LIMIT}
cgroup_cpu_ms_per_sec: {NSJAIL_CPU_MS_PER_SEC}

clone_newnet: true
clone_newuser: true
clone_newns: true
clone_newpid: true
clone_newipc: true

seccomp_string: "{NSJAIL_SECCOMP_POLICY.strip().replace(chr(10), ' ')}"

{mount_block}
"""


# ---------------------------------------------------------------------------
# Sandboxed subprocess execution
# ---------------------------------------------------------------------------


def _result_from_dict(d: dict) -> SimulationResult:
    """Reconstruct a SimulationResult from a dictionary."""
    regime = RegimeSummary(**d.pop("regime"))
    return SimulationResult(regime=regime, **d)


def _make_failed_result(
    seed: int, config: ChallengeConfig, error: str,
) -> SimulationResult:
    """Build a failed SimulationResult for sandbox-level errors."""
    from .process import true_probability

    initial_prob = true_probability(
        config.process.initial_score, config.process.n_steps, config.process,
    )
    return SimulationResult(
        seed=seed,
        failed=True,
        error=error,
        regime=RegimeSummary.from_config(config, initial_probability=initial_prob),
        total_edge=0.0,
        retail_edge=0.0,
        arb_edge=0.0,
        traded_quantity=0.0,
        traded_notional=0.0,
        fill_count=0,
        average_net_inventory=0.0,
        average_abs_inventory=0.0,
        max_abs_inventory=0.0,
        final_cash=config.starting_cash,
        final_yes_inventory=0.0,
        final_no_inventory=0.0,
        settlement_outcome=0.0,
        final_wealth=config.starting_cash,
    )


def _read_text_with_limit(fileobj, *, limit: int) -> tuple[str | None, int]:
    """Read UTF-8 text from a temporary file if it is within the size limit."""
    fileobj.flush()
    fileobj.seek(0, os.SEEK_END)
    size = fileobj.tell()
    fileobj.seek(0)
    if size > limit:
        return None, size
    data = fileobj.read()
    return data.decode("utf-8", errors="replace"), size


def run_sandboxed_simulation(
    strategy_path: str,
    config: ChallengeConfig,
    variance: ParameterVariance,
    seed: int,
    *,
    nsjail_path: str | None = None,
    timeout: int = 300,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> SimulationResult:
    """Run a single simulation in a sandboxed subprocess.

    The subprocess loads the strategy with Python-level restrictions
    (blocked imports / builtins).  When *nsjail_path* is provided the
    process is additionally wrapped in an OS-level namespace jail.
    """
    worker_module = "orderbook_pm_challenge._sandbox_worker"
    python_bin = sys.executable

    task_payload = json.dumps({
        "strategy_path": str(Path(strategy_path).resolve()),
        "config": asdict(config),
        "variance": asdict(variance),
        "seed": seed,
    })

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            if nsjail_path:
                package_path = str(Path(__file__).parent.resolve())
                nsjail_cfg = _generate_nsjail_config(
                    python_bin, strategy_path, package_path, time_limit=timeout,
                )
                cfg_fd, cfg_path = tempfile.mkstemp(suffix=".cfg", prefix="nsjail_pm_")
                try:
                    with os.fdopen(cfg_fd, "w") as f:
                        f.write(nsjail_cfg)
                    cmd = [nsjail_path, "--config", cfg_path, "--", python_bin, "-m", worker_module]
                    proc = subprocess.run(
                        cmd,
                        input=(task_payload + "\n").encode("utf-8"),
                        stdout=stdout_file,
                        stderr=stderr_file,
                        timeout=timeout,
                    )
                finally:
                    os.unlink(cfg_path)
            else:
                cmd = [python_bin, "-m", worker_module]
                proc = subprocess.run(
                    cmd,
                    input=(task_payload + "\n").encode("utf-8"),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return _make_failed_result(
                seed,
                config,
                f"Sandbox worker timed out after {timeout} seconds",
            )

        stdout_text, stdout_size = _read_text_with_limit(stdout_file, limit=max_output_bytes)
        stderr_text, stderr_size = _read_text_with_limit(stderr_file, limit=max_output_bytes)

    if stdout_text is None:
        return _make_failed_result(
            seed,
            config,
            f"Sandbox worker exceeded stdout limit of {max_output_bytes} bytes ({stdout_size} bytes written)",
        )
    if stderr_text is None:
        return _make_failed_result(
            seed,
            config,
            f"Sandbox worker exceeded stderr limit of {max_output_bytes} bytes ({stderr_size} bytes written)",
        )

    # Try to parse the worker's JSON output regardless of exit code -
    # the worker writes structured errors to stdout before exiting.
    try:
        output = json.loads(stdout_text.strip())
    except (json.JSONDecodeError, ValueError):
        output = None

    if output is not None:
        if output.get("success"):
            return _result_from_dict(output["result"])
        return _make_failed_result(seed, config, output.get("error", "Unknown sandbox error"))

    # No valid JSON on stdout - fall back to stderr / exit code
    error = stderr_text.strip() or f"Sandbox worker exited with code {proc.returncode}"
    return _make_failed_result(seed, config, error)
