from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def rust_backend_available() -> bool:
    try:
        from . import _rust_sim  # noqa: F401
    except ImportError:
        return False
    return True


def require_rust_backend():
    if not rust_backend_available():
        raise RuntimeError(
            "Rust engine requested but orderbook_pm_challenge._rust_sim is unavailable. "
            "Build the extension with `uv sync --dev` or select `--engine python`."
        )
    from . import _rust_sim

    return _rust_sim


def resolve_engine_backend(requested_engine: str, *, sandbox: bool) -> str:
    if requested_engine not in {"auto", "python", "rust"}:
        raise ValueError(f"Unsupported engine: {requested_engine}")
    if sandbox:
        if requested_engine == "rust":
            raise ValueError("Rust engine is not available in sandbox mode")
        return "python"
    if requested_engine == "auto":
        return "rust" if rust_backend_available() else "python"
    if requested_engine == "rust":
        require_rust_backend()
    return requested_engine
