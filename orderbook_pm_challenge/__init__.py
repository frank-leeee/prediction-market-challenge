"""Local simulator for the FIFO orderbook prediction market challenge."""

from .config import (
    ChallengeConfig,
    CompetitorConfig,
    JumpDiffusionConfig,
    ParameterVariance,
    RetailFlowConfig,
)
from .backend import resolve_engine_backend, rust_backend_available
from .engine import SimulationEngine
from .results import BatchResult, RegimeSummary, SimulationResult
from .runner import run_batch, sample_config
from .process import JumpDiffusionScoreProcess, true_probability
from .strategy import BaseStrategy, Strategy
from .types import (
    Action,
    CancelAll,
    CancelOrder,
    OwnOrderView,
    PlaceOrder,
    Side,
    StepState,
)

__all__ = [
    "Action",
    "BatchResult",
    "BaseStrategy",
    "CancelAll",
    "CancelOrder",
    "ChallengeConfig",
    "CompetitorConfig",
    "JumpDiffusionConfig",
    "JumpDiffusionScoreProcess",
    "OwnOrderView",
    "ParameterVariance",
    "PlaceOrder",
    "RegimeSummary",
    "RetailFlowConfig",
    "resolve_engine_backend",
    "rust_backend_available",
    "SimulationEngine",
    "SimulationResult",
    "Side",
    "StepState",
    "Strategy",
    "run_batch",
    "sample_config",
    "true_probability",
]
