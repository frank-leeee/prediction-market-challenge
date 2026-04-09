from __future__ import annotations

from typing import Protocol, Sequence

from .types import Action, StepState


class Strategy(Protocol):
    """Strategy interface for local challenge runs."""

    def on_step(self, state: StepState) -> Sequence[Action]:
        """Return the actions to apply for the next timestep."""


class BaseStrategy:
    """Convenience base class for local competitor development."""

    def on_step(self, state: StepState) -> Sequence[Action]:
        return []
