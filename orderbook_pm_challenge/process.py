from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import JumpDiffusionConfig


def standard_normal_cdf(x: float) -> float:
    """Standard normal CDF via erf."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _poisson_weights(mean: float, tail_mass: float) -> list[float]:
    """Return Poisson pmf weights up to a negligible tail."""

    if mean <= 0.0:
        return [1.0]

    weights: list[float] = []
    weight = math.exp(-mean)
    cumulative = 0.0
    n = 0

    while True:
        weights.append(weight)
        cumulative += weight
        if 1.0 - cumulative <= tail_mass:
            break
        n += 1
        weight *= mean / n

    total = sum(weights)
    return [w / total for w in weights]


def true_probability(
    score: float,
    steps_remaining: int,
    config: JumpDiffusionConfig,
) -> float:
    """Exact conditional probability that the terminal score finishes above zero.

    The future increment is modeled as:

    - Gaussian diffusion over the remaining steps
    - plus a compound Poisson number of Gaussian jumps
    """

    if steps_remaining <= 0:
        return 1.0 if score > config.terminal_threshold else 0.0

    horizon = float(steps_remaining)
    jump_mean = config.jump_intensity * horizon
    weights = _poisson_weights(jump_mean, config.poisson_tail_mass)

    probability = 0.0
    for n_jumps, weight in enumerate(weights):
        future_mean = n_jumps * config.jump_mean
        future_variance = (
            horizon * config.diffusion_sigma * config.diffusion_sigma
            + n_jumps * config.jump_sigma * config.jump_sigma
        )
        shifted_score = score - config.terminal_threshold + future_mean

        if future_variance <= 0.0:
            conditional = 1.0 if shifted_score > 0.0 else 0.0
        else:
            conditional = standard_normal_cdf(shifted_score / math.sqrt(future_variance))

        probability += weight * conditional

    return min(1.0, max(0.0, probability))


def _sample_poisson(rng: random.Random, mean: float) -> int:
    """Knuth Poisson sampler for the small per-step means used here."""

    if mean <= 0.0:
        return 0

    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


@dataclass
class JumpDiffusionScoreProcess:
    """Latent score process used by the simulator and arbitrageur."""

    config: JumpDiffusionConfig
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._score = self.config.initial_score
        self._step = 0

    @property
    def current_score(self) -> float:
        return self._score

    @property
    def current_step(self) -> int:
        return self._step

    def steps_remaining(self) -> int:
        return max(0, self.config.n_steps - self._step)

    def current_true_probability(self) -> float:
        return true_probability(self._score, self.steps_remaining(), self.config)

    def step(self) -> float:
        """Advance one step and return the new score."""

        diffusion_move = self._rng.gauss(0.0, self.config.diffusion_sigma)
        jump_count = _sample_poisson(self._rng, self.config.jump_intensity)
        jump_move = 0.0
        for _ in range(jump_count):
            jump_move += self._rng.gauss(self.config.jump_mean, self.config.jump_sigma)

        self._score += diffusion_move + jump_move
        self._step += 1
        return self._score
