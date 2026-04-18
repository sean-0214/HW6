from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"


@dataclass(frozen=True)
class Q96Params:
    spot: np.ndarray = field(default_factory=lambda: np.array([100.0, 50.0, 100.0]))
    strike: float = 90.0
    rate: float = 0.05
    dividend_yield: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.02, 0.01]))
    weights: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.4, 0.4]))
    cov: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.09, 0.01, -0.02], [0.01, 0.08, -0.01], [-0.02, -0.01, 0.07]]
        )
    )
    maturity: float = 1.0
    n_sims: int = 10_000
    seed: int = 123


@dataclass(frozen=True)
class Q98Params:
    spot: float = 50.0
    rate: float = 0.05
    sigma1: float = 0.3
    sigma2: float = 0.2
    rho: float = 0.5
    q1: float = 0.02
    q2: float = 0.01
    maturity: float = 0.25
    n_contracts: int = 100
    n_sims: int = 20_000
    seed: int = 7


@dataclass(frozen=True)
class Q911Params:
    spot: float = 50.0
    strike: float = 50.0
    rate: float = 0.05
    sigma: float = 0.2
    dividend_yield: float = 0.02
    barrier: float = 45.0
    maturity: float = 1.0
    n_steps: int = 200
    n_sims: int = 50_000
    seed: int = 2024


@dataclass(frozen=True)
class Q101Params:
    spot: float = 50.0
    strike: float = 50.0
    rate: float = 0.05
    sigma: float = 0.2
    dividend_yield: float = 0.02
    barrier: float = 45.0
    maturity: float = 1.0
    mc_steps: int = 200
    mc_sims: int = 50_000
    mc_seed: int = 2024
    cn_time_steps: int = 100
    cn_space_steps_above: int = 500
    cn_log_dist_above: float = 5.0


@dataclass(frozen=True)
class Q102Params:
    spot: float = 50.0
    strike: float = 50.0
    rate: float = 0.05
    sigma: float = 0.2
    dividend_yield: float = 0.02
    barrier: float = 60.0
    maturity: float = 1.0
    cn_time_steps: int = 100
    cn_space_steps_below: int = 200
    cn_log_dist_below: float = 4.0


@dataclass(frozen=True)
class Q103Params:
    spot: float = 50.0
    strike: float = 50.0
    rate: float = 0.05
    sigma: float = 0.2
    dividend_yield: float = 0.02
    maturity: float = 1.0
    n_time: int = 6_000
    n_space: int = 200
    s_max: float = 200.0

