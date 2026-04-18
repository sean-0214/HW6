from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seeded_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def summary_stats(x: np.ndarray, percentiles: Iterable[float] = (5, 50, 95)) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    stats: dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
    }
    pct = np.percentile(x, list(percentiles))
    for p, v in zip(percentiles, pct):
        stats[f"p{int(p)}"] = float(v)
    return stats


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, rate: float, sigma: float, dividend_yield: float, maturity: float) -> float:
    if maturity <= 0:
        return max(spot - strike, 0.0)
    if sigma <= 0:
        forward = spot * math.exp((rate - dividend_yield) * maturity)
        return math.exp(-rate * maturity) * max(forward - strike, 0.0)

    vol_sqrt = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * maturity) / vol_sqrt
    d2 = d1 - vol_sqrt
    return spot * math.exp(-dividend_yield * maturity) * _norm_cdf(d1) - strike * math.exp(-rate * maturity) * _norm_cdf(d2)


def black_scholes_put(spot: float, strike: float, rate: float, sigma: float, dividend_yield: float, maturity: float) -> float:
    call = black_scholes_call(spot, strike, rate, sigma, dividend_yield, maturity)
    return call - spot * math.exp(-dividend_yield * maturity) + strike * math.exp(-rate * maturity)
