from __future__ import annotations

import numpy as np

from utils.math_utils import seeded_rng


def down_and_out_call_mc(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    barrier: float,
    maturity: float,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
) -> tuple[float, float]:
    if barrier >= spot:
        raise ValueError("For a down-and-out call, barrier should be below spot.")

    dt = maturity / n_steps
    drift = (rate - dividend_yield - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    rng = seeded_rng(seed)
    z = rng.standard_normal(size=(n_sims, n_steps))

    log_s = np.empty((n_sims, n_steps + 1), dtype=float)
    log_s[:, 0] = np.log(spot)
    log_s[:, 1:] = log_s[:, [0]] + np.cumsum(drift + vol * z, axis=1)

    s_path = np.exp(log_s)

    knocked_out = np.any(s_path <= barrier, axis=1)
    s_t = s_path[:, -1]

    payoff = np.maximum(s_t - strike, 0.0)
    payoff[knocked_out] = 0.0

    disc_payoff = np.exp(-rate * maturity) * payoff
    price = float(np.mean(disc_payoff))
    se = float(np.std(disc_payoff, ddof=1) / np.sqrt(n_sims))
    return price, se
