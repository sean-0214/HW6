from __future__ import annotations

import numpy as np


def european_call_explicit(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    maturity: float,
    n_time: int,
    n_space: int,
    s_max: float,
) -> float:
    if s_max <= spot:
        raise ValueError("s_max must be larger than spot")

    dt = maturity / n_time
    ds = s_max / n_space

    if sigma > 0:
        dt_max = 0.9 / (sigma * sigma * (n_space * n_space))
        if dt > dt_max:
            raise ValueError(
                "Explicit method unstable for this grid. Increase n_time or reduce n_space."
            )

    s_grid = np.linspace(0.0, s_max, n_space + 1)

    v = np.maximum(s_grid - strike, 0.0)

    for n in range(n_time - 1, -1, -1):
        t = n * dt
        tau = maturity - t

        v_new = v.copy()

        v_new[0] = 0.0
        v_new[-1] = s_max * np.exp(-dividend_yield * tau) - strike * np.exp(-rate * tau)

        j = np.arange(1, n_space)
        jj = j.astype(float)

        a = 0.5 * dt * ((sigma * sigma) * (jj * jj) - (rate - dividend_yield) * jj)
        b = 1.0 - dt * ((sigma * sigma) * (jj * jj) + rate)
        c = 0.5 * dt * ((sigma * sigma) * (jj * jj) + (rate - dividend_yield) * jj)

        v_new[j] = a * v[j - 1] + b * v[j] + c * v[j + 1]
        v = v_new

    return float(np.interp(spot, s_grid, v))
