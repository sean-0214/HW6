from __future__ import annotations

import numpy as np

from src.pde_core import crank_nicolson_log_price


def up_and_out_put_cn(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    barrier: float,
    maturity: float,
    n_time: int,
    n_space_below: int,
    log_dist_below: float,
) -> float:
    if barrier <= spot:
        raise ValueError("For an up-and-out put, barrier should be above spot.")

    x0 = float(np.log(spot))
    x_bar = float(np.log(barrier))

    dx_guess = log_dist_below / n_space_below
    n_up = int(np.ceil((x_bar - x0) / dx_guess))
    dx = (x_bar - x0) / n_up

    n_down = int(np.ceil(log_dist_below / dx))
    x_min = x0 - n_down * dx

    x_grid = x_min + dx * np.arange(n_down + n_up + 1)
    s_grid = np.exp(x_grid)

    idx0 = n_down

    payoff = np.maximum(strike - s_grid, 0.0)
    payoff[-1] = 0.0

    tau_grid = np.linspace(0.0, maturity, n_time + 1)
    left_bc = strike * np.exp(-rate * tau_grid)
    right_bc = np.zeros_like(tau_grid)

    v0 = crank_nicolson_log_price(
        x_grid=x_grid,
        maturity=maturity,
        n_time=n_time,
        rate=rate,
        sigma=sigma,
        dividend_yield=dividend_yield,
        payoff=payoff,
        left_boundary=left_bc,
        right_boundary=right_bc,
    )

    return float(v0[idx0])
