from __future__ import annotations

import numpy as np
import pandas as pd

from src.pde_core import crank_nicolson_log_price
from src.q911_down_and_out_mc import down_and_out_call_mc


def down_and_out_call_cn(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    barrier: float,
    maturity: float,
    n_time: int,
    n_space_above: int,
    log_dist_above: float,
) -> float:
    if barrier >= spot:
        raise ValueError("For a down-and-out call, barrier should be below spot.")

    x0 = float(np.log(spot))
    x_bar = float(np.log(barrier))

    dx_guess = log_dist_above / n_space_above
    n_left = int(np.ceil((x0 - x_bar) / dx_guess))
    dx = (x0 - x_bar) / n_left

    n_right = int(np.ceil(log_dist_above / dx))
    x_max = x0 + n_right * dx

    x_grid = x_bar + dx * np.arange(n_left + n_right + 1)
    s_grid = np.exp(x_grid)

    idx0 = n_left

    payoff = np.maximum(s_grid - strike, 0.0)
    payoff[0] = 0.0

    tau_grid = np.linspace(0.0, maturity, n_time + 1)
    left_bc = np.zeros_like(tau_grid)
    right_bc = s_grid[-1] * np.exp(-dividend_yield * tau_grid) - strike * np.exp(-rate * tau_grid)

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


def compare_down_and_out_mc_cn(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    barrier: float,
    maturity: float,
    mc_steps: int,
    mc_sims: int,
    mc_seed: int | None,
    cn_time_steps: int,
    cn_space_steps_above: int,
    cn_log_dist_above: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    mc_price, mc_se = down_and_out_call_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=dividend_yield,
        barrier=barrier,
        maturity=maturity,
        n_steps=mc_steps,
        n_sims=mc_sims,
        seed=mc_seed,
    )

    cn_price = down_and_out_call_cn(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=dividend_yield,
        barrier=barrier,
        maturity=maturity,
        n_time=cn_time_steps,
        n_space_above=cn_space_steps_above,
        log_dist_above=cn_log_dist_above,
    )

    df = pd.DataFrame(
        [
            {"method": "MC", "price": mc_price, "standard_error": mc_se},
            {"method": "CN", "price": cn_price, "standard_error": np.nan},
        ]
    )

    meta = {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "sigma": sigma,
        "dividend_yield": dividend_yield,
        "barrier": barrier,
        "maturity": maturity,
    }
    return df, meta
