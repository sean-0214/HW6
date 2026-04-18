from __future__ import annotations

import numpy as np
import pandas as pd

from utils.math_utils import seeded_rng, summary_stats


def simulate_hedge_costs(
    spot: float,
    rate: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    q1: float,
    q2: float,
    maturity: float,
    n_sims: int,
    n_contracts: int = 100,
    strike: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    if strike is None:
        strike = float(spot)

    cov = np.array(
        [[sigma1 * sigma1, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 * sigma2]],
        dtype=float,
    )
    chol = np.linalg.cholesky(cov)

    rng = seeded_rng(seed)
    z = rng.standard_normal(size=(2, n_sims))
    shocks = chol @ z

    mu = np.array([rate - q1 - 0.5 * sigma1 * sigma1, rate - q2 - 0.5 * sigma2 * sigma2]).reshape(-1, 1)
    s0 = np.array([spot, spot]).reshape(-1, 1)

    st = s0 * np.exp(mu * maturity + np.sqrt(maturity) * shocks)
    s1_t = st[0, :]
    s2_t = st[1, :]

    call1_payoff = np.maximum(s1_t - strike, 0.0)
    call2_payoff = np.maximum(s2_t - strike, 0.0)

    cost_sep_calls = n_contracts * (s1_t - call1_payoff) + n_contracts * (s2_t - call2_payoff)

    basket = 0.5 * (s1_t + s2_t)
    basket_payoff = np.maximum(basket - strike, 0.0)
    cost_basket_call = n_contracts * (s1_t + s2_t) - 2 * n_contracts * basket_payoff

    max_underlying = np.maximum(s1_t, s2_t)
    max_call_payoff = np.maximum(max_underlying - strike, 0.0)
    cost_call_on_max = n_contracts * (s1_t + s2_t) - 2 * n_contracts * max_call_payoff

    return pd.DataFrame(
        {
            "European Calls": cost_sep_calls,
            "Basket Call": cost_basket_call,
            "Call on Max": cost_call_on_max,
        }
    )


def summarize_costs(cost_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in cost_df.columns:
        stats = summary_stats(cost_df[col].to_numpy())
        rows.append({"hedge": col, **stats})
    return pd.DataFrame(rows).set_index("hedge")
