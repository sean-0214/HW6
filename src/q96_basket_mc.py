from __future__ import annotations

import numpy as np

from utils.math_utils import seeded_rng


def european_basket_call_mc(
    spot: np.ndarray,
    strike: float,
    rate: float,
    cov: np.ndarray,
    dividend_yield: np.ndarray,
    weights: np.ndarray,
    maturity: float,
    n_sims: int,
    seed: int | None = None,
) -> tuple[float, float]:
    spot = np.asarray(spot, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)

    dim = spot.size
    if cov.shape != (dim, dim):
        raise ValueError("cov must be a square matrix with size equal to len(spot)")

    rng = seeded_rng(seed)
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(dim, n_sims))
    shocks = chol @ z

    drift = (rate - dividend_yield - 0.5 * np.diag(cov)).reshape(-1, 1)
    st = spot.reshape(-1, 1) * np.exp(drift * maturity + np.sqrt(maturity) * shocks)

    basket = weights.reshape(1, -1) @ st
    payoff = np.maximum(basket.ravel() - strike, 0.0)

    disc_payoff = np.exp(-rate * maturity) * payoff
    price = float(np.mean(disc_payoff))
    se = float(np.std(disc_payoff, ddof=1) / np.sqrt(n_sims))
    return price, se


def european_basket_call_mc_av(
    spot: np.ndarray,
    strike: float,
    rate: float,
    cov: np.ndarray,
    dividend_yield: np.ndarray,
    weights: np.ndarray,
    maturity: float,
    n_sims: int,
    seed: int | None = None,
) -> tuple[float, float]:
    spot = np.asarray(spot, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)

    dim = spot.size
    if cov.shape != (dim, dim):
        raise ValueError("cov must be a square matrix with size equal to len(spot)")

    rng = seeded_rng(seed)
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(dim, n_sims))
    shocks = chol @ z

    drift = (rate - dividend_yield - 0.5 * np.diag(cov)).reshape(-1, 1)

    st_pos = spot.reshape(-1, 1) * np.exp(drift * maturity + np.sqrt(maturity) * shocks)
    st_neg = spot.reshape(-1, 1) * np.exp(drift * maturity + np.sqrt(maturity) * (-shocks))

    basket_pos = (weights.reshape(1, -1) @ st_pos).ravel()
    basket_neg = (weights.reshape(1, -1) @ st_neg).ravel()

    payoff_pos = np.maximum(basket_pos - strike, 0.0)
    payoff_neg = np.maximum(basket_neg - strike, 0.0)

    paired_disc_payoff = np.exp(-rate * maturity) * 0.5 * (payoff_pos + payoff_neg)

    price = float(np.mean(paired_disc_payoff))
    se = float(np.std(paired_disc_payoff, ddof=1) / np.sqrt(n_sims))
    return price, se
