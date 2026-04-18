from __future__ import annotations

import numpy as np


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    n = diag.size
    if rhs.size != n:
        raise ValueError("rhs must have same length as diag")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("lower/upper must have length n-1")

    c_prime = np.empty(n - 1, dtype=float)
    d_prime = np.empty(n, dtype=float)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    denom_last = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom_last

    x = np.empty(n, dtype=float)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def crank_nicolson_log_price(
    x_grid: np.ndarray,
    maturity: float,
    n_time: int,
    rate: float,
    sigma: float,
    dividend_yield: float,
    payoff: np.ndarray,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    left_boundary = np.asarray(left_boundary, dtype=float)
    right_boundary = np.asarray(right_boundary, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("x_grid must be 1D")

    n_space = x_grid.size - 1
    if payoff.size != n_space + 1:
        raise ValueError("payoff must match x_grid length")
    if left_boundary.size != n_time + 1 or right_boundary.size != n_time + 1:
        raise ValueError("boundary arrays must have length n_time+1")

    dx = float(x_grid[1] - x_grid[0])
    if not np.allclose(np.diff(x_grid), dx):
        raise ValueError("x_grid must be uniform")

    dt = maturity / n_time

    nu = rate - dividend_yield - 0.5 * sigma * sigma

    alpha = 0.5 * sigma * sigma / (dx * dx) - nu / (2.0 * dx)
    beta = -sigma * sigma / (dx * dx) - rate
    gamma = 0.5 * sigma * sigma / (dx * dx) + nu / (2.0 * dx)

    n_int = n_space - 1

    lower_lhs = (-0.5 * dt * alpha) * np.ones(n_int - 1)
    diag_lhs = (1.0 - 0.5 * dt * beta) * np.ones(n_int)
    upper_lhs = (-0.5 * dt * gamma) * np.ones(n_int - 1)

    lower_rhs = (0.5 * dt * alpha)
    diag_rhs = (1.0 + 0.5 * dt * beta)
    upper_rhs = (0.5 * dt * gamma)

    v = payoff.copy()

    for k in range(0, n_time):
        v[0] = left_boundary[k]
        v[-1] = right_boundary[k]

        rhs = np.empty(n_int, dtype=float)
        for j in range(1, n_space):
            rhs[j - 1] = (
                lower_rhs * v[j - 1]
                + diag_rhs * v[j]
                + upper_rhs * v[j + 1]
            )

        rhs[0] -= (-0.5 * dt * alpha) * left_boundary[k + 1]
        rhs[-1] -= (-0.5 * dt * gamma) * right_boundary[k + 1]

        v_int = thomas_solve(lower_lhs, diag_lhs, upper_lhs, rhs)

        v[0] = left_boundary[k + 1]
        v[-1] = right_boundary[k + 1]
        v[1:n_space] = v_int

    return v
