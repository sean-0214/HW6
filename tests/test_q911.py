import numpy as np

from src.q911_down_and_out_mc import down_and_out_call_mc
from utils.math_utils import black_scholes_call


def test_q911_barrier_call_below_vanilla_call():
    spot = 50.0
    strike = 50.0
    rate = 0.05
    sigma = 0.2
    q = 0.02
    maturity = 1.0

    barrier_price, se = down_and_out_call_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=q,
        barrier=45.0,
        maturity=maturity,
        n_steps=200,
        n_sims=20_000,
        seed=2024,
    )

    vanilla = black_scholes_call(spot, strike, rate, sigma, q, maturity)

    assert np.isfinite(barrier_price)
    assert se >= 0
    assert barrier_price >= 0
    assert barrier_price <= vanilla + 0.25
