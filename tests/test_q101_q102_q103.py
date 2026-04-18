import numpy as np

from src.q101_barrier_cn_compare import down_and_out_call_cn
from src.q102_up_and_out_put_cn import up_and_out_put_cn
from src.q103_european_call_explicit import european_call_explicit
from utils.math_utils import black_scholes_call, black_scholes_put


def test_cn_down_and_out_call_nonnegative_and_below_vanilla():
    spot = 50.0
    strike = 50.0
    rate = 0.05
    sigma = 0.2
    q = 0.02
    maturity = 1.0

    cn = down_and_out_call_cn(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=q,
        barrier=45.0,
        maturity=maturity,
        n_time=80,
        n_space_above=300,
        log_dist_above=5.0,
    )

    vanilla = black_scholes_call(spot, strike, rate, sigma, q, maturity)

    assert np.isfinite(cn)
    assert cn >= 0
    assert cn <= vanilla + 0.5


def test_cn_up_and_out_put_nonnegative_and_below_vanilla():
    spot = 50.0
    strike = 50.0
    rate = 0.05
    sigma = 0.2
    q = 0.02
    maturity = 1.0

    cn = up_and_out_put_cn(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=q,
        barrier=60.0,
        maturity=maturity,
        n_time=80,
        n_space_below=250,
        log_dist_below=4.0,
    )

    vanilla = black_scholes_put(spot, strike, rate, sigma, q, maturity)

    assert np.isfinite(cn)
    assert cn >= 0
    assert cn <= vanilla + 0.5


def test_explicit_fd_call_reasonable_vs_black_scholes():
    spot = 50.0
    strike = 50.0
    rate = 0.05
    sigma = 0.2
    q = 0.02
    maturity = 1.0

    fd = european_call_explicit(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        dividend_yield=q,
        maturity=maturity,
        n_time=6000,
        n_space=200,
        s_max=200.0,
    )

    bs = black_scholes_call(spot, strike, rate, sigma, q, maturity)

    assert np.isfinite(fd)
    assert fd >= 0
    assert abs(fd - bs) / (bs + 1e-8) < 0.25
