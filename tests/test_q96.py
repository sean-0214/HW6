import numpy as np

from src.q96_basket_mc import european_basket_call_mc, european_basket_call_mc_av


def test_q96_runs_and_av_reduces_se_typically():
    spot = np.array([100.0, 50.0, 100.0])
    strike = 90.0
    rate = 0.05
    dividend_yield = np.array([0.01, 0.02, 0.01])
    weights = np.array([0.2, 0.4, 0.4])
    cov = np.array([[0.09, 0.01, -0.02], [0.01, 0.08, -0.01], [-0.02, -0.01, 0.07]])
    maturity = 1.0

    price_plain, se_plain = european_basket_call_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        cov=cov,
        dividend_yield=dividend_yield,
        weights=weights,
        maturity=maturity,
        n_sims=20_000,
        seed=123,
    )

    price_av, se_av = european_basket_call_mc_av(
        spot=spot,
        strike=strike,
        rate=rate,
        cov=cov,
        dividend_yield=dividend_yield,
        weights=weights,
        maturity=maturity,
        n_sims=20_000,
        seed=123,
    )

    assert np.isfinite(price_plain)
    assert np.isfinite(price_av)
    assert se_plain >= 0
    assert se_av >= 0

    assert se_av <= 1.10 * se_plain
