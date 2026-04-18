import numpy as np

from src.q98_hedge_simulation import simulate_hedge_costs, summarize_costs


def test_q98_costs_finite_and_shape_ok():
    costs = simulate_hedge_costs(
        spot=50.0,
        rate=0.05,
        sigma1=0.3,
        sigma2=0.2,
        rho=0.5,
        q1=0.02,
        q2=0.01,
        maturity=0.25,
        n_sims=5_000,
        n_contracts=100,
        strike=50.0,
        seed=7,
    )

    assert costs.shape[1] == 3
    assert np.isfinite(costs.to_numpy()).all()

    stats = summarize_costs(costs)
    assert (stats["std"] >= 0).all()
