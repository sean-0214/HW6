from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q98Params  # noqa: E402
from src.q98_hedge_simulation import simulate_hedge_costs, summarize_costs  # noqa: E402
from utils.math_utils import ensure_output_dir  # noqa: E402
from utils.plotting_utils import save_cost_histogram  # noqa: E402


def main() -> None:
    p = Q98Params()
    ensure_output_dir(OUTPUT_DIR)

    costs = simulate_hedge_costs(
        spot=p.spot,
        rate=p.rate,
        sigma1=p.sigma1,
        sigma2=p.sigma2,
        rho=p.rho,
        q1=p.q1,
        q2=p.q2,
        maturity=p.maturity,
        n_sims=p.n_sims,
        n_contracts=p.n_contracts,
        strike=p.spot,
        seed=p.seed,
    )

    stats = summarize_costs(costs)

    print("Q9.8 Hedged procurement cost simulation")
    with pd.option_context("display.max_columns", None):
        print(stats)

    stats.to_csv(OUTPUT_DIR / "q98_cost_summary.csv")
    costs.to_csv(OUTPUT_DIR / "q98_cost_paths.csv", index=False)

    save_cost_histogram(costs, OUTPUT_DIR / "q98_cost_hist.png", title="Q9.8 Hedge Cost Distributions")


if __name__ == "__main__":
    main()
