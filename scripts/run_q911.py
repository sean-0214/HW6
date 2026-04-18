from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q911Params  # noqa: E402
from src.q911_down_and_out_mc import down_and_out_call_mc  # noqa: E402
from utils.math_utils import ensure_output_dir  # noqa: E402


def main() -> None:
    p = Q911Params()
    ensure_output_dir(OUTPUT_DIR)

    price, se = down_and_out_call_mc(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        sigma=p.sigma,
        dividend_yield=p.dividend_yield,
        barrier=p.barrier,
        maturity=p.maturity,
        n_steps=p.n_steps,
        n_sims=p.n_sims,
        seed=p.seed,
    )

    print("Q9.11 Down-and-out call (Monte Carlo, discrete monitoring)")
    print(f"price={price:.6f}, SE={se:.6f}")

    out = pd.DataFrame([
        {
            "spot": p.spot,
            "strike": p.strike,
            "rate": p.rate,
            "sigma": p.sigma,
            "dividend_yield": p.dividend_yield,
            "barrier": p.barrier,
            "maturity": p.maturity,
            "n_steps": p.n_steps,
            "n_sims": p.n_sims,
            "price": price,
            "standard_error": se,
        }
    ])
    out.to_csv(OUTPUT_DIR / "q911_down_and_out_mc.csv", index=False)


if __name__ == "__main__":
    main()
