from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q96Params  # noqa: E402
from src.q96_basket_mc import european_basket_call_mc, european_basket_call_mc_av  # noqa: E402
from utils.math_utils import ensure_output_dir  # noqa: E402


def main() -> None:
    p = Q96Params()
    ensure_output_dir(OUTPUT_DIR)

    price_plain, se_plain = european_basket_call_mc(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        cov=p.cov,
        dividend_yield=p.dividend_yield,
        weights=p.weights,
        maturity=p.maturity,
        n_sims=p.n_sims,
        seed=p.seed,
    )

    price_av, se_av = european_basket_call_mc_av(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        cov=p.cov,
        dividend_yield=p.dividend_yield,
        weights=p.weights,
        maturity=p.maturity,
        n_sims=p.n_sims,
        seed=p.seed,
    )

    print("Q9.6 European basket call (Monte Carlo)")
    print(f"Plain MC:  price={price_plain:.6f}, SE={se_plain:.6f}")
    print(f"Antithetic: price={price_av:.6f}, SE={se_av:.6f}")

    df = pd.DataFrame(
        {
            "method": ["plain", "antithetic"],
            "price": [price_plain, price_av],
            "standard_error": [se_plain, se_av],
        }
    )
    df.to_csv(OUTPUT_DIR / "q96_basket_call_mc.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["method"], df["standard_error"], color=["tab:blue", "tab:orange"])
    ax.set_title("Q9.6 Standard Error Comparison")
    ax.set_ylabel("Standard error")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "q96_standard_error.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
