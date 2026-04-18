from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q103Params  # noqa: E402
from src.q103_european_call_explicit import european_call_explicit  # noqa: E402
from utils.math_utils import black_scholes_call, ensure_output_dir  # noqa: E402


def main() -> None:
    p = Q103Params()
    ensure_output_dir(OUTPUT_DIR)

    price_fd = european_call_explicit(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        sigma=p.sigma,
        dividend_yield=p.dividend_yield,
        maturity=p.maturity,
        n_time=p.n_time,
        n_space=p.n_space,
        s_max=p.s_max,
    )

    price_bs = black_scholes_call(p.spot, p.strike, p.rate, p.sigma, p.dividend_yield, p.maturity)

    print("Q10.3 European call (Explicit finite difference)")
    print(f"FD price={price_fd:.6f}")
    print(f"BS price={price_bs:.6f}")

    out = pd.DataFrame([
        {
            "spot": p.spot,
            "strike": p.strike,
            "rate": p.rate,
            "sigma": p.sigma,
            "dividend_yield": p.dividend_yield,
            "maturity": p.maturity,
            "n_time": p.n_time,
            "n_space": p.n_space,
            "s_max": p.s_max,
            "price_fd": price_fd,
            "price_bs": price_bs,
        }
    ])
    out.to_csv(OUTPUT_DIR / "q103_call_explicit_fd.csv", index=False)


if __name__ == "__main__":
    main()
