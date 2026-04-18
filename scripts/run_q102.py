from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q102Params  # noqa: E402
from src.q102_up_and_out_put_cn import up_and_out_put_cn  # noqa: E402
from utils.math_utils import ensure_output_dir  # noqa: E402


def main() -> None:
    p = Q102Params()
    ensure_output_dir(OUTPUT_DIR)

    price = up_and_out_put_cn(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        sigma=p.sigma,
        dividend_yield=p.dividend_yield,
        barrier=p.barrier,
        maturity=p.maturity,
        n_time=p.cn_time_steps,
        n_space_below=p.cn_space_steps_below,
        log_dist_below=p.cn_log_dist_below,
    )

    print("Q10.2 Up-and-out put (Crank–Nicolson)")
    print(f"price={price:.6f}")

    out = pd.DataFrame([
        {
            "spot": p.spot,
            "strike": p.strike,
            "rate": p.rate,
            "sigma": p.sigma,
            "dividend_yield": p.dividend_yield,
            "barrier": p.barrier,
            "maturity": p.maturity,
            "n_time": p.cn_time_steps,
            "n_space_below": p.cn_space_steps_below,
            "log_dist_below": p.cn_log_dist_below,
            "price": price,
        }
    ])
    out.to_csv(OUTPUT_DIR / "q102_up_and_out_put_cn.csv", index=False)


if __name__ == "__main__":
    main()
