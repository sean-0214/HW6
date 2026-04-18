from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from config import OUTPUT_DIR, Q101Params  # noqa: E402
from src.q101_barrier_cn_compare import compare_down_and_out_mc_cn  # noqa: E402
from utils.math_utils import ensure_output_dir  # noqa: E402


def main() -> None:
    p = Q101Params()
    ensure_output_dir(OUTPUT_DIR)

    df, meta = compare_down_and_out_mc_cn(
        spot=p.spot,
        strike=p.strike,
        rate=p.rate,
        sigma=p.sigma,
        dividend_yield=p.dividend_yield,
        barrier=p.barrier,
        maturity=p.maturity,
        mc_steps=p.mc_steps,
        mc_sims=p.mc_sims,
        mc_seed=p.mc_seed,
        cn_time_steps=p.cn_time_steps,
        cn_space_steps_above=p.cn_space_steps_above,
        cn_log_dist_above=p.cn_log_dist_above,
    )

    print("Q10.1 Down-and-out call: Monte Carlo vs Crank–Nicolson")
    with pd.option_context("display.max_columns", None):
        print(df)

    out = df.copy()
    for k, v in meta.items():
        out[k] = v

    out.to_csv(OUTPUT_DIR / "q101_mc_vs_cn.csv", index=False)


if __name__ == "__main__":
    main()
