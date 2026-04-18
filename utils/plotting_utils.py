from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_cost_histogram(cost_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(cost_df, bins=30, kde=True, ax=ax, multiple="layer", stat="density", common_norm=False)
    ax.set_title(title)
    ax.set_xlabel("Terminal cost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
