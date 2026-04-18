from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    scripts = [
        "run_q96.py",
        "run_q98.py",
        "run_q911.py",
        "run_q101.py",
        "run_q102.py",
        "run_q103.py",
    ]

    for s in scripts:
        path = repo_root / "scripts" / s
        print("=" * 70)
        print(f"Running {s} ...")
        subprocess.check_call([sys.executable, str(path)])


if __name__ == "__main__":
    main()
