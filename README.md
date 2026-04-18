# EF4820 Assignment 6 (Numerical Methods) — Clean Python Repo

This repository reproduces the *course-style* numerical-method workflow for EF4820 Assignment 6 Problems **9.6, 9.8, 9.11, 10.1, 10.2, 10.3**.

It is based on the same intended solution flow as the provided `HW6_Python.py` reference, but written with slightly cleaned up structure and original parameter naming.

## Problem map

- **Q9.6** `src/q96_basket_mc.py`  
  Monte Carlo pricing of a **European basket call** (plain MC + **antithetic variates**) using **Cholesky** to generate correlated shocks.

- **Q9.8** `src/q98_hedge_simulation.py`  
  Simulation of end-of-quarter procurement cost of buying **100 units each of two assets**, comparing hedges:
  - separate European calls
  - a 50/50 basket call
  - a call on the maximum

- **Q9.11** `src/q911_down_and_out_mc.py`  
  Monte Carlo pricing of a **discretely monitored down-and-out call**, knocking out a path if it touches/breaches the barrier at any monitoring time.

- **Q10.1** `src/q101_barrier_cn_compare.py`  
  Compare down-and-out call values from **Monte Carlo (Q9.11)** and **Crank–Nicolson (CN)**.

- **Q10.2** `src/q102_up_and_out_put_cn.py`  
  **Up-and-out put** priced with **Crank–Nicolson**, using the same course-style helper logic as Q10.1.

- **Q10.3** `src/q103_european_call_explicit.py`  
  **European call** priced with the **explicit finite difference** method for the Black–Scholes PDE.

Shared PDE helper logic is in `src/pde_core.py`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run each question

All scripts write tables/figures into the `output/` folder.

```bash
python scripts/run_q96.py
python scripts/run_q98.py
python scripts/run_q911.py
python scripts/run_q101.py
python scripts/run_q102.py
python scripts/run_q103.py
```

Run everything:

```bash
python scripts/run_all.py
```

## Tests

```bash
pytest -q
```

## Notes on “faithful but cleaned up”

- The Monte Carlo questions keep the same structure as the assignment flow: simulate correlated GBM terminal prices (or paths), compute discounted payoffs/costs, report **estimate + standard error**.
- The PDE questions follow the same *helper-style* CN logic (log-space grid, coefficient construction, backward stepping, and a tridiagonal solver), rather than introducing a different PDE framework.
- Code is modular enough to read question-by-question, but not abstracted into a large library.
