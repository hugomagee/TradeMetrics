"""Deterministic synthetic data generation with KNOWN ground truth.

Everything here is seeded (numpy ``default_rng``) and uses fixed date ranges —
no ``datetime.today()`` anywhere — so generated data is byte-identical across
runs and machines. The true population parameters are exposed as constants so
the validation notebook and tests can check that the engine's estimators
recover them.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Ground-truth parameters of the synthetic sample market (annualised).
# data/samples/*.csv are generated from these via tools/make_sample_data.py.
SAMPLE_MARKET = {
    "start": "2024-01-02",
    "end": "2024-12-31",
    "seed": 42,
    "nav0": 25_000.0,
    "spy0": 470.0,
    "qqq0": 400.0,
    "rf_ann": 0.02,             # risk-free rate the market is simulated under
    "spy_excess_mu_ann": 0.08,  # SPY excess drift (total drift = rf + this)
    "spy_vol_ann": 0.12,        # SPY volatility
    "qqq_loading": 1.15,        # QQQ beta to SPY
    "qqq_idio_vol_ann": 0.06,
    "beta": 1.30,               # portfolio beta to SPY
    "alpha_ann": 0.03,          # true portfolio CAPM alpha (3% a year)
    "idio_vol_ann": 0.08,       # portfolio idiosyncratic volatility
}


def simulate_market(params: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate a NAV series plus SPY/QQQ benchmark prices with known factor structure.

    The market obeys CAPM by construction under the stated risk-free rate:
    r_p - rf_d = alpha_d + beta * (r_spy - rf_d) + eps, eps ~ N(0, idio_vol_d),
    so regressing excess returns with rf = SAMPLE_MARKET["rf_ann"] recovers
    ``alpha_ann`` and ``beta`` up to sampling noise. Returns (nav, benchmark_prices).
    """
    p = {**SAMPLE_MARKET, **(params or {})}
    dates = pd.bdate_range(p["start"], p["end"])
    n = len(dates)
    rng = np.random.default_rng(p["seed"])
    rf_d = p["rf_ann"] / TRADING_DAYS

    spy_excess = rng.normal(
        p["spy_excess_mu_ann"] / TRADING_DAYS, p["spy_vol_ann"] / np.sqrt(TRADING_DAYS), n
    )
    spy_ret = rf_d + spy_excess
    qqq_ret = rf_d + p["qqq_loading"] * spy_excess + rng.normal(
        0.0, p["qqq_idio_vol_ann"] / np.sqrt(TRADING_DAYS), n
    )
    port_ret = (
        rf_d
        + p["alpha_ann"] / TRADING_DAYS
        + p["beta"] * spy_excess
        + rng.normal(0.0, p["idio_vol_ann"] / np.sqrt(TRADING_DAYS), n)
    )

    nav = pd.Series(p["nav0"] * np.cumprod(1 + port_ret), index=dates, name="nav")
    bench = pd.DataFrame(
        {
            "SPY": p["spy0"] * np.cumprod(1 + spy_ret),
            "QQQ": p["qqq0"] * np.cumprod(1 + qqq_ret),
        },
        index=dates,
    )
    bench.index.name = "date"
    nav.index.name = "date"
    return nav, bench


def simulate_returns(
    n: int,
    true_sharpe: float,
    vol_ann: float = 0.15,
    rf: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Daily returns whose POPULATION annualised Sharpe equals ``true_sharpe``.

    mean_daily = rf/252 + (true_sharpe/sqrt(252)) * vol_daily. Sample estimates
    will scatter around ``true_sharpe`` — that scatter is the point of the
    calibration experiments.
    """
    rng = np.random.default_rng(seed)
    vol_daily = vol_ann / np.sqrt(TRADING_DAYS)
    mean_daily = rf / TRADING_DAYS + (true_sharpe / np.sqrt(TRADING_DAYS)) * vol_daily
    return rng.normal(mean_daily, vol_daily, n)


def sample_trade_log() -> pd.DataFrame:
    """Fixed synthetic trade log exercising every FIFO code path.

    Longs with partial closes, multi-lot entries, a short hedge opened and
    partially covered, and a long→short flip in a single oversized sell.
    Prices and dates are invented; commissions are a flat 1.00 per execution.
    """
    rows = [
        # ticker, date, action, qty, price
        ("MSFT", "2024-01-15", "BOT", 20, 375.20),
        ("NVDA", "2024-01-22", "BOT", 8, 612.00),
        ("ORCL", "2024-02-05", "BOT", 30, 118.40),
        ("META", "2024-02-20", "BOT", 14, 488.00),
        ("AMD", "2024-03-04", "BOT", 25, 172.00),
        ("MSFT", "2024-03-11", "BOT", 10, 402.50),   # second lot
        ("PLTR", "2024-03-25", "BOT", 80, 18.20),
        ("GOOGL", "2024-04-08", "BOT", 16, 174.00),
        ("AMZN", "2024-04-22", "BOT", 18, 193.00),
        ("CRM", "2024-05-06", "BOT", 20, 298.00),
        ("TSLA", "2024-05-20", "BOT", 12, 241.00),
        ("SPY", "2024-06-03", "SLD", 10, 521.00),    # opens a short hedge
        ("MSFT", "2024-06-17", "SLD", 25, 441.00),   # closes lot 1 fully + 5 of lot 2
        ("AMD", "2024-07-08", "SLD", 15, 148.00),    # partial close at a loss
        ("TSLA", "2024-08-05", "SLD", 20, 210.00),   # flip: closes 12 long, opens 8 short
        ("NVDA", "2024-08-19", "SLD", 4, 890.00),    # partial close
        ("SPY", "2024-09-09", "BOT", 6, 549.00),     # covers 6 of the 10 short
        ("PLTR", "2024-09-23", "SLD", 40, 36.50),
        ("META", "2024-10-07", "SLD", 7, 583.00),
        ("TSLA", "2024-10-21", "BOT", 8, 220.00),    # covers the short flip
        ("CRM", "2024-11-04", "SLD", 10, 329.00),
        ("GOOGL", "2024-11-18", "SLD", 16, 172.50),
        ("AMZN", "2024-12-02", "SLD", 9, 207.00),
        ("ORCL", "2024-12-16", "SLD", 30, 168.00),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "datetime", "action", "qty", "price"])
    df["currency"] = "USD"
    df["commission"] = 1.00
    return df[["datetime", "ticker", "action", "qty", "price", "currency", "commission"]]
