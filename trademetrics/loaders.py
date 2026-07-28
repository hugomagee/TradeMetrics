"""CSV loading and trade-log cleaning.

All loaders read committed CSV files. Nothing here talks to a broker — an
optional, untested-in-repo IBKR sketch lives in ``examples/ibkr_loader.py``.
"""

from pathlib import Path

import pandas as pd

ACTION_MAP = {"BOT": 1, "BUY": 1, "SLD": -1, "SELL": -1}


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw trade log for the FIFO engine.

    - parses datetimes and sorts chronologically (stable order for ties)
    - maps action codes (BOT/BUY → +1, SLD/SELL → -1) into a ``side`` column
    - drops rows with unrecognised action codes (reported, not silent)
    - fills missing commissions with 0
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["side"] = df["action"].map(ACTION_MAP)
    invalid = int(df["side"].isna().sum())
    if invalid:
        print(f"clean_trades: dropped {invalid} row(s) with unrecognised action codes")
        df = df.dropna(subset=["side"])
    df["side"] = df["side"].astype(int)
    if "commission" not in df.columns:
        df["commission"] = 0.0
    df["commission"] = df["commission"].fillna(0.0)
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    return df


def load_trades(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean a trade-log CSV.

    Columns: datetime, ticker, action, qty, price, commission.
    """
    return clean_trades(pd.read_csv(csv_path))


def load_nav(csv_path: str | Path) -> pd.Series:
    """Load a daily NAV CSV (columns: date, nav) as a date-indexed Series."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return df["nav"].sort_index()


def load_benchmarks(csv_path: str | Path) -> pd.DataFrame:
    """Load benchmark price series (columns: date, then one column per ticker)."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return df.sort_index()
