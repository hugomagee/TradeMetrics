import pandas as pd
import pytest


def make_trades(rows: list[tuple]) -> pd.DataFrame:
    """Build a cleaned trade log from (datetime, ticker, side, qty, price, commission)."""
    df = pd.DataFrame(rows, columns=["datetime", "ticker", "side", "qty", "price", "commission"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@pytest.fixture
def nav_updown() -> pd.Series:
    """NAV with two drawdowns: 110->99 (-10%) then 120->90 (-25%)."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.Series([100.0, 110.0, 99.0, 105.0, 120.0, 90.0], index=dates, name="nav")
