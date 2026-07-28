"""Attribution tests plus an end-to-end determinism check on the demo."""

import pandas as pd
import pytest

from trademetrics.attribution import AttributionEngine
from trademetrics.demo import run_demo


@pytest.fixture
def realised():
    return pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "NVDA"],
        "net_pnl": [500.0, 200.0, -300.0, 100.0],
        "commission": [2.0, 1.0, 1.0, 1.0],
        "entry_date": pd.to_datetime(["2024-01-02"] * 4),
        "exit_date": pd.to_datetime(["2024-03-01"] * 4),
    })


@pytest.fixture
def nav():
    dates = pd.bdate_range("2024-01-01", periods=70)
    return pd.Series(
        [100.0 * (1.001 ** i) for i in range(70)], index=dates, name="nav"
    )


def test_pnl_by_ticker_aggregates_and_sorts(realised, nav):
    out = AttributionEngine(realised, nav).pnl_by_ticker()
    assert list(out["ticker"]) == ["NVDA", "MSFT", "TSLA"]
    assert out.loc[out["ticker"] == "NVDA", "net_pnl"].iloc[0] == pytest.approx(600.0)
    assert out.loc[out["ticker"] == "NVDA", "trades"].iloc[0] == 2


def test_pnl_by_sector_uses_the_supplied_map(realised, nav):
    sector_map = {"NVDA": "Semis", "MSFT": "Software", "TSLA": "Autos"}
    out = AttributionEngine(realised, nav, sector_map).pnl_by_sector()
    assert out.iloc[0]["sector"] == "Semis"
    assert out.iloc[0]["net_pnl"] == pytest.approx(600.0)


def test_unmapped_tickers_fall_into_other(realised, nav):
    out = AttributionEngine(realised, nav, {"NVDA": "Semis"}).pnl_by_ticker()
    assert set(out.loc[out["ticker"] != "NVDA", "sector"]) == {"Other"}


def test_top_contributors_and_detractors(realised, nav):
    engine = AttributionEngine(realised, nav)
    assert engine.top_contributors(1).iloc[0]["ticker"] == "NVDA"
    assert engine.top_detractors(1).iloc[0]["ticker"] == "TSLA"


def test_monthly_returns_measure_from_inception(realised, nav):
    monthly = AttributionEngine(realised, nav).returns_by_period("ME")
    # 70 business days from 2024-01-01 spans Jan, Feb, Mar, Apr
    assert len(monthly) >= 3
    compounded = (1 + monthly / 100).prod() - 1
    total = nav.iloc[-1] / nav.iloc[0] - 1
    assert compounded == pytest.approx(total, rel=1e-9)


def test_empty_realised_pnl_returns_empty_frames(nav):
    empty = pd.DataFrame(columns=["ticker", "net_pnl", "commission", "entry_date", "exit_date"])
    engine = AttributionEngine(empty, nav)
    assert engine.pnl_by_ticker().empty
    assert engine.pnl_by_sector().empty


def test_demo_is_deterministic_across_runs():
    first = run_demo()
    second = run_demo()
    assert first["summary"] == second["summary"]
    assert first["realised_by_ticker"] == second["realised_by_ticker"]


def test_demo_results_are_internally_consistent():
    results = run_demo()
    s = results["summary"]
    assert s["sharpe_ci_low"] < s["sharpe"] < s["sharpe_ci_high"]
    assert s["n_obs"] == 260
    assert s["max_drawdown"] <= 0
    assert 0 <= s["win_rate"] <= 1
    # every realised round-trip nets gross minus commission
    for trade in results["realised_trades"]:
        assert trade["net_pnl"] == pytest.approx(
            trade["gross_pnl"] - trade["commission"], abs=0.02
        )


def test_demo_period_is_fixed_not_relative_to_today():
    """Regression: the original demo used datetime.today(), so its output moved daily."""
    assert run_demo()["meta"]["period"] == "2024-01-02 to 2024-12-31"
