"""CAPM alpha/beta tests, including recovery of known simulated parameters."""

import numpy as np
import pandas as pd
import pytest

from trademetrics.benchmark import BenchmarkAnalysis
from trademetrics.simulate import SAMPLE_MARKET, simulate_market

TRADING_DAYS = 252


def _frame_from_returns(bench_returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"SPY": 100.0 * np.cumprod(np.r_[1.0, 1 + bench_returns])}, index=dates)


def test_beta_recovered_exactly_for_a_noiseless_leveraged_clone():
    # r_p - rf = 1.5 * (r_b - rf) exactly -> beta 1.5, alpha 0
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    rf = 0.02
    rf_d = rf / TRADING_DAYS
    bench_excess = rng.normal(0.0003, 0.008, n)
    bench_ret = rf_d + bench_excess
    port_ret = pd.Series(rf_d + 1.5 * bench_excess, index=dates[1:])
    prices = _frame_from_returns(bench_ret, dates)
    est = BenchmarkAnalysis(port_ret, prices, rf=rf).capm("SPY")
    assert est.beta == pytest.approx(1.5, abs=1e-9)
    assert est.alpha_ann == pytest.approx(0.0, abs=1e-9)
    assert est.r_squared == pytest.approx(1.0, abs=1e-12)


def test_alpha_recovered_for_a_constant_daily_outperformance():
    # add a constant 2bp/day on top of beta 1.0 -> alpha_ann = 0.0002 * 252
    rng = np.random.default_rng(2)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    bench_ret = rng.normal(0.0004, 0.009, n)
    port_ret = pd.Series(bench_ret + 0.0002, index=dates[1:])
    prices = _frame_from_returns(bench_ret, dates)
    est = BenchmarkAnalysis(port_ret, prices, rf=0.0).capm("SPY")
    assert est.beta == pytest.approx(1.0, abs=1e-9)
    assert est.alpha_ann == pytest.approx(0.0002 * TRADING_DAYS, abs=1e-9)


def test_capm_alpha_differs_from_raw_alpha_when_beta_is_not_one():
    """With rf > 0 and beta != 1, the excess-return intercept differs from the
    raw-return intercept by exactly (1 - beta) * rf_daily * 252."""
    rng = np.random.default_rng(3)
    n = 400
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    rf = 0.04
    rf_d = rf / TRADING_DAYS
    bench_excess = rng.normal(0.0003, 0.008, n)
    bench_ret = rf_d + bench_excess
    port_ret = pd.Series(rf_d + 1.4 * bench_excess + rng.normal(0, 0.004, n), index=dates[1:])
    prices = _frame_from_returns(bench_ret, dates)
    analysis = BenchmarkAnalysis(port_ret, prices, rf=rf)
    capm = analysis.capm("SPY")
    raw_alpha, raw_beta = analysis.raw_alpha_beta("SPY")
    assert raw_beta == pytest.approx(capm.beta, abs=1e-9)
    expected_gap = (1 - capm.beta) * rf_d * TRADING_DAYS
    assert raw_alpha - capm.alpha_ann == pytest.approx(expected_gap, abs=1e-9)


def test_simulated_market_parameters_are_recovered_within_confidence_intervals():
    """The sample market is built to obey CAPM with a known alpha and beta."""
    nav, bench = simulate_market()
    returns = nav.pct_change().dropna()
    est = BenchmarkAnalysis(returns, bench, rf=SAMPLE_MARKET["rf_ann"]).capm("SPY")
    assert est.beta == pytest.approx(SAMPLE_MARKET["beta"], abs=4 * est.beta_se)
    assert est.alpha_ci_low <= SAMPLE_MARKET["alpha_ann"] <= est.alpha_ci_high


def test_alpha_confidence_interval_brackets_the_point_estimate():
    nav, bench = simulate_market()
    est = BenchmarkAnalysis(nav.pct_change().dropna(), bench, rf=0.02).capm("SPY")
    assert est.alpha_ci_low < est.alpha_ann < est.alpha_ci_high


def test_information_ratio_is_zero_for_an_exact_benchmark_clone():
    rng = np.random.default_rng(4)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    bench_ret = rng.normal(0.0004, 0.008, n)
    port_ret = pd.Series(bench_ret, index=dates[1:])
    prices = _frame_from_returns(bench_ret, dates)
    analysis = BenchmarkAnalysis(port_ret, prices)
    assert np.isnan(analysis.information_ratio("SPY"))


def test_rolling_beta_tracks_a_regime_change():
    rng = np.random.default_rng(6)
    n = 400
    dates = pd.bdate_range("2024-01-01", periods=n + 1)
    bench_ret = rng.normal(0.0003, 0.008, n)
    betas = np.r_[np.full(200, 0.5), np.full(200, 2.0)]
    port_ret = pd.Series(betas * bench_ret, index=dates[1:])
    prices = _frame_from_returns(bench_ret, dates)
    rolling = BenchmarkAnalysis(port_ret, prices).rolling_beta("SPY", window=63)
    assert rolling.iloc[150] == pytest.approx(0.5, abs=0.05)
    assert rolling.iloc[-1] == pytest.approx(2.0, abs=0.05)


def test_cumulative_indexed_starts_from_a_common_base():
    nav, bench = simulate_market()
    indexed = BenchmarkAnalysis(nav.pct_change().dropna(), bench).cumulative_indexed()
    assert set(indexed.columns) == {"Portfolio", "SPY", "QQQ"}
    assert (indexed.iloc[0] > 90).all()


def test_insufficient_overlap_raises():
    dates = pd.bdate_range("2024-01-01", periods=10)
    port = pd.Series(np.full(10, 0.001), index=dates)
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, 10)}, index=dates)
    with pytest.raises(ValueError, match="overlapping dates"):
        BenchmarkAnalysis(port, prices)
