"""Metric tests against hand-computed fixtures.

Expected values are derived from the formulas by hand (see comments), not by
running the engine and pasting its output.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trademetrics.metrics import MetricsEngine, sharpe_with_ci

Z95 = 1.959963984540054


def test_total_and_annualised_return():
    # NAV 100 -> 121 over 10 daily returns (11 observations).
    # total = 0.21; annualised = 1.21**(252/10) - 1
    dates = pd.bdate_range("2024-01-01", periods=11)
    nav = pd.Series(np.linspace(100.0, 121.0, 11), index=dates)
    engine = MetricsEngine(nav)
    assert engine.total_return() == pytest.approx(0.21, abs=1e-12)
    expected = 1.21 ** (252 / 10) - 1
    assert engine.annualised_return() == pytest.approx(expected, rel=1e-12)


def test_annualised_volatility_matches_sample_std():
    dates = pd.bdate_range("2024-01-01", periods=7)
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav)
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert engine.annualised_volatility() == pytest.approx(expected, rel=1e-10)


def test_sharpe_with_ci_hand_computed():
    # r = [0.01, -0.005, 0.002, 0.007, -0.001, 0.004], rf = 0, n = 6
    # mean = 0.0028333333333333335, sd(ddof=1) = 0.005419102016632153
    # SR_daily = 0.5228...; SR_ann = SR_daily * sqrt(252) = 8.299857
    # SE_daily = sqrt((1 + SR_daily^2 / 2) / 6); SE_ann = SE_daily * sqrt(252) = 6.909460
    r = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    est = sharpe_with_ci(r, rf=0.0)
    assert est.sharpe == pytest.approx(8.299857, abs=1e-6)
    assert est.se == pytest.approx(6.909460, abs=1e-6)
    assert est.ci_low == pytest.approx(8.299857 - Z95 * 6.909460, abs=1e-6)
    assert est.ci_high == pytest.approx(8.299857 + Z95 * 6.909460, abs=1e-6)
    assert est.n_obs == 6


def test_sharpe_ci_is_symmetric_around_point_estimate():
    r = np.array([0.003, -0.002, 0.004, 0.001, -0.003, 0.002, 0.005])
    est = sharpe_with_ci(r)
    assert (est.ci_high - est.sharpe) == pytest.approx(est.sharpe - est.ci_low, rel=1e-12)


def test_sharpe_se_shrinks_with_sample_size():
    # Lo SE ∝ 1/sqrt(n): quadrupling n roughly halves the standard error.
    rng = np.random.default_rng(7)
    short = rng.normal(0.0004, 0.01, 250)
    long = np.concatenate([short] * 4)
    assert sharpe_with_ci(long).se == pytest.approx(sharpe_with_ci(short).se / 2, rel=0.05)


def test_sharpe_subtracts_risk_free_rate():
    r = np.full(50, 0.001)
    r = r + np.array([0.002, -0.002] * 25)  # non-zero variance, mean unchanged
    zero_rf = sharpe_with_ci(r, rf=0.0).sharpe
    with_rf = sharpe_with_ci(r, rf=0.05).sharpe
    assert with_rf < zero_rf


def test_sharpe_rejects_zero_volatility():
    with pytest.raises(ValueError, match="zero return volatility"):
        sharpe_with_ci(np.full(10, 0.001))


def test_sharpe_rejects_tiny_sample():
    with pytest.raises(ValueError, match="at least 2"):
        sharpe_with_ci(np.array([0.01]))


def test_sortino_uses_full_sample_downside_deviation():
    # r = [0.02, -0.01, 0.01, -0.03], target = 0 (rf = 0)
    # shortfalls = [0, -0.01, 0, -0.03]
    # downside_daily = sqrt((0 + 0.0001 + 0 + 0.0009) / 4) = sqrt(0.00025) = 0.0158113883
    # excess_ann = mean(r) * 252 = -0.0025 * 252 = -0.63
    # sortino = -0.63 / (0.0158113883 * sqrt(252)) = -2.509980
    returns = np.array([0.02, -0.01, 0.01, -0.03])
    dates = pd.bdate_range("2024-01-01", periods=5)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav, rf=0.0)
    assert engine.sortino_ratio() == pytest.approx(-2.509980, abs=1e-6)


def test_sortino_differs_from_negative_subset_std():
    """The old (wrong) formula used std of the negative returns only.

    With [0.02, -0.01, 0.01, -0.03] the correct downside deviation is
    sqrt(mean over ALL 4 obs) = 0.01581; std(ddof=1) over just the two negatives
    is 0.01414. They must not coincide.
    """
    returns = np.array([0.02, -0.01, 0.01, -0.03])
    correct = math.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    old_wrong = returns[returns < 0].std(ddof=1)
    assert correct == pytest.approx(0.0158113883, abs=1e-9)
    assert not math.isclose(correct, old_wrong, rel_tol=1e-3)


def test_sortino_is_nan_without_downside():
    dates = pd.bdate_range("2024-01-01", periods=5)
    nav = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    engine = MetricsEngine(nav, rf=0.0)
    assert math.isnan(engine.sortino_ratio())


def test_max_drawdown_hand_computed(nav_updown):
    # NAV 100, 110, 99, 105, 120, 90 -> worst peak-to-trough is 120 -> 90 = -25%
    engine = MetricsEngine(nav_updown)
    assert engine.max_drawdown() == pytest.approx(-0.25, abs=1e-12)


def test_drawdown_series_is_zero_at_new_highs(nav_updown):
    dd = MetricsEngine(nav_updown).drawdown_series()
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)   # 110 is a new high
    assert dd.iloc[2] == pytest.approx(-0.1)  # 99 vs peak 110
    assert dd.iloc[-1] == pytest.approx(-0.25)


def test_calmar_ratio_is_annualised_return_over_max_drawdown(nav_updown):
    engine = MetricsEngine(nav_updown)
    expected = engine.annualised_return() / 0.25
    assert engine.calmar_ratio() == pytest.approx(expected, rel=1e-12)


def test_var_parametric_hand_computed():
    # r as in the Sharpe fixture: mean 0.0028333333, sd 0.0054191020
    # VaR95 = -(mu + norm.ppf(0.05) * sd) = -(0.0028333333 - 1.6448536 * 0.0054191020)
    #       = 0.00608030
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    dates = pd.bdate_range("2024-01-01", periods=7)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav)
    assert engine.var_parametric(0.95) == pytest.approx(0.00608030, abs=1e-8)
    assert engine.var_parametric(0.99) == pytest.approx(0.00977338, abs=1e-8)


def test_var_parametric_includes_mean_term():
    """Shifting every return up by a constant must reduce VaR by that constant."""
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    dates = pd.bdate_range("2024-01-01", periods=7)
    nav_a = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    nav_b = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns + 0.01]), index=dates)
    var_a = MetricsEngine(nav_a).var_parametric(0.95)
    var_b = MetricsEngine(nav_b).var_parametric(0.95)
    assert var_b == pytest.approx(var_a - 0.01, abs=1e-9)


def test_var_supports_arbitrary_confidence_levels():
    """Regression: the original code had a hardcoded {0.95, 0.99} lookup that
    silently fell back to the 95% z-score for anything else."""
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    dates = pd.bdate_range("2024-01-01", periods=7)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav)
    v90, v95, v975 = (engine.var_parametric(c) for c in (0.90, 0.95, 0.975))
    assert v90 < v95 < v975


def test_var_historical_is_empirical_quantile():
    # 5% quantile of the six returns (numpy linear interpolation) = -0.004
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.001, 0.004])
    dates = pd.bdate_range("2024-01-01", periods=7)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav)
    assert engine.var_historical(0.95) == pytest.approx(0.004, abs=1e-9)


def test_historical_var_exceeds_parametric_on_fat_left_tail():
    rng = np.random.default_rng(3)
    returns = np.concatenate([rng.normal(0.0005, 0.004, 400), np.full(10, -0.08)])
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    engine = MetricsEngine(nav)
    assert engine.var_historical(0.99) > engine.var_parametric(0.99)


def test_rolling_sharpe_window_length():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.008, 120)
    dates = pd.bdate_range("2024-01-01", periods=121)
    nav = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
    rolling = MetricsEngine(nav).rolling_sharpe(window=63)
    assert rolling.isna().sum() == 62
    assert len(rolling) == 120


def test_trade_statistics_from_realised_pnl():
    realised = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "net_pnl": [100.0, -50.0, 25.0, -25.0],
        "entry_date": pd.to_datetime(["2024-01-01"] * 4),
        "exit_date": pd.to_datetime(["2024-01-11", "2024-01-21", "2024-01-06", "2024-01-11"]),
    })
    dates = pd.bdate_range("2024-01-01", periods=5)
    nav = pd.Series([100.0, 101.0, 99.0, 103.0, 102.0], index=dates)
    engine = MetricsEngine(nav, realised_pnl=realised)
    assert engine.win_rate() == pytest.approx(0.5)
    assert engine.profit_factor() == pytest.approx(125.0 / 75.0)
    assert engine.avg_holding_days() == pytest.approx((10 + 20 + 5 + 10) / 4)


def test_trade_statistics_require_realised_pnl(nav_updown):
    engine = MetricsEngine(nav_updown)
    with pytest.raises(ValueError, match="no realised P&L"):
        engine.win_rate()


def test_summary_always_carries_sharpe_ci(nav_updown):
    summary = MetricsEngine(nav_updown).summary()
    for key in ("sharpe", "sharpe_se", "sharpe_ci_low", "sharpe_ci_high", "n_obs"):
        assert key in summary
    assert summary["sharpe_ci_low"] < summary["sharpe"] < summary["sharpe_ci_high"]


def test_engine_rejects_too_few_nav_points():
    dates = pd.bdate_range("2024-01-01", periods=2)
    with pytest.raises(ValueError, match="at least 3 NAV"):
        MetricsEngine(pd.Series([100.0, 101.0], index=dates))
