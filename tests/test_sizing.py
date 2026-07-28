"""Position sizing tests: Kelly arithmetic, volatility targeting, and true ERC."""

import math

import numpy as np
import pytest

from trademetrics.sizing import PositionSizer, erc_weights, kelly_fraction


def test_kelly_fraction_hand_computed():
    # p = 0.6, avg_win = 0.20, avg_loss = 0.10 -> b = 2
    # f* = (2 * 0.6 - 0.4) / 2 = 0.8 / 2 = 0.4
    assert kelly_fraction(0.6, 0.20, 0.10) == pytest.approx(0.4, abs=1e-12)


def test_kelly_fraction_even_money_bet():
    # b = 1, p = 0.55 -> f* = (0.55 - 0.45) / 1 = 0.10
    assert kelly_fraction(0.55, 0.10, 0.10) == pytest.approx(0.10, abs=1e-12)


def test_kelly_fraction_is_zero_without_edge():
    # b = 1, p = 0.5 -> f* = 0; a negative edge clips at 0 rather than shorting
    assert kelly_fraction(0.5, 0.10, 0.10) == pytest.approx(0.0)
    assert kelly_fraction(0.3, 0.10, 0.10) == 0.0


def test_kelly_fraction_validates_inputs():
    with pytest.raises(ValueError, match="win_rate"):
        kelly_fraction(1.4, 0.2, 0.1)
    with pytest.raises(ValueError, match="positive magnitudes"):
        kelly_fraction(0.6, 0.2, 0.0)


def test_quarter_kelly_scales_full_kelly():
    sizer = PositionSizer(nav=100_000, kelly_multiplier=0.25)
    assert sizer.kelly_size(0.6, 0.20, 0.10) == pytest.approx(0.4 * 0.25)


def test_vol_target_size_hits_target_volatility():
    # target 15% vol on a 30%-vol name -> weight 0.5 (below the 5x risk cap)
    sizer = PositionSizer(nav=100_000, max_risk_per_trade=0.15)
    assert sizer.vol_target_size(ticker_vol=0.30, target_vol=0.15) == pytest.approx(0.5)


def test_vol_target_size_respects_cap():
    # 10%-vol name would want weight 1.5; cap is max_risk * 5 = 0.075
    sizer = PositionSizer(nav=100_000, max_risk_per_trade=0.015)
    assert sizer.vol_target_size(ticker_vol=0.10, target_vol=0.15) == pytest.approx(0.075)


def test_vol_target_rejects_non_positive_vol():
    with pytest.raises(ValueError, match="ticker_vol"):
        PositionSizer(nav=1000).vol_target_size(0.0)


def test_recommended_size_takes_the_most_conservative_constraint():
    sizer = PositionSizer(nav=50_000, max_risk_per_trade=0.015, kelly_multiplier=0.25)
    out = sizer.recommended_size(
        "NVDA", ticker_vol=0.38, win_rate=0.60, avg_win=0.20, avg_loss=0.10
    )
    assert out["recommended"] == pytest.approx(min(out["kelly_fraction"], out["vol_target"], 0.06))
    assert out["allocation"] == pytest.approx(out["recommended"] * 50_000)


def test_erc_weights_are_inverse_vol_for_uncorrelated_assets():
    # Uncorrelated assets with vols 10% and 20%: ERC weights ∝ 1/vol -> 2:1
    cov = np.diag([0.10**2, 0.20**2])
    w = erc_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert w[0] / w[1] == pytest.approx(2.0, rel=1e-4)


def test_erc_weights_equalise_risk_contributions():
    cov = np.array([
        [0.0400, 0.0090, 0.0060],
        [0.0090, 0.0225, 0.0045],
        [0.0060, 0.0045, 0.0100],
    ])
    w = erc_weights(cov)
    port_var = w @ cov @ w
    rc = w * (cov @ w) / port_var
    assert rc.max() - rc.min() < 1e-5
    assert rc.mean() == pytest.approx(1 / 3, rel=1e-4)


def test_erc_differs_from_vol_targeting_when_assets_are_correlated():
    """Vol targeting ignores correlation; ERC does not. They must not agree here."""
    vols = np.array([0.20, 0.20, 0.20])
    corr = np.array([
        [1.00, 0.90, 0.10],
        [0.90, 1.00, 0.10],
        [0.10, 0.10, 1.00],
    ])
    cov = np.outer(vols, vols) * corr
    w_erc = erc_weights(cov)
    w_voltarget = np.full(3, 1 / 3)  # equal vols -> vol targeting gives equal weights
    assert not np.allclose(w_erc, w_voltarget, atol=1e-3)
    # the near-duplicate pair must each get less weight than the diversifier
    assert w_erc[2] > w_erc[0]


def test_erc_rejects_non_square_covariance():
    with pytest.raises(ValueError, match="square"):
        erc_weights(np.ones((2, 3)))


def test_erc_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T / 6 + np.eye(6) * 0.01
    w = erc_weights(cov)
    assert w.min() >= -1e-9
    assert math.isclose(w.sum(), 1.0, abs_tol=1e-8)
