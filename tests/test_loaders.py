"""Loader and trade-log cleaning tests."""

import pandas as pd
import pytest

from trademetrics.loaders import clean_trades, load_benchmarks, load_nav, load_trades

SAMPLES = "data/samples"


def test_clean_trades_maps_action_codes():
    df = pd.DataFrame({
        "datetime": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "ticker": ["A", "A", "B", "B"],
        "action": ["BOT", "SLD", "BUY", "SELL"],
        "qty": [1, 1, 1, 1],
        "price": [10.0, 11.0, 12.0, 13.0],
        "commission": [1.0, 1.0, 1.0, 1.0],
    })
    out = clean_trades(df)
    assert list(out["side"]) == [1, -1, 1, -1]


def test_clean_trades_drops_unrecognised_actions(capsys):
    df = pd.DataFrame({
        "datetime": ["2024-01-02", "2024-01-03"],
        "ticker": ["A", "A"],
        "action": ["BOT", "TRANSFER"],
        "qty": [1, 1],
        "price": [10.0, 11.0],
        "commission": [0.0, 0.0],
    })
    out = clean_trades(df)
    assert len(out) == 1
    assert "dropped 1 row" in capsys.readouterr().out


def test_clean_trades_sorts_chronologically():
    df = pd.DataFrame({
        "datetime": ["2024-03-01", "2024-01-02", "2024-02-01"],
        "ticker": ["A", "A", "A"],
        "action": ["SLD", "BOT", "BOT"],
        "qty": [1, 1, 1],
        "price": [10.0, 11.0, 12.0],
        "commission": [0.0, 0.0, 0.0],
    })
    out = clean_trades(df)
    assert list(out["datetime"]) == sorted(out["datetime"])


def test_clean_trades_defaults_missing_commission_to_zero():
    df = pd.DataFrame({
        "datetime": ["2024-01-02"],
        "ticker": ["A"],
        "action": ["BOT"],
        "qty": [1],
        "price": [10.0],
    })
    out = clean_trades(df)
    assert out["commission"].iloc[0] == 0.0


def test_clean_trades_fills_nan_commission():
    df = pd.DataFrame({
        "datetime": ["2024-01-02"],
        "ticker": ["A"],
        "action": ["BOT"],
        "qty": [1],
        "price": [10.0],
        "commission": [None],
    })
    assert clean_trades(df)["commission"].iloc[0] == 0.0


def test_sample_nav_loads_as_sorted_date_indexed_series():
    nav = load_nav(f"{SAMPLES}/nav_sample.csv")
    assert isinstance(nav.index, pd.DatetimeIndex)
    assert nav.index.is_monotonic_increasing
    assert len(nav) == 261
    assert nav.notna().all()


def test_sample_trades_load_and_clean():
    trades = load_trades(f"{SAMPLES}/trades_sample.csv")
    assert set(trades["side"].unique()) <= {1, -1}
    assert (trades["qty"] > 0).all()
    assert trades["datetime"].is_monotonic_increasing


def test_sample_benchmarks_cover_the_nav_period():
    nav = load_nav(f"{SAMPLES}/nav_sample.csv")
    bench = load_benchmarks(f"{SAMPLES}/benchmarks_sample.csv")
    assert list(bench.columns) == ["SPY", "QQQ"]
    assert bench.index.equals(nav.index)


def test_sample_data_is_reproducible_from_the_generator():
    """The committed CSVs must be exactly what tools/make_sample_data.py emits."""
    from trademetrics.simulate import sample_trade_log, simulate_market

    nav_generated, bench_generated = simulate_market()
    nav_committed = load_nav(f"{SAMPLES}/nav_sample.csv")
    pd.testing.assert_series_equal(
        nav_generated.round(2), nav_committed, check_names=False, check_freq=False
    )
    bench_committed = load_benchmarks(f"{SAMPLES}/benchmarks_sample.csv")
    pd.testing.assert_frame_equal(
        bench_generated.round(2), bench_committed, check_freq=False
    )
    trades_committed = pd.read_csv(f"{SAMPLES}/trades_sample.csv")
    pd.testing.assert_frame_equal(sample_trade_log(), trades_committed)


def test_load_nav_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_nav("data/samples/does_not_exist.csv")
