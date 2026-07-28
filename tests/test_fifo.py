"""FIFO engine tests — every expected value worked by hand in the comments."""

import pandas as pd
import pytest

from tests.conftest import make_trades
from trademetrics.fifo import FifoEngine


def test_simple_long_round_trip():
    # buy 10@100 (comm 1), sell 10@110 (comm 1):
    # gross = 10*(110-100) = 100; commissions = 1+1 = 2; net = 98
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 10, 100.0, 1.0),
        ("2024-01-10", "AAA", -1, 10, 110.0, 1.0),
    ])
    realised = FifoEngine(trades).realised_pnl()
    assert len(realised) == 1
    row = realised.iloc[0]
    assert row["direction"] == "LONG"
    assert row["gross_pnl"] == pytest.approx(100.0)
    assert row["commission"] == pytest.approx(2.0)
    assert row["net_pnl"] == pytest.approx(98.0)


def test_partial_close_allocates_entry_commission_pro_rata():
    # buy 10@100 (comm 1 => 0.10/share), sell 4@105 (comm 0.8):
    # gross = 4*5 = 20; entry comm share = 4*0.10 = 0.40; exit = 0.80; net = 18.80
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 10, 100.0, 1.0),
        ("2024-01-10", "AAA", -1, 4, 105.0, 0.8),
    ])
    engine = FifoEngine(trades)
    row = engine.realised_pnl().iloc[0]
    assert row["gross_pnl"] == pytest.approx(20.0)
    assert row["commission"] == pytest.approx(1.2)
    assert row["net_pnl"] == pytest.approx(18.8)
    open_pos = engine.open_positions()
    assert len(open_pos) == 1
    assert open_pos.iloc[0]["qty"] == pytest.approx(6)
    assert open_pos.iloc[0]["entry_commission"] == pytest.approx(0.6)


def test_multi_lot_fifo_order():
    # buy 5@100 (comm 1), buy 5@110 (comm 1), sell 8@120 (comm 2):
    # lot1 (oldest) closes first: gross 5*20=100, comm 5*0.20+5*0.25=2.25, net 97.75
    # lot2 partially:            gross 3*10=30,  comm 3*0.20+3*0.25=1.35, net 28.65
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 5, 100.0, 1.0),
        ("2024-01-05", "AAA", 1, 5, 110.0, 1.0),
        ("2024-01-10", "AAA", -1, 8, 120.0, 2.0),
    ])
    engine = FifoEngine(trades)
    realised = engine.realised_pnl()
    assert len(realised) == 2
    assert realised.iloc[0]["entry_price"] == pytest.approx(100.0)  # oldest lot first
    assert realised.iloc[0]["net_pnl"] == pytest.approx(97.75)
    assert realised.iloc[1]["entry_price"] == pytest.approx(110.0)
    assert realised.iloc[1]["net_pnl"] == pytest.approx(28.65)
    open_pos = engine.open_positions()
    assert open_pos.iloc[0]["qty"] == pytest.approx(2)
    assert open_pos.iloc[0]["entry_price"] == pytest.approx(110.0)


def test_short_sale_round_trip():
    # sell 10@50 (comm 1), buy 10@45 (comm 1): short profits when price falls
    # gross = 10*(50-45) = 50; net = 48
    trades = make_trades([
        ("2024-01-02", "BBB", -1, 10, 50.0, 1.0),
        ("2024-01-10", "BBB", 1, 10, 45.0, 1.0),
    ])
    realised = FifoEngine(trades).realised_pnl()
    assert len(realised) == 1
    row = realised.iloc[0]
    assert row["direction"] == "SHORT"
    assert row["gross_pnl"] == pytest.approx(50.0)
    assert row["net_pnl"] == pytest.approx(48.0)


def test_short_losing_round_trip():
    # sell 10@50, cover at 55: gross = 10*(50-55) = -50; net = -52
    trades = make_trades([
        ("2024-01-02", "BBB", -1, 10, 50.0, 1.0),
        ("2024-01-10", "BBB", 1, 10, 55.0, 1.0),
    ])
    row = FifoEngine(trades).realised_pnl().iloc[0]
    assert row["gross_pnl"] == pytest.approx(-50.0)
    assert row["net_pnl"] == pytest.approx(-52.0)


def test_partial_short_cover():
    # sell 10@50 (comm 2 => 0.20/share), buy 4@40 (comm 1 => 0.25/share):
    # gross = 4*(50-40) = 40; comm = 4*0.20 + 4*0.25 = 1.80; net = 38.20
    # remaining short: 6@50
    trades = make_trades([
        ("2024-01-02", "BBB", -1, 10, 50.0, 2.0),
        ("2024-01-10", "BBB", 1, 4, 40.0, 1.0),
    ])
    engine = FifoEngine(trades)
    row = engine.realised_pnl().iloc[0]
    assert row["gross_pnl"] == pytest.approx(40.0)
    assert row["commission"] == pytest.approx(1.8)
    assert row["net_pnl"] == pytest.approx(38.2)
    open_pos = engine.open_positions()
    assert open_pos.iloc[0]["direction"] == "SHORT"
    assert open_pos.iloc[0]["qty"] == pytest.approx(6)


def test_flip_long_to_short_in_one_execution():
    # buy 10@100 (comm 1), sell 15@110 (comm 1.5 => 0.10/share):
    # closes the 10-lot: gross 100, comm 1 + 10*0.10 = 2.00, net 98
    # opens a NEW short of 5@110 carrying 5*0.10 = 0.50 entry commission
    trades = make_trades([
        ("2024-01-02", "CCC", 1, 10, 100.0, 1.0),
        ("2024-01-10", "CCC", -1, 15, 110.0, 1.5),
    ])
    engine = FifoEngine(trades)
    realised = engine.realised_pnl()
    assert len(realised) == 1
    assert realised.iloc[0]["net_pnl"] == pytest.approx(98.0)
    open_pos = engine.open_positions()
    assert len(open_pos) == 1
    row = open_pos.iloc[0]
    assert row["direction"] == "SHORT"
    assert row["qty"] == pytest.approx(5)
    assert row["entry_price"] == pytest.approx(110.0)
    assert row["entry_commission"] == pytest.approx(0.5)


def test_sell_with_no_open_lot_opens_short_not_dropped():
    # Regression: the original engine silently ignored sells with no open lot.
    trades = make_trades([
        ("2024-01-02", "SPY", -1, 10, 500.0, 1.0),
    ])
    engine = FifoEngine(trades)
    assert engine.realised_pnl().empty
    open_pos = engine.open_positions()
    assert len(open_pos) == 1
    assert open_pos.iloc[0]["direction"] == "SHORT"


def test_tickers_are_isolated():
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 10, 100.0, 0.0),
        ("2024-01-03", "BBB", -1, 5, 50.0, 0.0),
        ("2024-01-10", "AAA", -1, 10, 110.0, 0.0),
    ])
    engine = FifoEngine(trades)
    realised = engine.realised_pnl()
    assert list(realised["ticker"]) == ["AAA"]
    assert set(engine.open_positions()["ticker"]) == {"BBB"}


def test_zero_commission_net_equals_gross():
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 10, 100.0, 0.0),
        ("2024-01-10", "AAA", -1, 10, 110.0, 0.0),
    ])
    row = FifoEngine(trades).realised_pnl().iloc[0]
    assert row["net_pnl"] == pytest.approx(row["gross_pnl"])


def test_non_positive_quantity_raises():
    trades = make_trades([("2024-01-02", "AAA", 1, 0, 100.0, 0.0)])
    with pytest.raises(ValueError, match="non-positive quantity"):
        FifoEngine(trades).realised_pnl()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        FifoEngine(pd.DataFrame({"ticker": ["AAA"]}))


def test_realised_by_ticker_aggregation():
    trades = make_trades([
        ("2024-01-02", "AAA", 1, 10, 100.0, 1.0),
        ("2024-01-05", "AAA", -1, 5, 110.0, 0.5),   # net 5*10 - 0.5 - 0.5 = 49.0
        ("2024-01-08", "AAA", -1, 5, 90.0, 0.5),    # net 5*(-10) - 0.5 - 0.5 = -51.0
    ])
    agg = FifoEngine(trades).realised_by_ticker()
    assert len(agg) == 1
    assert agg.iloc[0]["net_pnl"] == pytest.approx(-2.0)
    assert agg.iloc[0]["trades"] == 2
