"""Long/short FIFO position matching with commission-adjusted P&L.

Every execution either closes existing lots (oldest first) or opens a new lot:

- A BUY first covers open short lots (oldest first); any remainder opens a long lot.
- A SELL first closes open long lots (oldest first); any remainder opens a short lot.

Commissions are allocated per share on both legs: a realised round-trip carries
its share of the opening commission and of the closing commission, so
``net_pnl = gross_pnl - entry_commission - exit_commission``.

A single execution may therefore both close a position and open one in the
opposite direction (a "flip"), producing one realised record and one new lot.
"""

from collections import deque
from dataclasses import dataclass

import pandas as pd

LONG = 1
SHORT = -1


@dataclass
class Lot:
    """An open lot: ``qty`` shares (always positive) at ``price``.

    ``commission_per_share`` is the opening commission divided by the opening
    quantity, so partial closes carry a proportional share of it.
    """

    qty: float
    price: float
    commission_per_share: float
    date: pd.Timestamp
    direction: int  # LONG or SHORT


class FifoEngine:
    """Match a cleaned trade log into realised round-trips and open positions.

    Parameters
    ----------
    trades : pd.DataFrame
        Cleaned trade log with columns ``datetime, ticker, side (+1/-1), qty,
        price, commission`` (see :func:`trademetrics.loaders.clean_trades`).
        Quantities are positive; ``side`` carries direction.
    """

    def __init__(self, trades: pd.DataFrame):
        required = {"datetime", "ticker", "side", "qty", "price", "commission"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"trade log missing columns: {sorted(missing)}")
        self.trades = trades.sort_values("datetime", kind="stable").reset_index(drop=True)
        self._realised: pd.DataFrame | None = None
        self._open: pd.DataFrame | None = None

    def _run(self) -> None:
        realised: list[dict] = []
        books: dict[str, deque[Lot]] = {}

        for row in self.trades.itertuples(index=False):
            book = books.setdefault(row.ticker, deque())
            side = int(row.side)
            remaining = float(row.qty)
            if remaining <= 0:
                raise ValueError(f"non-positive quantity {row.qty} for {row.ticker}")
            comm_per_share = float(row.commission) / remaining

            # Close opposing lots first, oldest first.
            while remaining > 0 and book and book[0].direction != side:
                lot = book[0]
                matched = min(remaining, lot.qty)
                # lot.direction is +1 when selling out of a long,
                # -1 when buying back a short.
                gross = matched * (float(row.price) - lot.price) * lot.direction
                entry_comm = matched * lot.commission_per_share
                exit_comm = matched * comm_per_share
                realised.append(
                    {
                        "ticker": row.ticker,
                        "direction": "LONG" if lot.direction == LONG else "SHORT",
                        "qty": matched,
                        "entry_date": lot.date,
                        "entry_price": lot.price,
                        "exit_date": row.datetime,
                        "exit_price": float(row.price),
                        "gross_pnl": gross,
                        "commission": entry_comm + exit_comm,
                        "net_pnl": gross - entry_comm - exit_comm,
                    }
                )
                lot.qty -= matched
                remaining -= matched
                if lot.qty == 0:
                    book.popleft()

            # Any remainder opens a new lot in the trade's direction.
            if remaining > 0:
                book.append(
                    Lot(
                        qty=remaining,
                        price=float(row.price),
                        commission_per_share=comm_per_share,
                        date=row.datetime,
                        direction=side,
                    )
                )

        self._realised = pd.DataFrame(
            realised,
            columns=[
                "ticker", "direction", "qty", "entry_date", "entry_price",
                "exit_date", "exit_price", "gross_pnl", "commission", "net_pnl",
            ],
        )
        open_rows = [
            {
                "ticker": ticker,
                "direction": "LONG" if lot.direction == LONG else "SHORT",
                "qty": lot.qty,
                "entry_date": lot.date,
                "entry_price": lot.price,
                "entry_commission": lot.qty * lot.commission_per_share,
            }
            for ticker, book in books.items()
            for lot in book
        ]
        self._open = pd.DataFrame(
            open_rows,
            columns=["ticker", "direction", "qty", "entry_date", "entry_price", "entry_commission"],
        )

    def realised_pnl(self) -> pd.DataFrame:
        """One row per realised (possibly partial) round-trip, in close order."""
        if self._realised is None:
            self._run()
        return self._realised.copy()

    def open_positions(self) -> pd.DataFrame:
        """Open lots remaining after all executions are processed."""
        if self._open is None:
            self._run()
        return self._open.copy()

    def realised_by_ticker(self) -> pd.DataFrame:
        """Net realised P&L aggregated per ticker, best first."""
        realised = self.realised_pnl()
        if realised.empty:
            return pd.DataFrame(columns=["ticker", "net_pnl", "gross_pnl", "commission", "trades"])
        grouped = realised.groupby("ticker").agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            commission=("commission", "sum"),
            trades=("net_pnl", "size"),
        )
        return grouped.sort_values("net_pnl", ascending=False).reset_index()
