"""OPTIONAL, UNTESTED-IN-REPO: sketch of an Interactive Brokers data loader.

This file is NOT imported by the trademetrics package, NOT covered by the test
suite, and NOT exercised in CI. It exists to show the shape an adapter would
take if you wanted to feed the engine from a live broker connection instead of
CSVs. It has no committed data behind it and no verified output.

Nothing in this repository depends on a broker connection. The engine's entire
public surface takes a NAV Series and a trades DataFrame; anything that can
produce those two objects can drive it — see trademetrics/loaders.py for the
CSV path that IS tested.

To use this you would need `pip install ib_insync` plus a running TWS or IB
Gateway with API access enabled. Treat the code below as a starting point to
verify yourself, not as a working integration.
"""

import pandas as pd

from trademetrics.loaders import clean_trades


def load_trades_from_ibkr(host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
    """Pull execution history from a running TWS/IB Gateway session.

    Untested. Returns a cleaned trade log in the format FifoEngine expects.
    """
    from ib_insync import IB, util  # imported lazily: not a project dependency

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    try:
        fills = ib.reqExecutions()
        records = [
            {
                "datetime": util.parseIBDatetime(f.execution.time),
                "ticker": f.contract.symbol,
                "action": f.execution.side,  # 'BOT' | 'SLD'
                "qty": f.execution.shares,
                "price": f.execution.price,
                "currency": f.contract.currency,
                "commission": f.commissionReport.commission,
            }
            for f in fills
        ]
    finally:
        ib.disconnect()
    return clean_trades(pd.DataFrame(records))


def load_nav_from_ibkr(*_args, **_kwargs):
    """Not implemented.

    IBKR exposes daily NAV through account-statement reports (Flex Queries)
    rather than the streaming API used above, so this would be a separate
    integration. It is deliberately left unimplemented rather than stubbed with
    something that looks like it works.
    """
    raise NotImplementedError(
        "No NAV pull is implemented. Export a NAV history CSV and use "
        "trademetrics.loaders.load_nav instead."
    )
