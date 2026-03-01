import pandas as pd

from src.strategy.momentum import momentum_signal
from src.backtest.engine import SimpleBacktester, BacktestConfig
from src.reporting.metrics import compute_metrics
from src.reporting.trades import build_trade_log


def run_momentum_sweep(
        df: pd.DataFrame,
        lookbacks: list[int],
        cfg: BacktestConfig,
        threshold: float = 0.0,
) -> pd.DataFrame:
    rows = []

    for lb in lookbacks:
        sig = momentum_signal(df["close"], lookback=lb, threshold=threshold)

        bt = SimpleBacktester(df, cfg)
        res = bt.run(sig)

        metrics = compute_metrics(res)
        trades = build_trade_log(res)

        metrics["lookback"] = lb
        metrics["Trades (closed)"] = len(trades)

        rows.append(metrics)

    out = pd.DataFrame(rows).set_index("lookback").sort_index()
    return out