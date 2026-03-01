import numpy as np
import pandas as pd

def compute_metrics(result: pd_DataFrame) -> dict:
    strat_ret = result["strat_ret"]
    equity = result["equity"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = len(result)

    cagr = (1 + total_return) ** (252 / n_days) - 1

    std = strat_ret.std()
    sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std()

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())

    trades = int((result["trade"] > 0).sum())

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Trades": trades,
    }