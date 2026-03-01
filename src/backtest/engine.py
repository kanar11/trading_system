from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    initial_cash: float = 10_000.0

class SimpleBacktester:
    def __init__(self, data: pd.DataFrame, config: BacktestConfig):
        if "close" not in data.columns:
            raise ValueError("data must contain 'close'")
        self.data = data.copy()
        self.cfg = config

    def run(self, signal: pd.Series) -> pd.DataFrame:
        df = self.data.copy()
        df["signal"] = signal.reindex(df.index).fillna(0).astype(int)

        df["ret"] = df["close"].pct_change().fillna(0)

        df["pos"] = df["signal"].shift(1).fillna(0).astype(int)

        df["trade"] = (df["pos"] - df["pos"].shift(1).fillna(0)).abs()

        cost_rate = (self.cfg.fee_bps + self.cfg.slippage_bps) / 10_000.0
        df["cost"] = df["trade"] * cost_rate

        df["strat_ret"] = df["pos"] * df["ret"] - df["cost"]
        df["equity"] = (1.00 + df["strat_ret"]).cumprod() * self.cfg.initial_cash

        return df