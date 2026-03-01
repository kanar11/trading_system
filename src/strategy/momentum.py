import pandas as pd

def momentum_signal(close: pd.Series, lookback:int = 20, threshold:float = 0.00) -> pd.Series:
    if lookback <= 0:
        raise ValueError("lookback must be greater than zero")

    rolling_ret = close.pct_change(lookback)

    signal = (rolling_ret > threshold).astype(int) - (rolling_ret < -threshold).astype(int)
    return signal.fillna(0).astype(int)