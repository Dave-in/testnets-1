"""Индикаторы. Все считаются только по прошлым данным (без заглядывания вперед)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME_PERIODS_PER_YEAR = {
    "1h": 24 * 365,
    "2h": 12 * 365,
    "4h": 6 * 365,
    "6h": 4 * 365,
    "12h": 2 * 365,
    "1d": 365,
}


def periods_per_year(timeframe: str) -> int:
    try:
        return TIMEFRAME_PERIODS_PER_YEAR[timeframe]
    except KeyError:
        raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}") from None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def realized_vol(close: pd.Series, lookback: int, timeframe: str) -> pd.Series:
    """Годовая волатильность по лог-доходностям."""
    rets = np.log(close).diff()
    return rets.rolling(lookback, min_periods=lookback).std() * np.sqrt(periods_per_year(timeframe))
