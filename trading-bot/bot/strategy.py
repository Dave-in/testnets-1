"""Консервативная трендовая стратегия (только спот, только лонг, без плеча).

Одна и та же логика используется и в бэктесте, и в paper/testnet/live —
чтобы результат на истории соответствовал тому, что бот делает вживую.

Идея:
  * Вход — только когда актив в устойчивом восходящем тренде:
    close > EMA(fast) > EMA(slow) и рост за momentum_lookback баров > 0.
  * Выход — трейлинг-стоп (макс. закрытие − k·ATR) или слом тренда.
  * Размер позиции — по волатильности: чем волатильнее актив, тем меньше доля.
    Весь портфель подгоняется под target_vol годовых. Остаток — в USDT.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .indicators import atr, ema, periods_per_year, realized_vol


@dataclass
class SymbolState:
    in_position: bool = False
    highest_close: float = 0.0
    stop: float = 0.0
    cooldown_left: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolState":
        return cls(**d)


def prepare(df: pd.DataFrame, cfg: StrategyConfig, timeframe: str) -> pd.DataFrame:
    """Добавляет к OHLCV колонки индикаторов."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.fast_ema)
    out["ema_slow"] = ema(out["close"], cfg.slow_ema)
    out["atr"] = atr(out, cfg.atr_period)
    out["vol"] = realized_vol(out["close"], cfg.vol_lookback, timeframe)
    out["mom"] = out["close"] / out["close"].shift(cfg.momentum_lookback) - 1
    return out


def _ready(row: pd.Series) -> bool:
    return all(
        not (isinstance(row[c], float) and math.isnan(row[c]))
        for c in ("ema_fast", "ema_slow", "atr", "vol", "mom", "close")
    )


def update_signal(row: pd.Series, st: SymbolState, cfg: StrategyConfig) -> bool:
    """Обновляет состояние актива по закрытию бара. Возвращает: хотим ли быть в лонге."""
    if not _ready(row):
        st.in_position = False
        return False

    close = float(row["close"])
    if st.in_position:
        st.highest_close = max(st.highest_close, close)
        st.stop = max(st.stop, st.highest_close - cfg.atr_stop_mult * float(row["atr"]))
        trend_broken = close < row["ema_slow"] or row["ema_fast"] < row["ema_slow"]
        if close <= st.stop or trend_broken:
            exit_position(st, cfg)
        return st.in_position

    if st.cooldown_left > 0:
        st.cooldown_left -= 1
        return False

    if close > row["ema_fast"] > row["ema_slow"] and row["mom"] > 0:
        st.in_position = True
        st.highest_close = close
        st.stop = close - cfg.atr_stop_mult * float(row["atr"])
    return st.in_position


def exit_position(st: SymbolState, cfg: StrategyConfig) -> None:
    st.in_position = False
    st.highest_close = 0.0
    st.stop = 0.0
    st.cooldown_left = cfg.reentry_cooldown_bars


def size_weights(
    vols: dict[str, float],
    returns: pd.DataFrame,
    caps: dict[str, float],
    cfg: StrategyConfig,
    timeframe: str,
    max_gross: float,
) -> dict[str, float]:
    """Веса для активов, по которым есть сигнал на лонг.

    vols    — годовая волатильность каждого актива с сигналом
    returns — лог-доходности за последние vol_lookback баров (для корреляций)
    """
    if not vols:
        return {}
    w = {s: min(caps[s], cfg.target_vol / v) if v > 0 else 0.0 for s, v in vols.items()}

    # Учитываем корреляцию: если активы ходят вместе, суммарный риск выше.
    syms = [s for s in w if w[s] > 0]
    if len(syms) > 1:
        sub = returns[syms].dropna()
        if len(sub) >= 5:
            cov = sub.cov().to_numpy() * periods_per_year(timeframe)
            vec = np.array([w[s] for s in syms])
            port_vol = float(np.sqrt(vec @ cov @ vec))
            if port_vol > cfg.target_vol:
                k = cfg.target_vol / port_vol
                w = {s: x * k for s, x in w.items()}

    gross = sum(w.values())
    if gross > max_gross:
        w = {s: x * max_gross / gross for s, x in w.items()}
    return w
