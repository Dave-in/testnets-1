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
    side: int = 0                # +1 лонг, −1 шорт, 0 вне рынка
    extreme_close: float = 0.0   # максимум закрытия (лонг) / минимум закрытия (шорт) с момента входа
    stop: float = 0.0
    cooldown_left: int = 0

    @property
    def in_position(self) -> bool:
        return self.side != 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolState":
        d = dict(d)
        if "in_position" in d:  # состояние, сохранённое версией без шортов
            d["side"] = 1 if d.pop("in_position") else 0
            d["extreme_close"] = d.pop("highest_close", 0.0)
        return cls(**d)


def prepare(df: pd.DataFrame, cfg: StrategyConfig, timeframe: str) -> pd.DataFrame:
    """Добавляет к OHLCV колонки индикаторов."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.fast_ema)
    out["ema_slow"] = ema(out["close"], cfg.slow_ema)
    out["atr"] = atr(out, cfg.atr_period)
    out["vol"] = realized_vol(out["close"], cfg.vol_lookback, timeframe)
    out["mom"] = out["close"] / out["close"].shift(cfg.momentum_lookback) - 1
    # максимум/минимум ПРЕДЫДУЩИХ баров (текущий не входит — иначе пробой невозможен)
    out["hh"] = out["high"].rolling(cfg.breakout_lookback).max().shift(1)
    if cfg.exit_lookback > 0:
        out["ll"] = out["low"].rolling(cfg.exit_lookback).min().shift(1)
    out["ll_entry"] = out["low"].rolling(cfg.breakout_lookback).min().shift(1)
    if cfg.exit_lookback > 0:
        out["hh_exit"] = out["high"].rolling(cfg.exit_lookback).max().shift(1)
    n = cfg.momentum_lookback
    path = out["close"].diff().abs().rolling(n).sum()
    out["quality"] = (out["close"] - out["close"].shift(n)).abs() / path
    out["regime_ok"] = True
    return out


def add_regime(frames: dict[str, pd.DataFrame], regime_df: pd.DataFrame | None,
               cfg: StrategyConfig) -> None:
    """Проставляет в каждый фрейм колонку regime_ok: рынок в целом растёт
    (regime_symbol закрылся выше своей EMA). regime_df — свечи regime_symbol."""
    if not cfg.regime_symbol or regime_df is None:
        return
    ok = regime_df["close"] > ema(regime_df["close"], cfg.regime_ema)
    for df in frames.values():
        df["regime_ok"] = ok.reindex(df.index).ffill().fillna(False).astype(bool)


def _ready(row: pd.Series) -> bool:
    return all(
        not (isinstance(row[c], float) and math.isnan(row[c]))
        for c in ("ema_fast", "ema_slow", "atr", "vol", "mom", "close")
    )


def update_signal(row: pd.Series, st: SymbolState, cfg: StrategyConfig) -> int:
    """Обновляет состояние актива по закрытию бара.
    Возвращает желаемую сторону: +1 лонг, −1 шорт, 0 вне рынка."""
    if not _ready(row):
        st.side = 0
        return 0

    close, atr_now = float(row["close"]), float(row["atr"])
    bull = bool(row["regime_ok"])          # рынок в целом растёт
    has_regime = bool(cfg.regime_symbol)

    if st.side > 0:
        st.extreme_close = max(st.extreme_close, close)
        st.stop = max(st.stop, st.extreme_close - cfg.atr_stop_mult * atr_now)
        trend_broken = close < row["ema_slow"] or row["ema_fast"] < row["ema_slow"]
        regime_bad = cfg.regime_exit and not bull
        channel_exit = cfg.exit_lookback > 0 and close < row["ll"]
        if close <= st.stop or trend_broken or regime_bad or channel_exit:
            exit_position(st, cfg)
        return st.side

    if st.side < 0:
        st.extreme_close = min(st.extreme_close, close)
        st.stop = min(st.stop, st.extreme_close + cfg.short_atr_mult * atr_now)
        trend_broken = close > row["ema_slow"] or row["ema_fast"] > row["ema_slow"]
        regime_bad = cfg.regime_exit and has_regime and bull
        channel_exit = cfg.exit_lookback > 0 and close > row["hh_exit"]
        if close >= st.stop or trend_broken or regime_bad or channel_exit:
            exit_position(st, cfg)
        return st.side

    if st.cooldown_left > 0:
        st.cooldown_left -= 1
        return 0

    breakout = cfg.entry_mode == "breakout"
    if cfg.min_trend_quality > 0 and not (row["quality"] >= cfg.min_trend_quality):
        return 0  # движение слишком «пилообразное» (или данных мало) — не входим
    if bull:
        if breakout:
            entry = close > row["hh"] and close > row["ema_slow"] and row["mom"] > 0
        else:
            entry = close > row["ema_fast"] > row["ema_slow"] and row["mom"] > 0
        if entry:
            st.side, st.extreme_close = 1, close
            st.stop = close - cfg.atr_stop_mult * atr_now
    if st.side == 0 and cfg.allow_short and (not has_regime or not bull):
        # шорт — только когда рынок в целом падает (если фильтр рынка включён)
        if breakout:
            entry = close < row["ll_entry"] and close < row["ema_slow"] and row["mom"] < 0
        else:
            entry = close < row["ema_fast"] < row["ema_slow"] and row["mom"] < 0
        if entry:
            st.side, st.extreme_close = -1, close
            st.stop = close + cfg.short_atr_mult * atr_now
    return st.side


def exit_position(st: SymbolState, cfg: StrategyConfig) -> None:
    st.side = 0
    st.extreme_close = 0.0
    st.stop = 0.0
    st.cooldown_left = cfg.reentry_cooldown_bars


def size_weights(
    sides: dict[str, int],
    vols: dict[str, float],
    stop_dist: dict[str, float],
    returns: pd.DataFrame,
    caps: dict[str, float],
    cfg: StrategyConfig,
    timeframe: str,
    max_gross: float,
) -> dict[str, float]:
    """Веса со знаком (+ лонг, − шорт) для активов с сигналом.

    sides     — сторона по каждому активу (+1 / −1)
    vols      — годовая волатильность
    stop_dist — расстояние до стопа в долях цены (для лимита риска на сделку)
    returns   — лог-доходности за последние vol_lookback баров (для корреляций)

    Три ограничителя размера, работает самый строгий:
      1) волатильность: чем сильнее скачет актив, тем меньше позиция;
      2) риск на сделку: срабатывание стопа стоит не больше risk_per_trade счёта;
      3) потолок доли актива (caps) и общее плечо (max_gross).
    """
    w: dict[str, float] = {}
    for s, side in sides.items():
        v = vols.get(s, 0.0)
        if not side or not v or v <= 0:
            continue
        size = min(caps[s], cfg.target_vol / v)
        d = stop_dist.get(s, 0.0)
        if cfg.risk_per_trade > 0 and d > 0:
            size = min(size, cfg.risk_per_trade / d)
        if side < 0:
            size *= cfg.short_scale
        if size > 0:
            w[s] = side * size
    if not w:
        return {}

    # Корреляция: лонги усиливают друг друга, лонг и шорт частично гасят.
    syms = list(w)
    if len(syms) > 1:
        sub = returns[syms].dropna()
        if len(sub) >= 5:
            cov = sub.cov().to_numpy() * periods_per_year(timeframe)
            vec = np.array([w[s] for s in syms])
            port_vol = float(np.sqrt(max(vec @ cov @ vec, 0.0)))
            if port_vol > cfg.target_vol:
                k = cfg.target_vol / port_vol
                w = {s: x * k for s, x in w.items()}

    gross = sum(abs(x) for x in w.values())
    if gross > max_gross:
        w = {s: x * max_gross / gross for s, x in w.items()}
    return w
