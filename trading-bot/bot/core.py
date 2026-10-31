"""Общий "мозг" бота: сигналы + размер позиций + защита от просадки.

Используется одинаково в бэктесте и в live-режимах.
"""
from __future__ import annotations

import pandas as pd

from .config import Config
from .risk import DrawdownGuard
from .strategy import SymbolState, exit_position, size_weights, update_signal


class Decider:
    def __init__(self, cfg: Config, initial_equity: float):
        self.cfg = cfg
        self.caps = cfg.portfolio.caps
        self.symbols = list(self.caps)
        self.states = {s: SymbolState() for s in self.symbols}
        self.guard = DrawdownGuard(cfg.risk, peak=initial_equity)

    def on_bar_close(self, rows: dict[str, pd.Series], returns: pd.DataFrame,
                     equity: float, ts=None) -> dict[str, float]:
        """Закрытие бара: возвращает целевые веса (доли капитала) по каждому активу."""
        mult = self.guard.on_bar_close(equity, ts)
        if self.guard.active:
            self.reset_signals()
        wants = {s: update_signal(rows[s], self.states[s], self.cfg.strategy)
                 for s in self.symbols if s in rows}
        vols = {s: float(rows[s]["vol"]) for s, w in wants.items() if w}
        weights = size_weights(vols, returns, self.caps, self.cfg.strategy,
                               self.cfg.exchange.timeframe, self.cfg.portfolio.max_gross_exposure)
        return {s: weights.get(s, 0.0) * mult for s in self.symbols}

    def check_stops(self, lows: dict[str, float], holdings: dict[str, float]) -> list[str]:
        """Какие позиции выбило трейлинг-стопом (lows — минимум бара или текущая цена)."""
        hit = []
        for s, units in holdings.items():
            st = self.states[s]
            if units > 0 and st.in_position and st.stop > 0 and lows.get(s, float("inf")) <= st.stop:
                hit.append(s)
        return hit

    def on_stop_hit(self, symbol: str) -> None:
        exit_position(self.states[symbol], self.cfg.strategy)

    def reset_signals(self) -> None:
        for s, st in self.states.items():
            if st.in_position:
                exit_position(st, self.cfg.strategy)

    def to_dict(self) -> dict:
        return {"states": {s: st.to_dict() for s, st in self.states.items()},
                "guard": self.guard.to_dict()}

    def load_dict(self, d: dict) -> None:
        for s, st in d.get("states", {}).items():
            if s in self.states:
                self.states[s] = SymbolState.from_dict(st)
        if "guard" in d:
            self.guard = DrawdownGuard.from_dict(self.cfg.risk, d["guard"])


def rebalance_deltas(targets: dict[str, float], current: dict[str, float],
                     threshold: float, min_trade: float = 0.01) -> dict[str, float]:
    """Какие веса реально менять. Мелкие подстройки пропускаем, чтобы не платить комиссии.

    threshold — относительный порог: 0.2 значит "меняем, если вес отличается
    на 20% от размера позиции"; min_trade — минимальная сделка в долях капитала."""
    out = {}
    for s, tgt in targets.items():
        cur = current.get(s, 0.0)
        diff = abs(tgt - cur)
        if tgt <= 0 and cur > 1e-9:
            out[s] = 0.0
        elif diff >= min_trade and diff >= threshold * max(tgt, cur):
            out[s] = tgt
    return out
