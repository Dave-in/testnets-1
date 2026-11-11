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
        st_cfg = self.cfg.strategy
        sides = {s: update_signal(rows[s], self.states[s], st_cfg)
                 for s in self.symbols if s in rows}
        sides = self._top_n(sides, rows)
        active = {s: side for s, side in sides.items() if side}
        vols = {s: float(rows[s]["vol"]) for s in active}
        stop_dist = {s: (st_cfg.atr_stop_mult if side > 0 else st_cfg.short_atr_mult)
                     * float(rows[s]["atr"]) / float(rows[s]["close"])
                     for s, side in active.items()}
        weights = size_weights(active, vols, stop_dist, returns, self.caps, st_cfg,
                               self.cfg.exchange.timeframe, self.cfg.portfolio.max_gross_exposure)
        return {s: weights.get(s, 0.0) * mult for s in self.symbols}

    def _top_n(self, sides: dict[str, int], rows: dict[str, pd.Series]) -> dict[str, int]:
        """Если сигналов больше, чем max_positions, — оставляем самые сильные
        (импульс в сторону сделки с поправкой на волатильность)."""
        n = self.cfg.strategy.max_positions
        active = [s for s, side in sides.items() if side]
        if n <= 0 or len(active) <= n:
            return sides
        score = {s: sides[s] * float(rows[s]["mom"]) / max(float(rows[s]["vol"]), 1e-9)
                 for s in active}
        keep = set(sorted(active, key=score.get, reverse=True)[:n])
        return {s: side if s in keep else 0 for s, side in sides.items()}

    def check_stops(self, lows: dict[str, float], holdings: dict[str, float],
                    highs: dict[str, float] | None = None) -> list[str]:
        """Какие позиции выбило трейлинг-стопом. lows/highs — минимум и максимум бара
        (в live оба равны текущей цене): лонг выбивает минимум, шорт — максимум."""
        highs = lows if highs is None else highs
        hit = []
        for s, units in holdings.items():
            st = self.states[s]
            if not units or st.stop <= 0 or st.side * units <= 0:
                continue
            if (units > 0 and lows.get(s, float("inf")) <= st.stop) or \
               (units < 0 and highs.get(s, 0.0) >= st.stop):
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
        if abs(tgt) < 1e-12:
            if abs(cur) > 1e-9:
                out[s] = 0.0          # выход всегда исполняем
        elif tgt * cur < 0:
            out[s] = tgt              # переворот: лонг ↔ шорт
        elif diff >= min_trade and diff >= threshold * max(abs(tgt), abs(cur)):
            out[s] = tgt
    return out
