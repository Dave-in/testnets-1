"""Бэктест: прогон стратегии по истории бар за баром.

Правила, чтобы не обманывать себя:
  * решение принимается на закрытии бара t, исполняется по open бара t+1;
  * каждая сделка платит комиссию и проскальзывание;
  * трейлинг-стоп и лимит просадки проверяются внутри бара по low;
    если цена открылась гэпом ниже стопа — выходим по open (хуже стопа).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .config import Config
from .core import Decider, rebalance_deltas
from .indicators import periods_per_year
from .strategy import add_regime, prepare


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    events: list[dict]
    metrics: dict
    benchmarks: dict[str, dict] = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: Config, data: dict[str, pd.DataFrame], trade_start: str | None = None):
        self.cfg = cfg
        self.symbols = cfg.portfolio.symbols
        missing = [s for s in self.symbols if s not in data]
        if missing:
            raise ValueError(f"Нет данных для: {missing}")
        tf = cfg.exchange.timeframe
        prepared = {s: prepare(data[s], cfg.strategy, tf) for s in self.symbols}
        if cfg.strategy.regime_symbol:
            if cfg.strategy.regime_symbol not in data:
                raise ValueError(f"Нет данных для фильтра режима: {cfg.strategy.regime_symbol}")
            add_regime(prepared, data[cfg.strategy.regime_symbol], cfg.strategy)
        index = sorted(set().union(*(df.index for df in prepared.values())))
        self.frames = {s: df.reindex(index) for s, df in prepared.items()}
        self.index = pd.DatetimeIndex(index)
        self.logret = pd.DataFrame(
            {s: np.log(df["close"]).diff() for s, df in self.frames.items()}, index=self.index
        )
        self.close_ff = pd.DataFrame({s: df["close"] for s, df in self.frames.items()}).ffill()

        first_ready = min(df["ema_slow"].first_valid_index() for df in prepared.values())
        start = first_ready
        if trade_start:
            start = max(start, pd.Timestamp(trade_start, tz="UTC"))
        self.start_i = int(self.index.searchsorted(start))
        if self.start_i >= len(self.index) - 1:
            raise ValueError("Слишком мало данных: не хватает истории для прогрева индикаторов")

    # --- исполнение -------------------------------------------------------
    def _trade(self, ts, sym, units, price, side_mult, reason):
        """units > 0 — покупка, < 0 — продажа. Возвращает изменение кэша."""
        ex = self.cfg.exchange
        fill = price * (1 + side_mult * ex.slippage)
        notional = abs(units) * fill
        fee = notional * ex.fee
        self.trades.append({"ts": ts, "symbol": sym, "side": "buy" if units > 0 else "sell",
                            "units": abs(units), "price": fill, "notional": notional,
                            "fee": fee, "reason": reason})
        return -units * fill - fee

    def _sell_all(self, ts, sym, price, reason):
        if self.units[sym] > 0:
            self.cash += self._trade(ts, sym, -self.units[sym], price, -1, reason)
            self.units[sym] = 0.0

    def _rebalance(self, ts, targets: dict[str, float], prices: dict[str, float]):
        equity = self.cash + sum(self.units[s] * prices[s] for s in self.symbols
                                 if self.units[s] > 0)
        # сначала продажи, потом покупки — чтобы хватило кэша
        order = sorted(targets, key=lambda s: targets[s] * equity - self.units[s] * prices[s])
        for s in order:
            p = prices.get(s)
            if p is None or np.isnan(p):
                continue
            delta_units = (targets[s] * equity - self.units[s] * p) / p
            if delta_units < 0:
                delta_units = max(delta_units, -self.units[s])
                self.cash += self._trade(ts, s, delta_units, p, -1, "rebalance")
                self.units[s] += delta_units
            elif delta_units > 0:
                ex = self.cfg.exchange
                max_units = self.cash / (p * (1 + ex.slippage) * (1 + ex.fee))
                delta_units = min(delta_units, max_units)
                if delta_units > 0:
                    self.cash += self._trade(ts, s, delta_units, p, +1, "rebalance")
                    self.units[s] += delta_units

    # --- основной цикл ----------------------------------------------------
    def run(self) -> BacktestResult:
        cfg = self.cfg
        capital = cfg.portfolio.initial_capital
        self.cash = capital
        self.units = {s: 0.0 for s in self.symbols}
        self.trades: list[dict] = []
        decider = Decider(cfg, capital)
        pending: dict[str, float] | None = None
        records = []
        lookback = cfg.strategy.vol_lookback

        for i in range(self.start_i, len(self.index)):
            ts = self.index[i]
            bar = {s: self.frames[s].iloc[i] for s in self.symbols}
            last_close = self.close_ff.iloc[i]
            opens = {s: bar[s]["open"] if not np.isnan(bar[s]["open"]) else last_close[s]
                     for s in self.symbols}

            # 1) исполняем решения, принятые на прошлом закрытии
            if pending:
                self._rebalance(ts, pending, opens)
                pending = None

            # 2) трейлинг-стопы внутри бара
            lows = {s: bar[s]["low"] for s in self.symbols if not np.isnan(bar[s]["low"])}
            for s in decider.check_stops(lows, self.units):
                stop = decider.states[s].stop
                self._sell_all(ts, s, min(opens[s], stop), "trailing_stop")
                decider.on_stop_hit(s)

            # 3) жесткий лимит просадки внутри бара
            held = [s for s in self.symbols if self.units[s] > 0]
            if held:
                eq_open = self.cash + sum(self.units[s] * opens[s] for s in held)
                eq_low = self.cash + sum(self.units[s] * lows.get(s, opens[s]) for s in held)
                if decider.guard.check_intrabar(eq_low, ts):
                    threshold = decider.guard.peak * (1 - cfg.risk.hard_drawdown)
                    lam = 0.0 if eq_open <= eq_low else (eq_open - threshold) / (eq_open - eq_low)
                    lam = float(np.clip(lam, 0.0, 1.0))
                    for s in held:
                        px = opens[s] + lam * (lows.get(s, opens[s]) - opens[s])
                        self._sell_all(ts, s, px, "drawdown_guard")
                    decider.reset_signals()

            # 4) закрытие бара: оценка капитала и новое решение
            equity = self.cash + sum(self.units[s] * last_close[s] for s in self.symbols
                                     if self.units[s] > 0)
            rows = {s: bar[s] for s in self.symbols}
            rets = self.logret.iloc[max(0, i - lookback + 1): i + 1]
            targets = decider.on_bar_close(rows, rets, equity, ts)
            current = {s: self.units[s] * last_close[s] / equity if equity > 0 else 0.0
                       for s in self.symbols}
            pending = rebalance_deltas(targets, current, cfg.strategy.rebalance_threshold,
                                       cfg.strategy.min_trade)

            rec = {"ts": ts, "equity": equity, "cash": self.cash,
                   "exposure": 1 - self.cash / equity if equity > 0 else 0.0,
                   "guard_peak": decider.guard.peak}
            rec.update({f"w_{s}": current[s] for s in self.symbols})
            records.append(rec)

        eq = pd.DataFrame(records).set_index("ts")
        trades = pd.DataFrame(self.trades, columns=["ts", "symbol", "side", "units", "price",
                                                    "notional", "fee", "reason"])
        ppy = periods_per_year(cfg.exchange.timeframe)
        m = metrics.compute(eq["equity"], ppy, trades, eq["exposure"],
                            cfg.portfolio.monthly_target)
        return BacktestResult(eq, trades, decider.guard.events, m, self._benchmarks(eq.index, ppy))

    def _benchmarks(self, idx: pd.DatetimeIndex, ppy: int) -> dict[str, dict]:
        """Просто купить и держать — для сравнения."""
        out = {}
        for s in self.symbols:
            c = self.close_ff[s].reindex(idx).dropna()
            if len(c) > 2:
                out[f"Buy&Hold {s}"] = metrics.compute(c / c.iloc[0], ppy)
        return out
