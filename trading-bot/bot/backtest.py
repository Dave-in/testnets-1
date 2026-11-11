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
    def __init__(self, cfg: Config, data: dict[str, pd.DataFrame], trade_start: str | None = None,
                 funding: dict[str, pd.Series] | None = None):
        self.cfg = cfg
        self.futures = cfg.exchange.market == "futures"
        if cfg.strategy.allow_short and not self.futures:
            raise ValueError('Шорты возможны только при [exchange] market = "futures"')
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

        self.funding_bar = self._funding_per_bar(funding or {})

        first_ready = min(df["ema_slow"].first_valid_index() for df in prepared.values())
        start = first_ready
        if trade_start:
            start = max(start, pd.Timestamp(trade_start, tz="UTC"))
        self.start_i = int(self.index.searchsorted(start))
        if self.start_i >= len(self.index) - 1:
            raise ValueError("Слишком мало данных: не хватает истории для прогрева индикаторов")

    def _funding_per_bar(self, funding: dict[str, pd.Series]) -> pd.DataFrame:
        """Суммарная ставка фандинга за каждый бар (NaN — истории нет)."""
        out = pd.DataFrame(index=self.index, columns=self.symbols, dtype=float)
        step = self.index[1] - self.index[0]
        for s, rates in funding.items():
            if s in out.columns and len(rates):
                bars = rates.index.floor(step)
                out[s] = rates.groupby(bars).sum().reindex(self.index)
                # внутри периода, где история есть, пропуск = выплат не было
                inside = (self.index >= rates.index[0]) & (self.index <= rates.index[-1])
                out.loc[inside, s] = out.loc[inside, s].fillna(0.0)
        self._default_funding = self.cfg.exchange.default_funding_8h * (step / pd.Timedelta(hours=8))
        return out

    # --- исполнение -------------------------------------------------------
    def _trade(self, ts, sym, units, price, reason):
        """units > 0 — покупка, < 0 — продажа (в т.ч. открытие шорта)."""
        if units == 0:
            return
        ex = self.cfg.exchange
        fill = price * (1 + np.sign(units) * ex.slippage)
        notional = abs(units) * fill
        fee = notional * ex.fee
        self.trades.append({"ts": ts, "symbol": sym, "side": "buy" if units > 0 else "sell",
                            "units": abs(units), "price": fill, "notional": notional,
                            "fee": fee, "reason": reason})
        self.cash += -units * fill - fee
        self.units[sym] += units

    def _close(self, ts, sym, price, reason):
        self._trade(ts, sym, -self.units[sym], price, reason)
        self.units[sym] = 0.0

    def _equity(self, prices) -> float:
        return self.cash + sum(u * prices[s] for s, u in self.units.items() if u)

    def _rebalance(self, ts, targets: dict[str, float], prices: dict[str, float]):
        equity = self._equity(prices)
        # сначала продажи, потом покупки — на споте иначе может не хватить кэша
        order = sorted(targets, key=lambda s: targets[s] * equity - self.units[s] * prices[s])
        for s in order:
            p = prices.get(s)
            if p is None or np.isnan(p):
                continue
            delta = (targets[s] * equity - self.units[s] * p) / p
            if not self.futures:
                ex = self.cfg.exchange
                delta = max(delta, -self.units[s])  # на споте нельзя продать больше, чем есть
                delta = min(delta, self.cash / (p * (1 + ex.slippage) * (1 + ex.fee)))
            self._trade(ts, s, delta, p, "rebalance")

    # --- основной цикл ----------------------------------------------------
    def run(self) -> BacktestResult:
        cfg = self.cfg
        capital = cfg.portfolio.initial_capital
        self.cash = capital
        self.units = {s: 0.0 for s in self.symbols}
        self.trades: list[dict] = []
        self.funding_total = 0.0
        decider = Decider(cfg, capital)
        pending: dict[str, float] | None = None
        records = []
        lookback = cfg.strategy.vol_lookback
        liquidated = False

        for i in range(self.start_i, len(self.index)):
            ts = self.index[i]
            bar = {s: self.frames[s].iloc[i] for s in self.symbols}
            last_close = self.close_ff.iloc[i]
            opens = {s: bar[s]["open"] if not np.isnan(bar[s]["open"]) else last_close[s]
                     for s in self.symbols}

            # 1) исполняем решения, принятые на прошлом закрытии
            if pending and not liquidated:
                self._rebalance(ts, pending, opens)
            pending = None

            # 2) трейлинг-стопы внутри бара (лонг выбивает минимум, шорт — максимум;
            #    при гэпе за стоп выходим по цене открытия, то есть хуже стопа)
            lows = {s: bar[s]["low"] for s in self.symbols if not np.isnan(bar[s]["low"])}
            highs = {s: bar[s]["high"] for s in self.symbols if not np.isnan(bar[s]["high"])}
            for s in decider.check_stops(lows, self.units, highs):
                stop = decider.states[s].stop
                px = min(opens[s], stop) if self.units[s] > 0 else max(opens[s], stop)
                self._close(ts, s, px, "trailing_stop")
                decider.on_stop_hit(s)

            # 3) худший случай внутри бара: ликвидация и жесткий лимит просадки
            held = [s for s in self.symbols if self.units[s]]
            if held:
                worst = {s: (lows if self.units[s] > 0 else highs).get(s, opens[s]) for s in held}
                eq_open = self.cash + sum(self.units[s] * opens[s] for s in held)
                eq_worst = self.cash + sum(self.units[s] * worst[s] for s in held)
                if eq_worst <= 0:  # плечо слишком велико: счёт обнулён
                    for s in held:
                        self._close(ts, s, worst[s], "liquidation")
                    self.cash = max(self.cash, 0.0)
                    decider.guard.halted = liquidated = True
                    decider.guard.events.append({"ts": ts, "event": "liquidation", "equity": 0.0,
                                                 "drawdown": 1.0})
                    decider.reset_signals()
                elif decider.guard.check_intrabar(eq_worst, ts):
                    threshold = decider.guard.peak * (1 - cfg.risk.hard_drawdown)
                    lam = 0.0 if eq_open <= eq_worst else (eq_open - threshold) / (eq_open - eq_worst)
                    lam = float(np.clip(lam, 0.0, 1.0))
                    for s in held:
                        self._close(ts, s, opens[s] + lam * (worst[s] - opens[s]), "drawdown_guard")
                    decider.reset_signals()

            # 4) фандинг (только фьючерсы): лонг платит положительную ставку, шорт получает
            if self.futures:
                for s, u in self.units.items():
                    if not u:
                        continue
                    rate = self.funding_bar[s].iloc[i]
                    if np.isnan(rate):  # истории нет: лонги платят типовую ставку, шорты — ноль
                        pay = max(u, 0.0) * last_close[s] * self._default_funding
                    else:
                        pay = u * last_close[s] * rate
                    self.cash -= pay
                    self.funding_total += pay

            # 5) закрытие бара: оценка капитала и новое решение
            equity = self._equity(last_close)
            rows = {s: bar[s] for s in self.symbols}
            rets = self.logret.iloc[max(0, i - lookback + 1): i + 1]
            targets = decider.on_bar_close(rows, rets, equity, ts)
            current = {s: self.units[s] * last_close[s] / equity if equity > 0 else 0.0
                       for s in self.symbols}
            pending = rebalance_deltas(targets, current, cfg.strategy.rebalance_threshold,
                                       cfg.strategy.min_trade)

            rec = {"ts": ts, "equity": equity, "cash": self.cash,
                   "exposure": sum(abs(w) for w in current.values()),
                   "net_exposure": sum(current.values()),
                   "guard_peak": decider.guard.peak}
            rec.update({f"w_{s}": current[s] for s in self.symbols})
            records.append(rec)

        eq = pd.DataFrame(records).set_index("ts")
        trades = pd.DataFrame(self.trades, columns=["ts", "symbol", "side", "units", "price",
                                                    "notional", "fee", "reason"])
        ppy = periods_per_year(cfg.exchange.timeframe)
        m = metrics.compute(eq["equity"], ppy, trades, eq["exposure"],
                            cfg.portfolio.monthly_target)
        if self.futures:
            m["funding_paid"] = float(self.funding_total)
        return BacktestResult(eq, trades, decider.guard.events, m, self._benchmarks(eq.index, ppy))

    def _benchmarks(self, idx: pd.DatetimeIndex, ppy: int) -> dict[str, dict]:
        """Просто купить и держать — для сравнения."""
        out = {}
        for s in self.symbols:
            c = self.close_ff[s].reindex(idx).dropna()
            if len(c) > 2:
                out[f"Buy&Hold {s}"] = metrics.compute(c / c.iloc[0], ppy)
        return out
