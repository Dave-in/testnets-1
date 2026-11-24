"""Состояние рынка по одной монете и признаки для стратегий.

Время — миллисекунды биржи. Раз в секунду движок вызывает on_second(): запоминаем цену,
закрываем минутные свечи. Признаки считаются только по прошлому.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .book import OrderBook

SECONDS_KEPT = 6 * 3600     # цены по секундам за 6 часов
MINUTES_KEPT = 24 * 60      # минутные закрытия за сутки


class SymbolMarket:
    def __init__(self, symbol: str, book=None) -> None:
        self.symbol = symbol
        self.book = book if book is not None else OrderBook()
        self.last_price = 0.0
        self.last_ts = 0
        self.trades: deque = deque()          # (ts, цена, объём в $ со знаком: + покупка тейкером)
        self.sec: deque = deque(maxlen=SECONDS_KEPT)       # цена на конец каждой секунды
        self.minute_closes: deque = deque(maxlen=MINUTES_KEPT)
        self.liqs: deque = deque()            # (ts, $ ликвидаций)
        self.liq_minutes: deque = deque(maxlen=6 * 60)     # сумма ликвидаций по минутам, 6 ч
        self._liq_this_minute = 0.0
        self.oi: deque = deque()              # (ts, открытый интерес в $)
        self.funding_rate = 0.0
        self.next_funding = 0                 # мс
        self.mark_price = 0.0
        self.turnover24h = 0.0
        # правила контракта Bybit (0 — не ограничено): шаг объёма, минимум, шаг цены и т.д.
        self.qty_step = 0.0
        self.min_qty = 0.0
        self.max_mkt_qty = 0.0
        self.min_notional = 0.0
        self.tick = 0.0
        self.funding_interval_h = 8.0         # у части монет фандинг раз в 1 или 4 часа
        self.tradable = True                  # False — только данные (ориентир), без сделок
        self._sigma_cache: tuple[int, float] = (-1, 0.0)

    # --- поступление данных ----------------------------------------------------
    def on_trade(self, ts: int, price: float, qty: float, taker_buy: bool) -> None:
        self.last_price, self.last_ts = price, ts
        self.trades.append((ts, price, price * qty * (1 if taker_buy else -1)))
        cut = ts - 15 * 60 * 1000
        while self.trades and self.trades[0][0] < cut:
            self.trades.popleft()

    def on_liquidation(self, ts: int, notional: float) -> None:
        self.liqs.append((ts, notional))
        self._liq_this_minute += notional
        cut = ts - 10 * 60 * 1000
        while self.liqs and self.liqs[0][0] < cut:
            self.liqs.popleft()

    def on_open_interest(self, ts: int, value: float) -> None:
        if value > 0:
            self.oi.append((ts, value))
            cut = ts - 3600 * 1000
            while self.oi and self.oi[0][0] < cut:
                self.oi.popleft()

    def warmup_minutes(self, closes: list[float]) -> None:
        """Предзагрузка минутных закрытий (с биржи по REST или из архива)."""
        for c in closes:
            if c > 0:
                self.minute_closes.append(float(c))
        if closes and not self.last_price:
            self.last_price = float(closes[-1])

    def on_second(self, sec: int) -> None:
        price = self.price()
        if price <= 0:
            return
        self.sec.append(price)
        if sec % 60 == 0:
            self.minute_closes.append(price)
            self.liq_minutes.append(self._liq_this_minute)
            self._liq_this_minute = 0.0

    # --- признаки -----------------------------------------------------------------
    def price(self) -> float:
        mid = self.book.mid() if self.book.ready else 0.0
        return mid or self.last_price

    def price_ago(self, seconds: int) -> float:
        """Цена seconds секунд назад (0 — если истории не хватает)."""
        if len(self.sec) <= seconds:
            return 0.0
        return self.sec[-1 - seconds]

    def ret(self, seconds: int) -> float:
        """Логарифмическая доходность за последние seconds секунд."""
        old = self.price_ago(seconds)
        return math.log(self.sec[-1] / old) if old > 0 and self.sec else 0.0

    def extreme(self, seconds: int, side: int) -> float:
        """Максимум (side=+1) или минимум (side=−1) цены за последние seconds секунд."""
        window = list(self.sec)[-seconds - 1:]
        if not window:
            return 0.0
        return max(window) if side > 0 else min(window)

    def sigma1m(self) -> float:
        """Волатильность минутной доходности (EWMA за ~4 часа). Кешируется на минуту."""
        n = len(self.minute_closes)
        if self._sigma_cache[0] == n:
            return self._sigma_cache[1]
        if n < 60:
            return 0.0
        c = np.fromiter(self.minute_closes, float)[-480:]
        r = np.diff(np.log(c))
        w = 0.5 ** (np.arange(len(r))[::-1] / 240)        # полураспад 4 часа
        s = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
        self._sigma_cache = (n, s)
        return s

    def minute_returns(self, n: int) -> np.ndarray:
        c = np.fromiter(self.minute_closes, float)[-(n + 1):]
        return np.diff(np.log(c)) if len(c) > 1 else np.array([])

    def flow(self, seconds: int, now: int) -> float:
        """Перекос агрессивных сделок за окно: (покупки − продажи) / (покупки + продажи)."""
        cut = now - seconds * 1000
        signed = total = 0.0
        for ts, _, v in reversed(self.trades):
            if ts < cut:
                break
            signed += v
            total += abs(v)
        return signed / total if total else 0.0

    def vwap(self, seconds: int, now: int) -> float:
        cut = now - seconds * 1000
        pv = vol = 0.0
        for ts, p, v in reversed(self.trades):
            if ts < cut:
                break
            pv += p * abs(v)
            vol += abs(v)
        return pv / vol if vol else self.price()

    def liq_sum(self, seconds: int, now: int) -> float:
        cut = now - seconds * 1000
        return sum(v for ts, v in self.liqs if ts >= cut)

    def last_liq_ts(self) -> int:
        return self.liqs[-1][0] if self.liqs else 0

    def liq_baseline(self) -> float:
        """Обычная сумма ликвидаций за минуту (среднее за 6 часов)."""
        return float(np.mean(self.liq_minutes)) if self.liq_minutes else 0.0

    def oi_change(self, seconds: int, now: int) -> float | None:
        """Изменение открытого интереса за окно, доля (None — данных мало)."""
        if len(self.oi) < 2:
            return None
        cut = now - seconds * 1000
        old = None
        for ts, v in self.oi:          # последнее значение на начало окна или раньше
            if ts > cut:
                break
            old = v
        if old is None or cut - self.oi[0][0] > 3600 * 1000:
            return None
        return self.oi[-1][1] / old - 1
