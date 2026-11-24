"""Состояние рынка по одной монете и признаки для стратегий.

Время — миллисекунды биржи. Раз в секунду движок вызывает on_second(): запоминаем цену,
закрываем минутные свечи. Признаки считаются только по прошлому.
"""
from __future__ import annotations

import itertools
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
        # Лента сделок — сводками по секундам, а не поштучно (поштучно это сотни тысяч мелких
        # объектов в час и постоянный рост памяти): [секунда, Σ цена·$, Σ $, Σ $ со знаком
        # (+ покупка тейкером)]. Хранится 15 минут.
        self.trades: deque = deque()
        self.sec: deque = deque(maxlen=SECONDS_KEPT)       # цена на конец каждой секунды
        self.minute_closes: deque = deque(maxlen=MINUTES_KEPT)
        self.liqs: deque = deque()            # (ts, $ ликвидаций)
        self.liq_minutes: deque = deque(maxlen=6 * 60)     # сумма ликвидаций по минутам, 6 ч
        self._liq_this_minute = 0.0
        self.oi: deque = deque()              # (ts, открытый интерес в $)
        self.minute_usd: deque = deque(maxlen=6 * 60)      # оборот за каждую минуту, $, 6 ч
        # поток ордеров (OFI, Cont–Kukanov–Stoikov): сумма за каждую секунду, час
        self.ofi_sec: deque = deque(maxlen=3600)
        self._ofi_this_sec = 0.0
        self._best: tuple | None = None       # (bid, объём, ask, объём) после прошлого обновления
        self.top_depth = 0.0                  # средний объём лучших уровней (для нормировки OFI)
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
        self.add_flow(ts, price, price * qty * (1 if taker_buy else -1))

    def add_flow(self, ts: int, price: float, signed_usd: float) -> None:
        """Добавить сделку в посекундную сводку ленты."""
        sec, usd = ts // 1000, abs(signed_usd)
        last = self.trades[-1] if self.trades else None
        if last is not None and last[0] == sec:
            last[1] += price * usd
            last[2] += usd
            last[3] += signed_usd
        else:
            self.trades.append([sec, price * usd, usd, signed_usd])
        cut = sec - 15 * 60
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
            # Bybit присылает открытый интерес несколько раз в секунду; для окна в минуты
            # хватает точки раз в 5 с — последнее значение за эти 5 с заменяет предыдущее
            if len(self.oi) > 1 and ts - self.oi[-2][0] < 5000:
                self.oi[-1] = (ts, value)
            else:
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
        self.ofi_sec.append(self._ofi_this_sec)
        self._ofi_this_sec = 0.0
        if sec % 60 == 0:
            self.minute_closes.append(price)
            self.liq_minutes.append(self._liq_this_minute)
            self._liq_this_minute = 0.0
            self.minute_usd.append(sum(usd for s, _, usd, _ in reversed(self.trades)
                                       if s > sec - 60) if self.trades else 0.0)

    def on_book_update(self) -> None:
        """Изменение лучших заявок → вклад в поток ордеров (OFI):
        e = [b≥b′]·qb − [b≤b′]·qb′ − [a≤a′]·qa + [a≥a′]·qa′ (штрих — до обновления)."""
        book = self.book
        if not book.ready or getattr(book, "approx", False):
            return
        (b, qb), = book.levels("bid", 1)
        (a, qa), = book.levels("ask", 1)
        prev = self._best
        self._best = (b, qb, a, qa)
        self.top_depth = (qb + qa) / 2 if not self.top_depth else \
            0.999 * self.top_depth + 0.001 * (qb + qa) / 2
        if prev is None:
            return
        pb, pqb, pa, pqa = prev
        e = (qb if b >= pb else 0.0) - (pqb if b <= pb else 0.0) \
            - (qa if a <= pa else 0.0) + (pqa if a >= pa else 0.0)
        self._ofi_this_sec += e

    def extreme_range(self, start_s: int, end_s: int, side: int) -> float:
        """Максимум/минимум цены в отрезке от start_s до end_s секунд назад (start_s < end_s)."""
        window = list(itertools.islice(reversed(self.sec), start_s, end_s + 1))
        if not window:
            return 0.0
        return max(window) if side > 0 else min(window)

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
        if not self.sec:
            return 0.0
        # только хвост окна, без копирования всей 6-часовой истории каждую секунду
        window = itertools.islice(reversed(self.sec), seconds + 1)
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
        cut = (now - seconds * 1000) // 1000
        signed = total = 0.0
        for sec, _, usd, s in reversed(self.trades):
            if sec < cut:
                break
            signed += s
            total += usd
        return signed / total if total else 0.0

    def vwap(self, seconds: int, now: int) -> float:
        cut = (now - seconds * 1000) // 1000
        pv = vol = 0.0
        for sec, p_usd, usd, _ in reversed(self.trades):
            if sec < cut:
                break
            pv += p_usd
            vol += usd
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
