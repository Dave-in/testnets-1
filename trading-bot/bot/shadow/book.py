"""Стакан заявок и расчёт исполнения по нему."""
from __future__ import annotations

import heapq


class OrderBook:
    """L2-стакан: цена → объём в монетах. Обновляется снимками и изменениями (как у Bybit)."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ts = 0

    def snapshot(self, bids, asks, ts: int) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.ts = ts

    def delta(self, bids, asks, ts: int) -> None:
        for side, rows in ((self.bids, bids), (self.asks, asks)):
            for p, q in rows:
                p, q = float(p), float(q)
                if q > 0:
                    side[p] = q
                else:
                    side.pop(p, None)
        self.ts = ts

    @property
    def ready(self) -> bool:
        return bool(self.bids) and bool(self.asks)

    def best_bid(self) -> float:
        return max(self.bids) if self.bids else 0.0

    def best_ask(self) -> float:
        return min(self.asks) if self.asks else 0.0

    def mid(self) -> float:
        return (self.best_bid() + self.best_ask()) / 2 if self.ready else 0.0

    def spread(self) -> float:
        return self.best_ask() - self.best_bid() if self.ready else 0.0

    def levels(self, side: str, n: int) -> list[tuple[float, float]]:
        """Лучшие n уровней: side='bid' (по убыванию цены) или 'ask' (по возрастанию)."""
        book = self.bids if side == "bid" else self.asks
        keys = heapq.nlargest(n, book) if side == "bid" else heapq.nsmallest(n, book)
        return [(p, book[p]) for p in keys]

    def size_at(self, side: str, price: float) -> float:
        return (self.bids if side == "bid" else self.asks).get(price, 0.0)

    def walk(self, side: str, qty: float) -> tuple[float, float]:
        """Рыночный ордер: покупка проходит по продажам (ask), продажа — по покупкам (bid).
        Возвращает (средняя цена, исполненный объём)."""
        filled = cost = 0.0
        for price, size in self.levels("ask" if side == "buy" else "bid", 200):
            take = min(size, qty - filled)
            filled += take
            cost += take * price
            if filled >= qty - 1e-12:
                break
        return (cost / filled if filled else 0.0), filled

    def imbalance(self, n: int = 10) -> float:
        """Перекос стакана по n лучшим уровням (в деньгах): +1 — только покупатели, −1 — только продавцы."""
        b = sum(p * q for p, q in self.levels("bid", n))
        a = sum(p * q for p, q in self.levels("ask", n))
        return (b - a) / (b + a) if b + a > 0 else 0.0

    def microprice(self) -> float:
        """«Справедливая» цена по лучшим уровням: ближе к той стороне, где объёма меньше."""
        if not self.ready:
            return 0.0
        (pb, qb), = self.levels("bid", 1)
        (pa, qa), = self.levels("ask", 1)
        return (pa * qb + pb * qa) / (qa + qb)


class CurveBook:
    """Приближённый стакан для повтора истории Binance: середина — по последней сделке,
    спред — один шаг цены, глубина — по накопленному объёму в процентах от цены
    (файлы bookDepth: ±1%, ±2% … ±5%, раз в ~30 секунд). Внутри 1% объём считается
    распределённым равномерно."""

    approx = True   # микроцены и уровней нет — стратегии это учитывают

    def __init__(self, tick: float) -> None:
        self.tick = tick
        self.last = 0.0
        self.ts = 0
        self.curve = {"bid": [(1.0, 0.0)], "ask": [(1.0, 0.0)]}  # (% от цены, накопленный $)

    @property
    def ready(self) -> bool:
        return self.last > 0 and self.curve["bid"][0][1] > 0 and self.curve["ask"][0][1] > 0

    def set_price(self, price: float, ts: int) -> None:
        self.last, self.ts = price, ts

    def set_curve(self, side: str, points: list[tuple[float, float]]) -> None:
        self.curve[side] = sorted(points)

    def best_bid(self) -> float:
        return self.last - self.tick / 2

    def best_ask(self) -> float:
        return self.last + self.tick / 2

    def mid(self) -> float:
        return self.last

    def spread(self) -> float:
        return self.tick

    def _density(self, side: str) -> float:
        pct, notional = self.curve[side][0]
        return notional / (pct / 100 * self.last) if self.last else 0.0   # $ на единицу цены

    def size_at(self, side: str, price: float) -> float:
        return self._density(side) * self.tick / max(price, 1e-12)        # монет на один шаг цены

    def walk(self, side: str, qty: float) -> tuple[float, float]:
        if not self.ready:
            return 0.0, 0.0
        s = "ask" if side == "buy" else "bid"
        dens = self._density(s)
        depth = qty * self.last / dens if dens > 0 else 0.0               # на сколько уйдём по цене
        start = self.best_ask() if side == "buy" else self.best_bid()
        avg = start + depth / 2 if side == "buy" else start - depth / 2
        return avg, qty

    def imbalance(self, n: int = 10) -> float:
        b, a = self.curve["bid"][0][1], self.curve["ask"][0][1]
        return (b - a) / (b + a) if b + a > 0 else 0.0

    def microprice(self) -> float:
        return self.last
