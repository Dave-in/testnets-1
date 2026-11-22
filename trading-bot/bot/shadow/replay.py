"""Повтор истории Binance Futures через теневой движок — проверка механики на реальных данных.

Источник — публичный архив data.binance.vision: тиковые сделки (aggTrades), глубина стакана
в процентах от цены раз в ~30 секунд (bookDepth), открытый интерес раз в 5 минут (metrics),
минутные свечи предыдущих суток для разогрева. Ликвидаций и полного стакана в архиве нет:
бот «Ликвидации» в повторе не торгует, бот «Стакан» работает по приближённой глубине.
"""
from __future__ import annotations

import heapq
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .book import CurveBook
from .engine import Engine

BASE = "https://data.binance.vision/data/futures/um/daily"


def _fetch(kind: str, symbol: str, day: str, cache: Path, interval: str = "") -> pd.DataFrame | None:
    name = f"{symbol}-{interval}-{day}" if interval else f"{symbol}-{kind}-{day}"
    path = cache / f"{name}.csv"
    if not path.exists():
        sub = f"klines/{symbol}/{interval}" if interval else f"{kind}/{symbol}"
        r = requests.get(f"{BASE}/{sub}/{name}.zip", timeout=120)
        if r.status_code != 200:
            return None
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            path.write_bytes(z.read(z.namelist()[0]))
    return pd.read_csv(path)


def load_day(symbol: str, day: str, cache: Path) -> dict:
    trades = _fetch("aggTrades", symbol, day, cache)
    if trades is None:
        raise FileNotFoundError(f"Нет сделок {symbol} за {day}")
    depth = _fetch("bookDepth", symbol, day, cache)
    metrics = _fetch("metrics", symbol, day, cache)
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    kl = _fetch("klines", symbol, prev, cache, interval="1m")
    return {"trades": trades, "depth": depth, "metrics": metrics, "warm": kl}


def _ms(col: pd.Series) -> pd.Series:
    return (pd.to_datetime(col, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)


def _tick(prices: np.ndarray) -> float:
    u = np.unique(prices[:200_000])
    d = np.diff(u)
    d = d[d > 0]
    return float(d.min()) if len(d) else float(prices[0]) * 1e-4


def _events(symbol: str, data: dict):
    """Генератор событий (ts, порядок, тип, данные) по одной монете."""
    t = data["trades"]
    ts = t["transact_time"].to_numpy(np.int64)
    px = t["price"].to_numpy(float)
    qty = t["quantity"].to_numpy(float)
    buy = ~t["is_buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
    depth_ev = []
    if data["depth"] is not None:
        d = data["depth"]
        d["ts"] = _ms(d["timestamp"])
        for when, g in d.sort_values("ts").groupby("ts", sort=True):
            bids = [(-float(p), float(n)) for p, n in zip(g.percentage, g.notional) if p < 0]
            asks = [(float(p), float(n)) for p, n in zip(g.percentage, g.notional) if p > 0]
            depth_ev.append((int(when), bids, asks))
    oi_ev = []
    if data["metrics"] is not None:
        mt = data["metrics"]
        mt["ts"] = _ms(mt["create_time"])
        oi_ev = sorted(zip(mt.ts.astype(int), mt.sum_open_interest_value.astype(float)))  # в архиве строки не по порядку
    di = oi = 0
    for i in range(len(ts)):
        while di < len(depth_ev) and depth_ev[di][0] <= ts[i]:
            yield depth_ev[di][0], 0, "depth", symbol, depth_ev[di][1:]
            di += 1
        while oi < len(oi_ev) and oi_ev[oi][0] <= ts[i]:
            yield oi_ev[oi][0], 1, "oi", symbol, oi_ev[oi][1]
            oi += 1
        yield int(ts[i]), 2, "trade", symbol, (px[i], qty[i], bool(buy[i]))


def run_replay(cfg: dict, symbols: list[str], days: list[str], cache: Path, state_dir: Path,
               funding: dict | None = None, progress=print) -> Engine:
    engine = Engine(cfg, state_dir, source="binance (повтор)")
    loaded = {}
    for s in symbols:
        loaded[s] = load_day(s, days[0], cache)
        first = loaded[s]["trades"]
        m = engine.add_market(s, CurveBook(_tick(first["price"].to_numpy(float))))
        if loaded[s]["warm"] is not None:
            m.warmup_minutes(loaded[s]["warm"].iloc[:, 4].astype(float).tolist())
    for n, day in enumerate(days):
        if n:
            loaded = {s: load_day(s, day, cache) for s in symbols}
        progress(f"{day}: сделок {sum(len(v['trades']) for v in loaded.values()):,}".replace(",", " "))
        stream = heapq.merge(*(_events(s, loaded[s]) for s in symbols), key=lambda e: (e[0], e[1]))
        for ts, _, kind, s, payload in stream:
            m = engine.markets[s]
            if kind == "trade":
                p, q, b = payload
                m.book.set_price(p, ts)
                if funding and s in funding:
                    _apply_funding(m, funding[s], ts)
                engine.on_trade(s, ts, p, q, b)
            elif kind == "depth":
                bids, asks = payload
                if bids:
                    m.book.set_curve("bid", bids)
                if asks:
                    m.book.set_curve("ask", asks)
            elif kind == "oi":
                m.on_open_interest(ts, payload)
    engine.refresh_all()
    return engine


def _apply_funding(m, rates: pd.Series, ts: int) -> None:
    """Ставка ближайшей будущей выплаты (в архиве — фактическая, у биржи — прогнозная)."""
    if m.next_funding and ts < m.next_funding:
        return
    idx = rates.index.searchsorted(pd.Timestamp(ts, unit="ms", tz="UTC"), side="right")
    if idx < len(rates):
        m.next_funding = int(rates.index[idx].value // 10**6)
        m.funding_rate = float(rates.iloc[idx])
