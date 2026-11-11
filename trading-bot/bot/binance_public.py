"""Публичные рыночные данные Binance напрямую по REST (без ключей и без ccxt).

Пробуем несколько адресов по очереди: основной api.binance.com закрыт в части
стран (ответ 451), зеркало data-api.binance.vision отдаёт те же свечи и цены.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from .data import _to_frame

HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
FUTURES_HOST = "https://fapi.binance.com"
TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "2h": 7_200_000,
                "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000}


class BinancePublic:
    def __init__(self, hosts: list[str] | None = None, timeout: float = 20.0):
        self.hosts = list(hosts or HOSTS)
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict):
        last: Exception | None = None
        for host in list(self.hosts):
            try:
                r = self.session.get(host + path, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    if host != self.hosts[0]:  # рабочий адрес — в начало списка
                        self.hosts.remove(host)
                        self.hosts.insert(0, host)
                    return r.json()
                last = RuntimeError(f"{host}{path}: HTTP {r.status_code} {r.text[:120]}")
            except requests.RequestException as e:
                last = e
        raise RuntimeError(f"Binance недоступен: {last}")

    @staticmethod
    def _pair(symbol: str) -> str:
        return symbol.replace("/", "")

    def klines(self, symbol: str, timeframe: str, start_ms: int | None = None,
               limit: int = 1000) -> list[list]:
        params = {"symbol": self._pair(symbol), "interval": timeframe, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        return [[k[0], *map(float, k[1:6])] for k in self._get("/api/v3/klines", params)]

    def history(self, symbol: str, timeframe: str, since: str,
                until: str | None = None) -> pd.DataFrame:
        """Все закрытые свечи с даты since."""
        step = TIMEFRAME_MS[timeframe]
        t = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
        now = int(time.time() * 1000)
        end = int(pd.Timestamp(until, tz="UTC").timestamp() * 1000) if until else now
        rows: list[list] = []
        while t < end:
            batch = self.klines(symbol, timeframe, t)
            if not batch:
                break
            rows.extend(batch)
            t = batch[-1][0] + step
            if len(batch) < 1000:
                break
        df = _to_frame(rows)
        closes_at = df.index + pd.Timedelta(milliseconds=step)
        return df[(closes_at <= pd.Timestamp(now, unit="ms", tz="UTC"))
                  & (df.index < pd.Timestamp(end, unit="ms", tz="UTC"))]

    # --- интерфейс MarketData для live-режимов -----------------------------
    def closed_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        df = _to_frame(self.klines(symbol, timeframe, limit=min(limit + 1, 1000)))
        step = pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe])
        if len(df) and df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]
        return df

    def funding_rates(self, symbols: list[str]) -> dict[str, float]:
        """Последняя ставка фандинга USDT-M фьючерсов (за один период выплаты)."""
        r = self.session.get(FUTURES_HOST + "/fapi/v1/premiumIndex", timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"fapi: HTTP {r.status_code}")
        want = {self._pair(s): s for s in symbols}
        return {want[d["symbol"]]: float(d["lastFundingRate"])
                for d in r.json() if d["symbol"] in want and d.get("lastFundingRate") not in ("", None)}

    def prices(self, symbols: list[str]) -> dict[str, float]:
        want = {self._pair(s): s for s in symbols}
        data = self._get("/api/v3/ticker/price", {})
        out = {want[d["symbol"]]: float(d["price"]) for d in data if d["symbol"] in want}
        missing = set(symbols) - set(out)
        if missing:
            raise RuntimeError(f"Нет цен для {sorted(missing)}")
        return out
