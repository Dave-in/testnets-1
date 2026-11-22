"""Живые данные Bybit (публичные, без ключей): выбор монет, разогрев, поток WebSocket.

Темы: orderbook.50 (стакан, 50 уровней), publicTrade (лента сделок), tickers (фандинг,
открытый интерес, маркировочная цена), allLiquidation (принудительные закрытия).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import requests

from .engine import Engine

log = logging.getLogger(__name__)
REST = "https://api.bybit.com"
WS = "wss://stream.bybit.com/v5/public/linear"


def _get(path: str, params: dict) -> dict:
    r = requests.get(REST + path, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit {path}: {data.get('retMsg')}")
    return data["result"]


def select_universe(cfg: dict) -> list[str]:
    """Монеты по ликвидности: оборот за сутки, возраст контракта, без исключённых."""
    u = cfg.get("universe", {})
    if u.get("symbols"):
        return list(u["symbols"])
    min_turnover = float(u.get("min_turnover_usd", 30e6))
    min_age_ms = int(float(u.get("min_age_days", 60)) * 86400 * 1000)
    exclude = set(u.get("exclude", []))
    top = int(u.get("top", 20))
    instruments = {}
    cursor = ""
    while True:
        res = _get("/v5/market/instruments-info", {"category": "linear", "limit": 1000, "cursor": cursor})
        for i in res["list"]:
            instruments[i["symbol"]] = i
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    now = int(time.time() * 1000)
    rows = []
    for t in _get("/v5/market/tickers", {"category": "linear"})["list"]:
        s, info = t["symbol"], instruments.get(t["symbol"], {})
        if (not s.endswith("USDT") or s in exclude or info.get("status") != "Trading"
                or info.get("contractType") != "LinearPerpetual"
                or now - int(info.get("launchTime", now)) < min_age_ms):
            continue
        turnover = float(t.get("turnover24h") or 0)
        if turnover >= min_turnover:
            rows.append((turnover, s))
    chosen = [s for _, s in sorted(rows, reverse=True)[:top]]
    ref = cfg.get("lag", {}).get("ref", "BTCUSDT")
    if ref not in chosen:
        chosen.insert(0, ref)
    return chosen


def warmup(engine: Engine, symbols: list[str]) -> None:
    """Минутные свечи за последние ~8 часов и текущие тикеры — чтобы сразу была волатильность."""
    for s in symbols:
        m = engine.add_market(s)
        try:
            kl = _get("/v5/market/kline", {"category": "linear", "symbol": s, "interval": 1, "limit": 500})
            m.warmup_minutes([float(k[4]) for k in reversed(kl["list"])])
        except Exception as e:  # noqa: BLE001
            log.warning("Разогрев %s не удался: %s", s, e)
    try:
        for t in _get("/v5/market/tickers", {"category": "linear"})["list"]:
            if t["symbol"] in engine.markets:
                apply_ticker(engine.markets[t["symbol"]], t, int(time.time() * 1000))
    except Exception as e:  # noqa: BLE001
        log.warning("Тикеры не получены: %s", e)


def apply_ticker(m, d: dict, ts: int) -> None:
    if d.get("fundingRate") not in (None, ""):
        m.funding_rate = float(d["fundingRate"])
    if d.get("nextFundingTime") not in (None, ""):
        m.next_funding = int(d["nextFundingTime"])
    if d.get("markPrice") not in (None, ""):
        m.mark_price = float(d["markPrice"])
    if d.get("turnover24h") not in (None, ""):
        m.turnover24h = float(d["turnover24h"])
    if d.get("openInterestValue") not in (None, ""):
        v = float(d["openInterestValue"])
        if not m.oi or m.oi[-1][1] != v:
            m.on_open_interest(ts, v)


def handle_message(engine: Engine, msg: dict) -> None:
    """Разобрать одно сообщение WebSocket и передать в движок."""
    topic = msg.get("topic", "")
    if not topic:
        return
    kind, _, symbol = topic.rpartition(".")
    m = engine.markets.get(symbol)
    if m is None:
        return
    ts = int(msg.get("ts") or msg.get("cts") or 0)
    data = msg.get("data")
    if kind == "orderbook.50":
        if msg.get("type") == "snapshot" or data.get("u") == 1:
            m.book.snapshot(data.get("b", []), data.get("a", []), ts)
        else:
            m.book.delta(data.get("b", []), data.get("a", []), ts)
        engine.advance(ts)
    elif kind == "publicTrade":
        for t in data:
            engine.on_trade(symbol, int(t["T"]), float(t["p"]), float(t["v"]), t["S"] == "Buy")
    elif kind == "tickers":
        apply_ticker(m, data, ts)
        engine.advance(ts)
    elif kind == "allLiquidation":
        for liq in data:
            m.on_liquidation(int(liq["T"]), float(liq["v"]) * float(liq["p"]))
        engine.advance(ts)


async def _stream(engine: Engine, symbols: list[str], stop: asyncio.Event) -> None:
    import aiohttp

    topics = [f"{t}.{s}" for s in symbols
              for t in ("orderbook.50", "publicTrade", "tickers", "allLiquidation")]
    backoff = 1
    async with aiohttp.ClientSession() as session:
        while not stop.is_set():
            try:
                async with session.ws_connect(WS, heartbeat=None, max_msg_size=0) as ws:
                    for i in range(0, len(topics), 10):
                        await ws.send_json({"op": "subscribe", "args": topics[i:i + 10]})
                    log.info("Bybit: подключено, тем %d", len(topics))
                    backoff = 1
                    last_ping = time.time()
                    while not stop.is_set():
                        if time.time() - last_ping > 20:
                            await ws.send_json({"op": "ping"})
                            last_ping = time.time()
                        try:
                            msg = await ws.receive(timeout=5)
                        except asyncio.TimeoutError:
                            engine.advance(int(time.time() * 1000))
                            continue
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            continue
                        handle_message(engine, json.loads(msg.data))
                        if any(b.stop_requested for b in engine.bots.values()):
                            stop.set()
            except Exception as e:  # noqa: BLE001 — сеть рвётся, это нормально: переподключаемся
                log.warning("Bybit: соединение потеряно (%s), повтор через %d с", e, backoff)
            if not stop.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
    engine.refresh_all()


def run_live(engine: Engine, stop: asyncio.Event | None = None) -> None:
    symbols = select_universe(engine.cfg)
    log.info("Теневые боты: монеты %s", ", ".join(symbols))
    warmup(engine, symbols)
    engine.refresh_all()
    asyncio.run(_stream(engine, symbols, stop or asyncio.Event()))
