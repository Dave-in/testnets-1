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


NON_CRYPTO = ["stock", "ETF", "commodity", "forex", "index"]   # поле symbolType у Bybit


def book_depth_usd(symbol: str, pct: float) -> float:
    """Объём стакана в $ в пределах pct % от середины — по более тонкой стороне."""
    ob = _get("/v5/market/orderbook", {"category": "linear", "symbol": symbol, "limit": 200})
    bids = [(float(p), float(q)) for p, q in ob.get("b", [])]
    asks = [(float(p), float(q)) for p, q in ob.get("a", [])]
    if not bids or not asks:
        return 0.0
    mid = (bids[0][0] + asks[0][0]) / 2
    lo, hi = mid * (1 - pct / 100), mid * (1 + pct / 100)
    return min(sum(p * q for p, q in bids if p >= lo), sum(p * q for p, q in asks if p <= hi))


def select_memes(cfg: dict) -> list[str]:
    """Мем-токены из списка [meme_universe], у которых на Bybit сейчас достаточно оборота,
    узкий спред и стакан. Топ по обороту здесь не исключается."""
    u = cfg.get("meme_universe", {})
    wanted = list(u.get("symbols", []))
    if not wanted:
        return []
    min_turnover = float(u.get("min_turnover_usd", 5e6))
    max_spread = float(u.get("max_spread_bps", 10.0)) / 1e4
    depth_pct = float(u.get("depth_pct", 0.1))
    min_depth = float(u.get("min_depth_usd", 3000.0))
    tickers = {t["symbol"]: t for t in _get("/v5/market/tickers", {"category": "linear"})["list"]}
    chosen = []
    for s in wanted:
        t = tickers.get(s)
        if t is None:
            continue
        bid, ask = float(t.get("bid1Price") or 0), float(t.get("ask1Price") or 0)
        spread = (ask - bid) / ((ask + bid) / 2) if bid > 0 and ask > 0 else 1.0
        if float(t.get("turnover24h") or 0) < min_turnover or spread > max_spread:
            continue
        try:
            if book_depth_usd(s, depth_pct) >= min_depth:
                chosen.append(s)
        except Exception as e:  # noqa: BLE001
            log.warning("Стакан %s не получен: %s", s, e)
    return chosen


def select_universe(cfg: dict) -> list[str]:
    """Монеты для торговли: только криптовалюты (без токенизированных акций, ETF, золота),
    без skip_top самых торгуемых за сутки, с оборотом, возрастом контракта, узким спредом
    и достаточной глубиной стакана. Ориентир для «Отставания» (BTC) сюда не входит —
    его поток подключается отдельно и не торгуется (см. run_live)."""
    u = cfg.get("universe", {})
    if u.get("symbols"):
        return list(u["symbols"])
    skip_top = int(u.get("skip_top", 0))
    min_turnover = float(u.get("min_turnover_usd", 30e6))
    min_age_ms = int(float(u.get("min_age_days", 60)) * 86400 * 1000)
    exclude = set(u.get("exclude", []))
    exclude_types = {t.lower() for t in u.get("exclude_types", NON_CRYPTO)}
    max_spread = float(u.get("max_spread_bps", 3.0)) / 1e4
    depth_pct = float(u.get("depth_pct", 0.1))
    min_depth = float(u.get("min_depth_usd", 6000.0))
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
    crypto = []                                   # (оборот, монета, возраст ок, спред)
    for t in _get("/v5/market/tickers", {"category": "linear"})["list"]:
        s, info = t["symbol"], instruments.get(t["symbol"], {})
        if (not s.endswith("USDT") or s in exclude or info.get("status") != "Trading"
                or info.get("contractType") != "LinearPerpetual"
                or (info.get("symbolType") or "").lower() in exclude_types):
            continue
        turnover = float(t.get("turnover24h") or 0)
        bid, ask = float(t.get("bid1Price") or 0), float(t.get("ask1Price") or 0)
        spread = (ask - bid) / ((ask + bid) / 2) if bid > 0 and ask > 0 else 1.0
        old_enough = now - int(info.get("launchTime", now)) >= min_age_ms
        crypto.append((turnover, s, old_enough, spread))
    crypto.sort(reverse=True)
    skipped = [s for _, s, _, _ in crypto[:skip_top]]
    if skipped:
        log.info("Самые торгуемые, не берём: %s", ", ".join(skipped))
    rows = [(v, s) for v, s, old_enough, spread in crypto[skip_top:]
            if old_enough and v >= min_turnover and spread <= max_spread]
    chosen = []
    for _, s in rows:
        if len(chosen) >= top:
            break
        try:
            depth = book_depth_usd(s, depth_pct)
        except Exception as e:  # noqa: BLE001
            log.warning("Стакан %s не получен: %s", s, e)
            continue
        if depth >= min_depth:
            chosen.append(s)
        else:
            log.info("%s пропущена: в пределах %.2f%% всего %.0f $", s, depth_pct, depth)
    return chosen


def apply_instrument(m, info: dict) -> None:
    """Правила контракта Bybit: шаг объёма, минимальный объём и сумма, шаг цены."""
    lot, pf = info.get("lotSizeFilter", {}), info.get("priceFilter", {})
    m.qty_step = float(lot.get("qtyStep") or 0)
    m.min_qty = float(lot.get("minOrderQty") or 0)
    m.max_mkt_qty = float(lot.get("maxMktOrderQty") or 0)
    m.min_notional = float(lot.get("minNotionalValue") or 0)
    m.tick = float(pf.get("tickSize") or 0)
    if info.get("fundingInterval"):
        m.funding_interval_h = float(info["fundingInterval"]) / 60     # Bybit даёт в минутах


def warmup(engine: Engine, symbols: list[str]) -> None:
    """Правила контрактов, минутные свечи за ~8 часов и тикеры — чтобы сразу была волатильность."""
    for s in symbols:
        m = engine.add_market(s)
        try:
            info = _get("/v5/market/instruments-info", {"category": "linear", "symbol": s})["list"]
            if info:
                apply_instrument(m, info[0])
        except Exception as e:  # noqa: BLE001
            log.warning("Правила контракта %s не получены: %s", s, e)
        try:
            kl = _get("/v5/market/kline", {"category": "linear", "symbol": s, "interval": 1, "limit": 500})
            rows = list(reversed(kl["list"]))[:-1]     # последняя минута ещё не закрыта
            m.warmup_minutes([float(k[4]) for k in rows])
            m.minute_usd.extend(float(k[6]) for k in rows if len(k) > 6)   # оборот минуты, $
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
            m._best = None                        # новый снимок — поток считаем заново
        else:
            m.book.delta(data.get("b", []), data.get("a", []), ts)
        engine.advance(ts)
        m.on_book_update()
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
    traded = select_universe(engine.cfg)
    log.info("Теневые боты: торгуем %d монет: %s", len(traded), ", ".join(traded))
    engine.universes = {"base": set(traded)}
    memes = []
    if any(b.universe == "meme" for b in engine.bots.values()):
        memes = select_memes(engine.cfg)
        engine.universes["meme"] = set(memes)
        log.info("Мем-бот: %d монет: %s", len(memes), ", ".join(memes) or "нет подходящих")
    ref = engine.cfg.get("lag", {}).get("ref", "BTCUSDT")
    symbols = list(dict.fromkeys(traded + memes))
    if ref not in symbols:
        symbols.insert(0, ref)
    warmup(engine, symbols)
    if ref not in traded and ref not in memes:  # ориентир для «Отставания»: данные есть, сделок нет
        engine.markets[ref].tradable = False
        log.info("%s — только ориентир, не торгуется", ref)
    engine.refresh_all()
    asyncio.run(_stream(engine, symbols, stop or asyncio.Event()))
