import json
import math

import numpy as np
import pandas as pd
import pytest

from bot import messages
from bot.shadow.book import CurveBook, OrderBook
from bot.shadow.bybit import handle_message
from bot.shadow.engine import Engine
from bot.shadow.market import SymbolMarket
from bot.shadow.sim import Account, Order
from bot.shadow import strategies as S

T0 = 1_790_000_000_000  # мс


def book(mid=100.0, step=0.1, size=10.0, levels=20, bid_size=None, ask_size=None):
    b = OrderBook()
    b.snapshot([(mid - step / 2 - i * step, bid_size or size) for i in range(levels)],
               [(mid + step / 2 + i * step, ask_size or size) for i in range(levels)], T0)
    return b


def market(prices_by_sec, minute_closes=None, sym="ETHUSDT", ob=None):
    m = SymbolMarket(sym, ob or book(prices_by_sec[-1]))
    m.warmup_minutes(minute_closes if minute_closes is not None else
                     list(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.001, 300)))))
    m.sec.extend(prices_by_sec)
    m.last_price = prices_by_sec[-1]
    return m


# --- стакан ----------------------------------------------------------------------

def test_order_book_snapshot_delta_walk():
    b = book(100.0, 0.1, 10.0, levels=3)
    assert b.best_bid() == pytest.approx(99.95) and b.best_ask() == pytest.approx(100.05)
    avg, filled = b.walk("buy", 15)                       # 10 по 100.05 и 5 по 100.15
    assert filled == 15 and avg == pytest.approx((10 * 100.05 + 5 * 100.15) / 15)
    b.delta([], [(100.05, 0)], T0 + 1)                    # лучший уровень продаж съели
    assert b.best_ask() == pytest.approx(100.15)
    b2 = book(100.0, bid_size=30.0, ask_size=10.0)
    assert b2.imbalance(10) == pytest.approx(0.5, abs=0.01)
    assert b2.microprice() > b2.mid()                     # покупателей больше → микроцена выше


def test_curve_book_slippage_grows_with_size():
    cb = CurveBook(tick=0.01)
    cb.set_price(100.0, T0)
    cb.set_curve("ask", [(1.0, 1_000_000.0)])
    cb.set_curve("bid", [(1.0, 1_000_000.0)])
    small, _ = cb.walk("buy", 10)        # 1 000 $
    big, _ = cb.walk("buy", 10_000)      # 1 000 000 $ — вся глубина в 1%
    assert 100.0 < small < big and big == pytest.approx(100.005 + 0.5, rel=1e-3)


# --- исполнение ----------------------------------------------------------------------

def test_limit_order_waits_for_queue_then_fills_as_maker():
    ob = book(100.0)
    acc = Account("t", 100.0)
    o = acc.submit(Order("X", "buy", "limit", 1.0, 99.95, "entry", meta={"sl_dist": 1.0}), ob, T0)
    acc.on_trade("X", 99.90, 50.0, taker_buy=False, book=ob, now=T0 + 100)  # ордер ещё в пути
    assert "X" not in acc.positions and not o.queued
    acc.process({"X": ob}, T0 + 250)                     # дошёл до биржи: встал за 10 монетами
    assert o.queue_ahead == 10.0
    acc.on_trade("X", 99.95, 5.0, taker_buy=False, book=ob, now=T0 + 301)
    acc.on_trade("X", 100.05, 50.0, taker_buy=True, book=ob, now=T0 + 302)  # покупки нас не касаются
    assert "X" not in acc.positions
    acc.on_trade("X", 99.95, 6.0, taker_buy=False, book=ob, now=T0 + 303)  # очередь 10 + наш 1 пройдены
    pos = acc.positions["X"]
    assert pos.entry == 99.95 and acc.stats["fees"] == pytest.approx(99.95 * 0.0002)


def test_limit_order_fills_when_price_trades_through():
    ob = book(100.0)
    acc = Account("t", 100.0)
    acc.submit(Order("X", "sell", "limit", 1.0, 100.05, "entry", meta={"sl_dist": 1.0}), ob, T0)
    acc.on_trade("X", 100.15, 0.1, taker_buy=True, book=ob, now=T0 + 300)
    assert acc.positions["X"].side == -1


def test_market_entry_then_stop_with_slippage_and_pnl():
    ob = book(100.0, 0.1, 1.0, levels=50)                # тонкий стакан: по 1 монете на уровне
    acc = Account("t", 1000.0, latency_ms=250)
    acc.submit(Order("X", "buy", "market", 3.0, meta={"sl_dist": 1.0, "tp_dist": 2.0,
                                                       "mid0": 100.0, "reason": "тест"}), ob, T0)
    acc.process({"X": ob}, T0 + 100)
    assert "X" not in acc.positions                      # задержка ещё не прошла
    acc.process({"X": ob}, T0 + 250)
    pos = acc.positions["X"]
    assert pos.entry == pytest.approx((100.05 + 100.15 + 100.25) / 3)       # прошли 3 уровня
    assert pos.slippage_bps > 0
    assert {o.purpose for o in acc.orders} == {"sl", "tp"}
    acc.on_trade("X", pos.sl - 0.01, 1.0, taker_buy=False, book=ob, now=T0 + 1000)  # стоп сработал
    ob2 = book(pos.sl - 0.5, 0.1, 1.0, levels=50)
    acc.process({"X": ob2}, T0 + 1250)
    assert "X" not in acc.positions and acc.orders == []
    assert acc.stats["closed"] == 1 and acc.stats["wins"] == 0
    assert acc.cash == pytest.approx(1000.0 + acc.stats["realized"])          # учёт сходится


def test_take_profit_and_funding_once():
    ob = book(100.0)
    acc = Account("t", 1000.0)
    acc.submit(Order("X", "buy", "market", 1.0, meta={"sl_dist": 1.0, "tp_dist": 1.0, "mid0": 100.0}),
               ob, T0)
    acc.process({"X": ob}, T0 + 300)
    m = SymbolMarket("X", ob)
    m.last_price, m.funding_rate, m.next_funding, m.mark_price = 100.0, 0.001, T0 + 5000, 100.0
    acc.housekeeping({"X": m}, T0 + 6000)
    acc.housekeeping({"X": m}, T0 + 7000)
    assert acc.stats["funding"] == pytest.approx(0.1)    # лонг платит 0.1% один раз
    tp = acc.positions["X"].tp
    acc.on_trade("X", tp + 0.1, 1.0, taker_buy=True, book=ob, now=T0 + 8000)
    assert "X" not in acc.positions and acc.stats["wins"] == 1
    assert acc.cash == pytest.approx(1000.0 + acc.stats["realized"])


def test_time_stop_and_trailing():
    ob = book(100.0)
    acc = Account("t", 1000.0, latency_ms=0)
    acc.submit(Order("X", "buy", "market", 1.0, meta={"sl_dist": 1.0, "trail": 1.0,
                                                       "time_stop_s": 10, "mid0": 100.0}), ob, T0)
    acc.process({"X": ob}, T0)
    m = SymbolMarket("X", book(105.0))
    acc.housekeeping({"X": m}, T0 + 1000)
    assert acc.positions["X"].sl == pytest.approx(104.0)  # стоп подтянулся за ценой
    acc.housekeeping({"X": m}, T0 + 11_000)               # время вышло
    acc.process({"X": m.book}, T0 + 11_000)
    assert "X" not in acc.positions and acc.stats["wins"] == 1


# --- стратегии ------------------------------------------------------------------------

def calm(n=900, price=100.0):
    return [price] * n


def test_fade_signal_after_exhausted_spike():
    prices = calm(600) + list(np.linspace(100, 103, 259)) + [103.0] + [102.8] * 40
    m = market(prices)
    sig = S.FadeSpike().check(m, T0, {})
    m.add_flow(T0 - 1000, 100.5, 1000.0)          # средняя цена ниже текущей
    sig = S.FadeSpike().check(m, T0, {})
    assert sig is not None and sig.side == -1 and sig.tp_dist > 0
    assert S.FadeSpike().check(market(calm()), T0, {}) is None


def test_leadlag_enters_alt_in_btc_direction():
    rng = np.random.default_rng(3)
    btc_min = list(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 400))))
    alt_min = list(10 * np.exp(np.log(np.array(btc_min) / 100)))         # β = 1
    btc = market(calm(200, btc_min[-1]) + [btc_min[-1] * 1.01], btc_min, "BTCUSDT")
    alt = market(calm(201, alt_min[-1]), alt_min, "SOLUSDT")
    sig = S.LeadLag().check(alt, T0, {"BTCUSDT": btc, "SOLUSDT": alt})
    assert sig is not None and sig.side == 1 and sig.tp_dist > 0
    assert S.LeadLag().check(btc, T0, {"BTCUSDT": btc}) is None        # сам BTC не торгуем


def test_liquidation_cascade_arms_then_enters_after_quiet():
    strat = S.LiquidationCascade()
    m = market(calm(600) + list(np.linspace(100, 96, 60)))
    m.on_liquidation(T0 - 5000, 400_000.0)
    assert strat.check(m, T0, {}) is None and m.symbol in strat.state   # каскад замечен
    m.sec.extend([96.3] * 20)                                           # 20 с тишины, без нового минимума
    sig = strat.check(m, T0 + 20_000, {})
    assert sig is not None and sig.side == 1                            # покупаем против каскада
    assert sig.tp_dist > 0 and sig.sl_dist > 0


def test_book_imbalance_needs_persistence_and_flow():
    ob = book(100.0, bid_size=40.0, ask_size=8.0)
    m = market(calm(), ob=ob)
    m.add_flow(T0 - 1000, 100.0, 5000.0)                       # агрессивные покупки
    strat = S.BookImbalance()
    sigs = [strat.check(m, T0 + i * 1000, {}) for i in range(5)]
    assert sigs[:4] == [None] * 4 and sigs[4] is not None                # 5 секунд подряд
    assert sigs[4].entry == "limit" and sigs[4].limit_price == ob.best_bid()


def test_oi_and_funding_signals():
    m = market(calm(300) + list(np.linspace(100, 102, 301)))
    m.on_open_interest(T0 - 400_000, 1e9)
    m.on_open_interest(T0 - 1000, 1.02e9)
    sig = S.OpenInterestFlow().check(m, T0, {})
    assert sig is not None and sig.side == 1 and sig.trail > 0
    f = market(calm(700) + list(np.linspace(100, 99, 300)))
    f.funding_rate = 0.001                                                 # толпа в лонгах
    f.add_flow(T0 - 1000, 99.8, 1000.0)
    sig = S.FundingSkew().check(f, T0, {})
    assert sig is not None and sig.side == -1


# --- движок, Telegram, Bybit -------------------------------------------------------------

def shadow_cfg(**over):
    cfg = {"bots": ["fade"], "save_every_s": 1,
           "account": {"capital": 100.0, "latency_ms": 0}, "fade": {}}
    cfg.update(over)
    return cfg


def test_engine_trades_journals_and_reports(tmp_path):
    eng = Engine(shadow_cfg(), tmp_path, source="тест")
    m = eng.add_market("ETHUSDT", book(100.0))
    m.warmup_minutes(list(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.001, 300)))))
    path = calm(600) + list(np.linspace(100, 104, 270)) + [104.0] + [103.9] * 40 + \
        list(np.linspace(103.9, 101, 200))
    t = T0
    for p in path:
        t += 1000
        m.book = book(p)
        eng.on_trade("ETHUSDT", t, p, 1.0, taker_buy=p < 104)
    bot = eng.bots["fade"]
    assert bot.account.stats["opened"] >= 1
    rows = (tmp_path / "shadow_fade_trades.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("ts,symbol,action") and any(",open," in r for r in rows[1:])
    snap = json.loads((tmp_path / "shadow_fade.json").read_text(encoding="utf-8"))["snapshot"]
    assert "Счёт:" in messages.status_text(snap) and snap["stats"]["closed"] >= 0
    assert "Сравнение счетов" in messages.group_report([("Тень", snap)])

    # управление: пауза и «закрыть всё» через control-файл
    bot.control_path.write_text(json.dumps({"command": "close-all"}))
    t += 1000
    eng.on_trade("ETHUSDT", t, 101.0, 1.0, True)
    eng.advance(t + 1000)
    assert bot.paused and not bot.account.positions


def test_engine_skips_signals_eaten_by_fees(tmp_path):
    eng = Engine(shadow_cfg(), tmp_path)
    m = eng.add_market("ETHUSDT", book(100.0))
    bot = eng.bots["fade"]
    eng._enter(bot, m, S.Signal(1, sl_dist=0.05, tp_dist=0.05), T0)     # тейк 0.05% < 2.5×0.11%
    assert not bot.account.orders
    eng._enter(bot, m, S.Signal(1, sl_dist=0.05, tp_dist=2.0), T0)      # стоп расширен до 0.4%
    o = bot.account.orders[0]
    assert o.meta["sl_dist"] == pytest.approx(0.4)
    assert o.qty * 100 <= 3 * 100 + 1e-9                                 # не больше 3 размеров счёта
    assert o.qty == pytest.approx(min(1.0 / 0.4, 3.0))


def test_bybit_message_parsing(tmp_path):
    eng = Engine(shadow_cfg(), tmp_path)
    m = eng.add_market("SOLUSDT")
    handle_message(eng, {"topic": "orderbook.50.SOLUSDT", "type": "snapshot", "ts": T0,
                         "data": {"s": "SOLUSDT", "b": [["99.9", "5"], ["99.8", "7"]],
                                  "a": [["100.1", "3"]], "u": 1}})
    handle_message(eng, {"topic": "orderbook.50.SOLUSDT", "type": "delta", "ts": T0 + 20,
                         "data": {"s": "SOLUSDT", "b": [["99.9", "0"]], "a": [["100.0", "2"]], "u": 2}})
    assert m.book.best_bid() == 99.8 and m.book.best_ask() == 100.0
    handle_message(eng, {"topic": "publicTrade.SOLUSDT", "ts": T0 + 30,
                         "data": [{"T": T0 + 30, "s": "SOLUSDT", "S": "Sell", "v": "2", "p": "99.8"}]})
    assert m.last_price == 99.8 and m.trades[-1][3] < 0
    handle_message(eng, {"topic": "tickers.SOLUSDT", "type": "delta", "ts": T0 + 40,
                         "data": {"symbol": "SOLUSDT", "fundingRate": "0.0003",
                                  "nextFundingTime": str(T0 + 3_600_000), "openInterestValue": "5000000"}})
    assert m.funding_rate == 0.0003 and m.oi[-1][1] == 5e6
    handle_message(eng, {"topic": "allLiquidation.SOLUSDT", "ts": T0 + 50,
                         "data": [{"T": T0 + 50, "s": "SOLUSDT", "S": "Buy", "v": "10", "p": "99"}]})
    assert m.liq_sum(60, T0 + 60) == pytest.approx(990.0)
    handle_message(eng, {"op": "pong"})                                    # служебные — пропускаем


def test_replay_events_are_time_ordered():
    from bot.shadow.replay import _events

    trades = pd.DataFrame({"transact_time": [T0 + 10, T0 + 20, T0 + 400_000],
                           "price": [1.0, 1.1, 1.2], "quantity": [1, 2, 3],
                           "is_buyer_maker": ["true", "false", "true"]})
    metrics = pd.DataFrame({"create_time": pd.to_datetime([T0 + 300_000, T0], unit="ms").astype(str),
                            "sum_open_interest_value": [2.0, 1.0]})     # не по порядку, как в архиве
    ev = list(_events("X", {"trades": trades, "depth": None, "metrics": metrics}))
    ts = [e[0] for e in ev]
    assert ts == sorted(ts)
    kinds = [e[2] for e in ev]
    assert kinds.count("oi") == 2 and kinds.count("trade") == 3
    trades_ev = [e[4] for e in ev if e[2] == "trade"]
    assert [t[2] for t in trades_ev] == [False, True, False]            # is_buyer_maker → тейкер


def test_default_config_loads_all_six_bots(tmp_path):
    from pathlib import Path

    from bot.config import tomllib

    root = Path(__file__).resolve().parents[1]
    with open(root / "config.shadow.toml", "rb") as f:
        cfg = tomllib.load(f)
    eng = Engine(cfg, tmp_path)
    assert list(eng.bots) == ["liq", "book", "lag", "oi", "fund", "fade"]
    assert len({b.title for b in eng.bots.values()}) == 6
    assert all(b.account.start == 100.0 for b in eng.bots.values())


def test_shadow_bots_join_group_reports_and_commands(tmp_path, monkeypatch):
    import bot.shadow.bybit as bybit
    from bot.group import Group, start_shadow
    from bot.telegram import CommandListener, Telegram

    started = []
    monkeypatch.setattr(bybit, "run_live", lambda engine, stop=None: started.append(engine))
    sent = []
    tg = Telegram("T", 111, lambda m, p, t: sent.append(p) or {"ok": True})
    cfg = {"bots": ["lag", "fade"], "account": {"capital": 100.0}}
    extra = start_shadow(cfg, tmp_path, tg)
    import time as _t
    for _ in range(50):
        if started:
            break
        _t.sleep(0.02)
    assert started and set(extra) == {"Тень · Отставание от BTC", "Тень · Откат (контроль)"}
    engine = started[0]
    engine.add_market("BTCUSDT", book(100.0))
    engine.advance(T0)
    engine.refresh_all()
    for b in engine.bots.values():                   # смена суток → отчёт за день
        b.day_snapshot = b.snapshot
    group = Group({}, tg, 1, extra)
    group.maybe_report()
    report = [p["text"] for p in sent if "Отчёт за день" in p["text"]]
    assert len(report) == 1 and "Тень · Откат (контроль)" in report[0]

    listener = CommandListener(Telegram("T", 111, lambda m, p, t: sent.append(p) or {"ok": True}),
                               bots=[(n, b.state_path, b.control_path) for n, b in extra.items()])
    assert "Тень · Отставание от BTC" in listener.handle("/status")
    listener.handle("/pause")
    engine.advance(T0 + 2000)
    assert all(b.paused for b in engine.bots.values())


def test_websocket_client_subscribes_parses_and_reconnects(tmp_path, monkeypatch):
    """Локальный сервер вместо Bybit: подписка, данные, обрыв, повторное подключение."""
    aiohttp = pytest.importorskip("aiohttp")
    import asyncio

    from aiohttp import web

    import bot.shadow.bybit as bybit

    subs, connects = [], []

    import time as _time
    T0 = int(_time.time() * 1000)                        # здесь время биржи ≈ местное

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connects.append(1)
        async for msg in ws:
            data = json.loads(msg.data)
            if data.get("op") == "subscribe":
                subs.extend(data["args"])
                if len(subs) >= 8:                       # 2 монеты × 4 темы
                    await ws.send_json({"topic": "orderbook.50.SOLUSDT", "type": "snapshot", "ts": T0,
                                        "data": {"b": [["99", "1"]], "a": [["101", "1"]], "u": 1}})
                    await ws.send_json({"topic": "publicTrade.SOLUSDT", "ts": T0 + 5,
                                        "data": [{"T": T0 + 5, "S": "Buy", "v": "1", "p": "100.5"}]})
                    if len(connects) == 1:
                        await ws.close()                 # обрыв связи
                        return ws
        return ws

    async def main():
        app = web.Application()
        app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(bybit, "WS", f"http://127.0.0.1:{port}/ws")
        eng = Engine({"bots": ["fade"]}, tmp_path)
        for s in ("SOLUSDT", "BTCUSDT"):
            eng.add_market(s)
        stop = asyncio.Event()
        task = asyncio.create_task(bybit._stream(eng, ["SOLUSDT", "BTCUSDT"], stop))
        for _ in range(100):
            if len(connects) >= 2 and eng.markets["SOLUSDT"].last_price:
                break
            await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 10)
        await runner.cleanup()
        return eng

    real_sleep = asyncio.sleep
    monkeypatch.setattr(bybit.asyncio, "sleep", lambda s: real_sleep(0.01))     # без долгих пауз
    eng = asyncio.run(main())
    m = eng.markets["SOLUSDT"]
    assert len(connects) >= 2                              # переподключился после обрыва
    assert {"orderbook.50.SOLUSDT", "publicTrade.BTCUSDT", "tickers.SOLUSDT",
            "allLiquidation.BTCUSDT"} <= set(subs)
    assert m.book.best_bid() == 99.0 and m.last_price == 100.5


def test_engine_fast_forwards_long_gaps(tmp_path):
    eng = Engine({"bots": ["fade"]}, tmp_path)
    eng.add_market("ETHUSDT", book(100.0))
    eng.advance(T0)
    eng.advance(T0 + 30 * 86400 * 1000)                  # месяц без данных — не зависаем
    assert eng.now == T0 + 30 * 86400 * 1000


def test_shadow_warns_once_when_bybit_is_blocked(tmp_path, monkeypatch):
    import bot.shadow.bybit as bybit
    from bot.group import start_shadow
    from bot.telegram import Telegram

    calls = []

    def blocked(engine, stop=None):
        calls.append(1)
        raise RuntimeError("403 Client Error: Forbidden")
    monkeypatch.setattr(bybit, "run_live", blocked)
    import threading
    import types

    import bot.group as group
    hold = threading.Event()
    fake_time = types.SimpleNamespace(sleep=lambda s: None if len(calls) < 3 else hold.wait())
    monkeypatch.setattr(group, "time", fake_time)          # без минутных пауз между попытками
    sent = []
    tg = Telegram("T", 111, lambda m, p, t: sent.append(p["text"]) or {"ok": True})
    start_shadow({"bots": ["fade"]}, tmp_path, tg)
    import time as _t
    for _ in range(100):
        if len(calls) >= 3:
            break
        _t.sleep(0.02)
    warnings = [x for x in sent if "не могут получить данные Bybit" in x]
    assert len(calls) >= 3 and len(warnings) == 1 and "Сингапуре" in warnings[0]


# --- исправления: лимит позиций, частичное закрытие, выбор монет -------------------------

def test_pending_entries_count_towards_position_and_leverage_limits(tmp_path):
    class Always(S.Strategy):
        key, title, defaults = "fade", "x", {}

        def check(self, m, now, markets):
            return S.Signal(1, m.price() * 0.01, m.price() * 0.05)

    e = Engine({"bots": ["fade"], "account": {"capital": 100.0, "max_positions": 3}}, tmp_path)
    e.bots["fade"].strategy = Always()
    for i in range(10):
        e.add_market(f"C{i}USDT", book(100.0, size=1e6)).last_price = 100.0
    for t in range(1, 5):
        e.advance(T0 + t * 1000)
    acc = e.bots["fade"].account
    prices = {s: 100.0 for s in e.markets}
    assert len(acc.positions) == 3
    assert acc.gross(prices) <= 10 * acc.equity(prices) + 1e-9


def test_partial_close_keeps_remainder_and_accounts_once():
    ob = book(100.0, 0.1, 10.0, levels=50)
    acc = Account("t", 1000.0)
    acc.submit(Order("X", "buy", "market", 3.0, meta={"sl_dist": 1.0, "mid0": 100.0}), ob, T0)
    acc.process({"X": ob}, T0 + 300)
    thin = book(101.0, 0.1, 1.0, levels=1)                  # в стакане всего 1 монета на покупку
    acc.close("X", thin, T0 + 1000, "manual")
    acc.process({"X": thin}, T0 + 1300)
    assert acc.positions["X"].qty == pytest.approx(2.0)      # закрыта треть, остаток жив
    assert acc.stats["closed"] == 0
    acc.process({"X": book(101.0, 0.1, 10.0)}, T0 + 1600)   # стакан восстановился
    assert "X" not in acc.positions and acc.stats["closed"] == 1
    assert acc.cash == pytest.approx(1000.0 + acc.stats["realized"])


def test_restored_take_profit_fills_only_through_level():
    acc = Account("t", 1000.0)
    acc.load_dict({"positions": {"X": {"symbol": "X", "side": 1, "qty": 1.0, "entry": 100.0,
                                       "opened": T0, "sl": 99.0, "tp": 101.0}}})
    ob = book(101.0)
    acc.on_trade("X", 101.0, 100.0, taker_buy=True, book=ob, now=T0 + 1)    # на уровне — мало
    assert "X" in acc.positions
    acc.on_trade("X", 101.1, 0.1, taker_buy=True, book=ob, now=T0 + 2)      # насквозь
    assert "X" not in acc.positions


def test_universe_skips_stocks_wide_spread_and_thin_books(monkeypatch):
    from bot.shadow import bybit
    old = "1600000000000"
    inst = [{"symbol": s, "status": "Trading", "contractType": "LinearPerpetual",
             "launchTime": old, "symbolType": t}
            for s, t in [("BTCUSDT", ""), ("NVDAUSDT", "stock"), ("WIDEUSDT", ""),
                         ("THINUSDT", ""), ("GOODUSDT", "")]]
    tick = {"BTCUSDT": (100.0, 100.01), "NVDAUSDT": (100.0, 100.01), "WIDEUSDT": (100.0, 100.1),
            "THINUSDT": (100.0, 100.01), "GOODUSDT": (100.0, 100.01)}

    def fake_get(path, params):
        if path.endswith("instruments-info"):
            return {"list": inst}
        if path.endswith("tickers"):
            return {"list": [{"symbol": s, "turnover24h": "1e9", "bid1Price": b, "ask1Price": a}
                             for s, (b, a) in tick.items()]}
        size = 1.0 if params["symbol"] == "THINUSDT" else 1000.0
        return {"b": [["100.0", size]], "a": [["100.01", size]]}

    monkeypatch.setattr(bybit, "_get", fake_get)
    assert sorted(bybit.select_universe({"universe": {}})) == ["BTCUSDT", "GOODUSDT"]


# --- Telegram: группа с темами, сообщение о каждой сделке с балансом и статистикой --------

class FakeTG:
    """Подмена API Telegram: запоминает вызовы, темы получают номера 100, 101, …"""

    def __init__(self):
        self.calls, self.next_topic = [], 100

    def __call__(self, method, payload, timeout):
        self.calls.append((method, payload))
        if method == "createForumTopic":
            self.next_topic += 1
            return {"ok": True, "result": {"message_thread_id": self.next_topic - 1}}
        return {"ok": True, "result": {}}

    def sent(self):
        return [p for m, p in self.calls if m == "sendMessage"]


def test_shadow_trade_message_has_bot_balance_and_stats():
    from bot.shadow.sim import FillReport
    ev = FillReport("ETHUSDT", "close", 1, 1.0, 101.0, 0.05, "tp", 0.0, 0.85, 0.0085)
    text = messages.shadow_trade(ev, "Ликвидации", {"equity": 101.7, "start": 100.0, "open": 1,
                                                    "closed": 4, "wins": 3, "realized": 1.9,
                                                    "fees": 0.6})
    assert "Ликвидации" in text and "ЗАКРЫТЬ ЛОНГ ETHUSDT" in text and "тейк" in text
    assert "101.70 USDT" in text and "+1.70%" in text and "Сделок: 4" in text and "75%" in text


def test_channel_binds_group_creates_topics_once_and_routes_by_topic(tmp_path):
    from bot.telegram import ShadowChannel, Telegram, TopicNotifier
    api = FakeTG()
    tg = Telegram("T", 111, api)
    ch = ShadowChannel(tg, tmp_path / "tg.json", start=False)
    private = []
    liq = TopicNotifier(ch, "Ликвидации", fallback=types_ns(send=lambda t: private.append(t) or True))
    TopicNotifier(ch, "Стакан")
    assert not liq.send("сделка", trade=True) and private == []     # группы нет — сделки молчат
    assert liq.send("предупреждение") and private == ["предупреждение"]
    assert "✅" in ch.bind("-1001")
    assert ch.topics == {"Ликвидации": 100, "Стакан": 101}
    ch.bind("-1001")                                                 # повторно — без дублей
    assert sum(m == "createForumTopic" for m, _ in api.calls) == 2
    assert liq.send("сделка", trade=True)
    ch.send_one(*ch.queue.get_nowait())
    last = api.sent()[-1]
    assert last["chat_id"] == "-1001" and last["message_thread_id"] == 100
    again = ShadowChannel(tg, tmp_path / "tg.json", start=False)    # после перезапуска
    assert again.chat_id == "-1001" and again.topics["Стакан"] == 101


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def test_group_commands_only_owner_binds_and_only_read_commands(tmp_path):
    from bot.telegram import CommandListener, ShadowChannel, Telegram
    api = FakeTG()
    tg = Telegram("T", 111, api)
    ch = ShadowChannel(tg, tmp_path / "tg.json", start=False)
    ch.register("Откат (контроль)")
    state = tmp_path / "s.json"
    state.write_text(json.dumps({"snapshot": {"equity": 100.0}}), encoding="utf-8")
    lst = CommandListener(tg, bots=[("Тень", state, tmp_path / "c.json")], group=ch)

    def feed(text, user, thread=None):
        msg = {"chat": {"id": -1001}, "from": {"id": user}, "text": text}
        if thread:
            msg.update(message_thread_id=thread, is_topic_message=True)
        lst.tg.updates = lambda: [{"update_id": 1, "message": msg}]
        lst.poll_once()

    feed("/here", user=999)                                # чужой не может привязать
    assert not ch.bound
    feed("/here@my_bot", user=111)
    assert ch.chat_id == "-1001" and "Откат (контроль)" in ch.topics
    feed("/pause", user=111)                               # управление в группе не работает
    assert not (tmp_path / "c.json").exists()
    n = len(api.sent())
    feed("/help", user=999, thread=100)
    reply = api.sent()[-1]
    assert len(api.sent()) == n + 1 and reply["chat_id"] == "-1001" and reply["message_thread_id"] == 100


def test_telegram_waits_and_retries_when_rate_limited(monkeypatch):
    import bot.telegram as T
    answers = [{"ok": False, "error_code": 429, "parameters": {"retry_after": 2}}, {"ok": True}]
    waited = []
    monkeypatch.setattr(T.time, "sleep", lambda s: waited.append(s))
    tg = T.Telegram("T", 111, lambda m, p, t: answers.pop(0))
    assert tg.send("x", wait_on_limit=True) and waited == [3.0]


def test_engine_sends_every_open_and_close_with_balance(tmp_path):
    sent = []

    class Notifier:
        def send(self, text, trade=False):
            sent.append((trade, text))

    cfg = shadow_cfg()
    cfg["notify_trades"] = True
    eng = Engine(cfg, tmp_path, notifier_for=lambda k: Notifier())
    m = eng.add_market("ETHUSDT", book(100.0))
    bot = eng.bots["fade"]
    eng._enter(bot, m, S.Signal(1, sl_dist=1.0, tp_dist=2.0, reason="тест"), T0)
    eng.advance(T0 + 1000)
    bot.account.close("ETHUSDT", m.book, eng.now, "manual")
    eng.advance(T0 + 2000)
    trades = [t for flag, t in sent if flag]
    assert len(trades) == 2 and "ОТКРЫТЬ ЛОНГ" in trades[0] and "ЗАКРЫТЬ ЛОНГ" in trades[1]
    assert all(bot.title in t and "Баланс" in t and "Сделок:" in t for t in trades)
    assert "Сделок: 1" in trades[1]


# --- реалистичность: правила контракта Bybit, задержка ордеров, комиссии в сообщениях -----

def test_entry_respects_bybit_lot_step_min_qty_and_tick(tmp_path):
    eng = Engine(shadow_cfg(), tmp_path)
    m = eng.add_market("PEPEUSDT", book(0.01, step=0.000001, size=1e9))
    m.last_price = 0.01
    m.qty_step, m.min_qty, m.tick, m.min_notional = 100.0, 100.0, 0.000001, 5.0
    bot = eng.bots["fade"]
    eng._enter(bot, m, S.Signal(-1, sl_dist=0.0001, tp_dist=0.001, entry="limit",
                                limit_price=0.0100004), T0)
    o = bot.account.orders[0]
    assert o.qty % 100 == 0 and o.qty > 0                   # объём кратен шагу Bybit
    assert o.price == pytest.approx(0.010001)               # продажа: цена вверх до шага
    m2 = eng.add_market("BIGUSDT", book(1000.0))
    m2.qty_step, m2.min_qty = 1.0, 1.0                       # 1 монета = 1000 $ > 3 размеров счёта
    eng._enter(bot, m2, S.Signal(1, sl_dist=5.0, tp_dist=50.0), T0)
    assert all(o.symbol != "BIGUSDT" for o in bot.account.orders)   # меньше минимума — пропуск


def test_protective_orders_rounded_and_delayed():
    ob = book(100.0, 0.1, 10.0, levels=50)
    acc = Account("t", 1000.0, latency_ms=250)
    acc.submit(Order("X", "buy", "market", 1.0, meta={"sl_dist": 0.333, "tp_dist": 0.777,
                                                       "mid0": 100.0, "tick": 0.1}), ob, T0)
    acc.process({"X": ob}, T0 + 250)
    pos = acc.positions["X"]
    sl = next(o for o in acc.orders if o.purpose == "sl")
    tp = next(o for o in acc.orders if o.purpose == "tp")
    assert round(sl.price * 10, 6) == int(round(sl.price * 10, 6)) and sl.price <= pos.entry - 0.333
    assert round(tp.price * 10, 6) == int(round(tp.price * 10, 6)) and tp.price >= pos.entry + 0.777
    acc.on_trade("X", sl.price - 1, 1.0, taker_buy=False, book=ob, now=T0 + 300)   # стоп ещё в пути
    assert not any(o.kind == "market" for o in acc.orders)
    acc.on_trade("X", sl.price - 1, 1.0, taker_buy=False, book=ob, now=T0 + 600)   # уже на бирже
    assert any(o.kind == "market" and o.purpose == "sl" for o in acc.orders)


def test_trade_messages_show_bybit_fees_and_breakdown(tmp_path):
    ob = book(100.0, 0.1, 10.0, levels=50)
    acc = Account("t", 1000.0)
    acc.submit(Order("X", "buy", "market", 1.0, meta={"sl_dist": 1.0, "tp_dist": 1.0, "mid0": 100.0}),
               ob, T0)
    acc.process({"X": ob}, T0 + 300)
    tp = acc.positions["X"].tp
    acc.on_trade("X", tp + 0.2, 1.0, taker_buy=True, book=ob, now=T0 + 1000)
    opened, closed = acc.events
    o_text, c_text = messages.shadow_trade(opened), messages.shadow_trade(closed)
    assert "Комиссия Bybit: 0.0550 USDT (тейкер, по рынку 0.055%)" in o_text
    assert "До комиссий" in c_text and "вход: тейкер" in c_text and "выход: мейкер, лимитка 0.02%" in c_text
    assert closed.gross - closed.trade_fees - closed.funding == pytest.approx(closed.pnl)
    assert closed.trade_fees == pytest.approx(acc.stats["fees"])


def test_old_journal_with_other_columns_is_set_aside(tmp_path):
    eng = Engine(shadow_cfg(), tmp_path)
    bot = eng.bots["fade"]
    bot.journal_path.write_text("ts,symbol,action\nold,row,x\n", encoding="utf-8")
    m = eng.add_market("ETHUSDT", book(100.0))
    eng._enter(bot, m, S.Signal(1, sl_dist=1.0, tp_dist=2.0), T0)
    eng.advance(T0 + 1000)
    assert (tmp_path / "shadow_fade_trades_old.csv").exists()
    head = bot.journal_path.read_text(encoding="utf-8").splitlines()[0]
    assert head.endswith("fee_type,fee_rate_pct,gross,trade_fees,funding")


# --- калибровка и правки стратегий по разбору первого часа --------------------------------

def test_funding_rate_is_compared_per_8_hours():
    def falling():
        f = market(calm(700) + list(np.linspace(100, 99, 300)))
        f.add_flow(T0 - 1000, 99.8, 1000.0)
        return f
    f8 = falling()
    f8.funding_rate, f8.funding_interval_h = 0.0001, 8.0                 # 0.01% за 8 ч — обычная
    assert S.FundingSkew().check(f8, T0, {}) is None
    f1 = falling()
    f1.funding_rate, f1.funding_interval_h = 0.0001, 1.0                 # 0.01% в час = 0.08% за 8 ч
    sig = S.FundingSkew().check(f1, T0, {})
    assert sig is not None and sig.side == -1 and "0.080% за 8 ч" in sig.reason


def test_book_imbalance_uses_top_levels_only():
    ob = OrderBook()
    bids = [(99.95 - i * 0.1, 10.0) for i in range(3)] + [(99.65 - i * 0.1, 500.0) for i in range(7)]
    asks = [(100.05 + i * 0.1, 10.0) for i in range(10)]
    ob.snapshot(bids, asks, T0)                                         # «стена» глубже 3-го уровня
    assert abs(ob.imbalance(3)) < 0.01 and ob.imbalance(10) > 0.5
    m = market(calm(), ob=ob)
    m.add_flow(T0 - 1000, 100.0, 5000.0)
    strat = S.BookImbalance()
    assert all(strat.check(m, T0 + i * 1000, {}) is None for i in range(6))


def test_trade_cost_includes_slippage_and_lag_needs_3x(tmp_path):
    cfg = shadow_cfg(bots=["lag"], lag={"min_reward_cost": 3.0})
    eng = Engine(cfg, tmp_path)
    bot = eng.bots["lag"]
    assert bot.min_reward_cost == 3.0
    thin = eng.add_market("ALTUSDT", book(100.0, 0.1, 0.2, levels=50))   # тонкий стакан
    thin.last_price = 100.0
    sig = S.Signal(1, sl_dist=0.5, tp_dist=0.6)                          # тейк 0.6%
    cost = eng.trade_cost(bot.account, thin, sig, 1.25)
    assert cost > 2 * 0.00055 + 0.001                                    # комиссии + заметное проскальзывание
    eng._enter(bot, thin, sig, T0)
    assert not bot.account.orders                                        # 0.6% < 3 × издержки
    obs = bot.strategy.take_observations()
    assert obs["тейк / издержки сделки"][0] < 3.0
    deep = eng.add_market("DEEPUSDT", book(100.0, 0.01, 1e6))
    deep.last_price = 100.0
    eng._enter(bot, deep, sig, T0)
    assert bot.account.orders                                            # в глубоком стакане проходит


def test_calibration_log_and_report(tmp_path):
    from bot.shadow.calib import report
    from bot.telegram import CommandListener, Telegram
    eng = Engine(shadow_cfg(bots=["fade", "liq"]), tmp_path)
    m = eng.add_market("ETHUSDT", book(100.0))
    m.warmup_minutes(list(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.001, 300)))))
    m.on_liquidation(T0 + 5_000, 7_000.0)
    t = T0 - T0 % 60_000
    for i, p in enumerate(calm(200) + list(np.linspace(100, 101, 100))):
        eng.on_trade("ETHUSDT", t + i * 1000, p, 1.0, True)
    path = tmp_path / "shadow_calibration.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "ts,bot,metric,value,threshold,ratio,symbol"
    assert any(",liq,ликвидации за окно ($)," in r for r in rows) and any(",fade,ход цены (σ)," in r for r in rows)
    text = report(path, 3)
    assert "Ликвидации" in text and "Откат (контроль)" in text and "99%" in text and "порог 20 000" in text
    lst = CommandListener(Telegram("T", 111, lambda *a: {"ok": True}),
                          bots=[("x", tmp_path / "s.json", tmp_path / "c.json")], calib_path=path)
    assert "Калибровка порогов" in lst.handle("/calib 1")
    assert "пока нет" in report(tmp_path / "нет.csv")


def test_tg_setup_takes_private_chat_not_group(tmp_path, monkeypatch):
    import types

    import bot.__main__ as cli
    import bot.telegram as T

    class FakeTG:
        def __init__(self, token, chat=None):
            self.chat_id, self.sent = chat, []

        def call(self, method, payload, timeout=15.0):
            return {"ok": True, "result": {"username": "my_bot"}}

        def call_patiently(self, method, payload, **kw):
            return self.call(method, payload)

        def updates(self, timeout=25):
            return [{"update_id": 1, "message": {"chat": {"id": -100555, "type": "supergroup",
                                                          "title": "Группа"}}},
                    {"update_id": 2, "message": {"chat": {"id": 777, "type": "private",
                                                          "first_name": "Я"}}}]

        def send(self, text, **kw):
            self.sent.append(text)
            return True

    monkeypatch.setattr(T, "Telegram", FakeTG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    cli.cmd_tg_setup(types.SimpleNamespace(wait=1))
    assert "TELEGRAM_CHAT_ID=777" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_universe_skips_most_traded_and_reference_is_data_only(tmp_path, monkeypatch):
    from bot.shadow import bybit
    old = "1600000000000"
    coins = [("BTCUSDT", 9e9), ("ETHUSDT", 3e9), ("HOTUSDT", 1e9), ("MIDUSDT", 5e7),
             ("LOWUSDT", 2e7), ("TINYUSDT", 1e6)]
    inst = [{"symbol": s, "status": "Trading", "contractType": "LinearPerpetual",
             "launchTime": old, "symbolType": ""} for s, _ in coins]

    def fake_get(path, params):
        if path.endswith("instruments-info"):
            return {"list": inst if "symbol" not in params else
                    [i for i in inst if i["symbol"] == params["symbol"]]}
        if path.endswith("tickers"):
            return {"list": [{"symbol": s, "turnover24h": str(v), "bid1Price": "100.0",
                              "ask1Price": "100.01"} for s, v in coins]}
        if path.endswith("kline"):
            return {"list": [[0, 0, 0, 0, "100"]] * 100}
        return {"b": [["100.0", 1000]], "a": [["100.01", 1000]]}

    monkeypatch.setattr(bybit, "_get", fake_get)
    cfg = {"universe": {"skip_top": 3, "min_turnover_usd": 1e7}, "bots": ["fade"]}
    assert bybit.select_universe(cfg) == ["MIDUSDT", "LOWUSDT"]        # топ-3 и мелочь — мимо

    eng = Engine(cfg, tmp_path)
    started = []

    async def fake_stream(engine, symbols, stop):
        started.append(symbols)
    monkeypatch.setattr(bybit, "_stream", fake_stream)
    bybit.run_live(eng)
    assert started[0][0] == "BTCUSDT" and not eng.markets["BTCUSDT"].tradable
    assert eng.markets["MIDUSDT"].tradable

    class Always(S.Strategy):
        key, title, defaults = "fade", "x", {}

        def check(self, m, now, markets):
            return S.Signal(1, m.price() * 0.01, m.price() * 0.05)

    eng.bots["fade"].strategy = Always()
    for s in eng.markets:
        eng.markets[s].book = book(100.0, size=1e6)
    eng.advance(T0)
    eng.advance(T0 + 2000)
    assert "BTCUSDT" not in {o.symbol for o in eng.bots["fade"].account.orders} | \
        set(eng.bots["fade"].account.positions)


def test_telegram_waits_out_rate_limit_and_timeouts(monkeypatch):
    import requests

    import bot.telegram as T
    waited = []
    monkeypatch.setattr(T.time, "sleep", lambda s: waited.append(s))
    answers = [requests.exceptions.ReadTimeout("hang"),
               {"ok": False, "error_code": 429, "parameters": {"retry_after": 23}},
               {"ok": True, "result": {"username": "my_bot"}}]

    def post(method, payload, timeout):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    tg = T.Telegram("T", 111, post)
    assert tg.call_patiently("getMe", {})["result"]["username"] == "my_bot"
    assert waited == [5, 24.0]                         # после зависания 5 с, после 429 — сколько просили
    tg2 = T.Telegram("T", 111, lambda m, p, t: {"ok": False, "error_code": 429,
                                                 "parameters": {"retry_after": 7}})
    with pytest.raises(T.RateLimited) as e:
        tg2.updates()
    assert e.value.retry_after == 7


def test_command_listener_backs_off_on_rate_limit(monkeypatch):
    import bot.telegram as T
    waited = []
    lst = T.CommandListener(T.Telegram("T", 111, lambda *a: {"ok": True}),
                            bots=[("x", "s.json", "c.json")])

    def fake_sleep(s):
        waited.append(s)
        lst.stop_event.set()
    monkeypatch.setattr(T.time, "sleep", fake_sleep)

    def limited():
        raise T.RateLimited(31)
    lst.poll_once = limited
    lst.run()
    assert waited == [32]


# --- память: сводки по секундам вместо поштучного хранения ---------------------------------

def test_trade_flow_is_stored_per_second_with_same_results():
    m = SymbolMarket("X", book(100.0))
    raw = []
    rng = np.random.default_rng(5)
    for i in range(3000):                                       # ~10 сделок в секунду, 5 минут
        ts = T0 + i * 100
        p = 100 + rng.normal(0, 0.1)
        q = float(rng.uniform(0.1, 2))
        buy = bool(rng.integers(0, 2))
        m.on_trade(ts, p, q, buy)
        raw.append((ts, p, p * q * (1 if buy else -1)))
    assert len(m.trades) == 300                                 # 300 секунд, а не 3000 сделок
    now = T0 + 3000 * 100
    for window in (30, 120):
        cut = (now - window * 1000) // 1000 * 1000
        sel = [(p, v) for ts, p, v in raw if ts >= cut]
        flow = sum(v for _, v in sel) / sum(abs(v) for _, v in sel)
        vwap = sum(p * abs(v) for p, v in sel) / sum(abs(v) for _, v in sel)
        assert m.flow(window, now) == pytest.approx(flow)
        assert m.vwap(window, now) == pytest.approx(vwap)
    m.on_trade(T0 + 20 * 60 * 1000, 100.0, 1.0, True)          # старше 15 минут — выброшено
    assert len(m.trades) == 1


def test_open_interest_sampled_every_5_seconds_and_extreme_unchanged():
    m = SymbolMarket("X")
    for i in range(600):                                        # 10 обновлений в секунду, минута
        m.on_open_interest(T0 + i * 100, 1e9 + i)
    assert len(m.oi) <= 14 and m.oi[-1][1] == 1e9 + 599         # последнее значение сохранено
    m.sec.extend([100.0, 105.0, 101.0, 99.0, 102.0])
    assert m.extreme(2, 1) == 102.0 and m.extreme(3, -1) == 99.0 and m.extreme(10, 1) == 105.0
