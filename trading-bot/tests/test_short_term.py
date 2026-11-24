"""Краткосрочные боты: плечо и изолированная маржа, 5 стратегий, свои списки монет, ветки."""
import math

import numpy as np
import pytest

from bot.shadow import short_term as ST
from bot.shadow import strategies as S
from bot.shadow.engine import Engine
from bot.shadow.market import SymbolMarket
from bot.shadow.sim import Account, Order
from tests.test_shadow import T0, book

RNG = np.random.default_rng(7)


def ou_closes(n=200, theta=0.3, sd=0.002, base=100.0):
    """Минутные закрытия процесса Орнштейна–Уленбека вокруг base (полупериод ln2/θ ≈ 2.3 мин)."""
    x, out = 0.0, []
    for _ in range(n):
        x += -theta * x + RNG.normal(0, sd)
        out.append(base * math.exp(x))
    return out


# --- плечо и изолированная маржа --------------------------------------------------------

def test_isolated_margin_liquidation_and_free_margin():
    acc = Account("t", 100.0, leverage=20.0)
    ob = book(100.0, 0.01, 1e6)
    acc.submit(Order("X", "buy", "market", 10.0, meta={"sl_dist": 5.0, "mid0": 100.0}), ob, T0)
    acc.process({"X": ob}, T0 + 300)
    pos = acc.positions["X"]
    assert pos.margin == pytest.approx(pos.entry * 10 / 20)                 # 1000 $ / ×20
    assert pos.liq_price == pytest.approx(pos.entry * (1 - 1 / 20 + 0.005))  # ≈ −4.5%
    assert acc.used_margin({"X": 100.0}) == pytest.approx(pos.margin)
    m = SymbolMarket("X", ob)
    m.last_price, m.mark_price = 100.0, pos.liq_price - 0.01                # марк-цена ниже ликвидации
    acc.housekeeping({"X": m}, T0 + 1000)
    assert any(o.purpose == "liquidation" for o in acc.orders)


def test_engine_sizes_with_leverage_caps_and_shows_leverage(tmp_path):
    cfg = {"bots": ["vwap"], "save_every_s": 1, "notify_trades": True,
           "account": {"capital": 100.0, "latency_ms": 0},
           "vwap": {"account": {"risk_per_trade": 0.01, "leverage": 20.0,
                                "max_position_leverage": 10.0, "max_leverage": 20.0}}}
    sent = []

    class N:
        def send(self, text, trade=False):
            sent.append(text)

    eng = Engine(cfg, tmp_path, notifier_for=lambda k: N())
    m = eng.add_market("X", book(100.0, 0.01, 1e6))
    m.last_price = 100.0
    bot = eng.bots["vwap"]
    assert bot.account.leverage == 20.0
    eng._enter(bot, m, S.Signal(1, sl_dist=0.4, tp_dist=5.0), T0)         # стоп 0.4% → ×2.5
    eng.advance(T0 + 1000)
    pos = bot.account.positions["X"]
    assert pos.qty * pos.entry == pytest.approx(250, rel=0.01)
    assert any("плечо ×2.5" in t and "ликвидация" in t for t in sent)
    bot.risk = 0.5                                                          # огромный риск —
    m2 = eng.add_market("Y", book(100.0, 0.01, 1e6))
    m2.last_price = 100.0
    eng._enter(bot, m2, S.Signal(1, sl_dist=0.4, tp_dist=5.0), T0 + 1000)
    o = bot.account.orders[-1]
    assert o.qty * 100 <= 10 * 100 * 1.001                                  # но не больше ×10 на позицию


# --- 7 и 11: возврат к VWAP ----------------------------------------------------------------

def vwap_market(dev, closes):
    m = SymbolMarket("X", book(100.0 * math.exp(dev), 0.001, 1e6))
    m.warmup_minutes(closes)
    m.last_price = 100.0 * math.exp(dev)
    m.add_flow(T0 - 5000, 100.0, 1e6)                                       # VWAP за 15 мин = 100
    return m


def test_vwap_reversion_enters_against_deviation_with_limit():
    closes = ou_closes()
    strat = ST.VWAPReversion()
    hl, sd, vr = strat._regime(vwap_market(0.0, closes))
    assert hl < 10 and vr < 0.9                                              # режим возврата
    strat = ST.VWAPReversion()
    sig = strat.check(vwap_market(2.5 * sd, closes), T0, {})
    assert sig is not None and sig.side == -1 and sig.entry == "limit"
    assert sig.tp_dist == pytest.approx(100.0 * (math.exp(2.5 * sd) - 1), rel=0.05)   # тейк — к VWAP
    assert ST.VWAPReversion().check(vwap_market(1.0 * sd, closes), T0, {}) is None      # мало
    trend = SymbolMarket("T", book(110.0))                                   # тренд — не входим
    trend.warmup_minutes(list(np.linspace(100, 110, 200)))
    trend.add_flow(T0 - 5000, 100.0, 1e6)
    assert ST.VWAPReversion().check(trend, T0, {}) is None


def test_meme_bot_is_more_aggressive_and_trades_only_memes(tmp_path):
    assert ST.MemeVWAP.defaults["z_entry"] < ST.VWAPReversion.defaults["z_entry"]
    cfg = {"bots": ["meme"], "account": {"capital": 100.0}, "meme": {"universe": "meme",
           "account": {"risk_per_trade": 0.03, "leverage": 20.0, "max_position_leverage": 20.0}}}
    eng = Engine(cfg, tmp_path)
    seen = []

    class Spy(ST.MemeVWAP):
        def check(self, m, now, markets):
            seen.append(m.symbol)
    eng.bots["meme"].strategy = Spy()
    for s in ("DOGEUSDT", "SUIUSDT"):
        eng.add_market(s, book(100.0)).last_price = 100.0
    eng.universes = {"base": {"SUIUSDT"}, "meme": {"DOGEUSDT"}}
    eng.advance(T0)
    eng.advance(T0 + 2000)
    assert set(seen) == {"DOGEUSDT"} and eng.bots["meme"].risk == 0.03


# --- 8: пробой сжатия ----------------------------------------------------------------------

def test_squeeze_breakout_needs_squeeze_breakout_and_volume():
    wide = list(100 + 2 * np.sin(np.arange(380) / 3))                       # широкие полосы
    calm = [100 + 0.01 * RNG.normal() for _ in range(25)]                   # последние минуты — сжатие
    m = SymbolMarket("X", book(100.3))
    m.warmup_minutes(wide + calm)
    m.minute_usd.extend([1000.0 + RNG.normal(0, 50) for _ in range(60)])
    m.sec.extend([100.0] * 1900)                                            # диапазон 30 мин: 100
    m.last_price = 100.3
    strat = ST.SqueezeBreakout()
    assert strat.check(m, T0, {}) is None                                   # объёма ещё нет
    m.add_flow(T0 - 2000, 100.3, 5000.0)                                    # всплеск объёма
    sig = strat.check(m, T0, {})
    assert sig is not None and sig.side == 1 and sig.entry == "market" and sig.trail > 0


# --- 9: парный арбитраж --------------------------------------------------------------------

def pair_markets(shift=0.0, seed=11):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.002, 400))
    noise = np.zeros(400)
    for i in range(1, 400):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.0008)
    a = SymbolMarket("AUSDT", book(100.0 * math.exp(common[-1] + noise[-1] + shift), 0.001, 1e6))
    b = SymbolMarket("BUSDT", book(50.0 * math.exp(common[-1]), 0.001, 1e6))
    a.warmup_minutes(list(100 * np.exp(common + noise)))
    b.warmup_minutes(list(50 * np.exp(common)))
    return {"AUSDT": a, "BUSDT": b}


def test_pairs_finds_cointegrated_pair_and_enters_both_legs(tmp_path):
    strat = ST.PairsCointegration()
    mk = pair_markets(shift=0.0)
    assert strat.check(mk["AUSDT"], T0, mk) is None                         # спред у средней
    assert list(strat.state["pairs"]) == ["AUSDT"] and strat.state["pairs"]["AUSDT"][0] == "BUSDT"
    alpha, beta = strat.state["pairs"]["AUSDT"][1:]
    z, sd = strat._z(mk["AUSDT"], mk["BUSDT"], alpha, beta)
    a_price = mk["AUSDT"].price() * math.exp((2.5 - z) * sd)                 # разошлись на 2.5σ
    mk["AUSDT"].book = book(a_price, 0.001, 1e6)
    sig = strat.check(mk["AUSDT"], T0, mk)
    assert isinstance(sig, S.PairSignal)
    (s1, l1), (s2, l2) = sig.legs
    assert (s1, l1.side, s2, l2.side) == ("AUSDT", -1, "BUSDT", 1)          # A дорогая → шорт A
    cfg = {"bots": ["pairs"], "account": {"capital": 100.0, "latency_ms": 0},
           "pairs": {"account": {"leverage": 20.0, "max_position_leverage": 10.0,
                                 "max_leverage": 20.0, "max_positions": 4}}}
    eng = Engine(cfg, tmp_path)
    eng.markets.update(mk)
    bot = eng.bots["pairs"]
    eng._enter_pair(bot, sig, T0, None)
    assert {o.symbol for o in bot.account.orders} == {"AUSDT", "BUSDT"}
    assert all(o.meta["tag"] == "AUSDT/BUSDT" for o in bot.account.orders)


def test_pairs_manage_closes_on_reversion_and_lone_leg():
    strat = ST.PairsCointegration()
    mk = pair_markets(shift=0.0)
    strat.check(mk["AUSDT"], T0, mk)
    alpha, beta = strat.state["pairs"]["AUSDT"][1:]
    # цена A ровно на «справедливом» уровне спреда → z около 0
    z, sd = strat._z(mk["AUSDT"], mk["BUSDT"], alpha, beta)
    acc = Account("t", 100.0)
    for s, side in (("AUSDT", "sell"), ("BUSDT", "buy")):
        acc.submit(Order(s, side, "market", 1.0, meta={"sl_dist": 5.0, "mid0": 100.0,
                                                       "tag": "AUSDT/BUSDT"}), mk[s].book, T0)
    acc.process({s: m.book for s, m in mk.items()}, T0 + 300)
    out = strat.manage(acc, mk, T0 + 1000)
    if abs(z) <= strat.p["z_exit"]:
        assert set(out) == {("AUSDT", "signal"), ("BUSDT", "signal")}
    del acc.positions["BUSDT"]                                              # осталась одна нога
    (action, sym, sig, notional, tag, lone), = strat.manage(acc, mk, T0 + 2000)
    assert (action, sym, sig.side, sig.entry, tag, lone) == ("open", "BUSDT", 1, "market",
                                                              "AUSDT/BUSDT", "AUSDT")
    assert notional == pytest.approx(acc.positions["AUSDT"].qty * acc.positions["AUSDT"].entry
                                     * beta, rel=1e-6)                      # сумма × β
    acc.orders.append(Order("BUSDT", "buy", "market", 1.0, purpose="entry"))
    assert strat.manage(acc, mk, T0 + 3000) == []                           # нога уже в пути


# --- 10: поток ордеров ---------------------------------------------------------------------

def test_order_flow_regression_predicts_direction():
    m = SymbolMarket("X", book(100.0))
    m.warmup_minutes(ou_closes(100))
    m.top_depth = 10.0
    ofi = RNG.normal(0, 10, 2000)
    logp = np.zeros(2000)
    for t in range(1, 2000):                                     # цена следует за OFI с задержкой 15 с
        logp[t] = logp[t - 1] + 0.00001 * ofi[max(t - 15, 0)] + RNG.normal(0, 0.00002)
    m.ofi_sec.extend(ofi)
    m.sec.extend(100 * np.exp(logp))
    strat = ST.OrderFlowImbalance()
    lam, a, r2 = strat._fit(m, T0)
    assert lam > 0 and r2 > 0.05
    m.ofi_sec.extend([60.0] * 10)                                           # сильные покупки сейчас
    m.sec.extend([m.sec[-1]] * 10)
    strat.state.clear()
    sig = strat.check(m, T0, {})
    assert sig is not None and sig.side == 1 and sig.entry == "limit" and sig.tp_dist > 0


def test_ofi_from_book_updates():
    m = SymbolMarket("X", book(100.0, 0.1, 10.0))
    m.on_book_update()                                                     # первое — без вклада
    m.book.delta([(99.95, 25.0)], [], T0)                                   # на покупке +15
    m.on_book_update()
    assert m._ofi_this_sec == pytest.approx(25.0 - 10.0)
    m.on_second(T0 // 1000)
    assert m.ofi_sec[-1] == pytest.approx(15.0) and m._ofi_this_sec == 0.0


# --- ветки и запуск без дневных счетов -----------------------------------------------------

def test_stale_topics_are_renamed_for_new_bots(tmp_path):
    from bot.telegram import ShadowChannel, Telegram, TopicNotifier
    calls = []

    def post(method, payload, timeout):
        calls.append((method, payload))
        if method == "createForumTopic":
            return {"ok": True, "result": {"message_thread_id": 900 + len(calls)}}
        return {"ok": True}

    path = tmp_path / "tg.json"
    path.write_text('{"chat_id": "-1", "topics": {"Спот-быстрый": 1, "Фьючерсы": 2, "Ликвидации": 3}}',
                    encoding="utf-8")
    ch = ShadowChannel(Telegram("T", 111, post), path, start=False)
    for t in ("Ликвидации", "Возврат к VWAP", "Пробой сжатия", "Мемы: возврат к VWAP"):
        TopicNotifier(ch, t)
    ch.ensure_topics()
    assert ch.topics["Возврат к VWAP"] == 1 and ch.topics["Пробой сжатия"] == 2
    assert "Спот-быстрый" not in ch.topics and ch.topics["Ликвидации"] == 3
    assert [m for m, _ in calls].count("editForumTopic") == 2
    assert [m for m, _ in calls].count("createForumTopic") == 1               # мемам — новая


def test_group_starts_without_daily_accounts(tmp_path):
    from bot.group import load_group
    p = tmp_path / "bots.toml"
    p.write_text('[shadow]\nenabled = true\nconfig = "config.shadow.toml"\n', encoding="utf-8")
    assert load_group(p) == []
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_group(p)


def test_default_config_has_eleven_bots(tmp_path):
    import tomllib
    with open("config.shadow.toml", "rb") as f:
        cfg = tomllib.load(f)
    eng = Engine(cfg, tmp_path)
    assert len(eng.bots) == 11
    assert eng.bots["meme"].universe == "meme" and eng.bots["meme"].account.leverage == 20.0
    assert eng.bots["oi"].account.leverage is None                          # старые не тронуты
    assert eng.bots["ofi"].min_reward_cost == 3.0


def test_position_on_coin_outside_universe_is_kept_and_streamed(tmp_path, monkeypatch):
    from bot.shadow import bybit
    eng = Engine({"bots": ["fade"], "account": {"capital": 100.0}}, tmp_path)
    acc = eng.bots["fade"].account
    acc.positions["OLDUSDT"] = __import__("bot.shadow.sim", fromlist=["Position"]).Position(
        "OLDUSDT", 1, 1.0, 10.0, T0)
    acc.housekeeping({}, T0)                                 # нет данных по монете — не падаем
    monkeypatch.setattr(bybit, "select_universe", lambda cfg: ["NEWUSDT"])
    monkeypatch.setattr(bybit, "warmup", lambda engine, symbols: [engine.add_market(s) for s in symbols])
    started = []

    async def fake_stream(engine, symbols, stop):
        started.append(symbols)
    monkeypatch.setattr(bybit, "_stream", fake_stream)
    bybit.run_live(eng)
    assert "OLDUSDT" in started[0] and "OLDUSDT" not in eng.universes["base"]
