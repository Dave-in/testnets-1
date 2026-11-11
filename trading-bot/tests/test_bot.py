import json

import numpy as np
import pandas as pd
import pytest

from bot.backtest import Backtester
from bot.brokers import PaperBroker
from bot.config import Config, RiskConfig
from bot.core import rebalance_deltas
from bot.data import synthetic
from bot.live import Runner
from bot.risk import DrawdownGuard


def make_cfg(profile="core", **risk) -> Config:
    cfg = Config()
    cfg.portfolio.profiles = {
        "core": {"ETH/USDT": 1.0},
        "balanced": {"ETH/USDT": 0.5, "BTC/USDT": 0.4, "SOL/USDT": 0.15},
    }
    cfg.portfolio.profile = profile
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    return cfg


def trend_frame(n=600, up_until=450, seed=0) -> pd.DataFrame:
    """Плавный рост, затем обвал — у стратегии должен быть вход и выход."""
    rng = np.random.default_rng(seed)
    rets = np.where(np.arange(n) < up_until, 0.004, -0.02) + rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.005,
                         "low": np.minimum(open_, close) * 0.995, "close": close,
                         "volume": 1.0}, index=idx)


def test_enters_uptrend_and_exits_crash():
    cfg = make_cfg()
    res = Backtester(cfg, {"ETH/USDT": trend_frame()}).run()
    buys = res.trades[res.trades.side == "buy"]
    sells = res.trades[res.trades.side == "sell"]
    assert len(buys) > 0 and len(sells) > 0
    # после обвала бот вне рынка
    assert res.equity["w_ETH/USDT"].iloc[-1] == 0
    assert res.metrics["max_drawdown"] < 0.07


def test_no_lookahead():
    """Изменение будущих цен не должно влиять на прошлые решения."""
    cfg = make_cfg("balanced")
    data = synthetic(cfg.portfolio.symbols, bars=900, seed=3)
    base = Backtester(cfg, data).run().equity["equity"]

    cut = data["ETH/USDT"].index[700]
    changed = {}
    for s, df in data.items():
        df = df.copy()
        df.loc[df.index > cut, ["open", "high", "low", "close"]] *= 3.0
        changed[s] = df
    other = Backtester(cfg, changed).run().equity["equity"]
    pd.testing.assert_series_equal(base[base.index <= cut], other[other.index <= cut])


@pytest.mark.parametrize("seed", range(1, 6))
def test_halt_mode_respects_hard_drawdown(seed):
    cfg = make_cfg("balanced", hard_stop_action="halt", hard_drawdown=0.065)
    data = synthetic(cfg.portfolio.symbols, bars=2000, seed=seed)
    res = Backtester(cfg, data).run()
    # синтетика почти без гэпов, допускаем проскальзывание и комиссию
    assert res.metrics["max_drawdown"] <= 0.065 + 0.005


def test_fees_are_charged():
    cfg = make_cfg()
    cfg.exchange.fee = 0.01
    res = Backtester(cfg, {"ETH/USDT": trend_frame()}).run()
    assert res.metrics["fees_paid"] > 0
    assert np.isclose(res.trades["fee"], res.trades["notional"] * 0.01).all()


def test_rebalance_deltas():
    assert rebalance_deltas({"A": 0.0}, {"A": 0.10}, 0.2) == {"A": 0.0}       # выход всегда
    assert rebalance_deltas({"A": 0.11}, {"A": 0.10}, 0.2) == {}              # мелочь
    assert rebalance_deltas({"A": 0.15}, {"A": 0.10}, 0.2) == {"A": 0.15}
    assert rebalance_deltas({"A": 0.005}, {"A": 0.0}, 0.2) == {}              # < min_trade


def test_guard_soft_hard_and_resume():
    g = DrawdownGuard(RiskConfig(soft_drawdown=0.02, hard_drawdown=0.06, soft_scale=0.5,
                                 cooldown_bars=2, resume_drawdown=0.03), peak=100)
    assert g.on_bar_close(100) == 1.0
    assert g.on_bar_close(96) == pytest.approx(0.75)  # посередине между soft и hard
    assert g.on_bar_close(93) == 0.0 and g.active
    assert g.on_bar_close(93) == 0.0
    assert g.on_bar_close(93) == 0.0 and not g.active
    assert g.drawdown(93) == pytest.approx(0.03)      # возобновили с урезанным риском


def test_guard_halt():
    g = DrawdownGuard(RiskConfig(hard_stop_action="halt", hard_drawdown=0.05), peak=100)
    assert g.check_intrabar(94)
    assert all(g.on_bar_close(200) == 0.0 for _ in range(50))


class FakeMarket:
    def __init__(self, frames):
        self.frames = frames
        self.i = 300

    def closed_ohlcv(self, symbol, timeframe, limit):
        return self.frames[symbol].iloc[max(0, self.i - limit): self.i]

    def prices(self, symbols):
        return {s: float(self.frames[s]["open"].iloc[self.i]) for s in symbols}


def test_paper_runner_trades_and_persists(tmp_path):
    cfg = make_cfg()
    market = FakeMarket({"ETH/USDT": trend_frame()})
    state = tmp_path / "new_dir" / "paper.json"  # папки еще нет
    runner = Runner(cfg, market, PaperBroker(0.001, 0.0005), state)
    for _ in range(100):
        runner.step()
        market.i += 1
    assert runner.book.units["ETH/USDT"] > 0
    saved = json.loads(state.read_text())
    assert saved["book"]["units"]["ETH/USDT"] == pytest.approx(runner.book.units["ETH/USDT"])

    # перезапуск — состояние восстанавливается
    again = Runner(cfg, market, PaperBroker(0.001, 0.0005), state)
    assert again.book.units == runner.book.units
    assert again.last_bar == runner.last_bar

    # обвал — бот выходит из позиции
    for _ in range(140):
        again.step()
        market.i += 1
    assert again.book.units["ETH/USDT"] == 0
    assert (tmp_path / "new_dir" / "paper_trades.csv").exists()


class FakeExchange:
    def __init__(self, fees):
        self.fees = fees

    def load_markets(self):
        pass

    def market(self, symbol):
        return {"limits": {"cost": {"min": 5}}}

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.4f}"

    def create_order(self, symbol, type_, side, amount):
        return {"id": "1", "status": "closed", "filled": amount, "average": 2000.0,
                "cost": amount * 2000.0, "fees": self.fees}


def test_exchange_broker_fee_in_base_currency():
    from bot.brokers import ExchangeBroker

    b = ExchangeBroker(FakeExchange([{"currency": "ETH", "cost": 0.001}]), fee=0.001)
    f = b.market_order("ETH/USDT", "buy", 1.0, 2000.0)
    assert f.units == pytest.approx(0.999) and f.cost == pytest.approx(2000.0)
    assert b.market_order("ETH/USDT", "buy", 0.001, 2000.0) is None  # меньше минимума

    b = ExchangeBroker(FakeExchange([{"currency": "USDT", "cost": 2.0}]), fee=0.001)
    f = b.market_order("ETH/USDT", "sell", 1.0, 2000.0)
    assert f.units == 1.0 and f.cost == pytest.approx(1998.0)


def test_control_commands(tmp_path):
    from bot.live import control_path

    cfg = make_cfg()
    market = FakeMarket({"ETH/USDT": trend_frame()})
    state = tmp_path / "paper.json"
    runner = Runner(cfg, market, PaperBroker(0.001, 0.0005), state)
    for _ in range(100):
        runner.step()
        market.i += 1
    assert runner.book.units["ETH/USDT"] > 0

    control_path(state).write_text(json.dumps({"command": "close-all"}))
    runner.step()
    assert runner.book.units["ETH/USDT"] == 0 and runner.paused
    for _ in range(10):  # на паузе бот не входит заново
        market.i += 1
        runner.step()
    assert runner.book.units["ETH/USDT"] == 0

    control_path(state).write_text(json.dumps({"command": "resume"}))
    for _ in range(5):
        market.i += 1
        runner.step()
    assert not runner.paused

    control_path(state).write_text(json.dumps({"command": "stop"}))
    runner.run()  # должен выйти сам, без бесконечного цикла
    assert runner.stop_requested


def test_top_n_keeps_strongest():
    from bot.core import Decider

    cfg = make_cfg("balanced")
    cfg.strategy.max_positions = 1
    d = Decider(cfg, 10_000)
    rows = {"ETH/USDT": pd.Series({"mom": 0.5, "vol": 0.5}),
            "BTC/USDT": pd.Series({"mom": 0.2, "vol": 0.5}),
            "SOL/USDT": pd.Series({"mom": 0.9, "vol": 3.0})}
    wants = d._top_n({s: True for s in rows}, rows)
    assert wants == {"ETH/USDT": True, "BTC/USDT": False, "SOL/USDT": False}


# --- сигнальный бот: Telegram, виртуальный счёт, фильтр рынка ---------------

class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


def test_runner_sends_signals_and_tracks_pnl(tmp_path):
    cfg = make_cfg()
    market = FakeMarket({"ETH/USDT": trend_frame()})
    note = FakeNotifier()
    runner = Runner(cfg, market, PaperBroker(0.001, 0.0005), tmp_path / "s.json", note)
    for _ in range(240):  # рост → вход, затем обвал → выход
        runner.step()
        market.i += 1
    buys = [m for m in note.sent if "КУПИТЬ ETH/USDT" in m]
    sells = [m for m in note.sent if "ПРОДАТЬ ETH/USDT" in m]
    assert buys and "Стоп:" in buys[0] and "Виртуальный счёт" in buys[0]
    assert sells and "Итог сделки" in sells[-1]
    assert runner.stats["closed"] >= 1
    # учёт сходится: стартовый капитал + зафиксированный результат = кэш (позиций нет)
    assert runner.book.units["ETH/USDT"] == 0
    assert runner.book.cash == pytest.approx(cfg.portfolio.initial_capital + runner.stats["realized"])
    snap = json.loads((tmp_path / "s.json").read_text())["snapshot"]
    assert snap["equity"] == pytest.approx(runner.book.cash) and snap["positions"] == []


def test_daily_report_on_new_day(tmp_path):
    cfg = make_cfg()
    cfg.telegram.daily_report = True
    market = FakeMarket({"ETH/USDT": trend_frame()})
    note = FakeNotifier()
    runner = Runner(cfg, market, PaperBroker(0.001, 0.0005), tmp_path / "s.json", note)
    runner.step()                      # первый запуск — отчёта ещё нет
    assert not any("Отчёт за день" in m for m in note.sent)
    runner.marks["day"] = "2000-01-01"  # «вчера»
    runner.step()
    assert sum("Отчёт за день" in m for m in note.sent) == 1
    runner.step()                      # в тот же день повторно не шлём
    assert sum("Отчёт за день" in m for m in note.sent) == 1


def test_telegram_commands(tmp_path):
    from bot.telegram import CommandListener, Telegram

    sent, queue = [], []

    def post(method, payload, timeout):
        if method == "getUpdates":
            items, queue[:] = list(queue), []
            return {"ok": True, "result": items}
        sent.append(payload)
        return {"ok": True}

    def msg(i, chat, text):
        return {"update_id": i, "message": {"chat": {"id": chat}, "text": text}}

    state, control = tmp_path / "s.json", tmp_path / "s_control.json"
    listener = CommandListener(Telegram("T", 111, post), state, control)

    queue.append(msg(1, 999, "/closeall"))        # чужой чат — игнор
    listener.poll_once()
    assert not sent and not control.exists()

    queue.append(msg(2, 111, "/status"))          # состояния ещё нет
    listener.poll_once()
    assert "ещё не сохранил" in sent[-1]["text"]

    cfg = make_cfg()
    runner = Runner(cfg, FakeMarket({"ETH/USDT": trend_frame()}), PaperBroker(0.001, 0.0005), state)
    runner.step()
    queue.extend([msg(3, 111, "/status"), msg(4, 111, "/report@my_bot"), msg(5, 111, "/pause")])
    listener.poll_once()
    texts = [p["text"] for p in sent]
    assert any("Статус" in t and "Счёт:" in t and "USDT" in t for t in texts)
    assert any("С запуска" in t for t in texts)
    assert json.loads(control.read_text()) == {"command": "pause"}
    assert all(p["chat_id"] == "111" for p in sent)
    runner.step()
    assert runner.paused


def test_regime_filter_blocks_trading():
    cfg = make_cfg()
    cfg.strategy.regime_symbol = "BTC/USDT"
    eth = trend_frame()
    falling = trend_frame(up_until=0)             # «BTC» всё время падает
    res = Backtester(cfg, {"ETH/USDT": eth, "BTC/USDT": falling}).run()
    assert len(res.trades) == 0
    rising = trend_frame(up_until=10_000, seed=5)  # «BTC» всё время растёт
    res = Backtester(cfg, {"ETH/USDT": eth, "BTC/USDT": rising}).run()
    assert len(res.trades) > 0


def test_guard_scale_mode_never_stops():
    g = DrawdownGuard(RiskConfig(hard_stop_action="scale", soft_drawdown=0.05, hard_drawdown=0.15,
                                 soft_scale=0.4), peak=100)
    assert g.on_bar_close(100) == 1.0
    assert g.on_bar_close(90) == pytest.approx(0.7)   # −10%: посередине
    assert g.on_bar_close(50) == pytest.approx(0.4)   # глубже лимита — минимальный размер
    assert not g.check_intrabar(50) and not g.active
    assert g.peak == 100                               # пик не сбрасывается


def test_load_csv_with_text_dates(tmp_path):
    from bot.data import load_csv

    df = trend_frame(n=5)
    path = tmp_path / "x.csv"
    df.to_csv(path)                                    # даты строками — как пишет download
    back = load_csv(path)
    assert len(back) == 5 and back["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1])


def test_binance_public_falls_back_to_mirror():
    from bot.binance_public import BinancePublic

    class Resp:
        def __init__(self, code, data):
            self.status_code, self._data, self.text = code, data, str(data)

        def json(self):
            return self._data

    class Session:
        calls = []

        def get(self, url, params, timeout):
            self.calls.append(url)
            if url.startswith("https://blocked"):
                return Resp(451, {"msg": "restricted location"})
            return Resp(200, [{"symbol": "ETHUSDT", "price": "2000.5"},
                              {"symbol": "BTCUSDT", "price": "80000"}])

    pub = BinancePublic(hosts=["https://blocked", "https://mirror"])
    pub.session = Session()
    assert pub.prices(["ETH/USDT"]) == {"ETH/USDT": 2000.5}
    assert pub.hosts[0] == "https://mirror"           # рабочий адрес запомнили
    with pytest.raises(RuntimeError):
        pub.prices(["NOPE/USDT"])


# --- фьючерсы: шорты, риск на сделку, фандинг ---------------------------------

def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Зеркальный рынок: там, где исходный рос, этот падает (и наоборот)."""
    k = 100.0 * 300.0
    return pd.DataFrame({"open": k / df["open"], "high": k / df["low"], "low": k / df["high"],
                         "close": k / df["close"], "volume": df["volume"]}, index=df.index)


def futures_cfg() -> Config:
    cfg = make_cfg()
    cfg.exchange.market = "futures"
    cfg.exchange.fee = 0.0005
    cfg.strategy.allow_short = True
    return cfg


def test_shorts_require_futures():
    cfg = make_cfg()
    cfg.strategy.allow_short = True
    with pytest.raises(ValueError):
        Backtester(cfg, {"ETH/USDT": trend_frame()})


def test_backtest_shorts_profit_in_downtrend():
    cfg = futures_cfg()
    res = Backtester(cfg, {"ETH/USDT": mirror(trend_frame())}).run()
    w, eq = res.equity["w_ETH/USDT"], res.equity["equity"]
    fall_end = trend_frame().index[449]              # до этого бара рынок падает
    assert w[:fall_end].min() < 0 and w[:fall_end].max() <= 1e-9   # в падении — только шорты
    assert eq[:fall_end].iloc[-1] > cfg.portfolio.initial_capital  # и на падении заработали
    assert w.iloc[-1] >= 0                           # после разворота вверх шорта уже нет


def test_risk_per_trade_caps_position_size():
    from bot.strategy import size_weights

    cfg = futures_cfg()
    cfg.strategy.target_vol = 10.0                   # волатильность не ограничивает
    cfg.strategy.risk_per_trade = 0.01
    rets = pd.DataFrame({"A": [0.0] * 30, "B": [0.0] * 30})
    w = size_weights({"A": 1, "B": -1}, {"A": 0.5, "B": 0.5}, {"A": 0.10, "B": 0.05}, rets,
                     {"A": 1.0, "B": 1.0}, cfg.strategy, "1d", max_gross=5.0)
    assert w["A"] == pytest.approx(0.10)             # 1% риска / 10% до стопа
    assert w["B"] == pytest.approx(-0.20)            # шорт со знаком минус
    cfg.strategy.short_scale = 0.5
    w = size_weights({"B": -1}, {"B": 0.5}, {"B": 0.05}, rets, {"B": 1.0}, cfg.strategy, "1d", 5.0)
    assert w["B"] == pytest.approx(-0.10)
    w = size_weights({"A": 1, "B": -1}, {"A": 0.5, "B": 0.5}, {"A": 0.001, "B": 0.001}, rets,
                     {"A": 1.0, "B": 1.0}, cfg.strategy, "1d", max_gross=1.0)
    assert sum(abs(x) for x in w.values()) == pytest.approx(1.0)   # общее плечо не выше лимита


def test_stops_for_longs_and_shorts():
    from bot.core import Decider

    d = Decider(futures_cfg(), 10_000)
    st = d.states["ETH/USDT"]
    st.side, st.stop = -1, 110.0
    assert d.check_stops({"ETH/USDT": 90.0}, {"ETH/USDT": -1.0}, {"ETH/USDT": 109.0}) == []
    assert d.check_stops({"ETH/USDT": 90.0}, {"ETH/USDT": -1.0}, {"ETH/USDT": 111.0}) == ["ETH/USDT"]
    st.side, st.stop = 1, 95.0
    assert d.check_stops({"ETH/USDT": 96.0}, {"ETH/USDT": 1.0}, {"ETH/USDT": 120.0}) == []
    assert d.check_stops({"ETH/USDT": 94.0}, {"ETH/USDT": 1.0}, {"ETH/USDT": 120.0}) == ["ETH/USDT"]


def test_rebalance_handles_shorts_and_flips():
    assert rebalance_deltas({"A": -0.10}, {"A": 0.0}, 0.2) == {"A": -0.10}     # открыть шорт
    assert rebalance_deltas({"A": -0.11}, {"A": -0.10}, 0.2) == {}             # мелочь
    assert rebalance_deltas({"A": 0.0}, {"A": -0.10}, 0.2) == {"A": 0.0}       # закрыть шорт
    assert rebalance_deltas({"A": 0.01}, {"A": -0.10}, 0.9) == {"A": 0.01}     # переворот всегда


def test_funding_reduces_long_and_pays_short():
    idx = trend_frame().index
    rate = pd.Series(0.001, index=idx)               # 0.1% за бар — заметно
    cfg = futures_cfg()
    cfg.strategy.allow_short = False                 # сначала только лонги
    base = Backtester(cfg, {"ETH/USDT": trend_frame()}, funding={}).run()
    cfg.exchange.default_funding_8h = 0.0
    free = Backtester(cfg, {"ETH/USDT": trend_frame()}, funding={}).run()
    paid = Backtester(cfg, {"ETH/USDT": trend_frame()}, funding={"ETH/USDT": rate}).run()
    assert paid.metrics["funding_paid"] > 0
    assert paid.equity["equity"].iloc[-1] < free.equity["equity"].iloc[-1]
    assert base.metrics["funding_paid"] > 0          # без истории лонги платят типовую ставку
    cfg.strategy.allow_short = True
    falling = mirror(trend_frame()).iloc[:450]       # только падение: бот всё время в шорте
    short = Backtester(cfg, {"ETH/USDT": falling}, funding={"ETH/USDT": rate}).run()
    assert short.metrics["funding_paid"] < 0         # шорт получает положительную ставку


class FundingMarket(FakeMarket):
    rates = {"ETH/USDT": 0.001}

    def funding_rates(self, symbols):
        return {s: self.rates[s] for s in symbols if s in self.rates}


def test_futures_runner_shorts_signals_and_funding(tmp_path, monkeypatch):
    import bot.live as live

    cfg = futures_cfg()
    market = FundingMarket({"ETH/USDT": mirror(trend_frame())})
    note = FakeNotifier()
    runner = Runner(cfg, market, PaperBroker(0.0005, 0.0005), tmp_path / "f.json", note)
    for _ in range(60):
        runner.step()
        market.i += 1
    assert runner.book.units["ETH/USDT"] < 0
    opened = [m for m in note.sent if "ОТКРЫТЬ ШОРТ ETH/USDT" in m]
    assert opened and "Стоп:" in opened[0] and "Риск по позиции до стопа" in opened[0]
    assert runner.snapshot["positions"][0]["side"] == -1 and runner.snapshot["market"] == "futures"
    assert "ШОРТ" in __import__("bot.messages", fromlist=["x"]).status_text(runner.snapshot)

    # фандинг: прошёл один 8-часовой период → шорт получает положительную ставку
    cash0, now = runner.book.cash, live.time.time()
    monkeypatch.setattr(live.time, "time", lambda: now + 8 * 3600)
    prices = market.prices(["ETH/USDT"])
    runner.apply_funding(prices)
    expected = runner.book.units["ETH/USDT"] * prices["ETH/USDT"] * 0.001   # отрицательное число
    assert runner.book.cash - cash0 == pytest.approx(-expected) and expected < 0
    assert runner.stats["funding"] == pytest.approx(expected)
    runner.apply_funding(prices)                     # тот же период — второй раз не начисляем
    assert runner.stats["funding"] == pytest.approx(expected)
    monkeypatch.setattr(live.time, "time", lambda: now)

    for _ in range(180):                             # разворот вверх → шорт закрывается
        runner.step()
        market.i += 1
    assert runner.book.units["ETH/USDT"] >= 0        # шорта больше нет (бот мог уйти в лонг)
    assert any("ЗАКРЫТЬ ШОРТ ETH/USDT" in m and "Итог сделки" in m for m in note.sent)
    assert runner.stats["closed"] >= 1
    # учёт сходится: капитал = старт + зафиксированное (минус комиссии и фандинг) + бумажный результат
    prices = market.prices(["ETH/USDT"])
    u, entry = runner.book.units["ETH/USDT"], runner.book.entry["ETH/USDT"]
    unrealized = u * (prices["ETH/USDT"] - entry)
    assert runner.book.equity(prices) == pytest.approx(
        cfg.portfolio.initial_capital + runner.stats["realized"] + unrealized)


def test_state_from_version_without_shorts_still_loads(tmp_path):
    from bot.strategy import SymbolState

    st = SymbolState.from_dict({"in_position": True, "highest_close": 120.0, "stop": 100.0,
                                "cooldown_left": 0})
    assert st.side == 1 and st.extreme_close == 120.0 and st.in_position
    cfg = make_cfg()
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "book": {"cash": 4000.0, "units": {"ETH/USDT": 2.0}, "cost": {"ETH/USDT": 6000.0}},
        "decider": {"states": {"ETH/USDT": {"in_position": True, "highest_close": 3100.0,
                                            "stop": 2800.0, "cooldown_left": 0}},
                    "guard": {"peak": 10000.0, "paused_left": 0, "halted": False, "events": []}},
        "last_bar": None}))
    r = Runner(cfg, FakeMarket({"ETH/USDT": trend_frame()}), PaperBroker(0.001, 0.0005), path)
    assert r.book.entry["ETH/USDT"] == pytest.approx(3000.0)
    assert r.decider.states["ETH/USDT"].side == 1


# --- фильтр чистоты тренда и группа счетов ------------------------------------

def test_trend_quality_filter_blocks_choppy_entries():
    """Ровный рост проходит фильтр, «пила» с тем же итоговым ростом — нет."""
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")

    def frame(rets):
        close = 100 * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                             "low": np.minimum(open_, close) * 0.998, "close": close,
                             "volume": 1.0}, index=idx)

    smooth = frame(np.full(n, 0.004))
    saw = frame(0.004 + 0.05 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0))  # тот же дрейф, но пила
    cfg = make_cfg()
    cfg.strategy.min_trend_quality = 0.3
    assert len(Backtester(cfg, {"ETH/USDT": smooth}).run().trades) > 0
    assert len(Backtester(cfg, {"ETH/USDT": saw}).run().trades) == 0
    cfg.strategy.min_trend_quality = 0.0                 # без фильтра в пилу бот входит
    assert len(Backtester(cfg, {"ETH/USDT": saw}).run().trades) > 0


def test_split_message_keeps_everything():
    from bot.telegram import split_message

    text = "\n\n".join(f"блок {i} " + "x" * 500 for i in range(20))
    parts = split_message(text, limit=1500)
    assert len(parts) > 1 and all(len(p) <= 1500 for p in parts)
    assert "\n\n".join(parts) == text
    assert split_message("коротко") == ["коротко"]


def test_group_runs_accounts_side_by_side(tmp_path):
    from bot.group import Group, Labeled
    from bot.live import control_path
    from bot.telegram import CommandListener, Telegram

    sent, queue = [], []

    def post(method, payload, timeout):
        if method == "getUpdates":
            items, queue[:] = list(queue), []
            return {"ok": True, "result": items}
        sent.append(payload["text"])
        return {"ok": True}

    tg = Telegram("T", 111, post)
    runners = {}
    for name, fut, signals in (("Спот", False, True), ("Фьючерсы", True, True), ("База", False, False)):
        cfg = futures_cfg() if fut else make_cfg()
        cfg.telegram.daily_report = False
        market = FakeMarket({"ETH/USDT": mirror(trend_frame()) if fut else trend_frame()})
        runners[name] = Runner(cfg, market, PaperBroker(0.001, 0.0005), tmp_path / f"{name}.json",
                               Labeled(tg, name) if signals else None)
    group = Group(runners, tg, poll_seconds=1)
    for _ in range(80):
        group.step_all()
        for r in runners.values():
            r.market.i += 1
    assert any(t.startswith("<b>[Спот]</b>") and "КУПИТЬ" in t for t in sent)
    assert any(t.startswith("<b>[Фьючерсы]</b>") and "ОТКРЫТЬ ШОРТ" in t for t in sent)
    assert not any("[База]" in t for t in sent)          # без сигналов
    assert runners["База"].book.units["ETH/USDT"] > 0    # но торгует

    # общий отчёт за день — один на все счета, когда у всех сменились сутки
    sent.clear()
    for r in runners.values():
        r.marks["day"] = "2000-01-01"
    group.step_all()
    group.maybe_report()
    reports = [t for t in sent if "Отчёт за день" in t]
    assert len(reports) == 1 and all(n in reports[0] for n in ("Спот", "Фьючерсы", "База"))
    assert "Сравнение счетов" in reports[0]
    group.maybe_report()
    assert len([t for t in sent if "Отчёт за день" in t]) == 1

    # команды: статус по всем счетам, пауза — всем
    bots = [(n, r.state_path, control_path(r.state_path)) for n, r in runners.items()]
    listener = CommandListener(Telegram("T", 111, post), bots=bots)
    sent.clear()
    queue.append({"update_id": 1, "message": {"chat": {"id": 111}, "text": "/status"}})
    queue.append({"update_id": 2, "message": {"chat": {"id": 111}, "text": "/pause"}})
    listener.poll_once()
    assert all(f"[{n}]" in sent[0] for n in runners)
    group.step_all()
    assert all(r.paused for r in runners.values())

    # сбой одного счёта не останавливает остальные
    def boom(symbols):
        raise RuntimeError("биржа недоступна")
    runners["Спот"].market.prices = boom
    before = runners["Фьючерсы"].snapshot["ts"]
    group.step_all()
    assert group.errors["Спот"] == 1 and group.errors["Фьючерсы"] == 0
    assert runners["Фьючерсы"].snapshot["ts"] >= before


def test_bots_toml_and_configs_load():
    from pathlib import Path

    from bot.config import load_config
    from bot.group import load_group
    from bot.live import state_path_for

    root = Path(__file__).resolve().parents[1]
    specs = load_group(root / "bots.toml")
    cfgs = [load_config(root / s.config) for s in specs]
    assert len({str(state_path_for(c)) for c in cfgs}) == len(cfgs)   # у каждого свой счёт
    assert all(c.exchange.mode == "paper" for c in cfgs)              # только виртуальные деньги
    trend = load_config(root / "config.trend.toml")
    assert trend.strategy.min_trend_quality == 0.3 and trend.strategy.fast_ema == 10


# --- режим попыток (Турбо) и учёт ошибок ---------------------------------------

def test_challenge_attempt_simulation():
    from bot import challenge

    idx = pd.date_range("2024-01-01", periods=9, freq="1D", tz="UTC")
    # рост x10 за три бара, затем падение ниже 20%, затем незавершённая попытка
    eq = pd.Series([1, 3, 6, 12, 6, 2.0, 2.2, 2.4, 2.6], index=idx, dtype=float)
    att = challenge.sequential_attempts(eq, target=10, bust=0.2)
    assert [a["outcome"] for a in att] == ["win", "bust", "open"]
    assert att[0]["final"] == pytest.approx(12.0) and att[1]["final"] == pytest.approx(2 / 12)
    odds = challenge.start_date_odds(eq, 10, 0.2, horizon_days=3, step_days=1)
    assert odds["win"] > 0 and odds["win"] + odds["bust"] + odds["open"] == pytest.approx(1.0)
    assert "побед 1" in challenge.summary_text(eq, 10, 0.2, stake=50)


def test_guard_off_never_scales():
    g = DrawdownGuard(RiskConfig(hard_stop_action="off"), peak=100)
    assert g.on_bar_close(100) == 1.0 and g.on_bar_close(10) == 1.0
    assert not g.check_intrabar(1) and not g.active


def test_learning_blocks_losing_pair_and_cuts_risk():
    from bot.core import Decider

    cfg = make_cfg()
    cfg.strategy.learn_window, cfg.strategy.learn_block_bars = 2, 5
    cfg.strategy.learn_loss_streak, cfg.strategy.learn_risk_cut = 2, 0.5
    d = Decider(cfg, 10_000)
    d._record("ETH/USDT", 1, entry=100.0, risk=0.10, exit_price=90.0)      # −1R
    assert d.learn["streak"] == 1 and not d.learn["blocked"]
    d._record("ETH/USDT", 1, entry=100.0, risk=0.10, exit_price=95.0)      # −0.5R → связка в минусе
    assert d.learn["blocked"] == {"ETH/USDT|long": 5} and d.learn["streak"] == 2
    lessons = d.pop_lessons()
    assert any("отключаю" in t for t in lessons) and any("уменьшаю размер" in t for t in lessons)
    assert d.pop_lessons() == []

    # отключённая связка не открывается, размер после серии минусов урезан вдвое
    frame = trend_frame()
    from bot.strategy import prepare
    row = prepare(frame, cfg.strategy, "1d").iloc[300]
    rets = pd.DataFrame({"ETH/USDT": np.log(frame["close"]).diff().iloc[270:301]})
    assert d.on_bar_close({"ETH/USDT": row}, rets, 10_000)["ETH/USDT"] == 0.0
    d.learn["blocked"].clear()
    cut = d.on_bar_close({"ETH/USDT": row}, rets, 10_000)["ETH/USDT"]
    d2 = Decider(make_cfg(), 10_000)
    full = d2.on_bar_close({"ETH/USDT": row}, rets, 10_000)["ETH/USDT"]
    assert full > 0 and cut == pytest.approx(full * 0.5)
    d._record("ETH/USDT", 1, entry=100.0, risk=0.10, exit_price=130.0)     # прибыль → серия сброшена
    assert d.learn["streak"] == 0
    # состояние уроков переживает перезапуск
    d3 = Decider(cfg, 10_000)
    d3.load_dict(json.loads(json.dumps(d.to_dict())))
    assert d3.learn["hist"] == d.learn["hist"]


def turbo_cfg() -> Config:
    cfg = futures_cfg()
    cfg.strategy.allow_short = False
    cfg.portfolio.initial_capital = 50.0
    cfg.portfolio.max_gross_exposure = 5.0
    cfg.portfolio.profiles = {"core": {"ETH/USDT": 5.0}}
    cfg.strategy.target_vol = 1000.0
    cfg.strategy.risk_per_trade = 0.2
    cfg.risk.hard_stop_action = "off"
    cfg.challenge.enabled = True
    return cfg


def test_challenge_win_and_bust_restart_attempt(tmp_path):
    cfg = turbo_cfg()
    market = FakeMarket({"ETH/USDT": trend_frame()})
    note = FakeNotifier()
    runner = Runner(cfg, market, PaperBroker(0.0005, 0.0005), tmp_path / "t.json", note)
    for _ in range(40):
        runner.step()
        market.i += 1
    assert runner.book.units["ETH/USDT"] > 0                      # в позиции с плечом
    assert runner.snapshot["leverage"] > 1.0
    assert runner.snapshot["challenge"]["n"] == 1

    # победа: счёт вырос в 10 раз → позиции закрыты, новая попытка с 50 USDT
    px = market.prices(["ETH/USDT"])
    runner.book.cash += 50.0 * 10
    runner.check_challenge(px)
    assert runner.attempts[-1]["outcome"] == "win" and runner.attempts[-1]["mult"] >= 10
    assert runner.attempt["n"] == 2 and runner.book.cash == 50.0
    assert runner.book.units["ETH/USDT"] == 0
    assert any("Цель достигнута" in m for m in note.sent)

    # слив: осталось меньше 20% → попытка закрыта, начата третья
    runner.book.cash = 9.0
    runner.check_challenge(px)
    assert [a["outcome"] for a in runner.attempts] == ["win", "bust"]
    assert runner.attempt["n"] == 3 and runner.book.cash == 50.0
    assert any("слита" in m and "побед 1, сливов 1" in m for m in note.sent)

    runner.step()
    ch = runner.snapshot["challenge"]
    assert ch["won"] == 1 and ch["lost"] == 1 and ch["spent"] == 150.0
    from bot import messages
    assert "Попытка №3" in messages.status_text(runner.snapshot)
    again = Runner(cfg, market, PaperBroker(0.0005, 0.0005), tmp_path / "t.json")
    assert again.attempt["n"] == 3 and len(again.attempts) == 2   # история попыток сохраняется


def test_turbo_config_is_paper_only_and_bounded():
    from pathlib import Path

    from bot.config import load_config
    from bot.group import load_group

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.turbo.toml")
    assert cfg.exchange.mode == "paper" and cfg.exchange.market == "futures"
    assert cfg.challenge.enabled and cfg.challenge.target_mult == 10
    assert cfg.portfolio.max_gross_exposure <= 5.0 and cfg.portfolio.initial_capital == 50.0
    assert any(s.config == "config.turbo.toml" for s in load_group(root / "bots.toml"))
