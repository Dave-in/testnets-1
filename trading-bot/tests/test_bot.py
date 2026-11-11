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
