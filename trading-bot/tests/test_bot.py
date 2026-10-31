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
    state = tmp_path / "paper.json"
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
    assert (tmp_path / "paper_trades.csv").exists()


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
