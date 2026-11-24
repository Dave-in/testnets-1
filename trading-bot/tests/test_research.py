"""Данные для улучшения ботов: сигналы с путём цены, сделки с MFE/MAE, лаборатория."""
import json

import numpy as np

from bot.shadow import lab
from bot.shadow import strategies as S
from bot.shadow.engine import Engine
from bot.shadow.research import PATH_LEN, PATH_STEP
from tests.test_shadow import T0, book


class Always(S.Strategy):
    key, title, defaults = "fade", "x", {}

    def check(self, m, now, markets):
        self.observe("сила сигнала", 3.2, 3.0, m.symbol)
        return S.Signal(1, m.price() * 0.005, m.price() * 0.01, reason="тест", time_stop_s=120)


def run_engine(tmp_path, prices, cost_tp=10.0):
    cfg = {"bots": ["fade"], "save_every_s": 1, "account": {"capital": 100.0, "latency_ms": 0},
           "fade": {"min_reward_cost": cost_tp}}
    eng = Engine(cfg, tmp_path)
    eng.bots["fade"].strategy = Always()
    m = eng.add_market("ETHUSDT", book(100.0, 0.01, 1e6))
    m.warmup_minutes(list(100 + np.random.default_rng(1).normal(0, 0.05, 200)))
    t = T0
    for p in prices:
        t += 1000
        m.book = book(p, 0.01, 1e6)
        eng.on_trade("ETHUSDT", t, p, 1.0, True)
    return eng, t


def test_signals_recorded_with_status_features_and_price_path(tmp_path):
    # тейк 1% < 10 × издержки — сигнал отсеян; через час записан путь цены
    prices = [100.0] * 30 + list(np.linspace(100, 101, PATH_LEN + 30))
    run_engine(tmp_path, prices, cost_tp=10.0)
    rows = [json.loads(x) for x in (tmp_path / "research" / "signals_fade.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert rows, "сигнал не записан"
    r = rows[0]
    assert r["status"] == "издержки: тейк меньше порога" and r["side"] == 1
    assert r["features"]["сила сигнала"] == 3.2 and "sigma1m_pct" in r["market"]
    assert len(r["path"]) == PATH_LEN // PATH_STEP + 1 and r["after_60m_pct"] > 0.5
    assert len(rows) <= len(prices) // 60 + 1                     # не чаще раза в минуту


def test_trades_recorded_with_mfe_mae_r_and_after_exit(tmp_path):
    # вход, цена сначала +0.4%, потом −0.6% → стоп; через час после выхода — +1%
    up = list(np.linspace(100, 100.4, 20))
    down = list(np.linspace(100.4, 99.3, 20))
    after = list(np.linspace(99.3, 101, PATH_LEN + 60))
    eng, _ = run_engine(tmp_path, [100.0] * 5 + up + down + after, cost_tp=0.1)
    rows = [json.loads(x) for x in (tmp_path / "research" / "trades_fade.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    t = rows[0]
    assert t["exit_reason"] == "sl" and t["r_multiple"] < 0
    assert t["mfe_pct"] > 0.3 and t["mae_pct"] < -0.4                # был в плюсе, потом стоп
    assert t["after_60m_pct"] > 0 and "signal" in t["features"] and "market" in t["features"]


def test_lab_simulation_and_report(tmp_path):
    d = tmp_path / "research"
    d.mkdir()
    rng = np.random.default_rng(3)
    with (d / "signals_oi.jsonl").open("w", encoding="utf-8") as f:
        for i in range(60):
            drift = 0.00002 if i % 2 else -0.00002                   # чётные — против сигнала
            path = [0.0] + list(np.cumsum(rng.normal(drift, 0.0003, PATH_LEN // PATH_STEP)))
            f.write(json.dumps({"ts": f"2026-11-24T{10 + i // 60:02d}:{i % 60:02d}:00+00:00",
                                "side": 1, "status": "вход", "path": path,
                                "features": {"сила": (i % 2) * 10 + i % 5},
                                "market": {"spread_bps": 1.0 + i % 3}, "after_15m_pct": path[90] * 100}) + "\n")
    assert lab.simulate([0, 0.003, 0.02], 1, 0.5, 1.0, 5) == 1.0 - lab.COST_PCT
    assert lab.simulate([0, -0.006, 0.02], 1, 0.5, 1.0, 5) == -0.5 - lab.COST_PCT
    text = lab.report(tmp_path, "oi", 3)
    assert "СИГНАЛЫ" in text and "ПЕРЕБОР" in text and "сила" in text
