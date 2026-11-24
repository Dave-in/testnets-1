"""Данные для улучшения ботов (state/research/), отдельно от журналов сделок.

signals_<бот>.jsonl — каждый сигнал стратегии, вошёл бот или нет (и почему нет), с признаками
    сигнала и рынка и путём цены на час вперёд (точка каждые 10 с) — по ним «лаборатория»
    проверяет другие пороги, стопы и тейки без ожидания новых сделок.
trades_<бот>.jsonl — каждая закрытая сделка: результат в $ и в R (доля от риска при входе),
    ход цены внутри сделки (MFE — лучший, MAE — худший), длительность, причина выхода,
    признаки на входе и ход цены через 1/5/15/60 мин после выхода.

Запись делается, когда прошёл час после события: путь цены берётся из посекундной истории
(6 ч), поэтому каждую секунду ничего не считается. Перезапуск теряет только незавершённые записи.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

HORIZONS = (60, 300, 900, 3600)      # с: ход цены через 1, 5, 15, 60 мин
PATH_STEP = 10                       # с: шаг пути цены
PATH_LEN = 3600                      # с: длина пути цены
SIGNAL_GAP_MS = 60_000               # один и тот же сигнал (бот+монета+сторона) — не чаще раза в минуту


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _r(x, n=6):
    return round(float(x), n) if x is not None and math.isfinite(float(x)) else None


def market_context(m, markets: dict, now: int, ref: str = "BTCUSDT") -> dict:
    """Состояние рынка по монете: волатильность, спред, глубина, перекос стакана, поток сделок,
    ход цены и BTC, фандинг за 8 ч, изменение открытого интереса, объём минуты, час."""
    price = m.price()
    book = m.book
    ctx = {"price": price, "hour_utc": datetime.fromtimestamp(now / 1000, tz=timezone.utc).hour,
           "sigma1m_pct": m.sigma1m() * 100, "ret60_pct": m.ret(60) * 100,
           "ret300_pct": m.ret(300) * 100, "flow30": m.flow(30, now),
           "funding8h_pct": m.funding_rate * 8 / (m.funding_interval_h or 8) * 100,
           "turnover24h_musd": m.turnover24h / 1e6}
    if book.ready and price:
        ctx["spread_bps"] = book.spread() / price * 1e4
        ctx["obi3"] = book.imbalance(3)
        if not getattr(book, "approx", False):
            lo, hi = price * 0.999, price * 1.001
            ctx["depth01_usd"] = min(sum(p * q for p, q in book.levels("bid", 50) if p >= lo),
                                     sum(p * q for p, q in book.levels("ask", 50) if p <= hi))
    oi = m.oi_change(300, now)
    if oi is not None:
        ctx["oi_chg300_pct"] = oi * 100
    if len(m.minute_usd) >= 30:
        vols = list(m.minute_usd)[-60:]
        mean = sum(vols) / len(vols)
        sd = (sum((v - mean) ** 2 for v in vols) / len(vols)) ** 0.5
        cur = sum(usd for s, _, usd, _ in reversed(m.trades) if s > now // 1000 - 60)
        ctx["vol_z"] = (cur - mean) / sd if sd > 0 else 0.0
    b = markets.get(ref)
    if b is not None and b is not m:
        ctx["btc_ret60_pct"] = b.ret(60) * 100
        ctx["btc_ret300_pct"] = b.ret(300) * 100
    return {k: _r(v) for k, v in ctx.items()}


def price_path(m, t0: int, now: int) -> list | None:
    """Относительный путь цены p/p₀ − 1 от момента t0 на PATH_LEN с вперёд, шаг PATH_STEP."""
    age = (now - t0) // 1000                       # сколько секунд назад было t0
    p0 = m.price_ago(age)
    if not p0:
        return None
    out = []
    for s in range(0, PATH_LEN + 1, PATH_STEP):
        p = m.price_ago(max(age - s, 0))
        out.append(round(p / p0 - 1, 6) if p else None)
    return out


class Recorder:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir) / "research"
        self.pending: list = []                    # (когда записать, вид, бот, монета, t0, запись)
        self.last_signal: dict = {}

    def on_signal(self, bot: str, m, sig, status: str, features: dict, markets: dict, now: int) -> None:
        key = (bot, m.symbol, sig.side)
        if now - self.last_signal.get(key, -SIGNAL_GAP_MS) < SIGNAL_GAP_MS:
            return
        self.last_signal[key] = now
        price = m.price()
        rec = {"ts": _iso(now), "bot": bot, "symbol": m.symbol, "side": sig.side, "status": status,
               "entry": sig.entry, "price": price, "reason": sig.reason,
               "sl_pct": _r(sig.sl_dist / price * 100) if price else None,
               "tp_pct": _r(sig.tp_dist / price * 100) if price else None,
               "trail_pct": _r(sig.trail / price * 100) if price else None,
               "hold_s": sig.time_stop_s,
               "features": {k: _r(v) for k, v in features.items()},
               "market": market_context(m, markets, now)}
        self.pending.append((now + PATH_LEN * 1000, "signals", bot, m.symbol, now, rec))

    def on_close(self, bot: str, ev, m, markets: dict, now: int) -> None:
        if not ev.opened or not ev.entry:
            return
        hold = max((now - ev.opened) // 1000, 0)
        favorable = []
        for k in range(min(hold, len(m.sec) - 1), -1, -1):       # путь цены внутри сделки
            p = m.price_ago(k)
            if p:
                favorable.append(ev.side * (p / ev.entry - 1))
        mfe = max(favorable) if favorable else 0.0
        mae = min(favorable) if favorable else 0.0
        risk_pct = ev.risk_usd / (ev.entry * ev.qty) if ev.qty and ev.entry else 0.0
        rec = {"opened": _iso(ev.opened), "closed": _iso(now), "bot": bot, "symbol": ev.symbol,
               "side": ev.side, "entry": ev.entry, "exit": ev.price, "exit_reason": ev.reason,
               "hold_s": hold, "pnl": _r(ev.pnl), "ret_pct": _r((ev.ret or 0) * 100),
               "r_multiple": _r(ev.pnl / ev.risk_usd) if ev.risk_usd else None,
               "risk_pct": _r(risk_pct * 100), "fees": _r(ev.trade_fees),
               "mfe_pct": _r(mfe * 100), "mae_pct": _r(mae * 100),
               "mfe_r": _r(mfe / risk_pct) if risk_pct else None,
               "mae_r": _r(mae / risk_pct) if risk_pct else None,
               "features": ev.features or {}}
        self.pending.append((now + PATH_LEN * 1000, "trades", bot, ev.symbol, now, rec))

    def tick(self, markets: dict, now: int) -> None:
        if not self.pending or self.pending[0][0] > now:
            return
        due = [p for p in self.pending if p[0] <= now]
        self.pending = [p for p in self.pending if p[0] > now]
        for _, kind, bot, symbol, t0, rec in due:
            m = markets.get(symbol)
            path = price_path(m, t0, now) if m is not None else None
            if path is not None:
                side = rec["side"]
                after = {f"after_{h // 60}m_pct": _r(side * path[h // PATH_STEP] * 100)
                         for h in HORIZONS if path[h // PATH_STEP] is not None}
                if kind == "signals":
                    rec["path"] = path
                    rec.update(after)
                    vals = [side * x for x in path if x is not None]
                    rec["best_60m_pct"] = _r(max(vals) * 100)
                    rec["worst_60m_pct"] = _r(min(vals) * 100)
                else:
                    rec.update(after)
            self._write(kind, bot, rec)

    def _write(self, kind: str, bot: str, rec: dict) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with (self.dir / f"{kind}_{bot}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("Не удалось записать данные для анализа")
