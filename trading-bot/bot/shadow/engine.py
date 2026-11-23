"""Движок теневой торговли: общий поток данных, у каждого бота свой виртуальный счёт.

Время берётся из данных биржи, поэтому живой режим и повтор истории работают одинаково.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from .. import messages
from .market import SymbolMarket
from .sim import Account, Order
from .strategies import STRATEGIES

log = logging.getLogger(__name__)

DEFAULT_BOTS = ["liq", "book", "lag", "oi", "fund", "fade"]


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class ShadowBot:
    """Один бот: стратегия + виртуальный счёт + файлы состояния (совместимы с Telegram-командами)."""

    def __init__(self, key: str, cfg: dict, state_dir: Path, notifier=None):
        cls = STRATEGIES[key]
        params = {k: v for k, v in cfg.get(key, {}).items() if k in cls.defaults}
        self.key, self.title = key, cfg.get(key, {}).get("title", cls.title)
        self.strategy = cls(**params)
        acc = cfg.get("account", {})
        self.risk = float(acc.get("risk_per_trade", 0.01))
        self.max_leverage = float(acc.get("max_leverage", 10.0))
        self.max_positions = int(acc.get("max_positions", 3))
        self.min_notional = float(acc.get("min_notional", 5.0))
        self.max_position_leverage = float(acc.get("max_position_leverage", 3.0))
        self.min_stop_pct = float(acc.get("min_stop_pct", 0.004))
        self.min_reward_cost = float(acc.get("min_reward_cost", 2.5))
        self.account = Account(key, float(acc.get("capital", 100.0)),
                               (float(acc.get("maker_fee", 0.0002)), float(acc.get("taker_fee", 0.00055))),
                               int(acc.get("latency_ms", 250)))
        self.state_path = Path(state_dir) / f"shadow_{key}.json"
        self.control_path = Path(state_dir) / f"shadow_{key}_control.json"
        self.journal_path = Path(state_dir) / f"shadow_{key}_trades.csv"
        self.notifier = notifier
        self.paused = False
        self.stop_requested = False
        self.snapshot: dict = {}
        self.day_snapshot: dict | None = None
        self.marks: dict = {}
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.account.load_dict(st.get("account", {}))
            self.paused = st.get("paused", False)
            self.marks = st.get("marks", {})

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"account": self.account.to_dict(), "paused": self.paused,
                                   "marks": self.marks, "snapshot": self.snapshot},
                                  ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.state_path)


class Engine:
    def __init__(self, cfg: dict, state_dir: str | Path, notifier_for=None, source: str = "bybit"):
        self.cfg = cfg
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.notify_trades = bool(cfg.get("notify_trades", False))
        self.markets: dict[str, SymbolMarket] = {}
        keys = cfg.get("bots", DEFAULT_BOTS)
        self.bots = {k: ShadowBot(k, cfg, self.state_dir, notifier_for(k) if notifier_for else None)
                     for k in keys}
        self.now = 0
        self.next_sec = 0
        self.save_every_s = int(cfg.get("save_every_s", 60))

    # --- данные ---------------------------------------------------------------------
    def add_market(self, symbol: str, book=None) -> SymbolMarket:
        m = self.markets.get(symbol)
        if m is None:
            m = self.markets[symbol] = SymbolMarket(symbol, book)
        return m

    def on_trade(self, symbol: str, ts: int, price: float, qty: float, taker_buy: bool) -> None:
        m = self.markets.get(symbol)
        if m is None:
            return
        self.advance(ts)
        m.on_trade(ts, price, qty, taker_buy)
        for bot in self.bots.values():
            bot.account.on_trade(symbol, price, qty, taker_buy, m.book, ts)

    def advance(self, ts: int) -> None:
        """Продвинуть время: исполнить отложенные рыночные ордера, отработать секунды."""
        if ts <= self.now:
            return
        if not self.next_sec:
            self.next_sec = ts // 1000 + 1
        if ts // 1000 - self.next_sec > 600:          # долгая пауза без данных: перематываем,
            self.next_sec = ts // 1000 - 600           # отрабатываем только последние 10 минут
        while self.next_sec * 1000 <= ts:
            self.now = self.next_sec * 1000
            self._process_orders()
            self._second(self.next_sec)
            self.next_sec += 1
        self.now = ts
        self._process_orders()

    def _process_orders(self) -> None:
        books = {s: m.book for s, m in self.markets.items()}
        for bot in self.bots.values():
            if bot.account.orders:
                bot.account.process(books, self.now)
                self._drain(bot)

    # --- раз в секунду ----------------------------------------------------------------
    def _second(self, sec: int) -> None:
        now = sec * 1000
        for m in self.markets.values():
            m.on_second(sec)
        for bot in self.bots.values():
            self._control(bot)
            acc = bot.account
            acc.housekeeping(self.markets, now)
            if acc.check_liquidation(self.markets, now):
                self._notify(bot, "💥 Счёт достиг уровня ликвидации — все позиции закрываются.")
            if not bot.paused:
                for sym, m in self.markets.items():
                    # ордера на вход, которые ещё не исполнились, тоже занимают место
                    if len(acc.positions) + len(acc.pending_entries()) >= bot.max_positions:
                        break
                    if acc.busy(sym, now):
                        continue
                    try:
                        sig = bot.strategy.check(m, now, self.markets)
                    except Exception:  # noqa: BLE001 — ошибка в одной монете не валит остальное
                        log.exception("[%s] ошибка стратегии на %s", bot.key, sym)
                        continue
                    if sig is not None:
                        self._enter(bot, m, sig, now)
            self._drain(bot)
            if sec % self.save_every_s == 0:
                self.refresh(bot)

    def _enter(self, bot: ShadowBot, m: SymbolMarket, sig, now: int) -> None:
        acc, price = bot.account, m.price()
        prices = {s: mk.price() for s, mk in self.markets.items()}
        equity = acc.equity(prices)
        if equity <= 0 or sig.sl_dist <= 0 or not price:
            return
        # Слишком близкий стоп расширяем до min_stop_pct: позиция станет меньше, риск на сделку
        # тот же, а комиссия — меньшая доля риска. Сигналы с тейком меньше min_reward_cost
        # стоимостей сделки туда-обратно пропускаем: на них комиссии съедают всё.
        floor = bot.min_stop_pct * price
        sl_dist = max(sig.sl_dist, floor)
        trail = max(sig.trail, floor) if sig.trail else 0.0
        round_trip = acc.taker_fee * 2
        if sig.tp_dist and sig.tp_dist / price < bot.min_reward_cost * round_trip:
            return
        qty = bot.risk * equity / sl_dist                         # стоп = risk·капитала
        room = bot.max_leverage * equity - acc.committed(prices)
        qty = min(qty, max(room, 0.0) / price, bot.max_position_leverage * equity / price)
        if qty * price < bot.min_notional:
            return
        meta = {"sl_dist": sl_dist, "tp_dist": sig.tp_dist, "time_stop_s": sig.time_stop_s,
                "trail": trail, "reason": sig.reason, "mid0": m.book.mid() or price}
        side = "buy" if sig.side > 0 else "sell"
        if sig.entry == "limit":
            o = Order(m.symbol, side, "limit", qty, sig.limit_price, "entry",
                      expire=now + sig.limit_ttl_s * 1000, meta=meta)
        else:
            o = Order(m.symbol, side, "market", qty, meta=meta)
        acc.submit(o, m.book, now)

    def _drain(self, bot: ShadowBot) -> None:
        for ev in bot.account.events:
            row = {"ts": iso(self.now), "symbol": ev.symbol, "action": ev.action,
                   "side": "long" if ev.side > 0 else "short", "qty": round(ev.qty, 8),
                   "price": ev.price, "fee": round(ev.fee, 6), "reason": ev.reason,
                   "slippage_bps": round(ev.slippage_bps, 2),
                   "pnl": "" if ev.pnl is None else round(ev.pnl, 6)}
            new = not bot.journal_path.exists()
            with bot.journal_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            if self.notify_trades and bot.notifier is not None:
                self._notify(bot, messages.shadow_trade(ev, bot.title, self.account_summary(bot)),
                             trade=True)
        bot.account.events.clear()

    def account_summary(self, bot: ShadowBot) -> dict:
        """Баланс и статистика счёта — для сообщения о сделке."""
        acc, st = bot.account, bot.account.stats
        prices = {s: m.price() for s, m in self.markets.items()}
        return {"equity": acc.equity(prices), "start": acc.start, "open": len(acc.positions),
                "closed": st["closed"], "wins": st["wins"], "realized": st["realized"],
                "fees": st["fees"]}

    # --- управление и отчёты -------------------------------------------------------------
    def _notify(self, bot: ShadowBot, text: str, trade: bool = False) -> None:
        if bot.notifier is not None:
            try:
                if trade:
                    bot.notifier.send(text, trade=True)
                else:
                    bot.notifier.send(text)
            except Exception:  # noqa: BLE001
                log.exception("Не удалось отправить уведомление")

    def _control(self, bot: ShadowBot) -> None:
        if not bot.control_path.exists():
            return
        try:
            cmd = json.loads(bot.control_path.read_text(encoding="utf-8")).get("command")
        except ValueError:
            cmd = None
        bot.control_path.unlink(missing_ok=True)
        if cmd == "pause":
            bot.paused = True
        elif cmd == "resume":
            bot.paused = False
        elif cmd == "close-all":
            for s in list(bot.account.positions):
                bot.account.close(s, self.markets[s].book, self.now, "manual")
            bot.account.orders = [o for o in bot.account.orders if o.purpose != "entry"]
            bot.paused = True
        elif cmd == "stop":
            bot.stop_requested = True

    def refresh(self, bot: ShadowBot) -> None:
        """Снимок для /status и /report, смена суток, сохранение."""
        acc = bot.account
        prices = {s: m.price() for s, m in self.markets.items()}
        equity = acc.equity(prices)
        acc.peak = max(acc.peak, equity)
        day = iso(self.now)[:10]
        if bot.marks.get("day") != day:
            if bot.marks.get("day") and bot.snapshot:
                bot.day_snapshot = bot.snapshot
            bot.marks.update(day=day, day_equity=equity)
            if bot.marks.get("month") != day[:7]:
                bot.marks.update(month=day[:7], month_equity=equity)
        positions = []
        for s, p in acc.positions.items():
            px = prices.get(s, p.entry)
            positions.append({"symbol": s, "side": p.side, "units": p.side * p.qty, "entry": p.entry,
                              "value": p.qty * px, "weight": p.qty * px / equity if equity else 0.0,
                              "ret": p.side * (px / p.entry - 1), "stop": p.sl})
        st = acc.stats
        bot.snapshot = {
            "ts": iso(self.now), "mode": "теневой", "exchange": self.source, "market": "futures",
            "allow_short": True, "equity": equity, "cash": acc.cash, "start_equity": acc.start,
            "drawdown": 1 - equity / acc.peak if acc.peak else 0.0, "peak": acc.peak,
            "paused": bot.paused, "halted": False, "guard_paused_left": 0, "regime_ok": True,
            "leverage": acc.gross(prices) / equity if equity > 0 else 0.0,
            "day_ret": equity / bot.marks.get("day_equity", equity) - 1,
            "month_ret": equity / bot.marks.get("month_equity", equity) - 1,
            "total_ret": equity / acc.start - 1, "positions": positions,
            "stats": {"closed": st["closed"], "wins": st["wins"], "realized": st["realized"],
                      "fees": st["fees"], "funding": st["funding"]},
            "shadow": {"strategy": bot.title, "opened": st["opened"],
                       "avg_slippage_bps": st["slippage_bps"] / st["opened"] if st["opened"] else 0.0},
        }
        bot.save()

    def refresh_all(self) -> None:
        for bot in self.bots.values():
            self.refresh(bot)

    def summary(self) -> list[dict]:
        out = []
        for bot in self.bots.values():
            st, acc = bot.account.stats, bot.account
            prices = {s: m.price() for s, m in self.markets.items()}
            out.append({"bot": bot.title, "trades": st["closed"], "open": len(acc.positions),
                        "win_rate": st["wins"] / st["closed"] if st["closed"] else math.nan,
                        "net": acc.equity(prices) - acc.start, "fees": st["fees"],
                        "funding": st["funding"],
                        "avg_slippage_bps": st["slippage_bps"] / st["opened"] if st["opened"] else math.nan})
        return out
