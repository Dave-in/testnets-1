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
from .sim import Account, Order, round_to_tick
from .research import Recorder, market_context
from .strategies import STRATEGIES, PairSignal

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
        # общие настройки счёта [account] + свои у бота ([<бот>.account] в конфиге)
        acc = {**cfg.get("account", {}), **cfg.get(key, {}).get("account", {})}
        self.universe = cfg.get(key, {}).get("universe", "base")   # base | meme — какие монеты
        self.risk = float(acc.get("risk_per_trade", 0.01))
        self.max_leverage = float(acc.get("max_leverage", 10.0))
        self.max_positions = int(acc.get("max_positions", 3))
        self.min_notional = float(acc.get("min_notional", 5.0))
        self.max_position_leverage = float(acc.get("max_position_leverage", 3.0))
        self.min_stop_pct = float(acc.get("min_stop_pct", 0.004))
        # тейк во столько раз больше издержек сделки (комиссии + проскальзывание туда-обратно);
        # у бота может быть свой множитель в его разделе конфига
        self.min_reward_cost = float(cfg.get(key, {}).get("min_reward_cost",
                                                          acc.get("min_reward_cost", 2.5)))
        self.account = Account(key, float(acc.get("capital", 100.0)),
                               (float(acc.get("maker_fee", 0.0002)), float(acc.get("taker_fee", 0.00055))),
                               int(acc.get("latency_ms", 250)),
                               leverage=float(acc["leverage"]) if acc.get("leverage") else None)
        self.state_path = Path(state_dir) / f"shadow_{key}.json"
        self.control_path = Path(state_dir) / f"shadow_{key}_control.json"
        self.journal_path = Path(state_dir) / f"shadow_{key}_trades.csv"
        self.journal_checked = False
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
        # списки монет по именам (base — общий, meme — мемы); бот торгует только своим.
        # Пусто — без ограничений (все подключённые монеты)
        self.universes: dict[str, set] = {}
        keys = cfg.get("bots", DEFAULT_BOTS)
        self.bots = {k: ShadowBot(k, cfg, self.state_dir, notifier_for(k) if notifier_for else None)
                     for k in keys}
        self.now = 0
        self.next_sec = 0
        self.save_every_s = int(cfg.get("save_every_s", 60))
        # данные для улучшения ботов (state/research/): сигналы, сделки, путь цены
        self.research = Recorder(self.state_dir) if cfg.get("research", True) else None
        self._status = ""

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
            try:                                  # выход по сигналу стратегии (например, пары)
                for item in bot.strategy.manage(acc, self.markets, now) or ():
                    if item[0] == "open":         # добрать ногу пары по рынку
                        _, sym, sig, notional, tag, lone = item
                        if not self._enter(bot, self.markets[sym], sig, now, notional, tag,
                                           check_cost=False) and lone in acc.positions:
                            acc.close(lone, self.markets[lone].book, now, "pair_leg")
                        continue
                    sym, purpose = item
                    if sym in acc.positions:
                        acc.close(sym, self.markets[sym].book, now, purpose)
            except Exception:  # noqa: BLE001
                log.exception("[%s] ошибка сопровождения позиций", bot.key)
            allowed = self.universes.get(bot.universe)
            bot.strategy.allowed = allowed
            if not bot.paused:
                for sym, m in self.markets.items():
                    # ордера на вход, которые ещё не исполнились, тоже занимают место
                    if len(acc.positions) + len(acc.pending_entries()) >= bot.max_positions:
                        break
                    if not m.tradable or (allowed is not None and sym not in allowed) \
                            or acc.busy(sym, now):
                        continue
                    bot.strategy.cur = {}
                    try:
                        sig = bot.strategy.check(m, now, self.markets)
                    except Exception:  # noqa: BLE001 — ошибка в одной монете не валит остальное
                        log.exception("[%s] ошибка стратегии на %s", bot.key, sym)
                        continue
                    if sig is None:
                        continue
                    if isinstance(sig, PairSignal):
                        self._enter_pair(bot, sig, now, allowed)
                        first = sig.legs[0][1]
                        first.reason = first.reason or sig.reason
                        sig = first
                    else:
                        self._enter(bot, m, sig, now)
                    if self.research is not None:
                        try:
                            self.research.on_signal(bot.key, m, sig, self._status,
                                                    {**bot.strategy.cur, **sig.features},
                                                    self.markets, now)
                        except Exception:  # noqa: BLE001
                            log.exception("[%s] ошибка записи сигнала", bot.key)
            self._drain(bot)
            if sec % self.save_every_s == 0:
                self.refresh(bot)
        if self.research is not None:
            try:
                self.research.tick(self.markets, now)
            except Exception:  # noqa: BLE001
                log.exception("Ошибка записи данных для анализа")
        if sec % 60 == 0:
            try:
                self._calibration(sec)
            except OSError:
                log.exception("Не удалось записать журнал калибровки")

    def _size(self, bot: ShadowBot, m: SymbolMarket, sig, sl_dist: float, prices: dict,
              notional: float | None = None) -> float:
        """Объём позиции: стоп стоит risk·капитала (или заданная сумма), в пределах плеча
        на позицию и на счёт, свободной маржи и правил контракта Bybit. 0 — не входить."""
        acc, price = bot.account, m.price()
        equity = acc.equity(prices)
        if equity <= 0 or not price:
            return 0.0
        qty = notional / price if notional else bot.risk * equity / sl_dist
        room = bot.max_leverage * equity - acc.committed(prices)
        qty = min(qty, max(room, 0.0) / price, bot.max_position_leverage * equity / price)
        if acc.leverage:                          # изолированная маржа: хватает ли свободной
            free = equity - acc.used_margin(prices)
            qty = min(qty, max(free, 0.0) * acc.leverage / price)
        # правила контракта Bybit: предел рыночного ордера, шаг объёма (вниз), минимумы
        if m.max_mkt_qty and sig.entry != "limit":
            qty = min(qty, m.max_mkt_qty)
        if m.qty_step:
            qty = math.floor(qty / m.qty_step + 1e-9) * m.qty_step
        if qty <= 0 or qty < m.min_qty or qty * price < max(bot.min_notional, m.min_notional):
            return 0.0
        return qty

    def _enter(self, bot: ShadowBot, m: SymbolMarket, sig, now: int, notional: float | None = None,
               tag: str = "", check_cost: bool = True) -> bool:
        acc, price = bot.account, m.price()
        prices = {s: mk.price() for s, mk in self.markets.items()}
        if sig.sl_dist <= 0 or not price:
            self._status = "нет цены или стопа"
            return False
        # Слишком близкий стоп расширяем до min_stop_pct: позиция станет меньше, риск на сделку
        # тот же, а комиссия — меньшая доля риска.
        floor = bot.min_stop_pct * price
        sl_dist = max(sig.sl_dist, floor)
        trail = max(sig.trail, floor) if sig.trail else 0.0
        qty = self._size(bot, m, sig, sl_dist, prices, notional)
        if qty <= 0:
            self._status = "размер: пределы плеча, маржа или минимум биржи"
            return False
        # Сигналы, где тейк меньше min_reward_cost издержек сделки, пропускаем: на них комиссии
        # и проскальзывание съедают всё. Издержки — по текущему стакану для этого объёма.
        if sig.tp_dist and check_cost:
            cost = self.trade_cost(acc, m, sig, qty)
            ratio = (sig.tp_dist / price) / cost if cost > 0 else math.inf
            bot.strategy.observe("тейк / издержки сделки", ratio, bot.min_reward_cost, m.symbol)
            if ratio < bot.min_reward_cost:
                self._status = "издержки: тейк меньше порога"
                return False
        features = {"signal": {k: round(v, 6) for k, v in {**bot.strategy.cur, **sig.features}.items()}}
        if self.research is not None:
            features["market"] = market_context(m, self.markets, now)
        meta = {"sl_dist": sl_dist, "tp_dist": sig.tp_dist, "time_stop_s": sig.time_stop_s,
                "trail": trail, "reason": sig.reason, "mid0": m.book.mid() or price,
                "tick": m.tick, "tag": tag, "features": features}
        side = "buy" if sig.side > 0 else "sell"
        if sig.entry == "limit":
            limit = round_to_tick(sig.limit_price, m.tick, up=sig.side < 0)
            o = Order(m.symbol, side, "limit", qty, limit, "entry",
                      expire=now + sig.limit_ttl_s * 1000, meta=meta)
        else:
            o = Order(m.symbol, side, "market", qty, meta=meta)
        acc.submit(o, m.book, now)
        self._status = "вход"
        return True

    def _enter_pair(self, bot: ShadowBot, ps, now: int, allowed) -> None:
        """Сделка из двух ног (парный арбитраж): обе ноги или ни одной. Размер первой ноги —
        из риска на сделку, второй — сумма первой × β (нейтрально к общему движению рынка)."""
        acc = bot.account
        self._status = "пара: нет места для двух ног или нога занята"
        if len(acc.positions) + len(acc.pending_entries()) + 2 > bot.max_positions:
            return
        legs = []
        for sym, sig in ps.legs:
            m = self.markets.get(sym)
            if m is None or not m.tradable or acc.busy(sym, now) or \
                    (allowed is not None and sym not in allowed):
                return
            legs.append((m, sig))
        prices = {s: mk.price() for s, mk in self.markets.items()}
        equity = acc.equity(prices)
        (m1, s1), (m2, s2) = legs
        n1 = min(bot.risk * equity / ps.risk_frac, bot.max_position_leverage * equity)
        n2 = n1 * abs(ps.ratio)
        if self._size(bot, m1, s1, s1.sl_dist, prices, n1) <= 0 or \
                self._size(bot, m2, s2, s2.sl_dist, prices, n2) <= 0:
            self._status = "размер: пределы плеча, маржа или минимум биржи"
            return
        if self._enter(bot, m1, s1, now, n1, ps.tag, check_cost=False):
            self._enter(bot, m2, s2, now, n2, ps.tag, check_cost=False)

    @staticmethod
    def trade_cost(acc: Account, m: SymbolMarket, sig, qty: float) -> float:
        """Издержки сделки туда-обратно в долях цены: комиссии Bybit плюс проскальзывание.
        Вход лимиткой — комиссия мейкера без проскальзывания; выход считаем худшим —
        рыночным (стоп или время): комиссия тейкера и проход по стакану на этот объём."""
        book = m.book
        mid = book.mid() or m.price()
        if not mid:
            return acc.taker_fee * 2
        buy = sig.side > 0

        def slip(side: str) -> float:
            avg, filled = book.walk(side, qty)
            return abs(avg / mid - 1) if filled > 0 else 0.0

        entry = acc.maker_fee if sig.entry == "limit" else acc.taker_fee + slip("buy" if buy else "sell")
        exit_ = acc.taker_fee + slip("sell" if buy else "buy")
        return entry + exit_

    def _calibration(self, sec: int) -> None:
        """Раз в минуту: максимумы условий входа каждого бота — в state/shadow_calibration.csv."""
        rows = []
        for bot in self.bots.values():
            for metric, (value, threshold, symbol) in bot.strategy.take_observations().items():
                rows.append({"ts": iso(sec * 1000), "bot": bot.key, "metric": metric,
                             "value": round(value, 8), "threshold": threshold,
                             "ratio": round(value / threshold, 4) if threshold else "",
                             "symbol": symbol})
        if not rows:
            return
        path = self.state_dir / "shadow_calibration.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)

    def _drain(self, bot: ShadowBot) -> None:
        for ev in bot.account.events:
            row = {"ts": iso(self.now), "symbol": ev.symbol, "action": ev.action,
                   "side": "long" if ev.side > 0 else "short", "qty": round(ev.qty, 8),
                   "price": ev.price, "fee": round(ev.fee, 6), "reason": ev.reason,
                   "slippage_bps": round(ev.slippage_bps, 2),
                   "pnl": "" if ev.pnl is None else round(ev.pnl, 6),
                   "fee_type": "maker" if ev.maker else "taker",
                   "fee_rate_pct": round(ev.fee_rate * 100, 4),
                   "gross": "" if ev.gross is None else round(ev.gross, 6),
                   "trade_fees": "" if ev.trade_fees is None else round(ev.trade_fees, 6),
                   "funding": "" if ev.funding is None else round(ev.funding, 6)}
            new = not bot.journal_path.exists()
            if not new and not bot.journal_checked:
                # журнал от прежней версии с другими колонками — откладываем в сторону
                with bot.journal_path.open(encoding="utf-8") as f:
                    header = f.readline().strip()
                if header != ",".join(row):
                    bot.journal_path.replace(bot.journal_path.with_name(
                        bot.journal_path.stem + "_old.csv"))
                    new = True
            bot.journal_checked = True
            with bot.journal_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            if ev.action == "close" and self.research is not None and ev.symbol in self.markets:
                try:
                    self.research.on_close(bot.key, ev, self.markets[ev.symbol], self.markets,
                                           self.now)
                except Exception:  # noqa: BLE001
                    log.exception("[%s] ошибка записи сделки для анализа", bot.key)
            if self.notify_trades and bot.notifier is not None:
                summary = self.account_summary(bot)
                if ev.action == "open" and summary["equity"] > 0:
                    pos = bot.account.positions.get(ev.symbol)
                    summary["leverage"] = ev.qty * ev.price / summary["equity"]
                    summary["liq_price"] = pos.liq_price if pos is not None else 0.0
                self._notify(bot, messages.shadow_trade(ev, bot.title, summary), trade=True)
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
                if s in self.markets:
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
        # до первых данных биржи времени ещё нет (now = 0 → 1970-01-01): смену суток не считаем,
        # иначе каждый перезапуск сбрасывал бы баланс на начало дня и слал лишний отчёт
        if self.now and bot.marks.get("day") != day:
            if bot.marks.get("day") and bot.snapshot:
                bot.day_snapshot = bot.snapshot
                # итоги завершившегося дня — для отчёта за день
                bot.marks["last_day"] = {"day": bot.marks["day"],
                                         "start": bot.marks.get("day_equity", equity), "end": equity}
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
