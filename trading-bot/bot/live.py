"""Работа в реальном времени: paper (виртуальные деньги), testnet, live.

Цикл:
  1. каждые poll_seconds берем текущие цены → проверяем трейлинг-стопы и лимит просадки;
  2. когда закрылся новый бар — пересчитываем сигналы и ребалансируем портфель
     (та же логика Decider, что и в бэктесте);
  3. состояние сохраняется в JSON, поэтому бота можно перезапускать.

Управление на ходу (из другого терминала): python -m bot ctl pause|resume|close-all|stop.
Команда кладется в файл <state>_control.json, бот читает его на каждом шаге.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import messages
from .brokers import Book, ExchangeBroker, PaperBroker
from .config import Config
from .core import Decider, rebalance_deltas
from .data import _to_frame, make_exchange
from .strategy import add_regime, prepare

log = logging.getLogger(__name__)


class CcxtMarketData:
    """Публичные данные биржи (ключи не нужны)."""

    def __init__(self, exchange_name: str):
        self.ex = make_exchange(exchange_name)
        self.ex.load_markets()

    def closed_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        df = _to_frame(self.ex.fetch_ohlcv(symbol, timeframe, limit=limit + 1))
        step = pd.Timedelta(seconds=self.ex.parse_timeframe(timeframe))
        if len(df) and df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]  # последняя свеча еще не закрыта
        return df

    def prices(self, symbols: list[str]) -> dict[str, float]:
        tickers = self.ex.fetch_tickers(symbols)
        return {s: float(tickers[s]["last"]) for s in symbols}


class Runner:
    def __init__(self, cfg: Config, market, broker, state_path: str | Path, notifier=None):
        self.cfg = cfg
        self.market = market
        self.broker = broker
        self.notifier = notifier     # объект с методом send(text) или None
        self.symbols = cfg.portfolio.symbols
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.decider = Decider(cfg, cfg.portfolio.initial_capital)
        self.book = Book(cash=cfg.portfolio.initial_capital, units={s: 0.0 for s in self.symbols},
                         cost={s: 0.0 for s in self.symbols})
        self.pos_pnl = {s: 0.0 for s in self.symbols}   # зафиксировано по текущей позиции
        self.pos_invested = {s: 0.0 for s in self.symbols}  # всего вложено в текущую позицию
        self._prices: dict[str, float] = {}
        self.stats = {"closed": 0, "wins": 0, "realized": 0.0, "fees": 0.0}
        self.marks: dict = {}        # капитал на начало дня/месяца — для отчётов
        self.regime_ok = True
        self.snapshot: dict = {}
        self._dirty = True           # были сделки/новый бар → записать точку в журнал капитала
        self._last_journal_hour = ""
        self.last_bar: str | None = None
        self.paused = False          # пауза по команде: новых сделок нет, стопы работают
        self.stop_requested = False
        self.history_limit = min(1000, max(500, cfg.strategy.warmup * 4))
        self._load()

    # --- состояние --------------------------------------------------------
    def _load(self) -> None:
        if not self.state_path.exists():
            log.info("Новый запуск. Капитал бота: %.2f USDT", self.book.cash)
            return
        st = json.loads(self.state_path.read_text())
        self.book = Book(cash=st["book"]["cash"],
                         units={s: st["book"]["units"].get(s, 0.0) for s in self.symbols},
                         cost={s: st["book"].get("cost", {}).get(s, 0.0) for s in self.symbols})
        self.pos_pnl.update({s: v for s, v in st.get("pos_pnl", {}).items() if s in self.pos_pnl})
        self.pos_invested.update({s: v for s, v in st.get("pos_invested", {}).items()
                                  if s in self.pos_invested})
        self.stats.update(st.get("stats", {}))
        self.marks = st.get("marks", {})
        self.regime_ok = st.get("regime_ok", True)
        self.decider.load_dict(st["decider"])
        self.last_bar = st.get("last_bar")
        self.paused = st.get("paused", False)
        log.info("Состояние восстановлено из %s (последний бар %s)", self.state_path, self.last_bar)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "book": {"cash": self.book.cash, "units": self.book.units, "cost": self.book.cost},
            "pos_pnl": self.pos_pnl,
            "pos_invested": self.pos_invested,
            "stats": self.stats,
            "marks": self.marks,
            "regime_ok": self.regime_ok,
            "snapshot": self.snapshot,
            "decider": self.decider.to_dict(),
            "last_bar": self.last_bar,
            "paused": self.paused,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2, default=str))
        tmp.replace(self.state_path)

    def _journal(self, name: str, row: dict) -> None:
        path = self.state_path.parent / f"{self.state_path.stem}_{name}.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # --- торговля ---------------------------------------------------------
    def _execute(self, symbol: str, side: str, units: float, price: float, reason: str) -> None:
        if units <= 0:
            return
        if side == "buy":
            ex = self.cfg.exchange
            units = min(units, self.book.cash / (price * (1 + ex.slippage) * (1 + ex.fee)))
        else:
            units = min(units, self.book.units[symbol])
        fill = self.broker.market_order(symbol, side, units, price)
        if fill is None:
            return
        before = self.book.units[symbol]
        pnl = position_pnl = position_ret = None
        if side == "buy":
            self.book.cash -= fill.cost
            self.book.units[symbol] += fill.units
            self.book.cost[symbol] += fill.cost
            self.pos_invested[symbol] += fill.cost
        else:
            sold_cost = self.book.cost[symbol] * min(1.0, fill.units / before) if before > 0 else 0.0
            pnl = fill.cost - sold_cost
            self.book.cash += fill.cost
            self.book.units[symbol] = max(0.0, before - fill.units)
            self.book.cost[symbol] -= sold_cost
            self.pos_pnl[symbol] += pnl
            self.stats["realized"] += pnl
            if self.book.units[symbol] * price < 1.0:  # пыль после округления → позиция закрыта
                invested = self.pos_invested[symbol]
                position_pnl = self.pos_pnl[symbol]
                position_ret = position_pnl / invested if invested > 0 else None
                self.stats["closed"] += 1
                self.stats["wins"] += position_pnl > 0
                self.book.units[symbol] = self.book.cost[symbol] = 0.0
                self.pos_pnl[symbol] = self.pos_invested[symbol] = 0.0
        self.stats["fees"] += fill.fee
        self._dirty = True
        log.info("%s %s %.6f @ %.4f (комиссия %.4f) — %s",
                 side.upper(), symbol, fill.units, fill.price, fill.fee, reason)
        self._journal("trades", {"ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
                                 "side": side, "units": fill.units, "price": fill.price,
                                 "fee": fill.fee, "reason": reason,
                                 "pnl": "" if pnl is None else round(pnl, 4)})
        prices = {**self._prices, symbol: price}
        equity = self.book.cash + sum(u * prices.get(s, 0.0) for s, u in self.book.units.items())
        self._notify(messages.trade_message(
            symbol=symbol, side=side, fill_price=fill.price, notional=fill.units * fill.price,
            reason=reason, units_before=before, units_after=self.book.units[symbol],
            equity=equity, weight_after=self.book.units[symbol] * price / equity if equity else 0,
            stop=self.decider.states[symbol].stop, position_pnl=position_pnl,
            position_ret=position_ret, virtual=self.cfg.exchange.mode != "live"))

    def _notify(self, text: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.send(text)
        except Exception:  # noqa: BLE001 — уведомления не должны ронять торговлю
            log.exception("Не удалось отправить уведомление")

    def _sell_all(self, symbol: str, price: float, reason: str) -> None:
        self._execute(symbol, "sell", self.book.units[symbol], price, reason)

    def check_risk(self, prices: dict[str, float]) -> None:
        """Трейлинг-стопы и жесткий лимит просадки по текущим ценам."""
        for s in self.decider.check_stops(prices, self.book.units):
            self._sell_all(s, prices[s], "trailing_stop")
            self.decider.on_stop_hit(s)
        equity = self.book.equity(prices)
        if self.decider.guard.check_intrabar(equity, datetime.now(timezone.utc)):
            log.warning("ЛИМИТ ПРОСАДКИ: капитал %.2f, пик %.2f — закрываем всё",
                        equity, self.decider.guard.peak)
            self._notify(f"⚠️ <b>Лимит просадки</b>: счёт {messages.money(equity)} USDT, "
                         f"пик {messages.money(self.decider.guard.peak)}. Закрываю все позиции.")
            for s in self.symbols:
                self._sell_all(s, prices[s], "drawdown_guard")
            self.decider.reset_signals()

    def on_new_bar(self, prices: dict[str, float]) -> str | None:
        """Если закрылся новый бар — пересчет сигналов и ребалансировка."""
        tf = self.cfg.exchange.timeframe
        raw = {s: self.market.closed_ohlcv(s, tf, self.history_limit) for s in self.symbols}
        bar_ts = max(str(df.index[-1]) for df in raw.values() if len(df))
        if bar_ts == self.last_bar:
            return None

        frames = {s: prepare(df, self.cfg.strategy, tf) for s, df in raw.items()}
        regime = self.cfg.strategy.regime_symbol
        if regime:
            regime_df = raw[regime] if regime in raw else self.market.closed_ohlcv(
                regime, tf, self.history_limit)
            add_regime(frames, regime_df, self.cfg.strategy)
        rows = {s: df.iloc[-1] for s, df in frames.items() if len(df)}
        was_ok, self.regime_ok = self.regime_ok, all(bool(r["regime_ok"]) for r in rows.values())
        if was_ok != self.regime_ok:
            self._notify("🌤 Рынок снова растёт (BTC выше своей средней) — бот возвращается к сделкам."
                         if self.regime_ok else
                         "🌧 Рынок ослаб (BTC ниже своей средней) — бот выходит в USDT и ждёт.")
        n = self.cfg.strategy.vol_lookback
        rets = pd.DataFrame({s: np.log(df["close"]).diff().iloc[-n:] for s, df in frames.items()})
        close_prices = {s: float(rows[s]["close"]) for s in rows}
        equity = self.book.equity(close_prices)
        targets = self.decider.on_bar_close(rows, rets, equity, bar_ts)
        current = self.book.weights(prices)
        deltas = rebalance_deltas(targets, current, self.cfg.strategy.rebalance_threshold,
                                  self.cfg.strategy.min_trade)
        log.info("Бар %s: капитал %.2f, цели %s, текущие %s", bar_ts, equity,
                 {s: round(w, 3) for s, w in targets.items()},
                 {s: round(w, 3) for s, w in current.items()})

        eq_now = self.book.equity(prices)
        for s in sorted(deltas, key=lambda s: deltas[s] - current.get(s, 0.0)):
            diff_units = (deltas[s] - current.get(s, 0.0)) * eq_now / prices[s]
            side = "buy" if diff_units > 0 else "sell"
            self._execute(s, side, abs(diff_units), prices[s], "rebalance")
        self.last_bar = bar_ts
        self._dirty = True
        return bar_ts

    def make_snapshot(self, prices: dict[str, float]) -> dict:
        """Срез состояния для /status, /report и ctl status."""
        equity = self.book.equity(prices)
        start = self.cfg.portfolio.initial_capital
        positions = []
        for s, u in self.book.units.items():
            if u <= 0:
                continue
            value, cost = u * prices[s], self.book.cost[s]
            positions.append({"symbol": s, "units": u, "value": value,
                              "weight": value / equity if equity else 0.0,
                              "ret": value / cost - 1 if cost > 0 else 0.0,
                              "stop": self.decider.states[s].stop})
        guard = self.decider.guard
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": self.cfg.exchange.mode, "exchange": self.cfg.exchange.name,
            "equity": equity, "cash": self.book.cash, "start_equity": start,
            "drawdown": guard.drawdown(equity), "peak": guard.peak,
            "paused": self.paused, "halted": guard.halted, "guard_paused_left": guard.paused_left,
            "regime_ok": self.regime_ok,
            "day_ret": equity / self.marks.get("day_equity", equity) - 1,
            "month_ret": equity / self.marks.get("month_equity", equity) - 1,
            "total_ret": equity / start - 1,
            "positions": positions, "stats": dict(self.stats),
        }

    def _roll_marks(self, equity: float) -> None:
        """Смена суток (UTC): дневной отчёт и новые точки отсчёта для дня и месяца."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.marks.get("day") == today:
            return
        if self.marks.get("day") and self.cfg.telegram.daily_report:
            self._notify(messages.report_text(self.snapshot, "Отчёт за день"))
        self.marks.update(day=today, day_equity=equity)
        if self.marks.get("month") != today[:7]:
            self.marks.update(month=today[:7], month_equity=equity)

    @property
    def control_path(self) -> Path:
        return control_path(self.state_path)

    def apply_control(self, prices: dict[str, float]) -> None:
        if not self.control_path.exists():
            return
        cmd = json.loads(self.control_path.read_text()).get("command")
        self.control_path.unlink()
        log.warning("Команда управления: %s", cmd)
        if cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
            self.last_bar = None  # сразу пересчитать сигналы, не дожидаясь нового бара
        elif cmd == "close-all":
            for s in self.symbols:
                self._sell_all(s, prices[s], "manual_close_all")
            self.decider.reset_signals()
            self.paused = True
        elif cmd == "stop":
            self.stop_requested = True
        else:
            log.error("Неизвестная команда: %s", cmd)

    def step(self) -> None:
        prices = self._prices = self.market.prices(self.symbols)
        self.apply_control(prices)
        self.check_risk(prices)
        if not self.paused:
            self.on_new_bar(prices)
        equity = self.book.equity(prices)
        self.snapshot = self.make_snapshot(prices)
        self._roll_marks(equity)
        hour = self.snapshot["ts"][:13]
        if self._dirty or hour != self._last_journal_hour:  # не чаще раза в час + после сделок
            self._journal("equity", {"ts": self.snapshot["ts"], "equity": equity,
                                     "cash": self.book.cash, "peak": self.decider.guard.peak,
                                     "drawdown": self.decider.guard.drawdown(equity),
                                     "paused": self.paused})
            self._dirty, self._last_journal_hour = False, hour
        self._save()

    def _wait(self, seconds: int) -> None:
        """Пауза между шагами; прерывается, если пришла команда управления."""
        for _ in range(max(1, seconds)):
            if self.control_path.exists():
                return
            time.sleep(1)

    def run(self, once: bool = False) -> None:
        errors = 0
        while True:
            try:
                self.step()
                if errors >= 5:
                    self._notify("✅ Связь с биржей восстановлена, бот снова работает.")
                errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:  # сеть/биржа: логируем и пробуем на следующем круге
                errors += 1
                log.exception("Ошибка на шаге, повтор через %ss", self.cfg.live.poll_seconds)
                if errors == 5:
                    self._notify(f"⚠️ Бот 5 раз подряд не смог выполнить шаг: "
                                 f"{messages.escape(str(e))[:200]}. Продолжаю попытки.")
            if once or self.stop_requested:
                log.info("Бот остановлен. Позиции сохранены в %s", self.state_path)
                return
            self._wait(self.cfg.live.poll_seconds)


def control_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.stem}_control.json")


def state_path_for(cfg: Config) -> Path:
    return (Path(cfg.live.state_dir)
            / f"{cfg.exchange.mode}_{cfg.exchange.name}_{cfg.portfolio.profile}.json")


def build_runner(cfg: Config, confirm_live: bool = False) -> Runner:
    mode = cfg.exchange.mode
    name = cfg.exchange.name
    # сигналы всегда по ценам основной биржи
    if name == "binance":
        from .binance_public import BinancePublic
        market = BinancePublic()
    else:
        market = CcxtMarketData(name)
    if mode == "paper":
        broker = PaperBroker(cfg.exchange.fee, cfg.exchange.slippage)
    elif mode in ("testnet", "live"):
        if mode == "live" and not (confirm_live and os.getenv("BOT_LIVE_CONFIRM") == "yes"):
            raise SystemExit("Live-режим заблокирован. Нужны флаг --confirm-live и "
                             "переменная окружения BOT_LIVE_CONFIRM=yes.")
        key = os.getenv(f"{name.upper()}_API_KEY")
        secret = os.getenv(f"{name.upper()}_API_SECRET")
        if not key or not secret:
            raise SystemExit(f"Нет ключей: задайте {name.upper()}_API_KEY и {name.upper()}_API_SECRET")
        broker = ExchangeBroker(make_exchange(name, key, secret, sandbox=mode == "testnet"),
                                cfg.exchange.fee)
    else:
        raise SystemExit(f"Неизвестный режим: {mode}")
    state = state_path_for(cfg)
    notifier = None
    if cfg.telegram.enabled:
        from .telegram import CommandListener, Telegram

        token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat:
            raise SystemExit("Telegram включён, но нет TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID. "
                             "См. README, раздел «Telegram».")
        notifier = Telegram(token, chat)
        state.parent.mkdir(parents=True, exist_ok=True)
        CommandListener(Telegram(token, chat), state, control_path(state)).start()
    runner = Runner(cfg, market, broker, state, notifier)
    runner._notify(f"🤖 Бот запущен: режим <b>{mode}</b>, {name}, "
                   f"монет: {len(cfg.portfolio.symbols)}. /help — команды.")
    return runner
