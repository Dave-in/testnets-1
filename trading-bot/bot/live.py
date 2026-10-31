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

from .brokers import Book, ExchangeBroker, PaperBroker
from .config import Config
from .core import Decider, rebalance_deltas
from .data import _to_frame, make_exchange
from .strategy import prepare

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
    def __init__(self, cfg: Config, market, broker, state_path: str | Path):
        self.cfg = cfg
        self.market = market
        self.broker = broker
        self.symbols = cfg.portfolio.symbols
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.decider = Decider(cfg, cfg.portfolio.initial_capital)
        self.book = Book(cash=cfg.portfolio.initial_capital, units={s: 0.0 for s in self.symbols})
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
                         units={s: st["book"]["units"].get(s, 0.0) for s in self.symbols})
        self.decider.load_dict(st["decider"])
        self.last_bar = st.get("last_bar")
        self.paused = st.get("paused", False)
        log.info("Состояние восстановлено из %s (последний бар %s)", self.state_path, self.last_bar)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "book": {"cash": self.book.cash, "units": self.book.units},
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
        if side == "buy":
            self.book.cash -= fill.cost
            self.book.units[symbol] += fill.units
        else:
            self.book.cash += fill.cost
            self.book.units[symbol] = max(0.0, self.book.units[symbol] - fill.units)
            if self.book.units[symbol] * price < 1.0:  # пыль после округления
                self.book.units[symbol] = 0.0
        log.info("%s %s %.6f @ %.4f (комиссия %.4f) — %s",
                 side.upper(), symbol, fill.units, fill.price, fill.fee, reason)
        self._journal("trades", {"ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
                                 "side": side, "units": fill.units, "price": fill.price,
                                 "fee": fill.fee, "reason": reason})

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
            for s in self.symbols:
                self._sell_all(s, prices[s], "drawdown_guard")
            self.decider.reset_signals()

    def on_new_bar(self, prices: dict[str, float]) -> str | None:
        """Если закрылся новый бар — пересчет сигналов и ребалансировка."""
        tf = self.cfg.exchange.timeframe
        frames = {s: prepare(self.market.closed_ohlcv(s, tf, self.history_limit),
                             self.cfg.strategy, tf) for s in self.symbols}
        bar_ts = max(str(df.index[-1]) for df in frames.values() if len(df))
        if bar_ts == self.last_bar:
            return None

        rows = {s: df.iloc[-1] for s, df in frames.items() if len(df)}
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
        return bar_ts

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
        prices = self.market.prices(self.symbols)
        self.apply_control(prices)
        self.check_risk(prices)
        if not self.paused:
            self.on_new_bar(prices)
        equity = self.book.equity(prices)
        self._journal("equity", {"ts": datetime.now(timezone.utc).isoformat(), "equity": equity,
                                 "cash": self.book.cash, "peak": self.decider.guard.peak,
                                 "drawdown": self.decider.guard.drawdown(equity),
                                 "paused": self.paused})
        self._save()

    def run(self, once: bool = False) -> None:
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:  # сеть/биржа: логируем и пробуем на следующем круге
                log.exception("Ошибка на шаге, повтор через %ss", self.cfg.live.poll_seconds)
            if once or self.stop_requested:
                log.info("Бот остановлен. Позиции сохранены в %s", self.state_path)
                return
            time.sleep(self.cfg.live.poll_seconds)


def control_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.stem}_control.json")


def state_path_for(cfg: Config) -> Path:
    return (Path(cfg.live.state_dir)
            / f"{cfg.exchange.mode}_{cfg.exchange.name}_{cfg.portfolio.profile}.json")


def build_runner(cfg: Config, confirm_live: bool = False) -> Runner:
    mode = cfg.exchange.mode
    name = cfg.exchange.name
    market = CcxtMarketData(name)  # сигналы всегда по ценам основной биржи
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
    return Runner(cfg, market, broker, state_path_for(cfg))
