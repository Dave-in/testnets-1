"""Несколько виртуальных счетов в одном процессе: общий Telegram, общий цикл.

Нужно, чтобы сравнивать стратегии на живом рынке бок о бок. Два отдельных процесса с одним
токеном Telegram мешали бы друг другу читать команды, поэтому всё собрано в одном.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import messages
from .config import load_config, tomllib
from .live import Runner, build_parts, control_path, state_path_for

log = logging.getLogger(__name__)


@dataclass
class BotSpec:
    name: str
    config: str
    signals: bool = True


def load_group(path: str | Path) -> list[BotSpec]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    specs = [BotSpec(**b) for b in raw.get("bot", [])]
    if not specs:
        raise SystemExit(f"В {path} нет ни одного блока [[bot]]")
    if len({s.name for s in specs}) != len(specs):
        raise SystemExit(f"В {path} повторяются имена ботов")
    return specs


class Labeled:
    """Подписывает сообщения именем счёта."""

    def __init__(self, tg, label: str):
        self.tg, self.label = tg, label

    def send(self, text: str) -> bool:
        return self.tg.send(f"<b>[{messages.escape(self.label)}]</b>\n{text}")


class Group:
    def __init__(self, runners: dict[str, Runner], tg=None, poll_seconds: int = 60):
        self.runners = runners
        self.tg = tg
        self.poll_seconds = poll_seconds
        self.errors = {name: 0 for name in runners}
        self.report_wait = 0

    def step_all(self) -> None:
        for name, r in self.runners.items():
            try:
                r.step()
                if self.errors[name] >= 5:
                    self._send(f"✅ [{name}] связь с биржей восстановлена.")
                self.errors[name] = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — один счёт не должен ронять остальные
                self.errors[name] += 1
                log.exception("[%s] ошибка на шаге", name)
                if self.errors[name] == 5:
                    self._send(f"⚠️ [{name}] 5 раз подряд не удалось выполнить шаг: "
                               f"{messages.escape(str(e))[:200]}. Продолжаю попытки.")

    def _send(self, text: str) -> None:
        if self.tg is not None:
            self.tg.send(text)

    def maybe_report(self) -> None:
        """Общий отчёт за день — когда все счета перешли на новые сутки (или ждём слишком долго)."""
        rolled = [n for n, r in self.runners.items() if r.day_snapshot]
        if not rolled:
            return
        self.report_wait += 1
        if len(rolled) < len(self.runners) and self.report_wait < 15:
            return
        items = [(n, r.day_snapshot or r.snapshot) for n, r in self.runners.items()]
        self._send(messages.group_report([(n, s) for n, s in items if s], "Отчёт за день"))
        for r in self.runners.values():
            r.day_snapshot = None
        self.report_wait = 0

    def run(self, once: bool = False) -> None:
        while True:
            self.step_all()
            self.maybe_report()
            if once or any(r.stop_requested for r in self.runners.values()):
                log.info("Остановлено. Состояние счетов сохранено.")
                return
            for _ in range(max(1, self.poll_seconds)):  # прерываемся, если пришла команда
                if any(r.control_path.exists() for r in self.runners.values()):
                    break
                time.sleep(1)


def build_group(path: str | Path, confirm_live: bool = False) -> Group:
    specs = load_group(path)
    base = Path(path).parent
    cfgs = {s.name: load_config(base / s.config) for s in specs}
    if len({str(state_path_for(c)) for c in cfgs.values()}) != len(cfgs):
        raise SystemExit("У двух ботов совпадает файл состояния: задайте разные profile в конфигах")

    tg = None
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if any(c.telegram.enabled for c in cfgs.values()):
        if not token or not chat:
            raise SystemExit("Нет TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID в .env. См. README.")
        from .telegram import CommandListener, Telegram
        tg = Telegram(token, chat)

    runners: dict[str, Runner] = {}
    markets: dict[str, object] = {}
    for spec in specs:
        cfg = cfgs[spec.name]
        cfg.telegram.daily_report = False  # отчёт за день — один общий на все счета
        market, broker = build_parts(cfg, confirm_live, markets.get(cfg.exchange.name))
        markets[cfg.exchange.name] = market
        notifier = Labeled(tg, spec.name) if tg is not None and spec.signals else None
        runners[spec.name] = Runner(cfg, market, broker, state_path_for(cfg), notifier)

    if tg is not None:
        bots = [(n, r.state_path, control_path(r.state_path)) for n, r in runners.items()]
        CommandListener(Telegram(token, chat), bots=bots).start()
        lines = [f"• <b>{messages.escape(s.name)}</b>: {cfgs[s.name].exchange.market}, "
                 f"монет {len(cfgs[s.name].portfolio.symbols)}, "
                 f"{'сигналы включены' if s.signals else 'без сигналов, только в отчётах'}"
                 for s in specs]
        tg.send("🤖 Бот запущен, виртуальных счетов: " + str(len(specs)) + "\n"
                + "\n".join(lines) + "\n/help — команды.")
    poll = min(c.live.poll_seconds for c in cfgs.values())
    return Group(runners, tg, poll)
