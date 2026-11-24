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


def load_shadow(path: str | Path) -> dict | None:
    """Раздел [shadow] в bots.toml: теневые боты на стакане Bybit (None — выключены)."""
    with open(path, "rb") as f:
        sh = tomllib.load(f).get("shadow", {})
    if not sh.get("enabled", False):
        return None
    with open(Path(path).parent / sh.get("config", "config.shadow.toml"), "rb") as f:
        return tomllib.load(f)


def load_group(path: str | Path) -> list[BotSpec]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    specs = [BotSpec(**b) for b in raw.get("bot", [])]
    if not specs and not raw.get("shadow", {}).get("enabled", False):
        raise SystemExit(f"В {path} нет ни одного блока [[bot]] и выключены теневые боты")
    if len({s.name for s in specs}) != len(specs):
        raise SystemExit(f"В {path} повторяются имена ботов")
    return specs


class Labeled:
    """Подписывает сообщения именем счёта."""

    def __init__(self, tg, label: str):
        self.tg, self.label = tg, label

    def send(self, text: str, trade: bool = False) -> bool:
        return self.tg.send(f"<b>[{messages.escape(self.label)}]</b>\n{text}")


class Group:
    def __init__(self, runners: dict[str, Runner], tg=None, poll_seconds: int = 60,
                 extra: dict | None = None):
        self.runners = runners
        self.extra = extra or {}   # счета, которые работают сами (теневые боты): только отчёты
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
        accounts = {**self.runners, **self.extra}
        rolled = [n for n, r in accounts.items() if r.day_snapshot]
        if not rolled:
            return
        self.report_wait += 1
        if len(rolled) < len(accounts) and self.report_wait < 15:
            return
        items = [(n, r.day_snapshot or r.snapshot) for n, r in accounts.items()]
        self._send(messages.group_report([(n, s) for n, s in items if s], "Отчёт за день"))
        for r in accounts.values():
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
    # без дневных счетов Telegram включается, если он настроен в .env
    if any(c.telegram.enabled for c in cfgs.values()) or (not cfgs and token and chat):
        if not token or not chat:
            raise SystemExit("Нет TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID в .env. См. README.")
        from .telegram import CommandListener, Telegram
        tg = Telegram(token, chat)

    # Группа с темами: у каждого счёта с сигналами и у каждого теневого бота своя тема.
    # Пока группа не привязана (/here), сообщения идут в личный чат, как раньше.
    channel = None
    if tg is not None:
        from .telegram import ShadowChannel, TopicNotifier
        channel = ShadowChannel(tg, base / "state" / "shadow_telegram.json")

    runners: dict[str, Runner] = {}
    markets: dict[str, object] = {}
    for spec in specs:
        cfg = cfgs[spec.name]
        cfg.telegram.daily_report = False  # отчёт за день — один общий на все счета
        market, broker = build_parts(cfg, confirm_live, markets.get(cfg.exchange.name))
        markets[cfg.exchange.name] = market
        notifier = None
        if tg is not None and spec.signals:
            notifier = TopicNotifier(channel, spec.name, fallback=Labeled(tg, spec.name))
        runners[spec.name] = Runner(cfg, market, broker, state_path_for(cfg), notifier)

    extra = {}
    shadow_cfg = load_shadow(path)
    if shadow_cfg is not None:
        extra = start_shadow(shadow_cfg, base / "state", tg, channel)
    if channel is not None and channel.bound:
        channel.ensure_topics()                # тема для бота, добавленного после /here

    if tg is not None:
        bots = [(n, r.state_path, control_path(r.state_path)) for n, r in runners.items()]
        bots += [(n, b.state_path, b.control_path) for n, b in extra.items()]
        CommandListener(Telegram(token, chat), bots=bots, group=channel,
                        calib_path=(base / "state" / "shadow_calibration.csv") if extra else None
                        ).start()
        lines = [f"• <b>{messages.escape(s.name)}</b>: {cfgs[s.name].exchange.market}, "
                 f"монет {len(cfgs[s.name].portfolio.symbols)}, "
                 f"{'сигналы включены' if s.signals else 'без сигналов, только в отчётах'}"
                 for s in specs]
        if extra:
            lines.append(f"• <b>Теневые боты</b> ({len(extra)}): фьючерсы Bybit, исполнение по "
                         f"реальному стакану — " + ", ".join(messages.escape(n) for n in extra))
        if channel is not None and channel.titles and not channel.bound:
            lines.append("\n📢 Чтобы каждый бот писал в свою ветку группы: включите в группе "
                         "«Темы», сделайте этого бота администратором с правом «Управление "
                         "темами» и отправьте в группе /here")
        tg.send("🤖 Бот запущен, виртуальных счетов: " + str(len(specs) + len(extra)) + "\n"
                + "\n".join(lines) + "\n/help — команды.")
    poll = min((c.live.poll_seconds for c in cfgs.values()), default=60)
    return Group(runners, tg, poll, extra)


def start_shadow(cfg: dict, state_dir: Path, tg=None, channel=None) -> dict:
    """Запустить теневых ботов в отдельном потоке. Возвращает {подпись: бот} для отчётов.
    channel — группа с темами (ShadowChannel): каждый бот пишет сделки в свою тему."""
    import threading

    from .shadow.bybit import run_live
    from .shadow.engine import Engine

    engine = Engine(cfg, state_dir,
                    notifier_for=(lambda k: Labeled(tg, f"Тень · {k}")) if tg is not None else None)
    for bot in engine.bots.values():            # подписи в сообщениях — названиями стратегий
        if bot.notifier is not None:
            bot.notifier.label = f"Тень · {bot.title}"
            if channel is not None:
                from .telegram import TopicNotifier
                bot.notifier = TopicNotifier(channel, bot.title, fallback=bot.notifier)

    def worker():
        warned = False
        while True:                             # при сбое (биржа недоступна и т.п.) — перезапуск
            try:
                run_live(engine)
                return
            except Exception as e:  # noqa: BLE001
                log.exception("Теневые боты: сбой, перезапуск через 60 с")
                if tg is not None and not warned:
                    blocked = "403" in str(e) or "451" in str(e)
                    tg.send("⚠️ Теневые боты не могут получить данные Bybit: "
                            f"{messages.escape(str(e))[:200]}\n"
                            + ("Похоже, Bybit закрыт для страны, где стоит сервер (например, США). "
                               "Нужен VPS в другом регионе, например в Сингапуре. "
                               if blocked else "")
                            + "Остальные счета работают; повторяю попытки раз в минуту.")
                    warned = True
                time.sleep(60)

    threading.Thread(target=worker, daemon=True, name="shadow-bots").start()
    return {f"Тень · {b.title}": b for b in engine.bots.values()}
