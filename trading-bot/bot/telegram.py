"""Telegram: отправка сигналов/отчётов и приём команд.

Токен бота и chat_id берутся из окружения (или файла .env):
  TELEGRAM_BOT_TOKEN — выдаёт @BotFather
  TELEGRAM_CHAT_ID   — ваш чат; узнать: python -m bot tg-setup

Команды принимаются ТОЛЬКО из чата TELEGRAM_CHAT_ID — остальные игнорируются.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import requests

from . import messages

log = logging.getLogger(__name__)
API = "https://api.telegram.org"


def split_message(text: str, limit: int = 3800) -> list[str]:
    """Telegram принимает до 4096 символов: длинный текст режем по пустым строкам."""
    parts, cur = [], ""
    for block in text.split("\n\n"):
        if cur and len(cur) + len(block) + 2 > limit:
            parts.append(cur)
            cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    return parts + [cur] if cur else parts


class Telegram:
    def __init__(self, token: str, chat_id: str | int | None = None, post: Callable | None = None):
        self.token = token
        self.chat_id = str(chat_id) if chat_id else None
        self._post = post or self._http_post
        self.offset = 0

    def _http_post(self, method: str, payload: dict, timeout: float) -> dict:
        r = requests.post(f"{API}/bot{self.token}/{method}", json=payload, timeout=timeout)
        return r.json()

    def call(self, method: str, payload: dict, timeout: float = 15.0) -> dict:
        return self._post(method, payload, timeout)

    def send(self, text: str, chat_id: str | None = None) -> bool:
        """Отправка сообщения. Никогда не бросает исключений: торговля важнее уведомлений."""
        target = chat_id or self.chat_id
        if not target:
            return False
        ok = True
        for part in split_message(text):
            try:
                res = self.call("sendMessage", {"chat_id": target, "text": part,
                                                "parse_mode": "HTML",
                                                "disable_web_page_preview": True})
                if not res.get("ok"):
                    log.error("Telegram отклонил сообщение: %s", res.get("description"))
                ok = ok and bool(res.get("ok"))
            except Exception as e:  # noqa: BLE001
                log.error("Telegram недоступен: %s", e)
                ok = False
        return ok

    def updates(self, timeout: int = 25) -> list[dict]:
        res = self.call("getUpdates", {"offset": self.offset, "timeout": timeout,
                                       "allowed_updates": ["message"]}, timeout=timeout + 10)
        if not res.get("ok"):
            # например 409 Conflict: этим же токеном уже читает команды другой запущенный бот
            raise RuntimeError(f"getUpdates: {res.get('description') or res}")
        items = res.get("result", [])
        if items:
            self.offset = items[-1]["update_id"] + 1
        return items


class CommandListener(threading.Thread):
    """Фоновый поток: читает команды из Telegram и отвечает.

    Состояние берёт из JSON-файла бота, а команды управления кладёт в control-файл —
    то есть с торговым циклом общается только через файлы, без общей памяти."""

    def __init__(self, tg: Telegram, state_path: Path | None = None,
                 control_path: Path | None = None, bots: list[tuple] | None = None):
        """bots — список (имя, файл состояния, control-файл); для одного счёта
        достаточно state_path и control_path."""
        super().__init__(daemon=True, name="telegram-commands")
        self.tg = tg
        self.bots = [(n, Path(s), Path(c)) for n, s, c in (bots or [("", state_path, control_path)])]
        self.stop_event = threading.Event()

    def _snapshots(self) -> list[tuple[str, dict]]:
        out = []
        for name, state, _ in self.bots:
            try:
                snap = json.loads(state.read_text()).get("snapshot")
            except (OSError, ValueError):
                snap = None
            if snap:
                out.append((name, snap))
        return out

    def _control(self, command: str) -> None:
        for _, _, control in self.bots:
            control.write_text(json.dumps({"command": command}))

    def handle(self, text: str) -> str:
        cmd = text.strip().split()[0].split("@")[0].lower() if text.strip() else ""
        many = len(self.bots) > 1
        if cmd in ("/start", "/help"):
            return messages.HELP
        if cmd in ("/status", "/report"):
            snaps = self._snapshots()
            if not snaps:
                return "Бот ещё не сохранил состояние. Попробуйте через минуту."
            if cmd == "/report":
                return messages.group_report(snaps) if many else messages.report_text(snaps[0][1])
            return "\n\n".join(
                (f"<b>[{messages.escape(n)}]</b>\n" if many else "") + messages.status_text(s)
                for n, s in snaps)
        scope = " на всех счетах" if many else ""
        if cmd == "/pause":
            self._control("pause")
            return f"⏸ Пауза{scope}: новых сделок не будет, стопы и защита работают."
        if cmd == "/resume":
            self._control("resume")
            return f"▶️ Пауза снята{scope}."
        if cmd == "/closeall":
            self._control("close-all")
            return f"🔴 Закрываю все позиции{scope} и встаю на паузу. Вернуть в работу: /resume"
        return "Не знаю такой команды. /help — список."

    def poll_once(self) -> None:
        for upd in self.tg.updates():
            msg = upd.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            if chat != self.tg.chat_id:
                log.warning("Сообщение из чужого чата %s проигнорировано", chat)
                continue
            self.tg.send(self.handle(msg.get("text") or ""))

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:  # noqa: BLE001
                log.error("Ошибка приёма команд Telegram: %s", e)
                time.sleep(10)
