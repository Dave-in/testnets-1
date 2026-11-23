"""Telegram: отправка сигналов/отчётов и приём команд.

Токен бота и chat_id берутся из окружения (или файла .env):
  TELEGRAM_BOT_TOKEN — выдаёт @BotFather
  TELEGRAM_CHAT_ID   — ваш чат; узнать: python -m bot tg-setup

Команды принимаются ТОЛЬКО из чата TELEGRAM_CHAT_ID — остальные игнорируются.
"""
from __future__ import annotations

import json
import logging
import queue
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

    def send(self, text: str, chat_id: str | None = None, thread_id: int | None = None,
             wait_on_limit: bool = False) -> bool:
        """Отправка сообщения. Никогда не бросает исключений: торговля важнее уведомлений.
        thread_id — тема в группе с темами. wait_on_limit — при ответе «слишком часто» (429)
        подождать, сколько просит Telegram, и повторить (только из фонового потока)."""
        target = chat_id or self.chat_id
        if not target:
            return False
        ok = True
        for part in split_message(text):
            payload = {"chat_id": target, "text": part, "parse_mode": "HTML",
                       "disable_web_page_preview": True}
            if thread_id:
                payload["message_thread_id"] = thread_id
            for _ in range(5):
                try:
                    res = self.call("sendMessage", payload)
                except Exception as e:  # noqa: BLE001
                    log.error("Telegram недоступен: %s", e)
                    res = {"ok": False}
                    break
                retry = (res.get("parameters") or {}).get("retry_after")
                if res.get("error_code") == 429 and wait_on_limit and retry:
                    time.sleep(float(retry) + 1)
                    continue
                break
            if not res.get("ok") and res.get("description"):
                log.error("Telegram отклонил сообщение: %s", res.get("description"))
            ok = ok and bool(res.get("ok"))
        return ok

    def create_topic(self, chat_id: str, name: str) -> tuple[int | None, str]:
        """Создать тему в группе. Возвращает (номер темы, текст ошибки)."""
        try:
            res = self.call("createForumTopic", {"chat_id": chat_id, "name": name[:128]})
        except Exception as e:  # noqa: BLE001
            return None, str(e)
        if res.get("ok"):
            return int(res["result"]["message_thread_id"]), ""
        return None, str(res.get("description") or res)

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


class ShadowChannel:
    """Группа Telegram с темами: у каждого теневого бота своя тема.

    Группу привязывает владелец командой /here в самой группе (бот должен быть админом
    с правом «Управление темами»). Привязка и номера тем хранятся в файле и переживают
    перезапуск. Сообщения уходят из очереди в фоновом потоке не чаще ~19 в минуту —
    лимит Telegram для одной группы, — поэтому торговый цикл никогда не ждёт Telegram."""

    def __init__(self, tg: Telegram, path: str | Path, min_interval: float = 3.2,
                 start: bool = True):
        self.tg = tg
        self.path = Path(path)
        self.min_interval = min_interval
        self.titles: list[str] = []
        self.lock = threading.Lock()
        self.chat_id: str | None = None
        self.topics: dict[str, int] = {}
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
            self.chat_id, self.topics = st.get("chat_id"), dict(st.get("topics", {}))
        except (OSError, ValueError):
            pass
        self.queue: queue.Queue = queue.Queue(maxsize=2000)
        if start:
            threading.Thread(target=self._worker, daemon=True, name="telegram-shadow").start()

    @property
    def bound(self) -> bool:
        return bool(self.chat_id)

    def register(self, title: str) -> None:
        if title not in self.titles:
            self.titles.append(title)

    def bind(self, chat_id: str) -> str:
        """Привязать группу и создать темы (уже созданные в этой группе не дублируются)."""
        chat_id = str(chat_id)
        topics = dict(self.topics) if chat_id == self.chat_id else {}
        for title in self.titles:
            if title in topics:
                continue
            tid, err = self.tg.create_topic(chat_id, title)
            if tid is None:
                return ("⚠️ Не получилось создать тему «" + messages.escape(title) + "»: "
                        + messages.escape(err)[:200] + "\nПроверьте: в группе включены темы, "
                        "бот — администратор с правом «Управление темами». Потом снова /here.")
            topics[title] = tid
        with self.lock:
            self.chat_id, self.topics = chat_id, topics
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"chat_id": chat_id, "topics": topics},
                                            ensure_ascii=False), encoding="utf-8")
        return ("✅ Группа привязана. Темы ботов: " + ", ".join(messages.escape(t) for t in topics)
                + ".\nСюда будут приходить все сделки. /status и /report работают в этой группе.")

    def post(self, title: str, text: str) -> bool:
        """Поставить сообщение в очередь темы бота. False — группа не привязана."""
        with self.lock:
            chat, thread = self.chat_id, self.topics.get(title)
        if not chat:
            return False
        try:
            self.queue.put_nowait((chat, thread, text))
        except queue.Full:
            log.error("Очередь Telegram переполнена, сообщение пропущено")
        return True

    def send_one(self, chat: str, thread: int | None, text: str) -> None:
        if not self.tg.send(text, chat, thread, wait_on_limit=True) and thread:
            self.tg.send(text, chat, None, wait_on_limit=True)   # тему удалили — в общую

    def _worker(self) -> None:
        last = 0.0
        while True:
            chat, thread, text = self.queue.get()
            wait = self.min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            try:
                self.send_one(chat, thread, text)
            except Exception:  # noqa: BLE001
                log.exception("Ошибка отправки в группу Telegram")
            last = time.time()


class TopicNotifier:
    """Уведомления одного теневого бота: в его тему в группе. Пока группа не привязана,
    предупреждения идут в личный чат (fallback), а сделки не отправляются."""

    def __init__(self, channel: ShadowChannel, title: str, fallback=None):
        self.channel, self.title, self.fallback = channel, title, fallback
        channel.register(title)

    def send(self, text: str, trade: bool = False) -> bool:
        if self.channel.post(self.title, text):
            return True
        if not trade and self.fallback is not None:
            return self.fallback.send(text)
        return False


class CommandListener(threading.Thread):
    """Фоновый поток: читает команды из Telegram и отвечает.

    Состояние берёт из JSON-файла бота, а команды управления кладёт в control-файл —
    то есть с торговым циклом общается только через файлы, без общей памяти."""

    GROUP_COMMANDS = ("/status", "/report", "/help", "/start")   # в группе — только просмотр

    def __init__(self, tg: Telegram, state_path: Path | None = None,
                 control_path: Path | None = None, bots: list[tuple] | None = None,
                 group: ShadowChannel | None = None):
        """bots — список (имя, файл состояния, control-файл); для одного счёта
        достаточно state_path и control_path. group — группа с темами теневых ботов."""
        super().__init__(daemon=True, name="telegram-commands")
        self.tg = tg
        self.group = group
        self.bots = [(n, Path(s), Path(c)) for n, s, c in (bots or [("", state_path, control_path)])]
        self.stop_event = threading.Event()

    def _snapshots(self) -> list[tuple[str, dict]]:
        out = []
        for name, state, _ in self.bots:
            try:
                snap = json.loads(state.read_text(encoding="utf-8")).get("snapshot")
            except (OSError, ValueError):
                snap = None
            if snap:
                out.append((name, snap))
        return out

    def _control(self, command: str, value: float | None = None) -> None:
        payload = {"command": command} if value is None else {"command": command, "value": value}
        for _, _, control in self.bots:
            control.write_text(json.dumps(payload))

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
        if cmd == "/cashout":
            self._control("cashout")
            return ("💰 Вывожу: закрываю позиции счёта в режиме попыток, фиксирую итог "
                    "и начинаю новую попытку. Итог придёт отдельным сообщением.")
        if cmd == "/target":
            parts = text.split()
            try:
                value = float(parts[1].lower().lstrip("x").replace(",", "."))
            except (IndexError, ValueError):
                value = 0.0
            if value <= 1.0:
                return "Укажите цель числом больше 1, например: /target 3"
            self._control("target", value)
            return f"🎯 Цель попытки будет x{value:g}."
        if cmd == "/closeall":
            self._control("close-all")
            return f"🔴 Закрываю все позиции{scope} и встаю на паузу. Вернуть в работу: /resume"
        return "Не знаю такой команды. /help — список."

    def poll_once(self) -> None:
        for upd in self.tg.updates():
            msg = upd.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            text = msg.get("text") or ""
            if chat == self.tg.chat_id:
                self.tg.send(self.handle(text))
                continue
            if self.group is not None and text.startswith("/"):
                cmd = text.strip().split()[0].split("@")[0].lower()
                # в личном чате id чата совпадает с id пользователя — так узнаём владельца
                owner = str((msg.get("from") or {}).get("id", "")) == self.tg.chat_id
                thread = msg.get("message_thread_id") if msg.get("is_topic_message") else None
                if cmd == "/here" and owner:
                    self.tg.send(self.group.bind(chat), chat, thread)
                    continue
                if chat == self.group.chat_id and cmd in self.GROUP_COMMANDS:
                    self.tg.send(self.handle(text), chat, thread)
                    continue
            log.warning("Сообщение из чужого чата %s проигнорировано", chat)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:  # noqa: BLE001
                log.error("Ошибка приёма команд Telegram: %s", e)
                time.sleep(10)
