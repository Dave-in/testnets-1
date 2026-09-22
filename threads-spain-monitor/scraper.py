#!/usr/bin/env python3
"""Сбор постов Threads через ваш собственный залогиненный браузер (запускать на своём ПК).

  python scraper.py login      открыть браузер, войти в Threads (один раз, сессия сохранится)
  python scraper.py collect    поиск по запросам из config.json + открытие топ-постов за ответами,
                               затем отчёт reports/latest.html

Нужно: pip install playwright  и  python -m playwright install chromium

Как это работает: браузер открывает поиск Threads как обычный пользователь, а скрипт
читает JSON, который сайт сам загружает (лайки, число ответов, ответы). Логин и cookies
хранятся только у вас в папке browser-profile/ — никуда не отправляются.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import monitor

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "browser-profile"
BASE = "https://www.threads.com"

DEFAULTS = {
    "scrolls_per_query": 5,
    "open_top_posts": 25,
    "max_replies_per_post": 30,
    "delay_seconds": [2.0, 5.0],
    "recent_too": True,
    "headless": False,
}


# ---------------------------------------------------------------- JSON → посты

def _post_text(d: dict) -> str:
    cap = d.get("caption")
    if isinstance(cap, dict) and cap.get("text"):
        return cap["text"]
    info = d.get("text_post_app_info") or {}
    frags = ((info.get("text_fragments") or {}).get("fragments")) or []
    return "".join(f.get("plaintext") or "" for f in frags if isinstance(f, dict))


def _is_post(d: dict) -> bool:
    return (isinstance(d, dict) and "like_count" in d and ("code" in d or "pk" in d)
            and ("caption" in d or "text_post_app_info" in d) and isinstance(d.get("user"), dict))


def normalize_post(d: dict) -> dict | None:
    text = _post_text(d)
    if not text.strip():
        return None
    info = d.get("text_post_app_info") or {}
    user = (d.get("user") or {}).get("username")
    code = d.get("code")
    ts = d.get("taken_at")
    return {
        "id": str(d.get("pk") or d.get("id")).split("_")[0],
        "code": code,
        "text": text,
        "username": user,
        "permalink": f"{BASE}/@{user}/post/{code}" if user and code else None,
        "timestamp": datetime.fromtimestamp(int(ts), timezone.utc).isoformat() if ts else None,
        "like_count": d.get("like_count"),
        "reply_count": info.get("direct_reply_count"),
        "repost_count": info.get("repost_count"),
        "is_reply": bool(info.get("reply_to_author")),
    }


def extract_posts(obj, out: list[dict]) -> None:
    """Рекурсивно ищет объекты постов в любом JSON Threads — не зависит от точной схемы."""
    if isinstance(obj, dict):
        if _is_post(obj):
            p = normalize_post(obj)
            if p:
                out.append(p)
        for v in obj.values():
            extract_posts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            extract_posts(v, out)


def parse_json_text(body: str):
    body = body.strip()
    if body.startswith("for (;;);"):
        body = body[9:]
    try:
        return json.loads(body)
    except ValueError:
        # иногда приходит несколько JSON подряд, по одному в строке
        out = []
        for line in body.splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out


# ---------------------------------------------------------------- браузер

class Collector:
    def __init__(self, page):
        self.page = page
        self.buffer: list[dict] = []
        page.on("response", self._on_response)

    def _on_response(self, resp):
        url = resp.url
        if "graphql" not in url and "/api/" not in url:
            return
        try:
            if "json" not in (resp.headers.get("content-type") or "") and "graphql" not in url:
                return
            extract_posts(parse_json_text(resp.text()), self.buffer)
        except Exception:
            pass  # тело недоступно (редирект, бинарь) — пропускаем

    def embedded(self) -> None:
        """Данные первой загрузки лежат в <script type="application/json">."""
        try:
            blobs = self.page.eval_on_selector_all(
                'script[type="application/json"]', "els => els.map(e => e.textContent)")
        except Exception:
            return
        for b in blobs:
            if b and "like_count" in b:
                extract_posts(parse_json_text(b), self.buffer)

    def take(self) -> list[dict]:
        self.embedded()
        seen, out = set(), []
        for p in self.buffer:
            if p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
        self.buffer = []
        return out


def pause(opts):
    lo, hi = opts["delay_seconds"]
    time.sleep(random.uniform(lo, hi))


def launch(pw, headless: bool):
    PROFILE_DIR.mkdir(exist_ok=True)
    return pw.chromium.launch_persistent_context(
        str(PROFILE_DIR), headless=headless, locale="ru-RU",
        viewport={"width": 1280, "height": 900})


def cmd_login(args, cfg) -> int:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        ctx = launch(pw, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(f"{BASE}/login")
        print("Войдите в Threads в открывшемся окне (через Instagram-аккаунт).")
        input("Когда увидите ленту — нажмите Enter здесь… ")
        ctx.close()
    print(f"Сессия сохранена в {PROFILE_DIR}")
    return 0


def cmd_collect(args, cfg) -> int:
    from playwright.sync_api import sync_playwright
    opts = {**DEFAULTS, **cfg.get("scraper", {})}
    if args.headless:
        opts["headless"] = True
    queries = args.query or cfg["search_queries"]
    if args.limit:
        queries = queries[: args.limit]
    db = monitor.open_db(args.db)
    fetched = new = 0

    with sync_playwright() as pw:
        ctx = launch(pw, headless=opts["headless"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        col = Collector(page)

        page.goto(BASE)
        page.wait_for_timeout(3000)
        if "/login" in page.url:
            print("Вы не вошли в Threads. Сначала: python scraper.py login", file=sys.stderr)
            ctx.close()
            return 2

        modes = [("default", None)] + ([("recent", "recent")] if opts["recent_too"] else [])
        for qi, q in enumerate(queries, 1):
            for mode, flt in modes:
                url = f"{BASE}/search?q={urllib.parse.quote(q)}&serp_type=default"
                if flt:
                    url += f"&filter={flt}"
                page.goto(url)
                page.wait_for_timeout(2500)
                for _ in range(opts["scrolls_per_query"]):
                    page.mouse.wheel(0, 2500)
                    pause(opts)
                posts = [p for p in col.take() if not p["is_reply"]]
                for rank, p in enumerate(posts):
                    fetched += 1
                    new += monitor.store_post(db, p, "browser", q, rank if mode == "default" else None)
                db.commit()
                print(f"[{qi}/{len(queries)}] {q} ({mode}): {len(posts)} постов")

        # открываем самые обсуждаемые посты и собираем ответы
        top = db.execute(
            "SELECT id, permalink FROM posts WHERE permalink IS NOT NULL AND COALESCE(reply_count,0) > 0 "
            "ORDER BY COALESCE(like_count,0) + 2*COALESCE(reply_count,0) DESC LIMIT ?",
            (opts["open_top_posts"],)).fetchall()
        for i, row in enumerate(top, 1):
            page.goto(row["permalink"])
            page.wait_for_timeout(2500)
            for _ in range(2):
                page.mouse.wheel(0, 2500)
                pause(opts)
            items = col.take()
            replies = [p for p in items if p["id"] != row["id"]][: opts["max_replies_per_post"]]
            main = next((p for p in items if p["id"] == row["id"]), None)
            if main:  # свежие цифры лайков/ответов
                monitor.upsert_post(db, main, "browser", None, None)
            for r in replies:
                monitor.upsert_reply(db, row["id"], r)
            db.commit()
            print(f"  ответы [{i}/{len(top)}]: {len(replies)}")
        ctx.close()

    db.execute("INSERT INTO runs (started,source,fetched,new_posts) VALUES (?,?,?,?)",
               (monitor.now_iso(), "browser", fetched, new))
    db.commit()
    print(f"Готово: {fetched} результатов, новых постов {new}")
    if args.no_report:
        return 0
    a = argparse.Namespace(db=args.db, days=args.days, top=20, out=str(monitor.REPORTS_DIR), quiet=True)
    return monitor.cmd_analyze(a, cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=monitor.DEFAULT_CONFIG)
    ap.add_argument("--db", type=Path, default=monitor.DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login")
    p = sub.add_parser("collect")
    p.add_argument("--query", action="append", help="свой запрос (можно несколько)")
    p.add_argument("--limit", type=int, help="взять только первые N запросов из конфига")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--headless", action="store_true", help="без окна браузера")
    p.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    cfg = monitor.load_config(args.config)
    return {"login": cmd_login, "collect": cmd_collect}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
