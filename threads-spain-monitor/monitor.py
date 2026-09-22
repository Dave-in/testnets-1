#!/usr/bin/env python3
"""Мониторинг Threads: что обсуждают про Испанию / Аликанте / Коста Бланку.

Команды:
  fetch            собрать посты через официальный Threads API (нужен THREADS_ACCESS_TOKEN)
  import FILE      загрузить посты из JSON/JSONL/CSV (например, собранные Claude через веб-поиск)
  analyze          посчитать тренды и написать отчёт в reports/
  run              fetch + analyze
  demo             прогнать анализ на демо-данных (без сети и токена)

Только стандартная библиотека Python 3.9+.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from html_report import render_html

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_DB = ROOT / "data" / "threads.db"
REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------------- config / text

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    """Термин — начало слова ("аликант" ловит "Аликанте", "аликантский").
    "=слово" — только целое слово. "?" — вопросительный знак."""
    out = []
    for term in terms:
        if term.startswith("_"):
            continue
        if term == "?":
            out.append((term, re.compile(r"\?")))
            continue
        exact = term.startswith("=")
        word = normalize(term.lstrip("="))
        body = re.escape(word)
        pattern = rf"(?<!\w){body}(?!\w)" if exact else rf"(?<!\w){body}"
        out.append((word, re.compile(pattern)))
    return out


class Classifier:
    def __init__(self, cfg: dict):
        self.locations = {k: compile_terms(v) for k, v in cfg["locations"].items()}
        self.intents = {k: compile_terms(v["terms"]) for k, v in cfg["intents"].items()}
        self.intent_weights = {k: v.get("weight", 1.0) for k, v in cfg["intents"].items()}
        self.topics = {k: compile_terms(v) for k, v in cfg["topics"].items()}

    @staticmethod
    def _hits(text: str, groups: dict) -> dict[str, list[str]]:
        found = {}
        for name, patterns in groups.items():
            words = [w for w, p in patterns if p.search(text)]
            if words:
                found[name] = words
        return found

    def classify(self, raw_text: str) -> dict:
        text = normalize(raw_text)
        return {
            "locations": self._hits(text, self.locations),
            "intents": self._hits(text, self.intents),
            "topics": self._hits(text, self.topics),
        }


# ---------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    text TEXT,
    username TEXT,
    permalink TEXT,
    timestamp TEXT,
    media_type TEXT,
    source TEXT,
    best_rank INTEGER,
    like_count INTEGER,
    reply_count INTEGER,
    repost_count INTEGER,
    first_seen TEXT,
    last_seen TEXT,
    seen_runs INTEGER DEFAULT 1,
    queries TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS replies (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    text TEXT,
    username TEXT,
    permalink TEXT,
    timestamp TEXT,
    like_count INTEGER,
    reply_count INTEGER,
    last_seen TEXT
);
CREATE INDEX IF NOT EXISTS replies_parent ON replies(parent_id);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT,
    source TEXT,
    fetched INTEGER,
    new_posts INTEGER
);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int_or_none(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def upsert_post(db: sqlite3.Connection, post: dict, source: str, query: str | None, rank: int | None) -> bool:
    """Возвращает True, если пост новый."""
    pid = str(post.get("id") or post.get("permalink") or "").strip()
    text = post.get("text") or ""
    if not pid or not text.strip():
        return False
    ts = now_iso()
    row = db.execute("SELECT best_rank, queries FROM posts WHERE id=?", (pid,)).fetchone()
    likes = _int_or_none(post.get("like_count") or post.get("likes"))
    replies = _int_or_none(post.get("reply_count") or (post.get("replies") if not isinstance(post.get("replies"), list) else None))
    reposts = _int_or_none(post.get("repost_count") or post.get("reposts"))
    if row is None:
        db.execute(
            "INSERT INTO posts (id,text,username,permalink,timestamp,media_type,source,best_rank,"
            "like_count,reply_count,repost_count,first_seen,last_seen,seen_runs,queries) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
            (pid, text, post.get("username"), post.get("permalink"), post.get("timestamp"),
             post.get("media_type"), source, rank, likes, replies, reposts, ts, ts,
             json.dumps([query] if query else [], ensure_ascii=False)),
        )
        return True
    queries = set(json.loads(row["queries"] or "[]"))
    if query:
        queries.add(query)
    best = row["best_rank"]
    if rank is not None and (best is None or rank < best):
        best = rank
    db.execute(
        "UPDATE posts SET last_seen=?, seen_runs=seen_runs+1, best_rank=?, queries=?,"
        " like_count=COALESCE(?, like_count), reply_count=COALESCE(?, reply_count),"
        " repost_count=COALESCE(?, repost_count) WHERE id=?",
        (ts, best, json.dumps(sorted(queries), ensure_ascii=False), likes, replies, reposts, pid),
    )
    return False


def upsert_reply(db: sqlite3.Connection, parent_id: str, r: dict) -> None:
    rid = str(r.get("id") or r.get("permalink") or "").strip()
    if not rid or not (r.get("text") or "").strip():
        return
    db.execute(
        "INSERT INTO replies (id,parent_id,text,username,permalink,timestamp,like_count,reply_count,last_seen)"
        " VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
        " text=excluded.text, like_count=COALESCE(excluded.like_count, like_count),"
        " reply_count=COALESCE(excluded.reply_count, reply_count), last_seen=excluded.last_seen",
        (rid, parent_id, r["text"], r.get("username"), r.get("permalink"), r.get("timestamp"),
         _int_or_none(r.get("like_count") or r.get("likes")),
         _int_or_none(r.get("reply_count") or r.get("replies_count")), now_iso()),
    )


def store_post(db: sqlite3.Connection, post: dict, source: str, query: str | None, rank: int | None) -> bool:
    """Пост + вложенный список ответов (поле "replies", если это список)."""
    is_new = upsert_post(db, post, source, query, rank)
    replies = post.get("replies")
    if isinstance(replies, list):
        pid = str(post.get("id") or post.get("permalink") or "").strip()
        for r in replies:
            upsert_reply(db, pid, r)
    return is_new


def load_replies(db: sqlite3.Connection) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in db.execute("SELECT * FROM replies ORDER BY COALESCE(like_count,0) DESC, COALESCE(reply_count,0) DESC"):
        out[r["parent_id"]].append({k: r[k] for k in ("id", "text", "username", "permalink", "timestamp", "like_count", "reply_count")})
    return out


# ---------------------------------------------------------------- Threads API

def threads_search(cfg: dict, token: str, query: str, search_type: str, since: int | None) -> list[dict]:
    api = cfg["api"]
    params = {
        "q": query,
        "search_type": search_type,
        "fields": api["fields"],
        "limit": api.get("limit_per_query", 25),
        "access_token": token,
    }
    if since:
        params["since"] = since
    url = f"{api['base_url']}/keyword_search?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp).get("data", [])
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Threads API {e.code} для '{query}': {body}") from None
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Сеть недоступна: {e.reason}") from None
    return []


def cmd_fetch(args, cfg) -> int:
    token = os.environ.get("THREADS_ACCESS_TOKEN")
    if not token:
        print("Нет THREADS_ACCESS_TOKEN. Как получить — см. README.md. "
              "Без токена используйте `import` (например, данные из веб-поиска Claude).", file=sys.stderr)
        return 2
    db = open_db(args.db)
    days = cfg["api"].get("lookback_days", 7)
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    queries = args.query or cfg["search_queries"]
    fetched = new = 0
    errors = []
    for q in queries:
        for st in cfg["api"]["search_types"]:
            try:
                posts = threads_search(cfg, token, q, st, since)
            except RuntimeError as e:
                errors.append(str(e))
                continue
            for rank, p in enumerate(posts):
                fetched += 1
                # ранг важен только для TOP: это сигнал популярности от самого Threads
                new += upsert_post(db, p, "threads_api", q, rank if st == "TOP" else None)
            time.sleep(0.3)
    db.execute("INSERT INTO runs (started,source,fetched,new_posts) VALUES (?,?,?,?)",
               (now_iso(), "threads_api", fetched, new))
    db.commit()
    print(f"Получено {fetched} результатов, новых постов: {new}")
    for e in errors[:5]:
        print("  ! " + e, file=sys.stderr)
    if errors and not fetched:
        return 1
    return 0


# ---------------------------------------------------------------- import

def read_posts_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(raw.splitlines()))
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            return obj.get("data") or obj.get("posts") or [obj]
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def cmd_import(args, cfg) -> int:
    db = open_db(args.db)
    posts = read_posts_file(Path(args.file))
    new = 0
    for rank, p in enumerate(posts):
        r = _int_or_none(p.get("rank"))
        new += store_post(db, p, args.source, p.get("query"), r)
    db.execute("INSERT INTO runs (started,source,fetched,new_posts) VALUES (?,?,?,?)",
               (now_iso(), args.source, len(posts), new))
    db.commit()
    print(f"Импортировано {len(posts)} записей, новых: {new}")
    return 0


# ---------------------------------------------------------------- analysis

def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    v = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", v)  # Threads отдаёт +0000
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        try:
            dt = datetime.strptime(v[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def score_post(row: sqlite3.Row, cls: dict, weights: dict, now: datetime, days: int) -> float:
    s = 0.0
    for intent in cls["intents"]:
        s += weights.get(intent, 1.0)
    s += 0.5 * len(cls["topics"])
    s += 0.7 if cls["locations"] and set(cls["locations"]) != {"Испания"} else 0.0
    eng = (row["like_count"] or 0) + 2 * (row["reply_count"] or 0) + 1.5 * (row["repost_count"] or 0)
    s += math.log1p(eng)
    if row["best_rank"] is not None:
        s += 3.0 / (1 + row["best_rank"])  # высоко в TOP-выдаче Threads
    s += 0.5 * (min(row["seen_runs"], 5) - 1)  # держится в выдаче несколько запусков подряд
    s += 0.3 * len(json.loads(row["queries"] or "[]"))
    ts = parse_ts(row["timestamp"]) or parse_ts(row["first_seen"])
    if ts:
        age_days = max((now - ts).total_seconds() / 86400, 0)
        s *= math.exp(-age_days / max(days, 1))
    return round(s, 3)


def analyze(db: sqlite3.Connection, cfg: dict, days: int, top_n: int, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    classifier = Classifier(cfg)
    weights = classifier.intent_weights
    cur_start = now - timedelta(days=days)
    prev_start = now - timedelta(days=2 * days)

    posts = []
    topic_cur, topic_prev = Counter(), Counter()
    intent_cur, loc_cur = Counter(), Counter()
    combo = Counter()
    topic_intents = defaultdict(Counter)
    topic_locs = defaultdict(Counter)
    topic_words = defaultdict(Counter)

    for row in db.execute("SELECT * FROM posts"):
        cls = classifier.classify(row["text"])
        if not cls["locations"]:
            continue  # нужен явный контекст Испании/побережья
        ts = parse_ts(row["timestamp"]) or parse_ts(row["first_seen"])
        if ts is None or ts < prev_start:
            continue
        in_current = ts >= cur_start
        topics = list(cls["topics"]) or ["Прочее"]
        if not in_current:
            for t in topics:
                topic_prev[t] += 1
            continue
        for t in topics:
            topic_cur[t] += 1
            for i in cls["intents"]:
                topic_intents[t][i] += 1
                combo[(t, i)] += 1
            for loc in cls["locations"]:
                topic_locs[t][loc] += 1
            for words in cls["topics"].get(t, []):
                topic_words[t][words] += 1
        for i in cls["intents"]:
            intent_cur[i] += 1
        for loc in cls["locations"]:
            loc_cur[loc] += 1
        posts.append({
            "id": row["id"],
            "text": row["text"],
            "username": row["username"],
            "permalink": row["permalink"],
            "timestamp": ts.isoformat(),
            "likes": row["like_count"], "replies": row["reply_count"], "reposts": row["repost_count"],
            "queries": json.loads(row["queries"] or "[]"),
            "top_rank": row["best_rank"],
            "score": score_post(row, cls, weights, now, days),
            **cls,
        })

    posts.sort(key=lambda p: p["score"], reverse=True)
    score_by_topic = defaultdict(float)
    for p in posts:
        for t in (list(p["topics"]) or ["Прочее"]):
            score_by_topic[t] += p["score"]

    trends = []
    for t, n in topic_cur.items():
        prev = topic_prev.get(t, 0)
        growth = (n - prev) / prev if prev else None
        trends.append({
            "topic": t,
            "posts": n,
            "prev_posts": prev,
            "growth": None if growth is None else round(growth, 2),
            "status": "новое" if prev == 0 else ("растёт" if n > prev * 1.2 else ("падает" if n < prev * 0.8 else "стабильно")),
            "heat": round(score_by_topic[t], 2),
            "intents": dict(topic_intents[t].most_common()),
            "locations": dict(topic_locs[t].most_common(3)),
            "keywords": [w for w, _ in topic_words[t].most_common(6)],
            "examples": [
                {"text": p["text"][:220], "permalink": p["permalink"], "score": p["score"]}
                for p in posts if t in p["topics"] or (t == "Прочее" and not p["topics"])
            ][:3],
        })
    trends.sort(key=lambda x: (x["heat"], x["posts"]), reverse=True)

    return {
        "generated_at": now.replace(microsecond=0).isoformat(),
        "window_days": days,
        "total_posts": len(posts),
        "intents": dict(intent_cur.most_common()),
        "locations": dict(loc_cur.most_common()),
        "trends": trends,
        "top_posts": posts[:top_n],
        "all_posts": posts,
        "polls": build_polls(trends, cfg, limit=6),
        "hot_combos": [{"topic": t, "intent": i, "posts": n} for (t, i), n in combo.most_common(10)],
    }


def build_polls(trends: list[dict], cfg: dict, limit: int) -> list[dict]:
    templates = cfg["poll_templates"]
    forms = cfg.get("location_forms", {})
    polls = []
    for tr in trends:
        if len(polls) >= limit:
            break
        tpl = templates.get(tr["topic"]) or templates["_default"]
        # подставляем самую упоминаемую конкретную локацию (не общую «Испанию», если есть выбор)
        locs = [l for l in tr["locations"] if l != "Испания"] or list(tr["locations"]) or ["Испания"]
        form = forms.get(locs[0], {"in": f" в {locs[0]}", "of": f" {locs[0]}"})
        q = tpl["q"].format(in_loc=form["in"], of_loc=form["of"])
        mood = max(tr["intents"], key=tr["intents"].get) if tr["intents"] else None
        polls.append({
            "topic": tr["topic"],
            "question": q,
            "options": tpl["options"],
            "why": f"постов: {tr['posts']}, статус: {tr['status']}"
                   + (f", преобладает «{mood}»" if mood else ""),
        })
    return polls


# ---------------------------------------------------------------- report

def fmt_growth(tr: dict) -> str:
    if tr["growth"] is None:
        return "🆕"
    pct = int(tr["growth"] * 100)
    return ("📈 +" if pct > 0 else "📉 " if pct < 0 else "➖ ") + f"{pct}%"


def render_markdown(res: dict) -> str:
    L = []
    L.append(f"# Тренды Threads: Испания / Аликанте / Коста Бланка")
    L.append(f"_Сформировано {res['generated_at']} · окно {res['window_days']} дн. · постов в анализе: {res['total_posts']}_\n")
    if not res["total_posts"]:
        L.append("Нет постов за период. Запустите `fetch` или `import`.")
        return "\n".join(L)

    L.append("## Настроение аудитории")
    L.append(" · ".join(f"**{k}** {v}" for k, v in res["intents"].items()) or "—")
    L.append("\n**Локации:** " + " · ".join(f"{k} {v}" for k, v in res["locations"].items()))

    L.append("\n## Темы в тренде")
    L.append("| # | Тема | Постов | Динамика | Жар | Эмоции/интент | Ключевые слова |")
    L.append("|---|------|-------:|----------|----:|---------------|----------------|")
    for i, t in enumerate(res["trends"][:12], 1):
        intents = ", ".join(f"{k} {v}" for k, v in list(t["intents"].items())[:3])
        L.append(f"| {i} | {t['topic']} | {t['posts']} | {fmt_growth(t)} | {t['heat']} | {intents} | {', '.join(t['keywords'])} |")

    L.append("\n## Горячие связки «тема × эмоция»")
    for c in res["hot_combos"][:8]:
        L.append(f"- **{c['topic']}** × _{c['intent']}_ — {c['posts']}")

    L.append("\n## Самые заметные посты")
    for p in res["top_posts"][:15]:
        text = p["text"].replace("\n", " ")[:260]
        tags = ", ".join(list(p["intents"]) + list(p["topics"]))
        link = f" — [открыть]({p['permalink']})" if p.get("permalink") else ""
        who = f"@{p['username']} · " if p.get("username") else ""
        L.append(f"- `{p['score']}` {who}{p['timestamp'][:10]} · _{tags}_{link}\n  > {text}")

    L.append("\n## Идеи опросов")
    for i, poll in enumerate(res["polls"], 1):
        L.append(f"{i}. **{poll['question']}** _(тема: {poll['topic']}; {poll['why']})_")
        for o in poll["options"]:
            L.append(f"   - {o}")
    return "\n".join(L) + "\n"


def cmd_analyze(args, cfg) -> int:
    db = open_db(args.db)
    res = analyze(db, cfg, args.days, args.top)
    md = render_markdown(res)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    (out_dir / f"report_{stamp}.md").write_text(md, encoding="utf-8")
    (out_dir / "latest.md").write_text(md, encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    html = render_html(res, load_replies(db))
    (out_dir / f"report_{stamp}.html").write_text(html, encoding="utf-8")
    (out_dir / "latest.html").write_text(html, encoding="utf-8")
    if not args.quiet:
        print(md)
    print(f"Отчёт: {out_dir / 'latest.html'}  (также latest.md и latest.json)", file=sys.stderr)
    return 0


def cmd_run(args, cfg) -> int:
    rc = cmd_fetch(args, cfg)
    if rc not in (0,):
        return rc
    return cmd_analyze(args, cfg)


def cmd_demo(args, cfg) -> int:
    demo_db = ROOT / "data" / "demo.db"
    if demo_db.exists():
        demo_db.unlink()
    db = open_db(demo_db)
    sample = read_posts_file(ROOT / "sample_posts.json")
    now = datetime.now(timezone.utc)
    for p in sample:
        # демо-данные хранят возраст поста в днях, чтобы демо всегда было «свежим»
        p = dict(p)
        p["timestamp"] = (now - timedelta(days=float(p.pop("age_days", 0)))).isoformat()
        now_ts = now
        for r in p.get("replies") or []:
            r["timestamp"] = (now_ts - timedelta(days=float(r.pop("age_days", 0)))).isoformat()
        store_post(db, p, "demo", p.get("query"), _int_or_none(p.get("rank")))
    db.commit()
    args.db = demo_db
    return cmd_analyze(args, cfg)


# ---------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def analyze_opts(p):
        p.add_argument("--days", type=int, default=7, help="окно анализа в днях (сравнивается с предыдущим таким же)")
        p.add_argument("--top", type=int, default=20, help="сколько топ-постов сохранить")
        p.add_argument("--out", default=str(REPORTS_DIR))
        p.add_argument("--quiet", action="store_true", help="не печатать отчёт в консоль")

    p = sub.add_parser("fetch"); p.add_argument("--query", action="append", help="свой запрос (можно несколько)")
    p = sub.add_parser("import"); p.add_argument("file"); p.add_argument("--source", default="import")
    p = sub.add_parser("analyze"); analyze_opts(p)
    p = sub.add_parser("run"); p.add_argument("--query", action="append"); analyze_opts(p)
    p = sub.add_parser("demo"); analyze_opts(p)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return {"fetch": cmd_fetch, "import": cmd_import, "analyze": cmd_analyze,
            "run": cmd_run, "demo": cmd_demo}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
