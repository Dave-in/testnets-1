"""Калибровка порогов по «почти сработавшим» сигналам.

Каждую минуту движок пишет в state/shadow_calibration.csv максимум каждого условия входа
по всем монетам (например, сумму ликвидаций за окно и порог). Здесь по этим записям
считаем распределение: как часто условие доходит до порога и где был бы порог «топ-1%».
Подкручивать пороги стоит по нескольким дням таких данных, а не по часу работы.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import messages
from .strategies import STRATEGIES


def _num(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def load(path: str | Path, days: float) -> dict:
    """{(бот, условие): {"values": [...], "thresholds": [...]}} за последние days дней."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = datetime.fromisoformat(rows[-1]["ts"])
    since = last - timedelta(days=days)
    out: dict = defaultdict(lambda: {"values": [], "thresholds": [], "first": None})
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if ts < since:
            continue
        item = out[(r["bot"], r["metric"])]
        item["values"].append(float(r["value"]))
        item["thresholds"].append(float(r["threshold"] or 0))
        item["first"] = item["first"] or ts
    for item in out.values():
        item["last"] = last
    return dict(out)


def report(path: str | Path, days: float = 3.0) -> str:
    data = load(path, days)
    if not data:
        return ("📐 Данных для калибровки пока нет: они копятся раз в минуту, "
                "первые появятся через пару минут после запуска.")
    first = min(v["first"] for v in data.values())
    last = max(v["last"] for v in data.values())
    hours = (last - first).total_seconds() / 3600
    lines = [f"📐 <b>Калибровка порогов</b> за {hours:.1f} ч (из запрошенных {days:g} дн.)",
             "Для каждой минуты взят максимум условия по всем монетам. "
             "«99%» — значение, которое превышается в 1% минут: кандидат в порог.",
             "Пороги лучше менять, когда данных хотя бы 2–3 дня."]
    by_bot: dict = defaultdict(list)
    for (bot, metric), v in data.items():
        by_bot[bot].append((metric, v))
    for key in [k for k in STRATEGIES if k in by_bot] + [k for k in by_bot if k not in STRATEGIES]:
        title = STRATEGIES[key].title if key in STRATEGIES else key
        lines += ["", f"🤖 <b>{messages.escape(title)}</b>"]
        for metric, v in sorted(by_bot[key]):
            x = np.array(v["values"])
            thr = float(np.median(v["thresholds"])) if v["thresholds"] else 0.0
            above = float(np.mean(x >= thr)) if thr else 0.0
            p50, p90, p99 = np.percentile(x, [50, 90, 99])
            lines.append(f"• {messages.escape(metric)}: порог {_num(thr)} · медиана {_num(p50)} · "
                         f"90% {_num(p90)} · 99% {_num(p99)} · макс {_num(float(x.max()))} · "
                         f"на пороге и выше {above * 100:.1f}% минут ({len(x)} мин)")
    return "\n".join(lines)
