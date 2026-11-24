"""Отчёт за день по всем теневым ботам: баланс на начало и конец дня, что бот сделал
(сделки, лонги/шорты, выходы, лучшая и худшая) и почему — а если сделок не было,
какое условие входа ни разу не выполнилось (по журналу калибровки).

Работает только с файлами в state/, поэтому одинаково годится для автоматического
отчёта в конце дня и для команды /daily (отчёт за текущий день на этот момент).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .. import messages

EXITS = {"tp": "тейк", "sl": "стоп", "time": "время", "manual": "команда", "liquidation": "ликвидация",
         "signal": "сигнал", "pair_leg": "нога пары"}


def _rows(path: Path, day: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("ts", "").startswith(day)]


def _why_idle(calib: list[dict], key: str) -> str:
    """Почему бот не торговал: условие, которое за день ни разу не дошло до порога (самое
    далёкое от него), или — если все доходили — что они не совпали по времени."""
    by_metric: dict = defaultdict(list)
    for r in calib:
        if r["bot"] == key and r.get("ratio"):
            by_metric[r["metric"]].append((float(r["ratio"]), float(r["value"]), float(r["threshold"])))
    if not by_metric:
        return "нет данных: бот ещё набирает историю"
    never = {m: max(v) for m, v in by_metric.items() if max(x[0] for x in v) < 1}
    if never:
        metric, (ratio, value, thr) = min(never.items(), key=lambda kv: kv[1][0])
        return (f"условие «{metric}» ни разу не выполнилось: максимум {_num(value)} при пороге "
                f"{_num(thr)} ({ratio * 100:.0f}% порога)")
    return "каждое условие входа выполнялось, но не одновременно"


def _num(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def bot_block(title: str, key: str, start: float, end: float, open_positions: int,
              trades: list[dict], calib: list[dict]) -> str:
    pnl = end - start
    pct = pnl / start if start else 0.0
    icon = "🟢" if pnl > 0.005 else ("🔴" if pnl < -0.005 else "⚪️")
    opens = [r for r in trades if r["action"] == "open"]
    closes = [r for r in trades if r["action"] == "close"]
    lines = [f"{icon} <b>{messages.escape(title)}</b>: {messages.money(start)} → "
             f"<b>{messages.money(end)}</b> USDT ({messages.signed(pnl)}, {messages.pct(pct)})"]
    if not opens and not closes:
        lines.append(f"Сделок не было: {messages.escape(_why_idle(calib, key))}."
                     + (f" Открыто позиций: {open_positions}." if open_positions else ""))
        return "\n".join(lines)
    wins = sum(1 for r in closes if r["pnl"] and float(r["pnl"]) > 0)
    fees = sum(float(r["fee"] or 0) for r in trades)
    sides = Counter("лонг" if r["side"] == "long" else "шорт" for r in opens)
    exits = Counter(EXITS.get(r["reason"], r["reason"]) for r in closes)
    lines.append(f"Сделок: открыто {len(opens)} ({', '.join(f'{k} {v}' for k, v in sides.items())}), "
                 f"закрыто {len(closes)}, прибыльных {wins}"
                 + (f" ({wins / len(closes) * 100:.0f}%)" if closes else "")
                 + f" · комиссии {messages.money(fees)} · открыто сейчас {open_positions}")
    if opens:
        lines.append(f"Почему входил ({len(opens)} раз), последний вход: "
                     f"{messages.escape(opens[-1]['symbol'])} — «{messages.escape(opens[-1]['reason'])}»")
    if exits:
        lines.append("Выходы: " + ", ".join(f"{k} {v}" for k, v in exits.most_common()))
    done = [r for r in closes if r["pnl"]]
    if done:
        best = max(done, key=lambda r: float(r["pnl"]))
        worst = min(done, key=lambda r: float(r["pnl"]))
        lines.append(f"Лучшая: {messages.signed(float(best['pnl']))} ({messages.escape(best['symbol'])})"
                     + (f" · худшая: {messages.signed(float(worst['pnl']))} "
                        f"({messages.escape(worst['symbol'])})" if worst is not best else ""))
    return "\n".join(lines)


def build(state_dir: str | Path, day: str, balances: dict | None = None,
          title: str = "Отчёт за день") -> str:
    """Отчёт по всем ботам из state/. balances — {ключ: (начало, конец)} для завершённого дня;
    без него берётся текущий снимок (начало дня → сейчас)."""
    state_dir = Path(state_dir)
    calib = _rows(state_dir / "shadow_calibration.csv", day)
    blocks, total_start, total_end = [], 0.0, 0.0
    for path in sorted(state_dir.glob("shadow_*.json")):
        key = path.stem[len("shadow_"):]
        if key.endswith("_control") or key == "telegram":
            continue
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        snap = st.get("snapshot") or {}
        if not snap:
            continue
        if balances and key in balances:
            start, end = balances[key]
        else:
            start = (st.get("marks") or {}).get("day_equity", snap.get("start_equity", 0.0))
            end = snap.get("equity", 0.0)
        total_start += start
        total_end += end
        trades = _rows(state_dir / f"shadow_{key}_trades.csv", day)
        title_ = (snap.get("shadow") or {}).get("strategy", key)
        blocks.append((end - start, bot_block(title_, key, start, end, len(snap.get("positions", [])),
                                              trades, calib)))
    if not blocks:
        return "Ботов с сохранённым состоянием пока нет."
    blocks.sort(key=lambda b: -b[0])                  # сначала заработавшие
    pnl = total_end - total_start
    head = (f"📅 <b>{messages.escape(title)} {day[8:10]}.{day[5:7]}</b> (UTC)\n"
            f"Всего по {len(blocks)} ботам: {messages.money(total_start)} → "
            f"<b>{messages.money(total_end)}</b> USDT ({messages.signed(pnl)})")
    return head + "\n\n" + "\n\n".join(b for _, b in blocks)
