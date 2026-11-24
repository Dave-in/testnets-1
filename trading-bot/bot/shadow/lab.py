"""«Лаборатория»: что улучшить в боте — по данным state/research/.

1. Сделки: результат в R, ход цены внутри сделки (MFE/MAE) и после выхода — стоп тесный
   или вход плохой; какие признаки на входе отличают прибыльные сделки от убыточных.
2. Сигналы (и принятые, и отсеянные) с путём цены на час вперёд: перебор стопа, тейка и
   времени удержания с учётом комиссий. Чтобы не подгонять под прошлое, данные делятся по
   времени: на первых 2/3 подбираем, на последней 1/3 проверяем.

Запуск: python -m bot shadow-lab --bot oi --days 3
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STOPS = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)          # %
TAKES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)           # %, 0 — без тейка
HOLDS = (5, 15, 30, 60)                          # мин
COST_PCT = 0.13                                  # комиссии тейкера туда-обратно + проскальзывание


def load(path: Path, days: float, ts_key: str) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return []
    last = max(datetime.fromisoformat(r[ts_key]) for r in rows)
    since = last - timedelta(days=days)
    return sorted((r for r in rows if datetime.fromisoformat(r[ts_key]) >= since), key=lambda r: r[ts_key])


def flat_features(rec: dict) -> dict:
    out = {}
    f = rec.get("features") or {}
    for group in ("signal", "market"):
        for k, v in (f.get(group) or {}).items():
            if isinstance(v, (int, float)):
                out[f"{k}"] = float(v)
    for k, v in f.items():                          # сигналы: признаки стратегии лежат плоско
        if isinstance(v, (int, float)):
            out[k] = float(v)
    for k, v in (rec.get("market") or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    out.pop("price", None)
    return out


def feature_split(rows: list[dict], value) -> list[tuple]:
    """Для каждого признака: результат в нижней и верхней трети значений. Сортировка — по разнице."""
    feats: dict = {}
    for r in rows:
        y = value(r)
        if y is None:
            continue
        for k, v in flat_features(r).items():
            feats.setdefault(k, []).append((v, y))
    out = []
    for k, pts in feats.items():
        if len(pts) < 30 or len({v for v, _ in pts}) < 3:
            continue
        pts.sort()
        n = len(pts) // 3
        lo, hi = pts[:n], pts[-n:]
        mlo, mhi = np.mean([y for _, y in lo]), np.mean([y for _, y in hi])
        out.append((abs(mhi - mlo), k, lo[-1][0], hi[0][0], mlo, mhi, n))
    return sorted(out, reverse=True)


def simulate(path: list, side: int, stop: float, take: float, hold_min: int) -> float | None:
    """Чистый результат сделки в % по пути цены (шаг 10 с): стоп, тейк, выход по времени."""
    end = min(len(path) - 1, hold_min * 6)
    last = None
    for x in path[1:end + 1]:
        if x is None:
            continue
        r = side * x * 100
        last = r
        if r <= -stop:
            return -stop - COST_PCT
        if take and r >= take:
            return take - COST_PCT
    return None if last is None else last - COST_PCT


def grid(signals: list[dict]) -> list[tuple]:
    res = []
    for stop in STOPS:
        for take in TAKES:
            for hold in HOLDS:
                vals = [simulate(s["path"], s["side"], stop, take, hold) for s in signals]
                vals = [v for v in vals if v is not None]
                if vals:
                    res.append((float(np.mean(vals)), stop, take, hold, len(vals),
                                sum(v > 0 for v in vals) / len(vals)))
    return sorted(res, reverse=True)


def telegram(state_dir: str | Path, args: list[str]) -> str:
    """/lab — список ботов и объём данных; /lab <бот> [дней] — отчёт лаборатории."""
    from .. import messages
    from .strategies import STRATEGIES
    d = Path(state_dir) / "research"

    def count(kind, key):
        p = d / f"{kind}_{key}.jsonl"
        return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0

    if not args or args[0].lower() not in STRATEGIES:
        lines = ["🧪 <b>Лаборатория</b>: что улучшить в боте — стоп тесный или вход плохой, какие "
                 "признаки отличают прибыльные сделки, какие стоп/тейк/время дали бы лучше результат.",
                 "Команда: <code>/lab ключ [дней]</code>, например <code>/lab oi</code> или "
                 "<code>/lab vwap 7</code>. Данные появляются через час после сигнала.", ""]
        for key, cls in STRATEGIES.items():
            lines.append(f"<code>{key}</code> — {messages.escape(cls.title)}: сделок {count('trades', key)}, "
                         f"сигналов {count('signals', key)}")
        return "\n".join(lines)
    key = args[0].lower()
    try:
        days = float(args[1].replace(",", ".")) if len(args) > 1 else 3.0
    except ValueError:
        days = 3.0
    title = STRATEGIES[key].title
    body = report(state_dir, key, days).replace(f"бот «{key}»", f"бот «{title}»", 1)
    return "🧪 " + messages.escape(body)


def report(state_dir: str | Path, bot: str, days: float = 3.0) -> str:
    d = Path(state_dir) / "research"
    trades = load(d / f"trades_{bot}.jsonl", days, "closed")
    signals = [s for s in load(d / f"signals_{bot}.jsonl", days, "ts") if s.get("path")]
    lines = [f"Лаборатория: бот «{bot}», данные за {days:g} дн. — сделок {len(trades)}, "
             f"сигналов с путём цены {len(signals)}"]
    if trades:
        pnl = [t["pnl"] or 0 for t in trades]
        rr = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
        wins = sum(p > 0 for p in pnl)
        lines += ["", "1. СДЕЛКИ",
                  f"   итог {sum(pnl):+.2f} USDT, прибыльных {wins}/{len(trades)} "
                  f"({wins / len(trades) * 100:.0f}%), средний результат {np.mean(rr):+.2f}R"
                  if rr else f"   итог {sum(pnl):+.2f} USDT"]
        exits = Counter(t["exit_reason"] for t in trades)
        lines.append("   выходы: " + ", ".join(f"{k} {v}" for k, v in exits.most_common()))
        losers = [t for t in trades if (t["pnl"] or 0) <= 0 and t.get("mfe_r") is not None]
        if losers:
            half = sum(t["mfe_r"] >= 0.5 for t in losers) / len(losers)
            one = sum(t["mfe_r"] >= 1.0 for t in losers) / len(losers)
            lines.append(f"   убыточные сделки: {half * 100:.0f}% были в плюсе на ≥0.5R, "
                         f"{one * 100:.0f}% — на ≥1R до разворота (много — тейк далеко/стоп тесный)")
        stopped = [t for t in trades if t["exit_reason"] == "sl"]
        for h in (15, 60):
            vals = [t.get(f"after_{h}m_pct") for t in stopped if t.get(f"after_{h}m_pct") is not None]
            if vals:
                lines.append(f"   после стопа через {h} мин цена в сторону сделки: в среднем "
                             f"{np.mean(vals):+.2f}%, вернулась выше входа в {np.mean([v > 0 for v in vals]) * 100:.0f}% "
                             f"случаев (много — стоп преждевременный)")
        holds_w = [t["hold_s"] / 60 for t in trades if (t["pnl"] or 0) > 0]
        holds_l = [t["hold_s"] / 60 for t in trades if (t["pnl"] or 0) <= 0]
        if holds_w and holds_l:
            lines.append(f"   держал: прибыльные — медиана {np.median(holds_w):.1f} мин, "
                         f"убыточные — {np.median(holds_l):.1f} мин")
        split = feature_split(trades, lambda t: t.get("pnl"))
        if split:
            lines.append("   признаки, сильнее всего разделяющие результат (нижняя треть → верхняя треть):")
            for diff, k, lo_max, hi_min, mlo, mhi, n in split[:6]:
                lines.append(f"     {k}: ≤{lo_max:.4g} → {mlo:+.2f} $/сделку; ≥{hi_min:.4g} → "
                             f"{mhi:+.2f} $/сделку (по {n})")
    if signals:
        status = Counter(s["status"] for s in signals)
        lines += ["", "2. СИГНАЛЫ (принятые и отсеянные)",
                  "   " + ", ".join(f"{k}: {v}" for k, v in status.most_common())]
        for h in (5, 15, 60):
            vals = [s.get(f"after_{h}m_pct") for s in signals if s.get(f"after_{h}m_pct") is not None]
            if vals:
                lines.append(f"   ход цены в сторону сигнала через {h} мин: в среднем {np.mean(vals):+.3f}%, "
                             f"в нужную сторону {np.mean([v > 0 for v in vals]) * 100:.0f}%")
        split = feature_split(signals, lambda s: s.get("after_15m_pct"))
        if split:
            lines.append("   признаки и ход цены через 15 мин (нижняя треть → верхняя треть):")
            for diff, k, lo_max, hi_min, mlo, mhi, n in split[:6]:
                lines.append(f"     {k}: ≤{lo_max:.4g} → {mlo:+.3f}%; ≥{hi_min:.4g} → {mhi:+.3f}% (по {n})")
        if len(signals) >= 30:
            cut = len(signals) * 2 // 3
            train, test = signals[:cut], signals[cut:]
            g = grid(train)
            test_map = {(s, t, h): (m, n, w) for m, s, t, h, n, w in grid(test)}
            lines += ["", f"3. ПЕРЕБОР стоп/тейк/время (с комиссиями {COST_PCT}% за сделку), "
                          f"подбор на первых {len(train)}, проверка на последних {len(test)} сигналах:"]
            for m, s, t, h, n, w in g[:5]:
                tm = test_map.get((s, t, h))
                lines.append(f"   стоп {s}% · тейк {t or 'нет'}{'%' if t else ''} · до {h} мин: "
                             f"подбор {m:+.3f}%/сделку ({w * 100:.0f}% прибыльных) → проверка "
                             + (f"{tm[0]:+.3f}% ({tm[2] * 100:.0f}%)" if tm else "нет данных"))
            best_test = max(test_map.values())[0] if test_map else None
            if best_test is not None and best_test <= 0:
                lines.append("   ни одно сочетание не даёт плюса на проверке: менять стоп и тейк мало, "
                             "нужно менять условия входа")
        else:
            lines.append("   для перебора нужно хотя бы 30 сигналов с путём цены — подождите")
    if not trades and not signals:
        lines.append("Данных пока нет: сигналы записываются через час после появления.")
    return "\n".join(lines)
