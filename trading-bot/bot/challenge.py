"""Режим «попыток»: оценка шансов по истории.

Бэктест даёт одну непрерывную кривую капитала. Размер позиций пропорционален счёту,
поэтому ту же последовательность сделок можно «нарезать» на попытки: стартуем с 1.0,
идём по доходностям бар за баром, пока не достигнем цели или не сольёмся.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _run_from(rets: np.ndarray, i: int, target: float, bust: float, max_bars: int | None):
    """Одна попытка, начатая на баре i. Возвращает (исход, номер последнего бара, итог)."""
    eq, end = 1.0, len(rets) if max_bars is None else min(len(rets), i + max_bars)
    for j in range(i, end):
        eq *= 1.0 + rets[j]
        if eq >= target:
            return "win", j, eq
        if eq <= bust:
            return "bust", j, eq
    return "open", end - 1, eq


def sequential_attempts(equity: pd.Series, target: float, bust: float) -> list[dict]:
    """Попытки одна за другой: закончилась одна — со следующего бара начинается новая."""
    rets = equity.pct_change().fillna(0.0).to_numpy()
    idx, out, i = equity.index, [], 1
    while i < len(rets):
        outcome, j, eq = _run_from(rets, i, target, bust, None)
        out.append({"start": idx[i], "end": idx[j], "outcome": outcome, "final": eq,
                    "days": (idx[j] - idx[i]).total_seconds() / 86400})
        i = j + 1
    return out


def start_date_odds(equity: pd.Series, target: float, bust: float, horizon_days: int,
                    step_days: int = 3) -> dict:
    """Шансы для случайной даты старта: доля попыток, которые за horizon_days
    дошли до цели, слились или остались в игре."""
    rets = equity.pct_change().fillna(0.0).to_numpy()
    per_day = int(round(pd.Timedelta(days=1) / (equity.index[1] - equity.index[0])))
    horizon, step = horizon_days * per_day, step_days * per_day
    res = [_run_from(rets, i, target, bust, horizon)
           for i in range(1, len(rets) - horizon, step)]
    n = max(len(res), 1)
    finals = np.array([r[2] for r in res]) if res else np.array([1.0])
    return {"starts": len(res), "horizon_days": horizon_days,
            "win": sum(r[0] == "win" for r in res) / n,
            "bust": sum(r[0] == "bust" for r in res) / n,
            "open": sum(r[0] == "open" for r in res) / n,
            "median_final": float(np.median(finals)),
            "mean_final": float(np.mean(np.clip(finals, 0, target)))}


def summary_text(equity: pd.Series, target: float, bust: float, stake: float) -> str:
    att = sequential_attempts(equity, target, bust)
    done = [a for a in att if a["outcome"] != "open"]
    wins = [a for a in done if a["outcome"] == "win"]
    lines = [f"Режим попыток: цель x{target:g}, слив — осталось меньше {bust * 100:.0f}% счёта",
             f"Попыток подряд за период: {len(done)} завершено, из них побед {len(wins)}, "
             f"сливов {len(done) - len(wins)}"]
    if done:
        spent = stake * len(done)
        got = stake * sum(a["final"] for a in done)
        num = lambda x, sign="": f"{x:{sign},.0f}".replace(",", " ")  # noqa: E731
        lines.append(f"Если каждый раз ставить {stake:g} USDT: вложено {num(spent)}, "
                     f"получено {num(got)}, итог {num(got - spent, '+')} USDT")
        lines.append("Средняя длительность попытки: "
                     f"{np.mean([a['days'] for a in done]):.0f} дн.")
    span_days = (equity.index[-1] - equity.index[0]).days
    for days in (90, 180, 365):
        if span_days > days * 2:
            o = start_date_odds(equity, target, bust, days)
            lines.append(f"Старт в случайный день, срок {days} дн.: цель {o['win'] * 100:.0f}%, "
                         f"слив {o['bust'] * 100:.0f}%, в игре {o['open'] * 100:.0f}% "
                         f"(медианный итог x{o['median_final']:.2f})")
    return "\n".join(lines)
