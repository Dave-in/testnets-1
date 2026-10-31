"""Отчет по бэктесту: таблица в консоль + HTML с графиками (без внешних библиотек)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .metrics import LABELS, fmt


def print_summary(res: BacktestResult) -> None:
    cols = {"Бот": res.metrics, **res.benchmarks}
    keys = [k for k in LABELS if k in res.metrics]
    width = max(len(LABELS[k]) for k in keys) + 2
    cw = max(23, *(len(n) + 2 for n in cols))
    header = "".join(f"{name:>{cw}}" for name in cols)
    print(f"{'':<{width}}{header}")
    for k in keys:
        vals = "".join(f"{fmt(k, m[k]) if k in m else '—':>{cw}}" for m in cols.values())
        print(f"{LABELS[k]:<{width}}{vals}")
    stops = [e for e in res.events if e["event"] == "hard_stop"]
    if stops:
        print(f"\nСрабатываний защиты от просадки: {len(stops)}")
        for e in stops[:10]:
            print(f"  {e['ts']}: просадка {e['drawdown'] * 100:.2f}%")


def _svg_line(series: dict[str, pd.Series], width=900, height=260, pct=False,
              log=False) -> str:
    colors = ["#2563eb", "#9ca3af", "#f59e0b", "#10b981", "#ef4444"]
    if log:  # логарифмическая шкала: иначе бот (±10%) не виден рядом с крипто-иксами
        series = {k: np.log10(v.clip(lower=1e-6)) for k, v in series.items()}
    allv = pd.concat(series.values())
    lo, hi = float(allv.min()), float(allv.max())
    if hi == lo:
        hi = lo + 1
    idx = next(iter(series.values())).index
    t0, t1 = idx[0].value, idx[-1].value
    pad = 40

    def xy(ts, v):
        x = pad + (ts.value - t0) / max(t1 - t0, 1) * (width - 2 * pad)
        y = height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)
        label = f"{v * 100:.1f}%" if pct else (f"{10 ** v:.3g}" if log else f"{v:,.0f}")
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="2" y="{y + 4:.1f}" class="axis">{label}</text>')
    for n, (name, s) in enumerate(series.items()):
        step = max(1, len(s) // 1500)
        pts = " ".join(xy(ts, v) for ts, v in s.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{colors[n % len(colors)]}" '
                     f'stroke-width="1.6" points="{pts}"/>')
    legend = " ".join(f'<span style="color:{colors[n % len(colors)]}">■ {html.escape(k)}</span>'
                      for n, k in enumerate(series))
    parts.append("</svg>")
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def write(res: BacktestResult, out_dir: str | Path, title: str, close: pd.DataFrame) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "equity.csv")
    res.trades.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"bot": res.metrics, "benchmarks": res.benchmarks,
         "events": [{**e, "ts": str(e["ts"])} for e in res.events]},
        ensure_ascii=False, indent=2, default=str))

    eq = res.equity["equity"]
    curves = {"Бот": eq / eq.iloc[0] * 100}
    for s in close.columns:
        c = close[s].reindex(eq.index).ffill().dropna()
        if len(c):
            curves[f"Buy&Hold {s}"] = c / c.iloc[0] * 100
    dd = {"Просадка бота": -(1 - eq / eq.cummax())}

    cols = {"Бот": res.metrics, **res.benchmarks}
    rows = "".join(
        f"<tr><td>{LABELS[k]}</td>"
        + "".join(f"<td>{fmt(k, m[k]) if k in m else '—'}</td>" for m in cols.values())
        + "</tr>"
        for k in LABELS if k in res.metrics
    )
    head = "".join(f"<th>{html.escape(n)}</th>" for n in cols)
    trades = res.trades.tail(30).iloc[::-1]
    trade_rows = "".join(
        f"<tr><td>{t.ts:%Y-%m-%d %H:%M}</td><td>{t.symbol}</td><td>{t.side}</td>"
        f"<td>{t.price:,.2f}</td><td>{t.notional:,.2f}</td><td>{t.reason}</td></tr>"
        for t in trades.itertuples()
    )
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#111}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
td,th{{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
.chart{{width:100%;height:auto}} .grid{{stroke:#e5e7eb}} .axis{{font-size:11px;fill:#6b7280}}
.legend{{font-size:13px;margin-top:16px}} .legend span{{margin-right:14px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<h2>Капитал (старт = 100, лог. шкала)</h2>{_svg_line(curves, log=True)}
<h2>Просадка бота</h2>{_svg_line(dd, height=180, pct=True)}
<h2>Метрики</h2><table><tr><th></th>{head}</tr>{rows}</table>
<h2>Последние сделки</h2>
<table><tr><th>Время</th><th>Актив</th><th>Сторона</th><th>Цена</th><th>Сумма</th><th>Причина</th></tr>
{trade_rows}</table>
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
