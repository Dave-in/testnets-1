"""Отчет по бэктесту: таблица в консоль + HTML с графиками (без внешних библиотек)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .metrics import LABELS, fmt, monthly_returns


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


def monthly_grid(equity: pd.Series) -> pd.DataFrame:
    """Таблица год × месяц с доходностью."""
    m = monthly_returns(equity)
    grid = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    return grid.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))


def print_monthly(equity: pd.Series, target: float) -> None:
    grid = monthly_grid(equity)
    print(f"\nДоходность по месяцам, %  (* — цель {target * 100:.0f}% выполнена)")
    print("год  " + "".join(f"{m:>7}" for m in range(1, 13)) + "    год")
    yearly = equity.resample("YE").last()
    prev = yearly.shift(1)
    prev.iloc[0] = equity.iloc[0]
    for year, row in grid.iterrows():
        cells = "".join("      ·" if pd.isna(v) else f"{v * 100:6.1f}{'*' if v >= target else ' '}"
                        for v in row)
        y = yearly[yearly.index.year == year]
        yr = (y.iloc[0] / prev[y.index[0]] - 1) * 100 if len(y) else float("nan")
        print(f"{year} {cells} {yr:6.1f}")


def _monthly_html(equity: pd.Series, target: float) -> str:
    grid = monthly_grid(equity)

    def cell(v):
        if pd.isna(v):
            return "<td></td>"
        cls = "hit" if v >= target else ("pos" if v > 0 else ("neg" if v < 0 else ""))
        return f'<td class="{cls}">{v * 100:.1f}</td>'

    head = "".join(f"<th>{m}</th>" for m in range(1, 13))
    body = "".join(f"<tr><td>{y}</td>{''.join(cell(v) for v in row)}</tr>"
                   for y, row in grid.iterrows())
    return f"<table class=\"months\"><tr><th>Год</th>{head}</tr>{body}</table>"


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


def write(res: BacktestResult, out_dir: str | Path, title: str, close: pd.DataFrame,
          monthly_target: float = 0.06) -> Path:
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
.months td{{text-align:center;font-size:13px}} .hit{{background:#86efac}} .pos{{background:#dcfce7}}
.neg{{background:#fee2e2}} .note{{color:#6b7280;font-size:13px}}
.months{{display:block;overflow-x:auto}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<h2>Капитал (старт = 100, лог. шкала)</h2>{_svg_line(curves, log=True)}
<h2>Просадка бота</h2>{_svg_line(dd, height=180, pct=True)}
<h2>Метрики</h2><table><tr><th></th>{head}</tr>{rows}</table>
<h2>Доходность по месяцам, %</h2>
<p class="note">Зеленый — цель {monthly_target * 100:.0f}% в месяц выполнена, светлый — плюс, красный — минус.</p>
{_monthly_html(eq, monthly_target)}
<h2>Последние сделки</h2>
<table><tr><th>Время</th><th>Актив</th><th>Сторона</th><th>Цена</th><th>Сумма</th><th>Причина</th></tr>
{trade_rows}</table>
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
