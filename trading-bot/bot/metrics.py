"""Метрики качества стратегии."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    return float((1 - equity / equity.cummax()).max()) if len(equity) else 0.0


def max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Самый долгий период "под водой" (от пика до восстановления пика)."""
    underwater = equity < equity.cummax()
    longest, start = pd.Timedelta(0), None
    for ts, uw in underwater.items():
        if uw and start is None:
            start = ts
        elif not uw and start is not None:
            longest, start = max(longest, ts - start), None
    if start is not None:
        longest = max(longest, equity.index[-1] - start)
    return longest


def compute(equity: pd.Series, periods_per_year: int, trades: pd.DataFrame | None = None,
            exposure: pd.Series | None = None) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(periods_per_year)
    downside = rets[rets < 0].std() * np.sqrt(periods_per_year)
    mdd = max_drawdown(equity)
    monthly = equity.resample("ME").last().pct_change().dropna()

    m = {
        "period": f"{equity.index[0]:%Y-%m-%d}…{equity.index[-1]:%Y-%m-%d}",
        "years": round(years, 2),
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": mdd,
        "max_dd_days": max_drawdown_duration(equity).days,
        "sharpe": (rets.mean() * periods_per_year) / vol if vol > 0 else 0.0,
        "sortino": (rets.mean() * periods_per_year) / downside if downside > 0 else 0.0,
        "calmar": cagr / mdd if mdd > 0 else 0.0,
        "worst_bar": float(rets.min()) if len(rets) else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        # среди месяцев, когда бот вообще что-то делал (не сидел весь месяц в USDT)
        "positive_months": float((monthly[monthly.abs() > 1e-9] > 0).mean())
        if (monthly.abs() > 1e-9).any() else 0.0,
    }
    if exposure is not None:
        m["avg_exposure"] = float(exposure.mean())
    if trades is not None and len(trades):
        m["trades"] = int(len(trades))
        m["fees_paid"] = float(trades["fee"].sum())
    return m


PCT_KEYS = {"total_return", "cagr", "volatility", "max_drawdown", "worst_bar",
            "worst_month", "positive_months", "avg_exposure"}

LABELS = {
    "period": "Период",
    "years": "Лет",
    "total_return": "Доходность всего",
    "cagr": "Доходность годовая (CAGR)",
    "volatility": "Волатильность (год.)",
    "max_drawdown": "Макс. просадка",
    "max_dd_days": "Макс. время под водой, дней",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar (CAGR / просадка)",
    "worst_bar": "Худший бар",
    "worst_month": "Худший месяц",
    "positive_months": "Прибыльных месяцев (из активных)",
    "avg_exposure": "Средняя загрузка капитала",
    "trades": "Сделок",
    "fees_paid": "Комиссии, USDT",
}


def fmt(key: str, value) -> str:
    if key in PCT_KEYS:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
