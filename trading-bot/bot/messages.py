"""Тексты для Telegram и консоли: сигналы, статус, отчёт. Чистые функции без сети."""
from __future__ import annotations

from html import escape  # noqa: F401 — используется и в live.py

REASONS = {
    "rebalance": "сигнал стратегии",
    "trailing_stop": "сработал трейлинг-стоп",
    "drawdown_guard": "защита от просадки",
    "manual_close_all": "команда «закрыть всё»",
}

HELP = (
    "<b>Команды</b>\n"
    "/status — счёт, просадка, открытые позиции\n"
    "/report — отчёт: день, месяц, всё время, статистика сделок\n"
    "/pause — не открывать новые сделки (стопы работают)\n"
    "/resume — снять паузу\n"
    "/closeall — продать всё в USDT и встать на паузу\n"
    "/help — эта справка"
)


def money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


def signed(x: float) -> str:
    return f"{x:+,.2f}".replace(",", " ")


def price(x: float) -> str:
    """Цена с разумным числом знаков: 67 250.5, 152.30, 0.4215, 0.00001234."""
    if x >= 1000:
        return f"{x:,.1f}".replace(",", " ")
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def pct(x: float, sign: bool = True) -> str:
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def trade_message(*, symbol: str, side: str, fill_price: float, notional: float, reason: str,
                  units_before: float, units_after: float, equity: float, weight_after: float,
                  stop: float = 0.0, position_pnl: float | None = None,
                  position_ret: float | None = None, virtual: bool = True) -> str:
    """Сообщение-сигнал о сделке, которую бот только что совершил на своём счёте."""
    if side == "buy":
        head = "🟢 <b>КУПИТЬ" if units_before <= 0 else "🟢 <b>ДОКУПИТЬ"
    else:
        head = "🔴 <b>ПРОДАТЬ" if units_after <= 0 else "🟠 <b>СОКРАТИТЬ"
    lines = [f"{head} {escape(symbol)}</b>",
             f"Цена: {price(fill_price)}",
             f"Сумма: {money(notional)} USDT"]
    if side == "buy" or units_after > 0:
        lines.append(f"Доля в счёте после сделки: {pct(weight_after, sign=False)}")
    if side == "buy" and stop > 0:
        lines.append(f"Стоп: {price(stop)} ({pct(stop / fill_price - 1)})")
    if side == "sell" and units_after <= 0 and position_pnl is not None:
        icon = "✅" if position_pnl >= 0 else "❌"
        ret = f" ({pct(position_ret)})" if position_ret is not None else ""
        lines.append(f"{icon} Итог сделки: {signed(position_pnl)} USDT{ret}")
    lines.append(f"Причина: {REASONS.get(reason, reason)}")
    lines.append(f"{'Виртуальный счёт' if virtual else 'Счёт'}: {money(equity)} USDT")
    return "\n".join(lines)


def _state_line(snap: dict) -> str:
    if snap.get("halted"):
        return "⛔️ остановлен защитой от просадки"
    if snap.get("paused"):
        return "⏸ пауза (по команде)"
    if snap.get("guard_paused_left"):
        return f"⏸ пауза защиты, осталось баров: {snap['guard_paused_left']}"
    if not snap.get("regime_ok", True):
        return "🌧 рынок слабый (фильтр по BTC) — бот вне рынка"
    return "▶️ торгует"


def _positions(snap: dict) -> list[str]:
    pos = snap.get("positions") or []
    if not pos:
        return ["Позиций нет — всё в USDT"]
    out = []
    for p in pos:
        out.append(f"• <b>{escape(p['symbol'])}</b>: {money(p['value'])} USDT "
                   f"({pct(p['weight'], sign=False)}), результат {pct(p['ret'])}, "
                   f"стоп {price(p['stop'])}")
    return out


def status_text(snap: dict) -> str:
    return "\n".join([
        f"<b>Статус</b> · {escape(snap['mode'])} · {escape(snap['exchange'])}",
        _state_line(snap),
        f"Счёт: <b>{money(snap['equity'])} USDT</b> (кэш {money(snap['cash'])})",
        f"Просадка от пика: {pct(snap['drawdown'], sign=False)}",
        "",
        *_positions(snap),
        "",
        f"Обновлено: {snap['ts'][:16].replace('T', ' ')} UTC",
    ])


def report_text(snap: dict, title: str = "Отчёт") -> str:
    st = snap.get("stats") or {}
    closed = st.get("closed", 0)
    winrate = f"{st.get('wins', 0) / closed * 100:.0f}%" if closed else "—"
    return "\n".join([
        f"<b>{escape(title)}</b> · {snap['ts'][:10]}",
        _state_line(snap),
        f"Счёт: <b>{money(snap['equity'])} USDT</b>",
        f"За день: {pct(snap['day_ret'])}",
        f"За месяц: {pct(snap['month_ret'])}",
        f"С запуска: {pct(snap['total_ret'])} (старт {money(snap['start_equity'])} USDT)",
        f"Просадка от пика: {pct(snap['drawdown'], sign=False)}",
        "",
        *_positions(snap),
        "",
        f"Закрытых сделок: {closed}, прибыльных: {winrate}",
        f"Зафиксировано: {signed(st.get('realized', 0.0))} USDT, "
        f"комиссии: {money(st.get('fees', 0.0))} USDT",
    ])
