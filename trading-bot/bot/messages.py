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


def _headline(side: str, before: float, after: float, futures: bool) -> str:
    """Что именно сделал бот — словами, понятными без знания знаков позиции."""
    if not futures:
        if side == "buy":
            return "🟢 <b>КУПИТЬ" if before <= 0 else "🟢 <b>ДОКУПИТЬ"
        return "🔴 <b>ПРОДАТЬ" if after <= 0 else "🟠 <b>СОКРАТИТЬ"
    if side == "buy":
        if before < 0:
            return "⚪️ <b>ЗАКРЫТЬ ШОРТ" if after == 0 else "🟠 <b>СОКРАТИТЬ ШОРТ"
        return "🟢 <b>ОТКРЫТЬ ЛОНГ" if before == 0 else "🟢 <b>УВЕЛИЧИТЬ ЛОНГ"
    if before > 0:
        return "⚪️ <b>ЗАКРЫТЬ ЛОНГ" if after == 0 else "🟠 <b>СОКРАТИТЬ ЛОНГ"
    return "🔴 <b>ОТКРЫТЬ ШОРТ" if before == 0 else "🔴 <b>УВЕЛИЧИТЬ ШОРТ"


def trade_message(*, symbol: str, side: str, fill_price: float, notional: float, reason: str,
                  units_before: float, units_after: float, equity: float, weight_after: float,
                  stop: float = 0.0, position_pnl: float | None = None,
                  position_ret: float | None = None, virtual: bool = True,
                  futures: bool = False) -> str:
    """Сообщение-сигнал о сделке, которую бот только что совершил на своём счёте."""
    opening = units_after != 0 and abs(units_after) > abs(units_before)
    closed = units_after == 0 and units_before != 0
    lines = [f"{_headline(side, units_before, units_after, futures)} {escape(symbol)}</b>",
             f"Цена: {price(fill_price)}",
             f"Сумма: {money(notional)} USDT"]
    if units_after != 0:
        lines.append(f"Доля в счёте после сделки: {pct(weight_after, sign=False)}")
    if opening and stop > 0:
        dist = abs(stop / fill_price - 1)
        lines.append(f"Стоп: {price(stop)} ({pct(stop / fill_price - 1)})")
        lines.append(f"Риск по позиции до стопа: {pct(weight_after * dist, sign=False)} счёта")
    if closed and position_pnl is not None:
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
        if snap.get("market") == "futures":
            return "🌧 рынок падает (BTC ниже средней) — бот ищет шорты"
        return "🌧 рынок слабый (фильтр по BTC) — бот вне рынка"
    return "▶️ торгует"


def _positions(snap: dict) -> list[str]:
    pos = snap.get("positions") or []
    if not pos:
        return ["Позиций нет — всё в USDT"]
    out = []
    for p in pos:
        side = " ШОРТ" if p.get("side", 1) < 0 else (" ЛОНГ" if snap.get("market") == "futures" else "")
        out.append(f"• <b>{escape(p['symbol'])}{side}</b>: {money(p['value'])} USDT "
                   f"({pct(p['weight'], sign=False)}), результат {pct(p['ret'])}, "
                   f"стоп {price(p['stop'])}")
    return out


def _leverage(snap: dict) -> list[str]:
    if snap.get("market") != "futures":
        return []
    return [f"Загрузка счёта (плечо): x{snap.get('leverage', 0.0):.2f}"]


def status_text(snap: dict) -> str:
    return "\n".join([
        f"<b>Статус</b> · {escape(snap['mode'])} · {escape(snap['exchange'])}",
        _state_line(snap),
        f"Счёт: <b>{money(snap['equity'])} USDT</b> (кэш {money(snap['cash'])})",
        f"Просадка от пика: {pct(snap['drawdown'], sign=False)}",
        *_leverage(snap),
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
        *_leverage(snap),
        "",
        *_positions(snap),
        "",
        f"Закрытых сделок: {closed}, прибыльных: {winrate}",
        f"Зафиксировано (чистыми): {signed(st.get('realized', 0.0))} USDT",
        f"в т.ч. комиссии: {money(st.get('fees', 0.0))} USDT"
        + (f", фандинг: {signed(-st.get('funding', 0.0) or 0.0)} USDT"
           if snap.get("market") == "futures" else ""),
    ])
