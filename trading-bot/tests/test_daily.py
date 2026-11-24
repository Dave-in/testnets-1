"""Отчёт за день по теневым ботам: цифры, сделки, причины простоя; отправка в General."""
import json

from bot.shadow import daily
from bot.shadow.engine import Engine
from tests.test_shadow import T0, book

DAY = "2026-11-24"


def write_state(d, key, title, start, end, positions=0):
    (d / f"shadow_{key}.json").write_text(json.dumps({
        "marks": {"day": DAY, "day_equity": start},
        "snapshot": {"equity": end, "start_equity": 100.0, "positions": [{}] * positions,
                     "shadow": {"strategy": title}}}, ensure_ascii=False), encoding="utf-8")


def test_daily_report_numbers_trades_and_idle_reasons(tmp_path):
    write_state(tmp_path, "fade", "Откат (контроль)", 100.0, 101.5)
    write_state(tmp_path, "book", "Стакан", 100.0, 100.0)
    head = "ts,symbol,action,side,qty,price,fee,reason,slippage_bps,pnl\n"
    (tmp_path / "shadow_fade_trades.csv").write_text(
        head
        + f"{DAY}T01:00:00,SUIUSDT,open,long,1,1.0,0.01,движение -1.2% за 5 мин выдохлось,0.5,\n"
        + f"{DAY}T01:05:00,SUIUSDT,close,long,1,1.1,0.01,tp,0.1,1.6\n"
        + "2026-11-23T23:00:00,ARBUSDT,close,long,1,1.0,0.01,sl,0.1,-5\n",       # вчера — не считаем
        encoding="utf-8")
    (tmp_path / "shadow_calibration.csv").write_text(
        "ts,bot,metric,value,threshold,ratio,symbol\n"
        + f"{DAY}T01:00:00,book,перекос стакана,0.9,0.5,1.8,X\n"
        + f"{DAY}T01:00:00,book,тейк / издержки сделки,2.41,2.5,0.964,X\n", encoding="utf-8")
    text = daily.build(tmp_path, DAY)
    assert "Всего по 2 ботам: 200.00 → <b>201.50</b> USDT (+1.50)" in text
    fade = text.split("\n\n")[1]                                            # заработавшие — первыми
    assert "Откат (контроль)" in fade and "100.00 → <b>101.50</b>" in fade
    assert "открыто 1 (лонг 1), закрыто 1, прибыльных 1 (100%)" in fade
    assert "выдохлось" in fade and "Выходы: тейк 1" in fade and "ARBUSDT" not in fade
    assert "«тейк / издержки сделки» ни разу не выполнилось: максимум 2.41 при пороге 2.50" in text


def test_day_rollover_ignores_startup_and_sends_report_to_general(tmp_path):
    from bot.group import Group
    from bot.telegram import ShadowChannel, Telegram

    eng = Engine({"bots": ["fade"], "save_every_s": 1, "account": {"capital": 100.0}}, tmp_path)
    eng.add_market("ETHUSDT", book(100.0))
    bot = eng.bots["fade"]
    eng.refresh_all()                                       # до данных биржи (now = 0)
    assert "day" not in bot.marks and bot.day_snapshot is None
    eng.advance(T0)
    eng.refresh_all()
    day0 = bot.marks["day"]
    eng.advance(T0 + 86_400_000)                             # следующие сутки
    eng.refresh_all()
    assert bot.marks["last_day"]["day"] == day0 and bot.day_snapshot is not None

    sent = []
    tg = Telegram("T", 111, lambda m, p, t: sent.append(p) or {"ok": True})
    ch = ShadowChannel(tg, tmp_path / "tg.json", start=False)
    ch.chat_id = "-100"
    Group({}, tg, 1, {"Тень": bot}, ch, tmp_path).maybe_report()
    chat, thread, text = ch.queue.get_nowait()
    assert chat == "-100" and thread is None and "Отчёт за день" in text
    assert not sent                                         # в личный чат не дублируется


def test_daily_command(tmp_path):
    from bot.telegram import CommandListener, Telegram
    write_state(tmp_path, "fade", "Откат (контроль)", 100.0, 99.0)
    lst = CommandListener(Telegram("T", 111, lambda *a: {"ok": True}),
                          bots=[("x", tmp_path / "s.json", tmp_path / "c.json")],
                          calib_path=tmp_path / "shadow_calibration.csv")
    assert "Отчёт за сегодня" in lst.handle("/daily") or "Ботов" in lst.handle("/daily")
