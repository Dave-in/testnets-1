"""Командная строка.

  python -m bot download  --since 2019-01-01              # скачать историю
  python -m bot backtest  --start 2020-01-01               # прогон по истории
  python -m bot backtest  --synthetic                      # проверка механики без интернета
  python -m bot walkforward --start 2020-01-01 --window 365 # по годам/окнам
  python -m bot run                                        # paper/testnet/live (по config)
  python -m bot ctl status | pause | resume | close-all | stop
  python -m bot logs -n 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import data, metrics, report
from .backtest import Backtester
from .config import Config, load_config
from .indicators import periods_per_year


def _cfg(args) -> Config:
    path = Path(args.config)
    if not path.exists():
        path = Path("config.example.toml")
        print(f"config.toml не найден — использую {path}", file=sys.stderr)
    cfg = load_config(path)
    if getattr(args, "profile", None):
        cfg.portfolio.profile = args.profile
    if getattr(args, "timeframe", None):
        cfg.exchange.timeframe = args.timeframe
    if getattr(args, "exchange", None):
        cfg.exchange.name = args.exchange
    return cfg


def _load_data(cfg: Config, args) -> dict[str, pd.DataFrame]:
    if args.synthetic:
        return data.synthetic(cfg.portfolio.symbols, bars=args.bars, timeframe=cfg.exchange.timeframe)
    # индикаторы прогреваются на данных ДО --start, торговля начинается с --start
    return data.load_cached(args.data_dir, cfg.exchange.name, cfg.portfolio.symbols,
                            cfg.exchange.timeframe, end=args.end)


def cmd_download(args) -> None:
    cfg = _cfg(args)
    symbols = args.symbols or sorted({s for p in cfg.portfolio.profiles.values() for s in p})
    data.download(cfg.exchange.name, symbols, cfg.exchange.timeframe, args.since, args.until,
                  args.data_dir)


def cmd_backtest(args) -> None:
    cfg = _cfg(args)
    frames = _load_data(cfg, args)
    res = Backtester(cfg, frames, trade_start=args.start).run()
    title = (f"Бэктест: {cfg.exchange.name} {cfg.exchange.timeframe}, профиль "
             f"{cfg.portfolio.profile} ({', '.join(cfg.portfolio.symbols)})"
             + (" — СИНТЕТИКА" if args.synthetic else ""))
    print(title + "\n")
    report.print_summary(res)
    close = pd.DataFrame({s: df["close"] for s, df in frames.items()})
    report.print_monthly(res.equity["equity"], cfg.portfolio.monthly_target)
    path = report.write(res, args.out, title, close, cfg.portfolio.monthly_target)
    print(f"\nОтчет: {path}  (equity.csv, trades.csv, summary.json рядом)")


def cmd_walkforward(args) -> None:
    """Проверка устойчивости: результат по отдельным окнам (например, по годам).

    Хорошая стратегия не обязана зарабатывать в каждом окне, но просадка
    должна оставаться в рамках лимита во всех."""
    cfg = _cfg(args)
    frames = _load_data(cfg, args)
    full = Backtester(cfg, frames, trade_start=args.start).run()
    eq = full.equity["equity"]
    ppy = periods_per_year(cfg.exchange.timeframe)
    print(f"{'Окно':<26}{'Доходн.':>10}{'Макс.просадка':>16}{'Sharpe':>9}")
    t = eq.index[0]
    while t < eq.index[-1]:
        t2 = t + pd.Timedelta(days=args.window)
        part = eq[(eq.index >= t) & (eq.index < t2)]
        if len(part) > 5:
            m = metrics.compute(part, ppy)
            print(f"{part.index[0]:%Y-%m-%d} … {part.index[-1]:%Y-%m-%d}"
                  f"{m['total_return'] * 100:>9.2f}%{m['max_drawdown'] * 100:>15.2f}%"
                  f"{m['sharpe']:>9.2f}")
        t = t2


def cmd_run(args) -> None:
    from .live import build_runner

    cfg = _cfg(args)
    if args.mode:
        cfg.exchange.mode = args.mode
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log", encoding="utf-8")],
    )
    logging.info("Старт: режим=%s биржа=%s профиль=%s", cfg.exchange.mode, cfg.exchange.name,
                 cfg.portfolio.profile)
    build_runner(cfg, confirm_live=args.confirm_live).run(once=args.once)


def cmd_ctl(args) -> None:
    """Управление запущенным ботом и просмотр состояния."""
    import json

    from .live import control_path, state_path_for

    cfg = _cfg(args)
    if args.mode:
        cfg.exchange.mode = args.mode
    state = state_path_for(cfg)
    if args.action != "status":
        state.parent.mkdir(parents=True, exist_ok=True)
        control_path(state).write_text(json.dumps({"command": args.action}))
        print(f"Команда '{args.action}' отправлена. Бот выполнит ее в течение "
              f"{cfg.live.poll_seconds} с (смотрите: python -m bot logs).")
        return
    if not state.exists():
        print(f"Нет состояния {state} — бот в этом режиме еще не запускался.")
        return
    st = json.loads(state.read_text())
    print(f"Режим: {cfg.exchange.mode}  биржа: {cfg.exchange.name}  профиль: {cfg.portfolio.profile}")
    print(f"Сохранено: {st.get('saved_at')}   последний бар: {st.get('last_bar')}")
    g = st["decider"]["guard"]
    paused = "ПАУЗА (команда)" if st.get("paused") else (
        "ОСТАНОВЛЕН защитой" if g.get("halted") else
        f"пауза защиты, баров: {g['paused_left']}" if g.get("paused_left") else "торгует")
    print(f"Статус: {paused}")
    eq_path = state.with_name(f"{state.stem}_equity.csv")
    if eq_path.exists():
        last = pd.read_csv(eq_path).iloc[-1]
        print(f"Капитал: {last['equity']:.2f} USDT  (кэш {last['cash']:.2f}, "
              f"пик {last['peak']:.2f}, просадка {last['drawdown'] * 100:.2f}%)")
    print("Позиции:")
    for s, u in st["book"]["units"].items():
        stop = st["decider"]["states"].get(s, {}).get("stop", 0)
        if u > 0:
            print(f"  {s}: {u:.6f}  стоп {stop:.4f}")
    if not any(u > 0 for u in st["book"]["units"].values()):
        print("  нет (всё в USDT)")


def cmd_logs(args) -> None:
    path = Path("logs/bot.log")
    if not path.exists():
        print("Лога пока нет: logs/bot.log")
        return
    for line in path.read_text(encoding="utf-8").splitlines()[-args.n:]:
        print(line)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m bot", description="Консервативный крипто-бот")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--profile", help="core | balanced | ваш профиль из конфига")
    p.add_argument("--timeframe", help="1d | 4h | ...")
    p.add_argument("--exchange", help="binance | bybit | ...")
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="скачать историю свечей")
    d.add_argument("--since", default="2018-01-01")
    d.add_argument("--until")
    d.add_argument("--symbols", nargs="*")
    d.set_defaults(func=cmd_download)

    for name, fn, hlp in (("backtest", cmd_backtest, "прогон по истории"),
                          ("walkforward", cmd_walkforward, "результаты по окнам")):
        b = sub.add_parser(name, help=hlp)
        b.add_argument("--start", help="дата начала торговли (прогрев — на данных до нее)")
        b.add_argument("--end")
        b.add_argument("--synthetic", action="store_true", help="синтетические данные (офлайн)")
        b.add_argument("--bars", type=int, default=2500)
        b.add_argument("--out", default="reports/latest")
        b.add_argument("--window", type=int, default=365, help="длина окна, дней")
        b.set_defaults(func=fn)

    r = sub.add_parser("run", help="paper / testnet / live")
    r.add_argument("--mode", choices=["paper", "testnet", "live"])
    r.add_argument("--once", action="store_true", help="один шаг и выход")
    r.add_argument("--confirm-live", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("ctl", help="управление: status | pause | resume | close-all | stop")
    c.add_argument("action", choices=["status", "pause", "resume", "close-all", "stop"])
    c.add_argument("--mode", choices=["paper", "testnet", "live"])
    c.set_defaults(func=cmd_ctl)

    lg = sub.add_parser("logs", help="последние строки лога")
    lg.add_argument("-n", type=int, default=40)
    lg.set_defaults(func=cmd_logs)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
