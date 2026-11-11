"""Исторические свечи: загрузка с биржи через ccxt, кеш в CSV, синтетика для тестов."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(data_dir: str | Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.csv"


def make_exchange(name: str, api_key: str | None = None, secret: str | None = None, sandbox=False):
    import ccxt  # импорт здесь, чтобы бэктест на CSV работал и без ccxt

    params = {"enableRateLimit": True}
    if api_key:
        params.update(apiKey=api_key, secret=secret)
    ex = getattr(ccxt, name)(params)
    if sandbox:
        ex.set_sandbox_mode(True)
    return ex


def _to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since: str, until: str | None = None,
                limit: int = 1000) -> pd.DataFrame:
    """Скачивает свечи постранично начиная с даты since (например '2019-01-01')."""
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    until_ms = exchange.parse8601(f"{until}T00:00:00Z") if until else exchange.milliseconds()
    step_ms = exchange.parse_timeframe(timeframe) * 1000
    rows: list[list] = []
    while since_ms < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + step_ms
        time.sleep(exchange.rateLimit / 1000)
    df = _to_frame(rows)
    df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    # последнюю свечу отбрасываем, если она еще не закрыта
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(milliseconds=step_ms) > now:
        df = df.iloc[:-1]
    return df


def download(exchange_name: str, symbols: list[str], timeframe: str, since: str,
             until: str | None, data_dir: str | Path) -> dict[str, Path]:
    if exchange_name == "binance":
        from .binance_public import BinancePublic

        pub = BinancePublic()
        fetch = lambda sym: pub.history(sym, timeframe, since, until)  # noqa: E731
    else:
        ex = make_exchange(exchange_name)
        ex.load_markets()
        fetch = lambda sym: fetch_ohlcv(ex, sym, timeframe, since, until)  # noqa: E731
    out = {}
    for sym in symbols:
        df = fetch(sym)
        path = cache_path(data_dir, exchange_name, sym, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        out[sym] = path
        print(f"{sym}: {len(df)} свечей {df.index[0]:%Y-%m-%d} … {df.index[-1]:%Y-%m-%d} → {path}")
    return out


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV с колонками ts,open,high,low,close,volume (ts — дата или unix ms)."""
    df = pd.read_csv(path)
    ts = df.columns[0]
    if pd.api.types.is_numeric_dtype(df[ts]):
        idx = pd.to_datetime(df[ts], unit="ms", utc=True)
    else:
        idx = pd.to_datetime(df[ts], utc=True)
    df = df.drop(columns=[ts]).set_index(idx)
    df.index.name = "ts"
    return df[COLUMNS].astype(float).sort_index()


def load_cached(data_dir: str | Path, exchange: str, symbols: list[str], timeframe: str,
                start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        path = cache_path(data_dir, exchange, sym, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"Нет данных для {sym}: {path}. Сначала: python -m bot download"
            )
        out[sym] = clip(load_csv(path), start, end)
    return out


def clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def synthetic(symbols: list[str], bars: int = 2000, timeframe: str = "1d",
              seed: int = 42) -> dict[str, pd.DataFrame]:
    """Синтетический "крипторынок": смена режимов (рост/падение/боковик), обвалы,
    коррелированные активы. Только для проверки механики — не для оценки прибыли."""
    rng = np.random.default_rng(seed)
    freq = {"1d": "1D", "4h": "4h", "1h": "1h"}[timeframe]
    idx = pd.date_range("2019-01-01", periods=bars, freq=freq, tz="UTC")
    scale = {"1d": 1.0, "4h": 1 / np.sqrt(6), "1h": 1 / np.sqrt(24)}[timeframe]

    regimes = np.zeros(bars, dtype=int)  # 0 боковик, 1 рост, 2 падение
    r = 0
    for i in range(bars):
        if rng.random() < 0.01:
            r = rng.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
        regimes[i] = r
    drift = np.select([regimes == 1, regimes == 2], [0.004, -0.004], 0.0) * scale**2
    market = drift + rng.normal(0, 0.035 * scale, bars)
    crash = rng.random(bars) < 0.004
    market[crash] -= rng.uniform(0.15, 0.35, crash.sum())

    out = {}
    for k, sym in enumerate(symbols):
        beta = 1.0 + 0.3 * k
        idio = rng.normal(0, 0.02 * scale * (1 + 0.5 * k), bars)
        rets = beta * market + idio
        close = 100 * (1 + k) * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, bars))
        wick = np.abs(rng.normal(0, 0.015 * scale, (2, bars)))
        high = np.maximum(open_, close) * (1 + wick[0])
        low = np.minimum(open_, close) * (1 - wick[1])
        out[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": rng.uniform(1e3, 1e4, bars)}, index=idx,
        ).rename_axis("ts")
    return out


# --- фандинг бессрочных фьючерсов Binance (USDT-M) ---------------------------

FUNDING_URL = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
               "{pair}/{pair}-fundingRate-{month}.zip")


def funding_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"binance_funding_{symbol.replace('/', '-')}.csv"


def _funding_month(pair: str, month: str) -> pd.DataFrame | None:
    import io
    import zipfile

    import requests

    for attempt in range(4):  # архив иногда рвёт соединение — повторяем
        try:
            r = requests.get(FUNDING_URL.format(pair=pair, month=month), timeout=30)
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if r.status_code != 200:
        return None  # месяца нет: монета ещё не торговалась или месяц не закрыт
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    # время выплаты у биржи бывает со сдвигом в миллисекунды — округляем до минуты
    return pd.DataFrame({"ts": pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.floor("min"),
                         "rate": df["last_funding_rate"].astype(float)})


def download_funding(symbols: list[str], since: str, data_dir: str | Path) -> None:
    """История ставок фандинга из публичного архива data.binance.vision (помесячно)."""
    from concurrent.futures import ThreadPoolExecutor

    months = [m.strftime("%Y-%m") for m in pd.period_range(since, pd.Timestamp.now(), freq="M")]
    for sym in symbols:
        pair = sym.replace("/", "")
        with ThreadPoolExecutor(4) as pool:
            parts = [p for p in pool.map(lambda m: _funding_month(pair, m), months) if p is not None]
        if not parts:
            print(f"{sym}: истории фандинга нет")
            continue
        df = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
        path = funding_path(data_dir, sym)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{sym}: фандинг {len(df)} выплат {df.ts.iloc[0]:%Y-%m-%d} … {df.ts.iloc[-1]:%Y-%m-%d}, "
              f"в среднем {df.rate.mean() * 3 * 365 * 100:.1f}% годовых → {path}")


def load_funding(data_dir: str | Path, symbols: list[str]) -> dict[str, pd.Series]:
    out = {}
    for sym in symbols:
        path = funding_path(data_dir, sym)
        if path.exists():
            df = pd.read_csv(path)
            out[sym] = pd.Series(df["rate"].to_numpy(),
                                 index=pd.DatetimeIndex(pd.to_datetime(
                                     df["ts"], utc=True, format="ISO8601")).floor("min")).sort_index()
    return out
