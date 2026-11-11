"""Загрузка конфига (TOML) в типизированные dataclass'ы."""
from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: пакет tomli с тем же интерфейсом
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExchangeConfig:
    name: str = "binance"
    mode: str = "paper"
    timeframe: str = "1d"
    fee: float = 0.001
    slippage: float = 0.0005
    market: str = "spot"              # spot | futures (бессрочные USDT-M: шорты, плечо, фандинг)
    default_funding_8h: float = 0.0001  # ставка фандинга, если биржа её не отдала (платят лонги)


@dataclass
class PortfolioConfig:
    profile: str = "core"
    initial_capital: float = 10_000.0
    max_gross_exposure: float = 1.0
    monthly_target: float = 0.06  # цель по доходности в месяц (для отчета)
    profiles: dict[str, dict[str, float]] = field(
        default_factory=lambda: {"core": {"ETH/USDT": 1.0}}
    )

    @property
    def caps(self) -> dict[str, float]:
        if self.profile not in self.profiles:
            raise ValueError(f"Профиль '{self.profile}' не найден в [portfolio.profiles]")
        return dict(self.profiles[self.profile])

    @property
    def symbols(self) -> list[str]:
        return list(self.caps)


@dataclass
class StrategyConfig:
    fast_ema: int = 50
    slow_ema: int = 200
    momentum_lookback: int = 90
    atr_period: int = 14
    atr_stop_mult: float = 3.0
    vol_lookback: int = 30
    target_vol: float = 0.12
    rebalance_threshold: float = 0.2
    min_trade: float = 0.01
    reentry_cooldown_bars: int = 5
    max_positions: int = 0  # 0 — без ограничения; N — держим только N сильнейших
    # Шорты (только market = "futures"). С фильтром рынка шортим, лишь когда рынок падает.
    allow_short: bool = False
    short_scale: float = 1.0      # размер шорта относительно лонга (0.5 — вдвое меньше)
    short_atr_stop_mult: float = 0.0  # стоп шорта в ATR; 0 — как у лонга
    risk_per_trade: float = 0.0   # макс. потеря счёта при срабатывании стопа (0.01 = 1%); 0 — выкл
    # Фильтр режима рынка: торгуем, только когда regime_symbol выше своей EMA(regime_ema).
    regime_symbol: str = ""       # например "BTC/USDT"; пусто — фильтр выключен
    regime_ema: int = 200
    regime_exit: bool = True      # True — при плохом режиме выходим из всех позиций
    # Фильтр чистоты тренда: входим, только если за momentum_lookback баров цена шла ровно.
    # Мера (efficiency ratio) = |цена сейчас − цена N баров назад| / сумма модулей изменений
    # за N баров: около 1 — ровное движение, около 0 — «пила». 0 — фильтр выключен.
    min_trend_quality: float = 0.0
    # Учёт собственных ошибок (0 — выключено). Бот ведёт счёт по каждой связке
    # «монета + направление» в единицах риска (R: +1 = заработал столько, сколько ставил на стоп).
    learn_window: int = 0         # если последние N сделок связки в сумме в минусе — отключить её
    learn_block_bars: int = 72    # на сколько баров отключать
    learn_loss_streak: int = 0    # после N убыточных сделок подряд (по всем монетам) резать риск
    learn_risk_cut: float = 0.5   # во сколько раз резать, пока не будет прибыльной сделки
    # Вход: "ema" — тренд по средним; "breakout" — пробой максимума за breakout_lookback баров
    entry_mode: str = "ema"
    breakout_lookback: int = 20
    exit_lookback: int = 0        # > 0: выход при закрытии ниже минимума за N баров

    @property
    def short_atr_mult(self) -> float:
        return self.short_atr_stop_mult or self.atr_stop_mult

    @property
    def warmup(self) -> int:
        return max(self.slow_ema, self.momentum_lookback, self.vol_lookback, self.atr_period,
                   self.regime_ema if self.regime_symbol else 0, self.breakout_lookback) + 1


@dataclass
class RiskConfig:
    soft_drawdown: float = 0.025
    hard_drawdown: float = 0.065
    cooldown_bars: int = 10
    resume_drawdown: float = 0.04
    soft_scale: float = 0.25
    hard_stop_action: str = "pause"   # pause | halt | scale | off (защита выключена)


@dataclass
class LiveConfig:
    poll_seconds: int = 60
    state_dir: str = "state"


@dataclass
class ChallengeConfig:
    """Режим «попыток»: маленький счёт, цель — умножить его в target_mult раз.
    Попытка заканчивается победой (цель) или сливом (осталось меньше bust_fraction),
    после чего начинается новая с тем же стартовым счётом."""
    enabled: bool = False
    target_mult: float = 10.0
    bust_fraction: float = 0.2


@dataclass
class TelegramConfig:
    enabled: bool = False       # токен и chat_id — в окружении / .env, не в конфиге
    daily_report: bool = True   # отчёт раз в сутки (в 00:00 UTC, после закрытия дня)


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        exchange=ExchangeConfig(**raw.get("exchange", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        live=LiveConfig(**raw.get("live", {})),
        telegram=TelegramConfig(**raw.get("telegram", {})),
        challenge=ChallengeConfig(**raw.get("challenge", {})),
    )
