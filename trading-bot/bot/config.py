"""Загрузка конфига (TOML) в типизированные dataclass'ы."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExchangeConfig:
    name: str = "binance"
    mode: str = "paper"
    timeframe: str = "1d"
    fee: float = 0.001
    slippage: float = 0.0005


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

    @property
    def warmup(self) -> int:
        return max(self.slow_ema, self.momentum_lookback, self.vol_lookback, self.atr_period) + 1


@dataclass
class RiskConfig:
    soft_drawdown: float = 0.025
    hard_drawdown: float = 0.065
    cooldown_bars: int = 10
    resume_drawdown: float = 0.04
    soft_scale: float = 0.25
    hard_stop_action: str = "pause"


@dataclass
class LiveConfig:
    poll_seconds: int = 60
    state_dir: str = "state"


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        exchange=ExchangeConfig(**raw.get("exchange", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        live=LiveConfig(**raw.get("live", {})),
    )
