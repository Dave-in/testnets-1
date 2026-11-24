"""Общие классы стратегий: сигнал, сигнал пары и базовый класс стратегии."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Signal:
    side: int                  # +1 лонг, −1 шорт
    sl_dist: float             # расстояние до стопа в цене
    tp_dist: float = 0.0       # расстояние до тейка (0 — без тейка)
    entry: str = "market"      # market | limit
    limit_price: float = 0.0
    limit_ttl_s: int = 15
    time_stop_s: int = 0
    trail: float = 0.0         # трейлинг-стоп (вместо фиксированного), в цене
    reason: str = ""
    features: dict = field(default_factory=dict)   # числа, на которых основан сигнал (для анализа)


@dataclass
class PairSignal:
    """Сделка из двух ног (парный арбитраж): legs — [(монета, Signal), (монета, Signal)].
    ratio — сумма второй ноги к сумме первой (β); risk_frac — какой доле суммы первой ноги
    равен убыток при стопе по спреду (для расчёта размера из риска на сделку)."""
    legs: list
    ratio: float
    risk_frac: float
    tag: str
    reason: str = ""


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


class Strategy:
    key = ""
    title = ""
    idea = ""
    defaults: dict = {}

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.key}: неизвестные параметры {sorted(unknown)}")
        self.p = {**self.defaults, **params}
        self.state: dict = {}
        self.obs: dict = {}
        self.cur: dict = {}                 # условия входа по монете последней проверки
        self.allowed: set | None = None     # монеты, которыми бот может торговать (задаёт движок)

    def check(self, m, now: int, markets: dict) -> Signal | PairSignal | None:  # pragma: no cover
        raise NotImplementedError

    def manage(self, acc, markets: dict, now: int):
        """Раз в секунду: какие открытые позиции закрыть по сигналу стратегии —
        [(монета, причина), …]. По умолчанию выход только по тейку, стопу и времени."""
        return ()

    # --- калибровка: «почти сработавшие» сигналы ------------------------------------------
    def observe(self, metric: str, value: float, threshold: float, symbol: str) -> None:
        """Запомнить значение условия входа. За минуту хранится максимум по всем монетам —
        по этим максимумам видно, как близко бот подходит к порогу и где порог поставить."""
        if value is None or not math.isfinite(value):
            return
        self.cur[metric] = float(value)         # значения по текущей монете — признаки сигнала
        cur = self.obs.get(metric)
        if cur is None or value > cur[0]:
            self.obs[metric] = (float(value), float(threshold), symbol)

    def take_observations(self) -> dict:
        out, self.obs = self.obs, {}
        return out
