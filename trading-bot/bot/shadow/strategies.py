"""Шесть стратегий теневой торговли. У каждой свой источник данных и своя гипотеза.

Все проверяются раз в секунду по каждой монете. Расстояния (стоп, тейк) — в единицах цены,
от цены фактического входа. σ1 — волатильность минутной доходности монеты; на горизонт
N минут волатильность пересчитывается как σ1·√N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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

    def check(self, m, now: int, markets: dict) -> Signal | None:  # pragma: no cover
        raise NotImplementedError


class LiquidationCascade(Strategy):
    key, title = "liq", "Ликвидации"
    idea = ("Каскад принудительных закрытий толкает цену дальше «справедливой». Когда каскад "
            "иссяк — ликвидаций нет и цена перестала обновлять экстремум — входим против "
            "движения и ждём частичного возврата.")
    defaults = {"window_s": 60, "mult": 8.0, "min_usd": 50_000.0, "move_sigma": 2.5,
                "quiet_s": 15, "wait_s": 180, "retrace": 0.5, "sl_frac": 0.3, "time_stop_s": 900}

    def check(self, m, now, markets):
        p, s1, price = self.p, m.sigma1m(), m.price()
        if not s1 or not price:
            return None
        st = self.state.get(m.symbol)
        if st is None:
            liq = m.liq_sum(p["window_s"], now)
            need = max(p["mult"] * m.liq_baseline() * p["window_s"] / 60, p["min_usd"])
            r = m.ret(p["window_s"])
            if liq >= need and abs(r) >= p["move_sigma"] * s1 * math.sqrt(p["window_s"] / 60):
                self.state[m.symbol] = {"t": now, "dir": _sign(r), "start": m.price_ago(p["window_s"])}
            return None
        elapsed = (now - st["t"]) // 1000
        if elapsed > p["wait_s"]:
            del self.state[m.symbol]
            return None
        d = st["dir"]
        ext = m.extreme(int(elapsed) + p["window_s"], d)
        recent = m.extreme(p["quiet_s"], d)
        quiet = now - m.last_liq_ts() >= p["quiet_s"] * 1000
        stalled = recent < ext if d > 0 else recent > ext
        if not (quiet and stalled):
            return None
        del self.state[m.symbol]
        move = abs(ext - st["start"])
        target = ext - d * p["retrace"] * move
        tp = (target - price) * -d
        if move <= 0 or tp <= 0:
            return None                                   # уже откатилось — поздно
        sl = abs(price - (ext + d * p["sl_frac"] * move))
        return Signal(-d, sl, tp, time_stop_s=p["time_stop_s"],
                      reason=f"каскад ликвидаций {'вниз' if d < 0 else 'вверх'}, ход {move / st['start'] * 100:.2f}%")


class BookImbalance(Strategy):
    key, title = "book", "Стакан"
    idea = ("Если в лучших уровнях стакана покупателей заметно больше, чем продавцов, "
            "и агрессивные сделки идут в ту же сторону, ближайшее движение вероятнее в эту "
            "сторону. Входим лимитным ордером (дешёвая комиссия), держим до 5 минут.")
    defaults = {"levels": 10, "obi": 0.5, "persist_s": 5, "micro": 0.2, "flow_s": 30, "flow": 0.2,
                "tp_sigma": 1.0, "sl_sigma": 1.0, "ttl_s": 15, "time_stop_s": 300}

    def check(self, m, now, markets):
        p, book = self.p, m.book
        if not book.ready or not m.sigma1m():
            return None
        obi = book.imbalance(p["levels"])
        d = _sign(obi) if abs(obi) >= p["obi"] else 0
        prev = self.state.get(m.symbol, (0, 0))
        streak = prev[1] + 1 if d and d == prev[0] else (1 if d else 0)
        self.state[m.symbol] = (d, streak)
        if streak < p["persist_s"]:
            return None
        spread = book.spread()
        if not getattr(book, "approx", False):     # в повторе истории микроцены нет
            micro = (book.microprice() - book.mid()) / spread if spread > 0 else 0.0
            if d * micro < p["micro"]:
                return None
        if d * m.flow(p["flow_s"], now) < p["flow"]:
            return None
        self.state[m.symbol] = (0, 0)
        s5 = m.sigma1m() * math.sqrt(5) * m.price()
        limit = book.best_bid() if d > 0 else book.best_ask()
        return Signal(d, p["sl_sigma"] * s5, p["tp_sigma"] * s5, entry="limit", limit_price=limit,
                      limit_ttl_s=p["ttl_s"], time_stop_s=p["time_stop_s"],
                      reason=f"перекос стакана {obi:+.2f}")


class LeadLag(Strategy):
    key, title = "lag", "Отставание от BTC"
    idea = ("Альткоины повторяют движения BTC с задержкой в секунды–минуты. Если BTC резко "
            "сдвинулся, а альт ещё нет — входим в альт в сторону BTC и ждём, пока он догонит.")
    defaults = {"ref": "BTCUSDT", "window_s": 60, "z": 2.5, "gap": 0.6, "beta_minutes": 360,
                "min_beta": 0.3, "tp_frac": 0.8, "sl_sigma": 1.0, "time_stop_s": 180}

    def _beta(self, m, ref) -> float:
        n = len(m.minute_closes)
        cached = self.state.get(("beta", m.symbol))
        if cached and cached[0] == n:
            return cached[1]
        a, b = m.minute_returns(self.p["beta_minutes"]), ref.minute_returns(self.p["beta_minutes"])
        k = min(len(a), len(b))
        beta = 0.0
        if k >= 60:
            a, b = a[-k:], b[-k:]
            var = float(np.var(b))
            beta = float(np.cov(a, b)[0, 1] / var) if var > 0 else 0.0
        self.state[("beta", m.symbol)] = (n, beta)
        return beta

    def check(self, m, now, markets):
        p = self.p
        ref = markets.get(p["ref"])
        if ref is None or m.symbol == p["ref"]:
            return None
        sb, sa, price = ref.sigma1m(), m.sigma1m(), m.price()
        if not sb or not sa or not price:
            return None
        rb = ref.ret(p["window_s"])
        if abs(rb) < p["z"] * sb * math.sqrt(p["window_s"] / 60):
            return None
        beta = self._beta(m, ref)
        if beta < p["min_beta"]:
            return None
        expected = beta * rb
        gap = expected - m.ret(p["window_s"])
        if _sign(gap) != _sign(rb) or abs(gap) < p["gap"] * abs(expected):
            return None
        d = _sign(rb)
        tp = price * (math.exp(abs(gap) * p["tp_frac"]) - 1)
        sl = max(price * abs(gap), p["sl_sigma"] * sa * price)
        return Signal(d, sl, tp, time_stop_s=p["time_stop_s"],
                      reason=f"BTC {rb * 100:+.2f}% за {p['window_s']} с, альт отстал на {gap * 100:.2f}%")


class OpenInterestFlow(Strategy):
    key, title = "oi", "Открытый интерес"
    idea = ("Если цена резко двинулась и одновременно вырос открытый интерес, в рынок пришли "
            "новые позиции, а не закрытия — такое движение чаще продолжается. Входим по "
            "движению с трейлинг-стопом.")
    defaults = {"window_s": 300, "z": 1.5, "oi_min": 0.005, "trail_sigma": 1.5, "time_stop_s": 1800}

    def check(self, m, now, markets):
        p, s1, price = self.p, m.sigma1m(), m.price()
        if not s1 or not price:
            return None
        doi = m.oi_change(p["window_s"], now)
        if doi is None or doi < p["oi_min"]:
            return None
        r = m.ret(p["window_s"])
        if abs(r) < p["z"] * s1 * math.sqrt(p["window_s"] / 60):
            return None
        trail = p["trail_sigma"] * s1 * math.sqrt(5) * price
        return Signal(_sign(r), trail, trail=trail, time_stop_s=p["time_stop_s"],
                      reason=f"цена {r * 100:+.2f}% и открытый интерес {doi * 100:+.1f}% за "
                             f"{p['window_s'] // 60} мин")


class FundingSkew(Strategy):
    key, title = "fund", "Фандинг"
    idea = ("Очень высокая ставка фандинга значит, что рынок перекошен в одну сторону (все в "
            "лонгах или все в шортах). Когда цена начинает идти против толпы, входим вместе с "
            "этим движением и вдобавок получаем фандинг.")
    defaults = {"rate": 0.0005, "window_s": 900, "sl_sigma": 2.0, "tp_sigma": 3.0,
                "time_stop_s": 4 * 3600}

    def check(self, m, now, markets):
        p, s1, price = self.p, m.sigma1m(), m.price()
        fr = m.funding_rate
        if not s1 or not price or abs(fr) < p["rate"]:
            return None
        d = -_sign(fr)                               # против толпы
        vwap = m.vwap(p["window_s"], now)
        if d * (price - vwap) <= 0 or d * m.ret(p["window_s"]) <= 0:
            return None
        s15 = s1 * math.sqrt(p["window_s"] / 60) * price
        return Signal(d, p["sl_sigma"] * s15, p["tp_sigma"] * s15, time_stop_s=p["time_stop_s"],
                      reason=f"фандинг {fr * 100:+.3f}% за период, цена пошла против толпы")


class FadeSpike(Strategy):
    key, title = "fade", "Откат (контроль)"
    idea = ("Контрольная стратегия: после резкого движения за 5 минут, когда оно выдохлось, "
            "входим против него до средней цены. На минутных свечах она проигрывала — "
            "сравним, что покажет исполнение по реальному стакану.")
    defaults = {"window_s": 300, "z": 3.0, "quiet_s": 30, "vwap_s": 900, "sl_sigma": 1.0,
                "time_stop_s": 900}

    def check(self, m, now, markets):
        p, s1, price = self.p, m.sigma1m(), m.price()
        if not s1 or not price:
            return None
        r = m.ret(p["window_s"])
        if abs(r) < p["z"] * s1 * math.sqrt(p["window_s"] / 60):
            return None
        d = _sign(r)
        ext, recent = m.extreme(p["window_s"], d), m.extreme(p["quiet_s"], d)
        if not (recent < ext if d > 0 else recent > ext):
            return None                               # ещё обновляет экстремум
        target = m.vwap(p["vwap_s"], now)
        tp = (price - target) * d
        if tp <= 0:
            return None
        sl = abs(ext - price) + p["sl_sigma"] * s1 * math.sqrt(5) * price
        return Signal(-d, sl, tp, time_stop_s=p["time_stop_s"],
                      reason=f"движение {r * 100:+.2f}% за {p['window_s'] // 60} мин выдохлось")


STRATEGIES = {cls.key: cls for cls in
              (LiquidationCascade, BookImbalance, LeadLag, OpenInterestFlow, FundingSkew, FadeSpike)}
