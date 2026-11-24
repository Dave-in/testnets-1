"""Краткосрочные стратегии с математической моделью: возврат к VWAP (процесс
Орнштейна–Уленбека), пробой сжатия, парный арбитраж (коинтеграция), поток ордеров
(OFI Конта + лямбда Кайла) и агрессивный возврат к VWAP на мем-токенах.

Все проверяются раз в секунду. Расстояния стопа и тейка — в единицах цены от цены входа.
Тяжёлые расчёты (регрессии) кешируются и пересчитываются раз в минуту или реже.
"""
from __future__ import annotations

import math

import numpy as np

from .base import PairSignal, Signal, Strategy, _sign


def _ou_fit(closes: np.ndarray, sma_n: int):
    """Отклонение x = ln P − ln SMA, регрессия Δx = a + b·x₋₁ → θ = −b, полупериод ln2/θ (мин),
    стандартное отклонение x и отношение дисперсий VR(q). None — мало данных."""
    if len(closes) < sma_n + 30:
        return None
    logc = np.log(closes)
    sma = np.convolve(closes, np.ones(sma_n) / sma_n, mode="valid")
    x = logc[sma_n - 1:] - np.log(sma)
    x0, dx = x[:-1], np.diff(x)
    var = float(np.var(x0))
    if var <= 0:
        return None
    b = float(np.cov(x0, dx)[0, 1] / var)
    theta = -b
    half_life = math.log(2) / theta if theta > 0 else math.inf
    return half_life, float(np.std(x)), logc


def variance_ratio(logc: np.ndarray, q: int) -> float:
    """VR(q) = Var(r_q) / (q·Var(r_1)) по перекрывающимся доходностям. < 1 — склонность к откатам."""
    r1 = np.diff(logc)
    rq = logc[q:] - logc[:-q]
    v1 = float(np.var(r1))
    return float(np.var(rq)) / (q * v1) if v1 > 0 else 1.0


class VWAPReversion(Strategy):
    key, title = "vwap", "Возврат к VWAP"
    idea = ("Цена колеблется вокруг средневзвешенной (VWAP) и возвращается к ней — но только в "
            "«боковом» режиме. Режим проверяем математически: регрессия процесса "
            "Орнштейна–Уленбека даёт полупериод возврата, отношение дисперсий — склонность к "
            "откатам. Входим лимиткой при сильном отклонении, тейк лимиткой на VWAP.")
    defaults = {"vwap_s": 900, "lookback_min": 120, "sma_n": 15, "z_entry": 2.0, "z_stop": 3.5,
                "max_half_life": 10.0, "max_vr": 0.9, "vr_q": 5, "ttl_s": 20, "max_hold_min": 20}

    def _regime(self, m):
        n = len(m.minute_closes)
        cached = self.state.get(("ou", m.symbol))
        if cached and cached[0] == n:
            return cached[1]
        p = self.p
        closes = np.fromiter(m.minute_closes, float)[-(p["lookback_min"] + p["sma_n"]):]
        fit = _ou_fit(closes, p["sma_n"])
        res = None
        if fit is not None:
            hl, std_x, logc = fit
            res = (hl, std_x, variance_ratio(logc[-(p["lookback_min"] + 1):], p["vr_q"]))
        self.state[("ou", m.symbol)] = (n, res)
        return res

    def check(self, m, now, markets):
        p, price, book = self.p, m.price(), m.book
        if not price or not book.ready:
            return None
        reg = self._regime(m)
        if reg is None:
            return None
        hl, std_x, vr = reg
        self.observe("скорость возврата 1/полупериод (1/мин)", 1 / hl if hl > 0 else 0.0,
                      1 / p["max_half_life"], m.symbol)
        if not (0 < hl <= p["max_half_life"] and vr < p["max_vr"] and std_x > 0):
            return None
        vwap = m.vwap(p["vwap_s"], now)
        z = math.log(price / vwap) / std_x
        self.observe("|z| отклонения от VWAP", abs(z), p["z_entry"], m.symbol)
        if abs(z) < p["z_entry"] or abs(z) >= p["z_stop"]:
            return None
        d = -_sign(z)                                   # против отклонения — к средней
        tp = abs(price - vwap)
        sl = (p["z_stop"] - abs(z)) * std_x * price
        limit = book.best_bid() if d > 0 else book.best_ask()
        hold = min(2 * hl, p["max_hold_min"]) * 60
        return Signal(d, sl, tp, entry="limit", limit_price=limit, limit_ttl_s=p["ttl_s"],
                      time_stop_s=int(hold),
                      reason=f"отклонение от VWAP {z:+.1f}σ, полупериод {hl:.0f} мин, VR {vr:.2f}")


class MemeVWAP(VWAPReversion):
    key, title = "meme", "Мемы: возврат к VWAP"
    idea = ("Тот же возврат к VWAP, но только на мем-токенах и агрессивнее: вход уже при 1.5σ, "
            "режим возврата мягче, держим дольше. Мемы часто дают резкие выбросы и откаты.")
    defaults = {**VWAPReversion.defaults, "z_entry": 1.5, "z_stop": 3.0, "max_half_life": 15.0,
                "max_vr": 0.95, "max_hold_min": 30}


class SqueezeBreakout(Strategy):
    key, title = "squeeze", "Пробой сжатия"
    idea = ("После затишья — когда полосы Боллинджера сжались до 10% самых узких значений за 6 ч — "
            "движение разгоняется. Входим в сторону пробоя 30-минутного диапазона, если объём "
            "минуты выше обычного на 2σ. Стоп за уровнем, тейк 3 ATR, трейлинг.")
    defaults = {"bb_n": 20, "bw_hist_min": 360, "bw_pct": 10.0, "squeeze_recent_min": 10,
                "range_s": 1800, "vol_z": 2.0, "vol_hist_min": 60, "atr_n": 14, "sl_atr": 1.0,
                "tp_atr": 3.0, "trail_atr": 1.5, "time_stop_s": 1800}

    def _bands(self, m):
        n = len(m.minute_closes)
        cached = self.state.get(("bw", m.symbol))
        if cached and cached[0] == n:
            return cached[1]
        p = self.p
        need = p["bw_hist_min"] + p["bb_n"]
        res = None
        if n >= need:
            c = np.fromiter(m.minute_closes, float)[-need:]
            win = np.lib.stride_tricks.sliding_window_view(c, p["bb_n"])
            bw = 4 * win.std(axis=1) / win.mean(axis=1)
            thr = float(np.percentile(bw, p["bw_pct"]))
            recent = float(bw[-p["squeeze_recent_min"]:].min())
            atr = float(np.mean(np.abs(np.diff(c[-(p["atr_n"] + 1):]))))
            res = (recent, thr, atr)
        self.state[("bw", m.symbol)] = (n, res)
        return res

    def check(self, m, now, markets):
        p, price = self.p, m.price()
        if not price or len(m.sec) < p["range_s"] + 1:
            return None
        bands = self._bands(m)
        if bands is None or len(m.minute_usd) < 30:
            return None
        recent_bw, thr, atr = bands
        self.observe("сжатие: порог / ширина полос", thr / recent_bw if recent_bw else 0.0, 1.0,
                     m.symbol)
        if recent_bw > thr or atr <= 0:
            return None
        hi = m.extreme_range(60, p["range_s"], 1)
        lo = m.extreme_range(60, p["range_s"], -1)
        d = 1 if price > hi else (-1 if price < lo else 0)
        if not d:
            return None
        vols = np.fromiter(m.minute_usd, float)[-p["vol_hist_min"]:]
        cur = sum(usd for s, _, usd, _ in reversed(m.trades) if s > now // 1000 - 60)
        sd = float(vols.std())
        vz = (cur - float(vols.mean())) / sd if sd > 0 else 0.0
        self.observe("объём минуты, z", vz, p["vol_z"], m.symbol)
        if vz < p["vol_z"]:
            return None
        level = hi if d > 0 else lo
        sl = abs(price - (level - d * p["sl_atr"] * atr))
        return Signal(d, sl, p["tp_atr"] * atr, trail=p["trail_atr"] * atr,
                      time_stop_s=p["time_stop_s"],
                      reason=f"пробой {'вверх' if d > 0 else 'вниз'} после сжатия, объём {vz:+.1f}σ")


def adf_t(e: np.ndarray) -> float:
    """t-статистика теста Дики–Фуллера без лагов: Δe = γ·e₋₁ + ε. Чем меньше, тем стационарнее."""
    e0, de = e[:-1], np.diff(e)
    denom = float(np.dot(e0, e0))
    if denom <= 0:
        return 0.0
    g = float(np.dot(e0, de)) / denom
    resid = de - g * e0
    se = math.sqrt(float(np.dot(resid, resid)) / max(len(de) - 1, 1) / denom)
    return g / se if se > 0 else 0.0


class PairsCointegration(Strategy):
    key, title = "pairs", "Парный арбитраж"
    idea = ("Две монеты, которые ходят вместе (корреляция ≥ 0.8, стационарный спред по тесту "
            "Энгла–Грейнджера), временно разошлись — покупаем отставшую и продаём убежавшую. "
            "Нейтрально к общему движению рынка. Выход, когда спред вернулся к средней.")
    defaults = {"refresh_min": 60, "lookback_min": 360, "z_min": 120, "corr_min": 0.8,
                "adf_t": -3.34, "max_pairs": 6, "z_entry": 2.0, "z_exit": 0.5, "z_stop": 4.0,
                "max_hold_min": 120, "ttl_s": 20}

    def _fit(self, a, b):
        """(α, β, t-статистика ADF, корреляция) по минутным ценам за lookback, или None."""
        p = self.p
        k = min(len(a.minute_closes), len(b.minute_closes), p["lookback_min"] + 1)
        if k < 120:
            return None
        la = np.log(np.fromiter(a.minute_closes, float)[-k:])
        lb = np.log(np.fromiter(b.minute_closes, float)[-k:])
        ra, rb = np.diff(la), np.diff(lb)
        if ra.std() == 0 or rb.std() == 0:
            return None
        corr = float(np.corrcoef(ra, rb)[0, 1])
        beta, alpha = np.polyfit(lb, la, 1)
        return float(alpha), float(beta), adf_t(la - alpha - beta * lb), corr

    def _refresh(self, markets, now):
        last = self.state.get("pairs_at", 0)
        if now - last < self.p["refresh_min"] * 60_000 and "pairs" in self.state:
            return
        p = self.p
        syms = sorted(s for s, m in markets.items()
                      if m.tradable and (self.allowed is None or s in self.allowed))
        found = []
        for i, sa in enumerate(syms):
            for sb in syms[i + 1:]:
                f = self._fit(markets[sa], markets[sb])
                if f and f[3] >= p["corr_min"] and f[2] <= p["adf_t"] and f[1] > 0:
                    found.append((f[2], sa, sb, f))
        pairs, used = {}, set()
        for t, sa, sb, f in sorted(found):          # самые стационарные спреды — первыми
            if sa in used or sb in used or len(pairs) >= p["max_pairs"]:
                continue
            pairs[sa] = (sb, f[0], f[1])
            used.update((sa, sb))
        self.state["pairs"], self.state["pairs_at"] = pairs, now
        self.observe("найдено пар", float(len(pairs)), 1.0, ",".join(pairs) or "—")

    def _z(self, a, b, alpha, beta):
        p = self.p
        k = min(len(a.minute_closes), len(b.minute_closes), p["z_min"])
        if k < 30 or not a.price() or not b.price():
            return None
        la = np.log(np.fromiter(a.minute_closes, float)[-k:])
        lb = np.log(np.fromiter(b.minute_closes, float)[-k:])
        e = la - alpha - beta * lb
        sd = float(e.std())
        if sd <= 0:
            return None
        now_e = math.log(a.price()) - alpha - beta * math.log(b.price())
        return (now_e - float(e.mean())) / sd, sd

    def check(self, m, now, markets):
        self._refresh(markets, now)
        pair = self.state["pairs"].get(m.symbol)
        if pair is None:
            return None
        sb, alpha, beta = pair
        b = markets.get(sb)
        if b is None or not m.book.ready or not b.book.ready:
            return None
        res = self._z(m, b, alpha, beta)
        if res is None:
            return None
        z, sd = res
        p = self.p
        self.observe("|z| спреда пары", abs(z), p["z_entry"], f"{m.symbol}/{sb}")
        if abs(z) < p["z_entry"] or abs(z) >= p["z_stop"]:
            return None
        d = -_sign(z)                         # z > 0: первая монета дорогая → шорт её, лонг второй
        hold = p["max_hold_min"] * 60
        legs = []
        for mk, side in ((m, d), (b, -d)):
            pr = mk.price()
            limit = mk.book.best_bid() if side > 0 else mk.book.best_ask()
            # аварийный стоп каждой ноги по цене; основной выход — по спреду (manage)
            legs.append((mk.symbol, Signal(side, max(2 * p["z_stop"] * sd, 0.01) * pr, 0.0,
                                           entry="limit", limit_price=limit,
                                           limit_ttl_s=p["ttl_s"], time_stop_s=hold,
                                           reason=f"спред {m.symbol}/{sb} {z:+.1f}σ, β {beta:.2f}")))
        return PairSignal(legs, ratio=beta, risk_frac=(p["z_stop"] - abs(z)) * sd,
                          tag=f"{m.symbol}/{sb}", reason=f"спред {z:+.1f}σ")

    def _params(self, sa, sb, markets):
        pair = self.state.get("pairs", {}).get(sa)
        if pair and pair[0] == sb:
            return pair[1], pair[2]
        f = self._fit(markets[sa], markets[sb])           # пара выпала из списка (или перезапуск)
        return (f[0], f[1]) if f else None

    def manage(self, acc, markets, now):
        """Выход по спреду: вернулся к ±z_exit — закрыть обе ноги; ушёл за z_stop — стоп.
        Если лимитка второй ноги не исполнилась — добрать её по рынку
        («open», монета, Signal, сумма, метка, первая нога); не вышло — закрыть первую."""
        out = []
        tags = {}
        for s, pos in acc.positions.items():
            if pos.tag:
                tags.setdefault(pos.tag, []).append(s)
        for tag, held in tags.items():
            sa, sb = tag.split("/")
            params = self._params(sa, sb, markets)
            if len(held) == 1:
                lone = acc.positions[held[0]]
                other = sb if held[0] == sa else sa
                if any(o.symbol == other and o.purpose == "entry" for o in acc.orders):
                    continue                      # вторая нога ещё в пути
                om = markets.get(other)
                if params is None or om is None or not om.price():
                    out.append((held[0], "pair_leg"))
                    continue
                beta = abs(params[1]) or 1.0
                notional = lone.qty * lone.entry * (beta if held[0] == sa else 1 / beta)
                hold = max((lone.time_stop - now) // 1000, 60) if lone.time_stop else \
                    self.p["max_hold_min"] * 60
                sig = Signal(-lone.side, 0.02 * om.price(), 0.0, time_stop_s=int(hold),
                             reason=f"вторая нога пары {tag} по рынку")
                out.append(("open", other, sig, notional, tag, held[0]))
                continue
            if params is None:
                continue
            res = self._z(markets[sa], markets[sb], *params)
            if res is None:
                continue
            z = res[0]
            if abs(z) <= self.p["z_exit"]:
                out += [(sa, "signal"), (sb, "signal")]
            elif abs(z) >= self.p["z_stop"]:
                out += [(sa, "sl"), (sb, "sl")]
        return out


class OrderFlowImbalance(Strategy):
    key, title = "ofi", "Поток ордеров"
    idea = ("Дисбаланс изменений лучших заявок стакана (OFI, Cont–Kukanov–Stoikov) за последние "
            "секунды предсказывает ближайшее движение. Силу связи для каждой монеты оцениваем "
            "регрессией хода цены за 30 с на OFI за 10 с (лямбда Кайла) по последнему часу. "
            "Входим лимиткой, когда прогноз хода в 3 раза больше издержек сделки.")
    defaults = {"ofi_window_s": 10, "horizon_s": 30, "fit_every_s": 60, "fit_window_s": 3600,
                "min_obs": 600, "min_r2": 0.05, "ttl_s": 5, "sl_sigma": 1.0, "time_stop_s": 120}

    def _fit(self, m, now):
        p = self.p
        cached = self.state.get(("fit", m.symbol))
        if cached and now - cached[0] < p["fit_every_s"] * 1000:
            return cached[1]
        res = None
        k = min(len(m.ofi_sec), len(m.sec), p["fit_window_s"])
        if k >= p["min_obs"] and m.top_depth > 0:
            ofi = np.fromiter(m.ofi_sec, float)[-k:] / m.top_depth
            px = np.log(np.fromiter(m.sec, float)[-k:])
            w, h = p["ofi_window_s"], p["horizon_s"]
            x = np.convolve(ofi, np.ones(w), mode="valid")[:-h]     # OFI за w с, кончая в t
            y = px[w - 1 + h:] - px[w - 1:-h]                         # ход цены за h с после t
            if len(x) > 100 and x.std() > 0:
                lam, a = np.polyfit(x, y, 1)
                pred = a + lam * x
                ss = float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else 0.0
                res = (float(lam), float(a), r2)
        self.state[("fit", m.symbol)] = (now, res)
        return res

    def check(self, m, now, markets):
        p, price, book = self.p, m.price(), m.book
        if not price or not book.ready or not m.sigma1m():
            return None
        fit = self._fit(m, now)
        if fit is None:
            return None
        lam, a, r2 = fit
        self.observe("R² связи OFI → цена", r2, p["min_r2"], m.symbol)
        if r2 < p["min_r2"] or lam <= 0:
            return None
        recent = list(m.ofi_sec)[-p["ofi_window_s"]:]
        x_now = sum(recent) / m.top_depth
        pred = a + lam * x_now                       # прогноз лог-хода за horizon_s
        d = _sign(pred)
        if not d:
            return None
        limit = book.best_bid() if d > 0 else book.best_ask()
        return Signal(d, p["sl_sigma"] * m.sigma1m() * price, abs(pred) * price, entry="limit",
                      limit_price=limit, limit_ttl_s=p["ttl_s"], time_stop_s=p["time_stop_s"],
                      reason=f"поток ордеров: прогноз {pred * 100:+.2f}% за {p['horizon_s']} с, "
                             f"R² {r2:.2f}")


SHORT_TERM = (VWAPReversion, SqueezeBreakout, PairsCointegration, OrderFlowImbalance, MemeVWAP)
