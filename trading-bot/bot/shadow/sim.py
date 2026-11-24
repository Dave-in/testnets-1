"""Виртуальный фьючерсный счёт и исполнение ордеров по реальному стакану.

Правила исполнения (осторожные — лучше недооценить результат, чем переоценить):
  * рыночный ордер исполняется через latency мс после сигнала проходом по уровням стакана;
  * лимитный ордер встаёт в конец очереди: исполнится, только когда по его цене пройдёт
    больше сделок, чем стояло впереди, плюс его собственный объём, или цена пройдёт насквозь;
  * стоп срабатывает по цене сделок и дальше исполняется как рыночный — с проскальзыванием;
  * лимитки и стопы тоже появляются на бирже только через latency мс после решения,
    а очередь перед лимиткой считается в момент, когда она «дошла» до биржи;
  * цены тейка и стопа округляются до шага цены Bybit в худшую для нас сторону;
  * комиссии Bybit по умолчанию: мейкер 0.02%, тейкер 0.055%; фандинг — по ставке биржи.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

_ids = itertools.count(1)


def round_to_tick(price: float, tick: float, up: bool) -> float:
    """Цена, кратная шагу цены биржи: up=True — вверх, False — вниз."""
    if not tick:
        return price
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if up else math.floor(steps + 1e-9)
    return round(steps * tick, 12)


@dataclass
class Order:
    symbol: str
    side: str              # buy | sell
    kind: str              # market | limit | stop
    qty: float
    price: float = 0.0     # цена лимитки или триггер стопа
    purpose: str = "entry"  # entry | tp | sl | time | manual
    created: int = 0
    exec_after: int = 0    # для рыночных: когда исполнять (задержка)
    expire: int = 0        # для лимитных входов: когда снять
    queue_ahead: float = 0.0
    meta: dict = field(default_factory=dict)
    id: int = field(default_factory=lambda: next(_ids))
    active_after: int = 0  # лимитка/стоп ещё «в пути» до биржи до этого момента
    queued: bool = True    # False — очередь на уровне ещё не измерена (лимитка в пути)


@dataclass
class Position:
    symbol: str
    side: int              # +1 лонг, −1 шорт
    qty: float
    entry: float
    opened: int
    sl: float = 0.0
    tp: float = 0.0
    time_stop: int = 0
    trail: float = 0.0     # расстояние трейлинг-стопа (0 — нет)
    best: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    reason: str = ""
    slippage_bps: float = 0.0
    funding_paid_for: int = 0
    realized: float = 0.0      # результат уже закрытых частей позиции (до комиссий)
    closed_qty: float = 0.0
    entry_maker: bool = False  # вход лимиткой (комиссия мейкера) или по рынку (тейкера)
    margin: float = 0.0        # изолированная маржа позиции (0 — счёт без модели маржи)
    liq_price: float = 0.0     # цена ликвидации позиции при этой марже
    tag: str = ""              # метка стратегии (например, пара парного арбитража)


@dataclass
class FillReport:
    """Что произошло — для журнала и уведомлений."""
    symbol: str
    action: str            # open | close
    side: int
    qty: float
    price: float
    fee: float
    reason: str
    slippage_bps: float = 0.0
    pnl: float | None = None
    ret: float | None = None
    maker: bool = False            # этот ордер исполнился как мейкер (лимитка)
    fee_rate: float = 0.0          # ставка комиссии Bybit этого ордера
    gross: float | None = None     # закрытие: результат до комиссий и фандинга
    trade_fees: float | None = None  # закрытие: все комиссии сделки (вход + выход)
    funding: float | None = None   # закрытие: фандинг за время сделки (+ — заплатили)
    entry_maker: bool = False
    entry_fee_rate: float = 0.0


class Account:
    def __init__(self, name: str, capital: float, fees=(0.0002, 0.00055), latency_ms: int = 250,
                 leverage: float | None = None, maintenance: float = 0.005):
        """leverage — настройка плеча на бирже при изолированной марже (как на Bybit):
        маржа позиции = сумма / плечо, ликвидация — когда убыток съел маржу до поддерживающей.
        None — без модели маржи (только общие пределы суммы позиций)."""
        self.name = name
        self.start = capital
        self.cash = capital
        self.maker_fee, self.taker_fee = fees
        self.latency = latency_ms
        self.leverage = leverage
        self.maintenance = maintenance
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.stats = {"closed": 0, "wins": 0, "realized": 0.0, "fees": 0.0, "funding": 0.0,
                      "slippage_bps": 0.0, "opened": 0}
        self.peak = capital
        self.cooldown: dict[str, int] = {}
        self.events: list[FillReport] = []      # движок забирает и рассылает

    # --- оценка ----------------------------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.side * p.qty * (prices.get(s, p.entry) - p.entry)
                               for s, p in self.positions.items())

    def gross(self, prices: dict[str, float]) -> float:
        return sum(p.qty * prices.get(s, p.entry) for s, p in self.positions.items())

    def pending_entries(self) -> list[Order]:
        return [o for o in self.orders if o.purpose == "entry"]

    def committed(self, prices: dict[str, float]) -> float:
        """Объём открытых позиций плюс ордеров на вход, которые ещё могут исполниться."""
        return self.gross(prices) + sum(o.qty * (o.price or prices.get(o.symbol, 0.0))
                                        for o in self.pending_entries())

    def used_margin(self, prices: dict[str, float]) -> float:
        """Маржа открытых позиций и ордеров на вход (только при модели изолированной маржи)."""
        if not self.leverage:
            return 0.0
        return (sum(p.margin for p in self.positions.values())
                + sum(o.qty * (o.price or prices.get(o.symbol, 0.0)) for o in self.pending_entries())
                / self.leverage)

    def liquidation_price(self, side: int, entry: float) -> float:
        lev = self.leverage
        return entry * (1 - side / lev + side * self.maintenance) if lev else 0.0

    def busy(self, symbol: str, now: int) -> bool:
        return (symbol in self.positions or self.cooldown.get(symbol, 0) > now
                or any(o.symbol == symbol and o.purpose == "entry" for o in self.orders))

    # --- ордера ---------------------------------------------------------------------
    def submit(self, order: Order, book, now: int) -> Order:
        order.created = now
        if order.kind == "market":
            order.exec_after = now + self.latency
        elif order.kind in ("limit", "stop"):
            # ордер доходит до биржи через latency; лимитка встаёт в конец очереди уровня,
            # который будет в стакане в момент прихода
            order.active_after = now + self.latency
            order.queued = order.kind != "limit"
        self.orders.append(order)
        return order

    def _activate(self, o: Order, book, now: int) -> bool:
        """Дошёл ли ордер до биржи; для лимитки при приходе измеряем очередь впереди."""
        if o.active_after > now:
            return False
        if not o.queued:
            o.queue_ahead = book.size_at("bid" if o.side == "buy" else "ask", o.price)
            o.queued = True
        return True

    def cancel(self, symbol: str, purposes: tuple[str, ...]) -> None:
        self.orders = [o for o in self.orders if not (o.symbol == symbol and o.purpose in purposes)]

    def on_trade(self, symbol: str, price: float, qty: float, taker_buy: bool, book, now: int) -> None:
        """Лента сделок: продвигаем очередь лимиток и проверяем стопы."""
        for o in list(self.orders):
            if o.symbol != symbol or o not in self.orders:
                continue
            if o.kind in ("limit", "stop") and not self._activate(o, book, now):
                continue                          # ордер ещё не дошёл до биржи
            if o.kind == "limit":
                # нашу покупку исполняют продавцы-тейкеры, нашу продажу — покупатели
                if (o.side == "buy" and taker_buy) or (o.side == "sell" and not taker_buy):
                    continue
                through = price < o.price if o.side == "buy" else price > o.price
                if price == o.price:
                    o.queue_ahead -= qty
                if through or o.queue_ahead <= -o.qty:
                    self.orders.remove(o)
                    self._fill(o, o.price, o.qty, maker=True, now=now, mid=book.mid(), book=book)
            elif o.kind == "stop":
                hit = price <= o.price if o.side == "sell" else price >= o.price
                if hit:
                    o.kind, o.exec_after = "market", now + self.latency

    def process(self, books: dict, now: int) -> None:
        """Исполнить рыночные ордера, у которых прошла задержка; снять просроченные лимитки."""
        for o in list(self.orders):
            if o not in self.orders:
                continue
            if o.kind == "market" and now >= o.exec_after:
                book = books[o.symbol]
                avg, filled = book.walk(o.side, o.qty)
                if filled <= 0:
                    continue                      # стакан пуст — ждём следующего шага
                self.orders.remove(o)
                self._fill(o, avg, filled, maker=False, now=now, mid=o.meta.get("mid0", book.mid()),
                           book=book)
            elif o.kind == "limit" and o.expire and now >= o.expire:
                self.orders.remove(o)
            elif o.kind in ("limit", "stop") and o.symbol in books:
                self._activate(o, books[o.symbol], now)

    # --- исполнение ------------------------------------------------------------------
    def _fill(self, o: Order, price: float, qty: float, maker: bool, now: int, mid: float,
              book=None) -> None:
        rate = self.maker_fee if maker else self.taker_fee
        fee = price * qty * rate
        sign = 1 if o.side == "buy" else -1
        slip = sign * (price / mid - 1) * 1e4 if mid else 0.0   # + — хуже ожидаемого
        self.cash -= fee
        self.stats["fees"] += fee
        pos = self.positions.get(o.symbol)
        if o.purpose == "entry":
            if pos is not None:
                return
            m = o.meta
            pos = Position(o.symbol, sign, qty, price, now, reason=m.get("reason", ""),
                           time_stop=now + int(m.get("time_stop_s", 0) * 1000) if m.get("time_stop_s") else 0,
                           trail=m.get("trail", 0.0), best=price, fees=fee, slippage_bps=slip,
                           entry_maker=maker, tag=m.get("tag", ""))
            if self.leverage:                     # изолированная маржа, как на Bybit
                pos.margin = price * qty / self.leverage
                pos.liq_price = self.liquidation_price(sign, price)
            tick = m.get("tick", 0.0)
            close_side = "sell" if sign > 0 else "buy"
            # защитные ордера ставятся после исполнения входа и тоже идут до биржи latency мс
            if m.get("sl_dist"):
                pos.sl = round_to_tick(price - sign * m["sl_dist"], tick, up=sign < 0)
                self.submit(Order(o.symbol, close_side, "stop", qty, pos.sl, "sl"), book, now)
            if m.get("tp_dist"):
                pos.tp = round_to_tick(price + sign * m["tp_dist"], tick, up=sign > 0)
                self.submit(Order(o.symbol, close_side, "limit", qty, pos.tp, "tp"), book, now)
            self.positions[o.symbol] = pos
            self.stats["opened"] += 1
            self.stats["slippage_bps"] += slip
            self.events.append(FillReport(o.symbol, "open", sign, qty, price, fee, pos.reason, slip,
                                          maker=maker, fee_rate=rate))
            return
        if pos is None:
            return
        pnl_gross = pos.side * (price - pos.entry) * qty
        self.cash += pnl_gross
        pos.fees += fee
        if qty < pos.qty * (1 - 1e-9):
            # стакана не хватило: закрыта часть, остаток закрываем следующим рыночным ордером
            pos.realized += pnl_gross
            pos.closed_qty += qty
            pos.qty -= qty
            for other in self.orders:
                if other.symbol == o.symbol and other.purpose in ("sl", "tp"):
                    other.qty = pos.qty
            if o.kind == "market":
                self.orders.append(Order(o.symbol, o.side, "market", pos.qty, purpose=o.purpose,
                                         created=now, exec_after=now + self.latency, meta=o.meta))
            return
        pnl_gross += pos.realized
        qty += pos.closed_qty
        net = pnl_gross - pos.fees - pos.funding
        self.stats["realized"] += net
        self.stats["closed"] += 1
        self.stats["wins"] += net > 0
        self.cancel(o.symbol, ("sl", "tp", "time"))
        del self.positions[o.symbol]
        self.cooldown[o.symbol] = now + 5 * 60 * 1000
        self.events.append(FillReport(o.symbol, "close", pos.side, qty, price, fee, o.purpose,
                                      slip, net, net / (pos.entry * qty), maker=maker,
                                      fee_rate=rate, gross=pnl_gross, trade_fees=pos.fees,
                                      funding=pos.funding, entry_maker=pos.entry_maker,
                                      entry_fee_rate=self.maker_fee if pos.entry_maker
                                      else self.taker_fee))

    def close(self, symbol: str, book, now: int, purpose: str) -> None:
        pos = self.positions.get(symbol)
        if pos is None or any(o.symbol == symbol and o.kind == "market" for o in self.orders):
            return
        self.cancel(symbol, ("sl", "tp", "time"))
        self.orders.append(Order(symbol, "sell" if pos.side > 0 else "buy", "market", pos.qty,
                                 purpose=purpose, created=now, exec_after=now + self.latency,
                                 meta={"mid0": book.mid()}))

    # --- раз в секунду ------------------------------------------------------------------
    def check_liquidation(self, markets: dict, now: int, maintenance: float = 0.005) -> bool:
        """Если капитал опустился до поддерживающей маржи — биржа закрыла бы всё."""
        prices = {s: m.price() for s, m in markets.items()}
        if self.positions and self.equity(prices) <= maintenance * self.gross(prices):
            for s in list(self.positions):
                self.close(s, markets[s].book, now, "liquidation")
            return True
        return False

    def housekeeping(self, markets: dict, now: int) -> None:
        for s, pos in list(self.positions.items()):
            m = markets[s]
            price = m.price()
            if pos.liq_price:                     # изолированная маржа: ликвидация по марк-цене
                mark = m.mark_price or price
                if (mark - pos.liq_price) * pos.side <= 0:
                    self.close(s, m.book, now, "liquidation")
                    continue
            if pos.time_stop and now >= pos.time_stop:
                self.close(s, m.book, now, "time")
                continue
            if pos.trail:
                pos.best = max(pos.best, price) if pos.side > 0 else min(pos.best, price)
                new_sl = pos.best - pos.side * pos.trail
                if (pos.side > 0 and new_sl > pos.sl) or (pos.side < 0 and (not pos.sl or new_sl < pos.sl)):
                    pos.sl = new_sl
                    for o in self.orders:
                        if o.symbol == s and o.purpose == "sl":
                            o.price = new_sl
            # фандинг: начисляем один раз за каждую выплату
            if m.next_funding and now >= m.next_funding and pos.funding_paid_for != m.next_funding:
                if pos.opened < m.next_funding:
                    pay = pos.side * pos.qty * (m.mark_price or price) * m.funding_rate
                    self.cash -= pay
                    pos.funding += pay
                    self.stats["funding"] += pay
                pos.funding_paid_for = m.next_funding

    def to_dict(self) -> dict:
        return {"cash": self.cash, "peak": self.peak, "stats": self.stats, "cooldown": self.cooldown,
                "positions": {s: asdict(p) for s, p in self.positions.items()}}

    def load_dict(self, d: dict) -> None:
        self.cash = d.get("cash", self.cash)
        self.peak = d.get("peak", self.peak)
        self.stats.update(d.get("stats", {}))
        self.cooldown = {k: int(v) for k, v in d.get("cooldown", {}).items()}
        self.positions = {s: Position(**p) for s, p in d.get("positions", {}).items()}
        for s, p in self.positions.items():   # восстановить защитные ордера
            close_side = "sell" if p.side > 0 else "buy"
            if p.sl:
                self.orders.append(Order(s, close_side, "stop", p.qty, p.sl, "sl"))
            if p.tp:
                # очередь на уровне тейка после перезапуска неизвестна: считаем, что мы в её
                # конце — тейк исполнится, только если цена пройдёт уровень насквозь
                self.orders.append(Order(s, close_side, "limit", p.qty, p.tp, "tp",
                                         queue_ahead=float("inf")))
