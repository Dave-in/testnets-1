"""Исполнение сделок: виртуальное (paper) или через биржу (testnet / live).

Бот ведет собственную "книгу": сколько USDT ему выделено и сколько монет он купил.
Поэтому на аккаунте могут лежать и другие средства — бот трогает только свои.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Book:
    cash: float
    units: dict[str, float] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(u * prices[s] for s, u in self.units.items() if u > 0)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        return {s: (u * prices[s] / eq if eq > 0 else 0.0) for s, u in self.units.items()}


@dataclass
class Fill:
    symbol: str
    side: str
    units: float      # сколько монет реально пришло/ушло
    price: float
    cost: float       # сколько USDT ушло (buy) или пришло (sell), уже с учетом комиссии
    fee: float


class PaperBroker:
    """Виртуальные деньги, реальные цены. Комиссия и проскальзывание — как в бэктесте."""

    def __init__(self, fee: float, slippage: float):
        self.fee = fee
        self.slippage = slippage

    def market_order(self, symbol: str, side: str, units: float, price: float) -> Fill:
        fill = price * (1 + self.slippage if side == "buy" else 1 - self.slippage)
        notional = units * fill
        fee = notional * self.fee
        cost = notional + fee if side == "buy" else notional - fee
        return Fill(symbol, side, units, fill, cost, fee)


class ExchangeBroker:
    """Реальные ордера через ccxt (testnet или live)."""

    def __init__(self, exchange, fee: float):
        self.ex = exchange
        self.fee = fee
        self.ex.load_markets()

    def min_notional(self, symbol: str) -> float:
        lim = self.ex.market(symbol).get("limits", {}).get("cost", {}) or {}
        return float(lim.get("min") or 5.0)

    def market_order(self, symbol: str, side: str, units: float, price: float) -> Fill | None:
        amount = float(self.ex.amount_to_precision(symbol, units))
        if amount <= 0 or amount * price < self.min_notional(symbol):
            log.info("Пропуск %s %s %.8f: меньше минимального ордера", side, symbol, units)
            return None
        order = self.ex.create_order(symbol, "market", side, amount)
        if order.get("status") != "closed" or not order.get("filled"):
            order = self.ex.fetch_order(order["id"], symbol)
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or price)
        notional = float(order.get("cost") or filled * avg)

        base, quote = symbol.split("/")
        base_fee = quote_fee = other_fee = 0.0
        fees = [f for f in (order.get("fees") or [order.get("fee")]) if f and f.get("cost")]
        for f in fees:
            c = float(f["cost"])
            if f.get("currency") == base:
                base_fee += c
            elif f.get("currency") == quote:
                quote_fee += c
            else:  # например, комиссия списана в BNB — в книге не учитываем, только оценка
                other_fee = notional * self.fee
        if not fees:
            quote_fee = notional * self.fee  # биржа не вернула комиссию — оцениваем
        fee_total = quote_fee + base_fee * avg + other_fee
        if side == "buy":
            return Fill(symbol, side, filled - base_fee, avg, notional + quote_fee, fee_total)
        return Fill(symbol, side, filled, avg, notional - quote_fee - base_fee * avg, fee_total)
