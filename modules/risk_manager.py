from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    CAPITAL_RISK_PER_TRADE,
    MAX_SIMULTANEOUS_POSITIONS,
    DAILY_DRAWDOWN_KILL_PCT,
    ANTI_LIQ_MARGIN_THRESHOLD,
    MAX_LOSS_PER_TRADE_USD,
    MAX_MARGIN_PER_TRADE_USD,
    MAX_NOTIONAL_USD,
    LEVERAGE_LOW,
    LEVERAGE_MEDIUM,
    LEVERAGE_HIGH,
    CONVICTION_MEDIUM_THRESH,
    CONVICTION_HIGH_THRESH,
    ATR_STOP_MULT,
    ATR_TP_MULT,
    LOG_DIR,
    LOG_LEVEL,
)
from config.logging_config import setup_logging

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="risk_manager")


@dataclass
class Position:
    symbol:        str
    direction:     str
    entry_price:   float
    quantity:      float
    leverage:      int
    stop_loss:     float
    take_profit:   float
    partial_close: float
    margin_used:   float
    opened_at:     int
    is_open:       bool = True
    partial_closed: bool = False
    realised_pnl:  float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def margin_ratio(self, current_price: float) -> float:
        if self.margin_used == 0:
            return 1.0
        upnl = self.unrealised_pnl(current_price)
        return (self.margin_used + upnl) / self.margin_used


@dataclass
class RiskCheckResult:
    approved:        bool
    reason:          str
    position_size:   float
    margin_required: float
    leverage:        int
    risk_usd:        float
    reward_usd:      float
    risk_reward:     float


@dataclass
class AccountState:
    equity:             float
    balance:            float
    daily_start_equity: float
    open_positions:     Dict[str, Position] = field(default_factory=dict)
    daily_pnl:          float = 0.0
    trades_today:       int   = 0
    kill_switch:        bool  = False

    @property
    def daily_drawdown_pct(self) -> float:
        if self.daily_start_equity == 0:
            return 0.0
        return (self.daily_start_equity - self.equity) / self.daily_start_equity

    @property
    def position_count(self) -> int:
        return len([p for p in self.open_positions.values() if p.is_open])

    @property
    def used_margin(self) -> float:
        return sum(p.margin_used for p in self.open_positions.values() if p.is_open)

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin


class RiskManager:
    """
    Central risk management system with hard margin cap per trade.
    No single trade can ever use more than MAX_MARGIN_PER_TRADE_USD.
    """

    def __init__(self, initial_equity: float = 10_000.0):
        self._lock = threading.RLock()
        self.account = AccountState(
            equity             = initial_equity,
            balance            = initial_equity,
            daily_start_equity = initial_equity,
        )
        self._daily_reset_hour = 0
        self._last_reset_day   = -1
        logger.info(f"RiskManager initialised — equity=${initial_equity:,.2f}")

    def check_trade(
        self,
        symbol:     str,
        direction:  str,
        price:      float,
        atr:        float,
        conviction: float,
    ) -> RiskCheckResult:
        with self._lock:
            self._check_daily_reset()

            if self.account.kill_switch:
                return self._reject("Kill switch active — daily drawdown limit hit")

            if self.account.position_count >= MAX_SIMULTANEOUS_POSITIONS:
                return self._reject(
                    f"Max positions reached ({MAX_SIMULTANEOUS_POSITIONS})"
                )

            if symbol in self.account.open_positions:
                existing = self.account.open_positions[symbol]
                if existing.is_open:
                    return self._reject(f"Position already open for {symbol}")

            if price <= 0 or atr <= 0:
                return self._reject(f"Invalid price={price} or atr={atr}")

            leverage  = self._get_leverage(conviction)
            stop_dist = atr * ATR_STOP_MULT

            if stop_dist <= 0:
                return self._reject("Stop distance is zero")

            # Step 1 — risk based sizing
            risk_usd = self.account.equity * CAPITAL_RISK_PER_TRADE
            risk_usd = min(risk_usd, MAX_LOSS_PER_TRADE_USD)
            quantity = risk_usd / stop_dist

            # Step 2 — cap notional value
            notional = quantity * price
            if notional > MAX_NOTIONAL_USD:
                quantity = MAX_NOTIONAL_USD / price
                notional = MAX_NOTIONAL_USD

            # Step 3 — calculate margin
            margin_req = notional / leverage

            # Step 4 — hard margin cap per trade
            if margin_req > MAX_MARGIN_PER_TRADE_USD:
                margin_req = MAX_MARGIN_PER_TRADE_USD
                notional   = margin_req * leverage
                quantity   = notional / price

            # Step 5 — check free margin
            if margin_req > self.account.free_margin:
                return self._reject(
                    f"Insufficient margin — need ${margin_req:.2f}, "
                    f"have ${self.account.free_margin:.2f}"
                )

            # Step 6 — recalculate actual risk with capped quantity
            actual_risk   = quantity * stop_dist
            reward_usd    = actual_risk * ATR_TP_MULT
            risk_reward   = ATR_TP_MULT

            if risk_reward < 1.0:
                return self._reject(
                    f"Risk/reward too low: {risk_reward:.2f}"
                )

            logger.info(
                f"Trade approved: {symbol} {direction} "
                f"qty={quantity:.6f} lev={leverage}x "
                f"margin=${margin_req:.2f} risk=${actual_risk:.2f} "
                f"rr={risk_reward:.1f}"
            )

            return RiskCheckResult(
                approved        = True,
                reason          = "All checks passed",
                position_size   = round(quantity, 6),
                margin_required = round(margin_req, 2),
                leverage        = leverage,
                risk_usd        = round(actual_risk, 2),
                reward_usd      = round(reward_usd, 2),
                risk_reward     = round(risk_reward, 2),
            )

    def open_position(
        self,
        symbol:        str,
        check:         RiskCheckResult,
        entry_price:   float,
        stop_loss:     float,
        take_profit:   float,
        partial_close: float,
        direction:     str,
    ) -> Position:
        with self._lock:
            pos = Position(
                symbol        = symbol,
                direction     = direction,
                entry_price   = entry_price,
                quantity      = check.position_size,
                leverage      = check.leverage,
                stop_loss     = stop_loss,
                take_profit   = take_profit,
                partial_close = partial_close,
                margin_used   = check.margin_required,
                opened_at     = int(time.time() * 1000),
            )
            self.account.open_positions[symbol] = pos
            self.account.equity      -= check.margin_required
            self.account.trades_today += 1
            logger.info(
                f"Position opened: {symbol} {direction} "
                f"@ {entry_price} qty={check.position_size} "
                f"lev={check.leverage}x margin=${check.margin_required:.2f}"
            )
            return pos

    def close_position(
        self,
        symbol:      str,
        exit_price:  float,
        partial:     bool  = False,
        partial_pct: float = 0.5,
    ) -> float:
        with self._lock:
            pos = self.account.open_positions.get(symbol)
            if not pos or not pos.is_open:
                logger.warning(f"No open position for {symbol}")
                return 0.0

            if partial and not pos.partial_closed:
                close_qty = pos.quantity * partial_pct
                pnl = self._calc_pnl(pos, exit_price, close_qty)
                pos.quantity       -= close_qty
                pos.partial_closed  = True
                pos.realised_pnl   += pnl
                self.account.equity     += pnl
                self.account.daily_pnl  += pnl
                logger.info(
                    f"Partial close {symbol} qty={close_qty:.6f} "
                    f"@ {exit_price} pnl=${pnl:.2f}"
                )
                return pnl
            else:
                pnl = self._calc_pnl(pos, exit_price, pos.quantity)
                pos.is_open       = False
                pos.realised_pnl += pnl
                self.account.equity    += pos.margin_used + pnl
                self.account.daily_pnl += pnl
                del self.account.open_positions[symbol]
                logger.info(
                    f"Position closed {symbol} @ {exit_price} "
                    f"pnl=${pnl:.2f}"
                )
                self._check_kill_switch()
                return pnl

    def update_equity(self, new_equity: float):
        with self._lock:
            self.account.equity  = new_equity
            self.account.balance = new_equity
            self._check_kill_switch()

    def check_anti_liquidation(
        self, symbol: str, current_price: float
    ) -> bool:
        with self._lock:
            pos = self.account.open_positions.get(symbol)
            if not pos or not pos.is_open:
                return False
            ratio = pos.margin_ratio(current_price)
            if ratio < ANTI_LIQ_MARGIN_THRESHOLD:
                logger.warning(
                    f"Anti-liquidation triggered for {symbol} "
                    f"margin_ratio={ratio:.2%}"
                )
                return True
            return False

    def get_open_positions(self) -> List[Position]:
        with self._lock:
            return [p for p in self.account.open_positions.values() if p.is_open]

    def get_account_state(self) -> AccountState:
        with self._lock:
            return self.account

    def reset_kill_switch(self):
        with self._lock:
            self.account.kill_switch = False
            logger.warning("Kill switch manually reset")

    def _get_leverage(self, conviction: float) -> int:
        if conviction >= CONVICTION_HIGH_THRESH:
            return LEVERAGE_HIGH
        elif conviction >= CONVICTION_MEDIUM_THRESH:
            return LEVERAGE_MEDIUM
        else:
            return LEVERAGE_LOW

    def _calc_pnl(
        self, pos: Position, exit_price: float, quantity: float
    ) -> float:
        if pos.direction == "BUY":
            return (exit_price - pos.entry_price) * quantity
        else:
            return (pos.entry_price - exit_price) * quantity

    def _check_kill_switch(self):
        dd = self.account.daily_drawdown_pct
        if dd >= DAILY_DRAWDOWN_KILL_PCT and not self.account.kill_switch:
            self.account.kill_switch = True
            logger.critical(
                f"KILL SWITCH ACTIVATED — daily drawdown "
                f"{dd:.2%} >= {DAILY_DRAWDOWN_KILL_PCT:.2%}"
            )

    def _check_daily_reset(self):
        today = time.gmtime().tm_yday
        if today != self._last_reset_day:
            self.account.daily_pnl         = 0.0
            self.account.trades_today      = 0
            self.account.daily_start_equity = self.account.equity
            self.account.kill_switch       = False
            self._last_reset_day           = today
            logger.info("Daily risk counters reset")

    def _reject(self, reason: str) -> RiskCheckResult:
        logger.warning(f"Trade rejected: {reason}")
        return RiskCheckResult(
            approved        = False,
            reason          = reason,
            position_size   = 0.0,
            margin_required = 0.0,
            leverage        = 0,
            risk_usd        = 0.0,
            reward_usd      = 0.0,
            risk_reward     = 0.0,
        )
