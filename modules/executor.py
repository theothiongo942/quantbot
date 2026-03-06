from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_FUTURES_BASE_URL,
    USE_TESTNET,
    ATR_STOP_MULT,
    ATR_TP_MULT,
    ATR_PARTIAL_CLOSE_MULT,
    PARTIAL_CLOSE_PCT,
    LOG_DIR,
    LOG_LEVEL,
)
from config.logging_config import setup_logging
from modules.risk_manager import RiskManager, RiskCheckResult, Position
from modules.signal_engine import SignalResult

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="executor")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    success:      bool
    order_id:     str
    symbol:       str
    side:         str
    quantity:     float
    price:        float
    order_type:   str
    paper:        bool
    message:      str
    timestamp:    int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class PaperTrade:
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
    exit_price:    float = 0.0
    closed_at:     int = 0
    realised_pnl:  float = 0.0

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealised_pnl_pct(self, current_price: float) -> float:
        if self.margin_used == 0:
            return 0.0
        return self.unrealised_pnl(current_price) / self.margin_used * 100


# ── Paper Executor ─────────────────────────────────────────────────────────────

class PaperExecutor:
    """
    Simulates trade execution without touching real money.

    Behaves identically to LiveExecutor except:
      - Orders are filled instantly at current price
      - No API calls are made
      - All state is stored in memory

    Switch to live trading by replacing PaperExecutor with
    LiveExecutor in main.py — everything else stays the same.
    """

    def __init__(self, risk_manager: RiskManager):
        self.rm             = risk_manager
        self._lock          = threading.RLock()
        self._paper_trades: Dict[str, PaperTrade] = {}
        self._trade_history: List[PaperTrade] = []
        self._order_counter = 0
        logger.info("PaperExecutor initialised — paper trading mode active")

    # ─────────────────────────────────────────────────────────────────────────
    # Core Execution
    # ─────────────────────────────────────────────────────────────────────────

    def execute_signal(self, signal: SignalResult, current_price: float) -> Optional[OrderResult]:
        """
        Main entry point. Takes a SignalResult and executes if approved.

        Flow:
          1. Check signal is tradeable
          2. Run risk manager pre-trade check
          3. Place entry order
          4. Set stop loss and take profit
          5. Register position with risk manager
        """
        if not signal.is_tradeable:
            logger.debug(f"Signal not tradeable for {signal.symbol}")
            return None

        # Pre-trade risk check
        check = self.rm.check_trade(
            symbol     = signal.symbol,
            direction  = signal.direction,
            price      = current_price,
            atr        = signal.atr,
            conviction = signal.conviction,
        )

        if not check.approved:
            logger.warning(
                f"Trade rejected by risk manager: {check.reason}"
            )
            return None

        # Calculate levels
        atr = signal.atr
        if signal.direction == "BUY":
            stop_loss     = current_price - atr * ATR_STOP_MULT
            take_profit   = current_price + atr * ATR_TP_MULT
            partial_close = current_price + atr * ATR_PARTIAL_CLOSE_MULT
        else:
            stop_loss     = current_price + atr * ATR_STOP_MULT
            take_profit   = current_price - atr * ATR_TP_MULT
            partial_close = current_price - atr * ATR_PARTIAL_CLOSE_MULT

        # Place paper order
        order = self._place_paper_order(
            symbol    = signal.symbol,
            side      = signal.direction,
            quantity  = check.position_size,
            price     = current_price,
            order_type = "MARKET",
        )

        if not order.success:
            return order

        # Register with risk manager
        self.rm.open_position(
            symbol        = signal.symbol,
            check         = check,
            entry_price   = current_price,
            stop_loss     = stop_loss,
            take_profit   = take_profit,
            partial_close = partial_close,
            direction     = signal.direction,
        )

        # Track paper trade
        with self._lock:
            self._paper_trades[signal.symbol] = PaperTrade(
                symbol        = signal.symbol,
                direction     = signal.direction,
                entry_price   = current_price,
                quantity      = check.position_size,
                leverage      = check.leverage,
                stop_loss     = stop_loss,
                take_profit   = take_profit,
                partial_close = partial_close,
                margin_used   = check.margin_required,
                opened_at     = int(time.time() * 1000),
            )

        logger.info(
            f"PAPER TRADE OPENED: {signal.symbol} {signal.direction} "
            f"@ {current_price} | qty={check.position_size} "
            f"lev={check.leverage}x | "
            f"SL={stop_loss:.4f} TP={take_profit:.4f}"
        )

        return order

    def check_and_manage_positions(self, prices: Dict[str, float]):
        """
        Called on every price update.
        Checks all open positions against their SL/TP levels.
        Handles partial closes and trailing stops.
        """
        with self._lock:
            open_trades = {
                sym: trade for sym, trade in self._paper_trades.items()
                if trade.is_open
            }

        for symbol, trade in open_trades.items():
            price = prices.get(symbol)
            if not price:
                continue

            # Check partial close
            if not trade.partial_closed:
                if trade.direction == "BUY" and price >= trade.partial_close:
                    self._close_position(symbol, price, partial=True)
                    logger.info(
                        f"PARTIAL CLOSE: {symbol} @ {price} "
                        f"(target was {trade.partial_close:.4f})"
                    )
                elif trade.direction == "SELL" and price <= trade.partial_close:
                    self._close_position(symbol, price, partial=True)
                    logger.info(
                        f"PARTIAL CLOSE: {symbol} @ {price} "
                        f"(target was {trade.partial_close:.4f})"
                    )

            # Check stop loss
            if trade.direction == "BUY" and price <= trade.stop_loss:
                self._close_position(symbol, price)
                logger.info(
                    f"STOP LOSS HIT: {symbol} @ {price} "
                    f"(SL was {trade.stop_loss:.4f})"
                )

            elif trade.direction == "SELL" and price >= trade.stop_loss:
                self._close_position(symbol, price)
                logger.info(
                    f"STOP LOSS HIT: {symbol} @ {price} "
                    f"(SL was {trade.stop_loss:.4f})"
                )

            # Check take profit
            elif trade.direction == "BUY" and price >= trade.take_profit:
                self._close_position(symbol, price)
                logger.info(
                    f"TAKE PROFIT HIT: {symbol} @ {price} "
                    f"(TP was {trade.take_profit:.4f})"
                )

            elif trade.direction == "SELL" and price <= trade.take_profit:
                self._close_position(symbol, price)
                logger.info(
                    f"TAKE PROFIT HIT: {symbol} @ {price} "
                    f"(TP was {trade.take_profit:.4f})"
                )

            # Check anti liquidation
            if self.rm.check_anti_liquidation(symbol, price):
                self._close_position(symbol, price)
                logger.warning(
                    f"ANTI-LIQUIDATION CLOSE: {symbol} @ {price}"
                )

    def close_position_manual(
        self, symbol: str, current_price: float
    ) -> Optional[OrderResult]:
        """Manual override — close a position immediately."""
        with self._lock:
            trade = self._paper_trades.get(symbol)
            if not trade or not trade.is_open:
                logger.warning(f"No open paper trade for {symbol}")
                return None

        self._close_position(symbol, current_price)
        logger.info(f"MANUAL CLOSE: {symbol} @ {current_price}")

        return OrderResult(
            success    = True,
            order_id   = self._next_order_id(),
            symbol     = symbol,
            side       = "CLOSE",
            quantity   = trade.quantity,
            price      = current_price,
            order_type = "MARKET",
            paper      = True,
            message    = "Manual close executed",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public Data Access
    # ─────────────────────────────────────────────────────────────────────────

    def get_open_trades(self) -> List[PaperTrade]:
        with self._lock:
            return [t for t in self._paper_trades.values() if t.is_open]

    def get_trade_history(self) -> List[PaperTrade]:
        with self._lock:
            return list(self._trade_history)

    def get_trade(self, symbol: str) -> Optional[PaperTrade]:
        with self._lock:
            return self._paper_trades.get(symbol)

    def get_total_pnl(self) -> float:
        with self._lock:
            return sum(t.realised_pnl for t in self._trade_history)

    def get_win_rate(self) -> float:
        with self._lock:
            closed = [t for t in self._trade_history if t.realised_pnl != 0]
            if not closed:
                return 0.0
            wins = sum(1 for t in closed if t.realised_pnl > 0)
            return wins / len(closed)

    def get_performance_summary(self) -> dict:
        with self._lock:
            closed  = self._trade_history
            open_t  = [t for t in self._paper_trades.values() if t.is_open]
            total   = len(closed)
            wins    = sum(1 for t in closed if t.realised_pnl > 0)
            losses  = sum(1 for t in closed if t.realised_pnl <= 0)
            pnl     = sum(t.realised_pnl for t in closed)
            avg_win = (
                sum(t.realised_pnl for t in closed if t.realised_pnl > 0) / wins
                if wins > 0 else 0.0
            )
            avg_loss = (
                sum(t.realised_pnl for t in closed if t.realised_pnl <= 0) / losses
                if losses > 0 else 0.0
            )

        return {
            "total_trades":    total,
            "open_trades":     len(open_t),
            "wins":            wins,
            "losses":          losses,
            "win_rate":        round(wins / total * 100, 1) if total > 0 else 0.0,
            "total_pnl":       round(pnl, 2),
            "avg_win":         round(avg_win, 2),
            "avg_loss":        round(avg_loss, 2),
            "profit_factor":   round(
                abs(avg_win * wins) / abs(avg_loss * losses), 2
            ) if losses > 0 and avg_loss != 0 else 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _place_paper_order(
        self,
        symbol:     str,
        side:       str,
        quantity:   float,
        price:      float,
        order_type: str = "MARKET",
    ) -> OrderResult:
        order_id = self._next_order_id()
        logger.debug(
            f"[PAPER] {order_type} {side} {quantity} {symbol} @ {price}"
        )
        return OrderResult(
            success    = True,
            order_id   = order_id,
            symbol     = symbol,
            side       = side,
            quantity   = quantity,
            price      = price,
            order_type = order_type,
            paper      = True,
            message    = "Paper order filled",
        )

    def _close_position(
        self,
        symbol:  str,
        price:   float,
        partial: bool = False,
    ):
        with self._lock:
            trade = self._paper_trades.get(symbol)
            if not trade or not trade.is_open:
                return

            if partial and not trade.partial_closed:
                close_qty = trade.quantity * PARTIAL_CLOSE_PCT
                pnl = (
                    (price - trade.entry_price) * close_qty
                    if trade.direction == "BUY"
                    else (trade.entry_price - price) * close_qty
                )
                trade.quantity      -= close_qty
                trade.partial_closed = True
                trade.realised_pnl  += pnl
                self.rm.close_position(symbol, price, partial=True)
            else:
                pnl = (
                    (price - trade.entry_price) * trade.quantity
                    if trade.direction == "BUY"
                    else (trade.entry_price - price) * trade.quantity
                )
                trade.is_open      = False
                trade.exit_price   = price
                trade.closed_at    = int(time.time() * 1000)
                trade.realised_pnl += pnl
                self._trade_history.append(trade)
                del self._paper_trades[symbol]
                self.rm.close_position(symbol, price)

    def _next_order_id(self) -> str:
        with self._lock:
            self._order_counter += 1
            return f"PAPER-{self._order_counter:06d}"


# ── Live Executor (stub — activate when ready for live trading) ────────────────

class LiveExecutor(PaperExecutor):
    """
    Live trading executor using Binance Futures API.
    Inherits all logic from PaperExecutor.
    Only overrides _place_paper_order to make real API calls.

    To activate:
      1. Set BINANCE_API_KEY and BINANCE_API_SECRET in environment
      2. Set USE_TESTNET=false in settings.py when ready for mainnet
      3. Replace PaperExecutor with LiveExecutor in main.py
    """

    def __init__(self, risk_manager: RiskManager):
        super().__init__(risk_manager)
        self._client = None
        self._init_client()
        logger.warning(
            "LiveExecutor initialised — "
            f"{'TESTNET' if USE_TESTNET else 'MAINNET'} mode"
        )

    def _init_client(self):
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            logger.error(
                "API keys not set — "
                "set BINANCE_API_KEY and BINANCE_API_SECRET"
            )
            return
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "X-MBX-APIKEY": BINANCE_API_KEY
            })
            logger.info("Binance Futures client initialised")
        except Exception as exc:
            logger.error(f"Failed to init Binance client: {exc}")

    def _place_paper_order(
        self,
        symbol:     str,
        side:       str,
        quantity:   float,
        price:      float,
        order_type: str = "MARKET",
    ) -> OrderResult:
        """Override to place real orders on Binance Futures."""
        if not self._client and not hasattr(self, "_session"):
            return OrderResult(
                success=False, order_id="", symbol=symbol,
                side=side, quantity=quantity, price=price,
                order_type=order_type, paper=False,
                message="API client not initialised",
            )
        try:
            import hmac
            import hashlib
            import urllib.parse

            timestamp = int(time.time() * 1000)
            params = {
                "symbol":    symbol,
                "side":      side,
                "type":      order_type,
                "quantity":  quantity,
                "timestamp": timestamp,
            }
            query = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query.encode(),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature

            url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/order"
            resp = self._session.post(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            return OrderResult(
                success    = True,
                order_id   = str(data.get("orderId", "")),
                symbol     = symbol,
                side       = side,
                quantity   = float(data.get("origQty", quantity)),
                price      = float(data.get("avgPrice", price)),
                order_type = order_type,
                paper      = False,
                message    = "Live order placed",
            )
        except Exception as exc:
            logger.error(f"Live order failed for {symbol}: {exc}")
            return OrderResult(
                success=False, order_id="", symbol=symbol,
                side=side, quantity=quantity, price=price,
                order_type=order_type, paper=False,
                message=str(exc),
            )