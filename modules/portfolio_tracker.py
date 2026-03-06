from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import LOG_DIR, LOG_LEVEL
from config.logging_config import setup_logging
from modules.executor import PaperExecutor, PaperTrade
from modules.risk_manager import RiskManager

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="portfolio_tracker")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class EquityPoint:
    timestamp: int
    equity:    float
    pnl:       float


@dataclass
class TradeStats:
    symbol:       str
    direction:    str
    entry_price:  float
    exit_price:   float
    quantity:     float
    pnl:          float
    pnl_pct:      float
    duration_ms:  int
    opened_at:    int
    closed_at:    int


@dataclass
class PortfolioSnapshot:
    timestamp:        int
    equity:           float
    balance:          float
    unrealised_pnl:   float
    daily_pnl:        float
    total_pnl:        float
    open_positions:   int
    total_trades:     int
    win_rate:         float
    profit_factor:    float
    daily_drawdown:   float
    kill_switch:      bool
    best_trade:       Optional[TradeStats]
    worst_trade:      Optional[TradeStats]
    open_trade_list:  List[dict]
    equity_curve:     List[dict]


# ── Portfolio Tracker ──────────────────────────────────────────────────────────

class PortfolioTracker:
    """
    Tracks all portfolio metrics in real time.

    Pulls data from:
      - PaperExecutor  — open trades, history, PnL
      - RiskManager    — account state, kill switch

    Provides:
      - get_snapshot()     — full portfolio state in one call
      - get_equity_curve() — list of equity points over time
      - get_trade_stats()  — detailed stats on closed trades
      - get_open_positions_display() — formatted for dashboard
    """

    EQUITY_CURVE_MAX = 500   # keep last 500 equity points

    def __init__(
        self,
        executor:     PaperExecutor,
        risk_manager: RiskManager,
        update_interval: int = 5,
    ):
        self.executor      = executor
        self.rm            = risk_manager
        self._lock         = threading.RLock()
        self._equity_curve: deque = deque(maxlen=self.EQUITY_CURVE_MAX)
        self._running      = False
        self._thread:       Optional[threading.Thread] = None
        self._update_interval = update_interval
        self._initial_equity  = risk_manager.account.balance

        # Seed first equity point
        self._record_equity_point()
        logger.info("PortfolioTracker initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._update_loop,
            name="portfolio_tracker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"PortfolioTracker started — "
            f"updating every {self._update_interval}s"
        )

    def stop(self):
        self._running = False
        logger.info("PortfolioTracker stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_snapshot(self, current_prices: Dict[str, float] = None) -> PortfolioSnapshot:
        """
        Returns a complete portfolio snapshot.
        Pass current_prices dict to get live unrealised PnL.
        """
        current_prices = current_prices or {}
        account        = self.rm.get_account_state()
        open_trades    = self.executor.get_open_trades()
        history        = self.executor.get_trade_history()
        perf           = self.executor.get_performance_summary()

        # Unrealised PnL across all open positions
        unrealised = sum(
            t.unrealised_pnl(current_prices.get(t.symbol, t.entry_price))
            for t in open_trades
        )

        # Best and worst closed trades
        best  = self._get_best_trade(history)
        worst = self._get_worst_trade(history)

        # Open positions formatted for display
        open_list = self._format_open_trades(open_trades, current_prices)

        # Equity curve as list of dicts
        with self._lock:
            curve = [
                {"timestamp": p.timestamp, "equity": p.equity, "pnl": p.pnl}
                for p in self._equity_curve
            ]

        return PortfolioSnapshot(
            timestamp       = int(time.time() * 1000),
            equity          = account.equity,
            balance         = account.balance,
            unrealised_pnl  = round(unrealised, 2),
            daily_pnl       = round(account.daily_pnl, 2),
            total_pnl       = round(self.executor.get_total_pnl(), 2),
            open_positions  = len(open_trades),
            total_trades    = perf["total_trades"],
            win_rate        = perf["win_rate"],
            profit_factor   = perf["profit_factor"],
            daily_drawdown  = round(account.daily_drawdown_pct * 100, 2),
            kill_switch     = account.kill_switch,
            best_trade      = best,
            worst_trade     = worst,
            open_trade_list = open_list,
            equity_curve    = curve,
        )

    def get_equity_curve(self) -> List[dict]:
        with self._lock:
            return [
                {"timestamp": p.timestamp, "equity": p.equity}
                for p in self._equity_curve
            ]

    def get_trade_stats(self) -> List[TradeStats]:
        history = self.executor.get_trade_history()
        return [self._to_trade_stats(t) for t in history if not t.is_open]

    def get_open_positions_display(
        self, current_prices: Dict[str, float] = None
    ) -> List[dict]:
        current_prices = current_prices or {}
        open_trades    = self.executor.get_open_trades()
        return self._format_open_trades(open_trades, current_prices)

    def get_daily_summary(self) -> dict:
        account = self.rm.get_account_state()
        perf    = self.executor.get_performance_summary()
        return {
            "date":            time.strftime("%Y-%m-%d", time.gmtime()),
            "trades_today":    account.trades_today,
            "daily_pnl":       round(account.daily_pnl, 2),
            "daily_drawdown":  round(account.daily_drawdown_pct * 100, 2),
            "kill_switch":     account.kill_switch,
            "win_rate":        perf["win_rate"],
            "open_positions":  account.position_count,
        }

    def get_performance_metrics(self) -> dict:
        perf    = self.executor.get_performance_summary()
        history = self.executor.get_trade_history()
        account = self.rm.get_account_state()

        total_return = (
            (account.equity - self._initial_equity) / self._initial_equity * 100
            if self._initial_equity > 0 else 0.0
        )

        avg_duration = 0
        if history:
            durations = [
                t.closed_at - t.opened_at
                for t in history
                if t.closed_at > 0
            ]
            avg_duration = int(sum(durations) / len(durations)) if durations else 0

        return {
            "total_trades":    perf["total_trades"],
            "wins":            perf["wins"],
            "losses":          perf["losses"],
            "win_rate":        perf["win_rate"],
            "total_pnl":       perf["total_pnl"],
            "avg_win":         perf["avg_win"],
            "avg_loss":        perf["avg_loss"],
            "profit_factor":   perf["profit_factor"],
            "total_return_pct": round(total_return, 2),
            "avg_duration_ms": avg_duration,
            "current_equity":  round(account.equity, 2),
            "initial_equity":  round(self._initial_equity, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_loop(self):
        while self._running:
            try:
                self._record_equity_point()
            except Exception as exc:
                logger.error(f"Portfolio update error: {exc}")
            time.sleep(self._update_interval)

    def _record_equity_point(self):
        account = self.rm.get_account_state()
        point   = EquityPoint(
            timestamp = int(time.time() * 1000),
            equity    = account.equity,
            pnl       = account.daily_pnl,
        )
        with self._lock:
            self._equity_curve.append(point)

    def _format_open_trades(
        self,
        trades: List[PaperTrade],
        prices: Dict[str, float],
    ) -> List[dict]:
        result = []
        for t in trades:
            price  = prices.get(t.symbol, t.entry_price)
            upnl   = t.unrealised_pnl(price)
            upnl_pct = t.unrealised_pnl_pct(price)
            result.append({
                "symbol":          t.symbol,
                "direction":       t.direction,
                "entry_price":     t.entry_price,
                "current_price":   price,
                "quantity":        t.quantity,
                "leverage":        t.leverage,
                "stop_loss":       t.stop_loss,
                "take_profit":     t.take_profit,
                "unrealised_pnl":  round(upnl, 2),
                "unrealised_pct":  round(upnl_pct, 2),
                "margin_used":     t.margin_used,
                "partial_closed":  t.partial_closed,
                "opened_at":       t.opened_at,
            })
        return result

    def _to_trade_stats(self, trade: PaperTrade) -> TradeStats:
        pnl_pct = (
            trade.realised_pnl / trade.margin_used * 100
            if trade.margin_used > 0 else 0.0
        )
        return TradeStats(
            symbol      = trade.symbol,
            direction   = trade.direction,
            entry_price = trade.entry_price,
            exit_price  = trade.exit_price,
            quantity    = trade.quantity,
            pnl         = round(trade.realised_pnl, 2),
            pnl_pct     = round(pnl_pct, 2),
            duration_ms = trade.closed_at - trade.opened_at,
            opened_at   = trade.opened_at,
            closed_at   = trade.closed_at,
        )

    def _get_best_trade(
        self, history: List[PaperTrade]
    ) -> Optional[TradeStats]:
        closed = [t for t in history if not t.is_open and t.realised_pnl > 0]
        if not closed:
            return None
        best = max(closed, key=lambda t: t.realised_pnl)
        return self._to_trade_stats(best)

    def _get_worst_trade(
        self, history: List[PaperTrade]
    ) -> Optional[TradeStats]:
        closed = [t for t in history if not t.is_open and t.realised_pnl < 0]
        if not closed:
            return None
        worst = min(closed, key=lambda t: t.realised_pnl)
        return self._to_trade_stats(worst)