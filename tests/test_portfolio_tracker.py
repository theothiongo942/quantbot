import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager import RiskManager
from modules.signal_engine import SignalResult
from modules.executor import PaperExecutor
from modules.portfolio_tracker import (
    PortfolioTracker,
    PortfolioSnapshot,
    TradeStats,
    EquityPoint,
)


def make_rm(equity=10_000.0) -> RiskManager:
    return RiskManager(initial_equity=equity)


def make_executor(equity=10_000.0) -> PaperExecutor:
    return PaperExecutor(risk_manager=make_rm(equity=equity))


def make_signal(
    symbol="BTCUSDT",
    direction="BUY",
    conviction=0.75,
    atr=500.0,
):
    return SignalResult(
        symbol        = symbol,
        timestamp     = int(time.time() * 1000),
        direction     = direction,
        conviction    = conviction,
        leverage_tier = 20,
        signal_scores = {},
        atr           = atr,
        stop_loss     = 0.0,
        take_profit   = 0.0,
        partial_close = 0.0,
        notes         = [],
    )


def make_tracker(equity=10_000.0):
    rm  = make_rm(equity=equity)
    ex  = PaperExecutor(risk_manager=rm)
    pt  = PortfolioTracker(executor=ex, risk_manager=rm, update_interval=60)
    return pt, ex, rm


class T01_EquityPoint(unittest.TestCase):

    def test_equity_point_fields(self):
        pt = EquityPoint(
            timestamp = int(time.time() * 1000),
            equity    = 10_000.0,
            pnl       = 0.0,
        )
        self.assertEqual(pt.equity, 10_000.0)
        self.assertEqual(pt.pnl, 0.0)
        self.assertGreater(pt.timestamp, 0)


class T02_TradeStats(unittest.TestCase):

    def test_trade_stats_fields(self):
        ts = TradeStats(
            symbol      = "BTCUSDT",
            direction   = "BUY",
            entry_price = 65000.0,
            exit_price  = 66000.0,
            quantity    = 0.4,
            pnl         = 400.0,
            pnl_pct     = 30.0,
            duration_ms = 60_000,
            opened_at   = int(time.time() * 1000),
            closed_at   = int(time.time() * 1000) + 60_000,
        )
        self.assertEqual(ts.symbol, "BTCUSDT")
        self.assertEqual(ts.pnl, 400.0)
        self.assertGreater(ts.duration_ms, 0)


class T03_InitialState(unittest.TestCase):

    def test_initial_equity_curve_has_one_point(self):
        pt, ex, rm = make_tracker()
        curve = pt.get_equity_curve()
        self.assertGreaterEqual(len(curve), 1)

    def test_initial_equity_correct(self):
        pt, ex, rm = make_tracker(equity=10_000)
        curve = pt.get_equity_curve()
        self.assertAlmostEqual(curve[0]["equity"], 10_000.0, delta=1.0)

    def test_initial_snapshot_no_trades(self):
        pt, ex, rm = make_tracker()
        snap = pt.get_snapshot()
        self.assertEqual(snap.total_trades, 0)
        self.assertEqual(snap.open_positions, 0)

    def test_initial_pnl_zero(self):
        pt, ex, rm = make_tracker()
        snap = pt.get_snapshot()
        self.assertEqual(snap.total_pnl, 0.0)
        self.assertEqual(snap.daily_pnl, 0.0)

    def test_initial_no_best_trade(self):
        pt, ex, rm = make_tracker()
        snap = pt.get_snapshot()
        self.assertIsNone(snap.best_trade)

    def test_initial_no_worst_trade(self):
        pt, ex, rm = make_tracker()
        snap = pt.get_snapshot()
        self.assertIsNone(snap.worst_trade)


class T04_Snapshot(unittest.TestCase):

    def setUp(self):
        self.pt, self.ex, self.rm = make_tracker(equity=10_000)

    def test_snapshot_returns_correct_type(self):
        snap = self.pt.get_snapshot()
        self.assertIsInstance(snap, PortfolioSnapshot)

    def test_snapshot_has_timestamp(self):
        snap = self.pt.get_snapshot()
        self.assertGreater(snap.timestamp, 0)

    def test_snapshot_equity_positive(self):
        snap = self.pt.get_snapshot()
        self.assertGreater(snap.equity, 0)

    def test_snapshot_with_open_trade(self):
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)
        snap = self.pt.get_snapshot({"BTCUSDT": 65000})
        self.assertEqual(snap.open_positions, 1)

    def test_snapshot_unrealised_pnl_profit(self):
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)
        snap = self.pt.get_snapshot({"BTCUSDT": 66000})
        self.assertGreater(snap.unrealised_pnl, 0)

    def test_snapshot_unrealised_pnl_loss(self):
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)
        snap = self.pt.get_snapshot({"BTCUSDT": 64000})
        self.assertLess(snap.unrealised_pnl, 0)

    def test_snapshot_kill_switch_false_initially(self):
        snap = self.pt.get_snapshot()
        self.assertFalse(snap.kill_switch)

    def test_snapshot_daily_drawdown_zero_initially(self):
        snap = self.pt.get_snapshot()
        self.assertEqual(snap.daily_drawdown, 0.0)


class T05_OpenPositionsDisplay(unittest.TestCase):

    def setUp(self):
        self.pt, self.ex, self.rm = make_tracker(equity=10_000)
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)

    def test_open_positions_returns_list(self):
        positions = self.pt.get_open_positions_display({"BTCUSDT": 65000})
        self.assertIsInstance(positions, list)

    def test_open_positions_has_one_entry(self):
        positions = self.pt.get_open_positions_display({"BTCUSDT": 65000})
        self.assertEqual(len(positions), 1)

    def test_position_has_required_fields(self):
        positions = self.pt.get_open_positions_display({"BTCUSDT": 65000})
        pos = positions[0]
        for key in [
            "symbol", "direction", "entry_price", "current_price",
            "quantity", "leverage", "stop_loss", "take_profit",
            "unrealised_pnl", "unrealised_pct", "margin_used",
        ]:
            self.assertIn(key, pos)

    def test_position_symbol_correct(self):
        positions = self.pt.get_open_positions_display({"BTCUSDT": 65000})
        self.assertEqual(positions[0]["symbol"], "BTCUSDT")

    def test_position_pnl_updates_with_price(self):
        pos_low  = self.pt.get_open_positions_display({"BTCUSDT": 64000})
        pos_high = self.pt.get_open_positions_display({"BTCUSDT": 66000})
        self.assertLess(
            pos_low[0]["unrealised_pnl"],
            pos_high[0]["unrealised_pnl"]
        )


class T06_TradeStatsAfterClose(unittest.TestCase):

    def setUp(self):
        self.pt, self.ex, self.rm = make_tracker(equity=10_000)
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)
        self.ex.check_and_manage_positions({"BTCUSDT": 67000})

    def test_trade_stats_recorded(self):
        stats = self.pt.get_trade_stats()
        self.assertEqual(len(stats), 1)

    def test_trade_stats_correct_symbol(self):
        stats = self.pt.get_trade_stats()
        self.assertEqual(stats[0].symbol, "BTCUSDT")

    def test_trade_stats_positive_pnl(self):
        stats = self.pt.get_trade_stats()
        self.assertGreater(stats[0].pnl, 0)

    def test_best_trade_recorded(self):
        snap = self.pt.get_snapshot()
        self.assertIsNotNone(snap.best_trade)

    def test_best_trade_symbol(self):
        snap = self.pt.get_snapshot()
        self.assertEqual(snap.best_trade.symbol, "BTCUSDT")

    def test_total_trades_increments(self):
        snap = self.pt.get_snapshot()
        self.assertEqual(snap.total_trades, 1)

    def test_total_pnl_positive_after_win(self):
        snap = self.pt.get_snapshot()
        self.assertGreater(snap.total_pnl, 0)


class T07_PerformanceMetrics(unittest.TestCase):

    def test_metrics_has_all_keys(self):
        pt, ex, rm = make_tracker()
        metrics = pt.get_performance_metrics()
        for key in [
            "total_trades", "wins", "losses", "win_rate",
            "total_pnl", "avg_win", "avg_loss", "profit_factor",
            "total_return_pct", "current_equity", "initial_equity",
        ]:
            self.assertIn(key, metrics)

    def test_initial_equity_correct(self):
        pt, ex, rm = make_tracker(equity=10_000)
        metrics = pt.get_performance_metrics()
        self.assertEqual(metrics["initial_equity"], 10_000.0)

    def test_total_return_zero_initially(self):
        pt, ex, rm = make_tracker()
        metrics = pt.get_performance_metrics()
        self.assertEqual(metrics["total_return_pct"], 0.0)


class T08_DailySummary(unittest.TestCase):

    def test_daily_summary_has_all_keys(self):
        pt, ex, rm = make_tracker()
        summary = pt.get_daily_summary()
        for key in [
            "date", "trades_today", "daily_pnl",
            "daily_drawdown", "kill_switch", "win_rate",
            "open_positions",
        ]:
            self.assertIn(key, summary)

    def test_daily_summary_no_trades_initially(self):
        pt, ex, rm = make_tracker()
        summary = pt.get_daily_summary()
        self.assertEqual(summary["trades_today"], 0)
        self.assertEqual(summary["daily_pnl"], 0.0)

    def test_kill_switch_reflected_in_summary(self):
        pt, ex, rm = make_tracker()
        rm.account.kill_switch = True
        summary = pt.get_daily_summary()
        self.assertTrue(summary["kill_switch"])


class T09_EquityCurve(unittest.TestCase):

    def test_equity_curve_grows_after_record(self):
        pt, ex, rm = make_tracker()
        pt._record_equity_point()
        curve = pt.get_equity_curve()
        self.assertGreaterEqual(len(curve), 2)

    def test_equity_curve_has_correct_keys(self):
        pt, ex, rm = make_tracker()
        curve = pt.get_equity_curve()
        self.assertIn("timestamp", curve[0])
        self.assertIn("equity", curve[0])

    def test_equity_curve_max_size(self):
        pt, ex, rm = make_tracker()
        for _ in range(600):
            pt._record_equity_point()
        curve = pt.get_equity_curve()
        self.assertLessEqual(len(curve), pt.EQUITY_CURVE_MAX)


class T10_BackgroundUpdates(unittest.TestCase):

    def test_tracker_starts_and_stops(self):
        pt, ex, rm = make_tracker()
        pt.start()
        time.sleep(1)
        pt.stop()
        self.assertFalse(pt._running)

    def test_equity_curve_updates_while_running(self):
        pt, ex, rm = make_tracker(equity=10_000)
        pt._update_interval = 1
        pt.start()
        time.sleep(2)
        pt.stop()
        curve = pt.get_equity_curve()
        self.assertGreaterEqual(len(curve), 2)


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 6: Portfolio Tracker Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  {len(result.failures)} FAILURES | {len(result.errors)} ERRORS")
    print("=" * 60)

    sys.exit(0 if result.wasSuccessful() else 1)