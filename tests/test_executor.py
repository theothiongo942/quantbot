import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager import RiskManager
from modules.signal_engine import SignalResult
from modules.executor import PaperExecutor, PaperTrade, OrderResult


def make_rm(equity=10_000.0) -> RiskManager:
    return RiskManager(initial_equity=equity)


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


def make_executor(equity=10_000.0) -> PaperExecutor:
    return PaperExecutor(risk_manager=make_rm(equity=equity))


class T01_OrderResult(unittest.TestCase):

    def test_order_result_fields(self):
        order = OrderResult(
            success=True, order_id="PAPER-000001",
            symbol="BTCUSDT", side="BUY",
            quantity=0.4, price=65000,
            order_type="MARKET", paper=True,
            message="Paper order filled",
        )
        self.assertTrue(order.success)
        self.assertTrue(order.paper)
        self.assertEqual(order.symbol, "BTCUSDT")

    def test_order_result_timestamp(self):
        order = OrderResult(
            success=True, order_id="PAPER-000001",
            symbol="BTCUSDT", side="BUY",
            quantity=0.4, price=65000,
            order_type="MARKET", paper=True,
            message="test",
        )
        self.assertGreater(order.timestamp, 0)


class T02_PaperTrade(unittest.TestCase):

    def make_trade(self, direction="BUY", entry=65000, qty=0.4):
        return PaperTrade(
            symbol        = "BTCUSDT",
            direction     = direction,
            entry_price   = entry,
            quantity      = qty,
            leverage      = 20,
            stop_loss     = entry - 500 if direction == "BUY" else entry + 500,
            take_profit   = entry + 1000 if direction == "BUY" else entry - 1000,
            partial_close = entry + 750 if direction == "BUY" else entry - 750,
            margin_used   = 1300.0,
            opened_at     = int(time.time() * 1000),
        )

    def test_unrealised_pnl_buy_profit(self):
        trade = self.make_trade("BUY", entry=65000)
        pnl = trade.unrealised_pnl(66000)
        self.assertGreater(pnl, 0)

    def test_unrealised_pnl_buy_loss(self):
        trade = self.make_trade("BUY", entry=65000)
        pnl = trade.unrealised_pnl(64000)
        self.assertLess(pnl, 0)

    def test_unrealised_pnl_sell_profit(self):
        trade = self.make_trade("SELL", entry=65000)
        pnl = trade.unrealised_pnl(64000)
        self.assertGreater(pnl, 0)

    def test_unrealised_pnl_sell_loss(self):
        trade = self.make_trade("SELL", entry=65000)
        pnl = trade.unrealised_pnl(66000)
        self.assertLess(pnl, 0)

    def test_unrealised_pnl_pct(self):
        trade = self.make_trade("BUY", entry=65000)
        pct = trade.unrealised_pnl_pct(66000)
        self.assertGreater(pct, 0)

    def test_zero_margin_pnl_pct(self):
        trade = self.make_trade("BUY")
        trade.margin_used = 0
        pct = trade.unrealised_pnl_pct(66000)
        self.assertEqual(pct, 0.0)


class T03_ExecuteSignal(unittest.TestCase):

    def test_execute_buy_signal(self):
        ex = make_executor()
        sig = make_signal(direction="BUY")
        result = ex.execute_signal(sig, current_price=65000)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertTrue(result.paper)

    def test_execute_sell_signal(self):
        ex = make_executor()
        sig = make_signal(direction="SELL")
        result = ex.execute_signal(sig, current_price=65000)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

    def test_neutral_signal_not_executed(self):
        ex = make_executor()
        sig = make_signal(direction="NEUTRAL", conviction=0.5)
        result = ex.execute_signal(sig, current_price=65000)
        self.assertIsNone(result)

    def test_order_id_is_paper(self):
        ex = make_executor()
        sig = make_signal()
        result = ex.execute_signal(sig, current_price=65000)
        self.assertIn("PAPER", result.order_id)

    def test_position_tracked_after_execute(self):
        ex = make_executor()
        sig = make_signal()
        ex.execute_signal(sig, current_price=65000)
        trades = ex.get_open_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].symbol, "BTCUSDT")

    def test_duplicate_symbol_rejected(self):
        ex = make_executor()
        sig = make_signal()
        ex.execute_signal(sig, current_price=65000)
        result2 = ex.execute_signal(sig, current_price=65000)
        self.assertIsNone(result2)

    def test_stop_loss_set_correctly_buy(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        trade = ex.get_trade("BTCUSDT")
        self.assertLess(trade.stop_loss, 65000)

    def test_stop_loss_set_correctly_sell(self):
        ex = make_executor()
        sig = make_signal(direction="SELL", atr=500)
        ex.execute_signal(sig, current_price=65000)
        trade = ex.get_trade("BTCUSDT")
        self.assertGreater(trade.stop_loss, 65000)

    def test_take_profit_set_correctly_buy(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        trade = ex.get_trade("BTCUSDT")
        self.assertGreater(trade.take_profit, 65000)

    def test_take_profit_set_correctly_sell(self):
        ex = make_executor()
        sig = make_signal(direction="SELL", atr=500)
        ex.execute_signal(sig, current_price=65000)
        trade = ex.get_trade("BTCUSDT")
        self.assertLess(trade.take_profit, 65000)


class T04_PositionManagement(unittest.TestCase):

    def setUp(self):
        self.ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        self.ex.execute_signal(sig, current_price=65000)

    def test_stop_loss_closes_position(self):
        self.ex.check_and_manage_positions({"BTCUSDT": 64000})
        trades = self.ex.get_open_trades()
        self.assertEqual(len(trades), 0)

    def test_take_profit_closes_position(self):
        self.ex.check_and_manage_positions({"BTCUSDT": 67000})
        trades = self.ex.get_open_trades()
        self.assertEqual(len(trades), 0)

    def test_partial_close_triggered(self):
        trade = self.ex.get_trade("BTCUSDT")
        partial_price = trade.partial_close + 1
        self.ex.check_and_manage_positions({"BTCUSDT": partial_price})
        trade = self.ex.get_trade("BTCUSDT")
        if trade:
            self.assertTrue(trade.partial_closed)

    def test_position_stays_open_in_range(self):
        self.ex.check_and_manage_positions({"BTCUSDT": 65100})
        trades = self.ex.get_open_trades()
        self.assertEqual(len(trades), 1)

    def test_missing_price_doesnt_crash(self):
        try:
            self.ex.check_and_manage_positions({})
        except Exception as e:
            self.fail(f"Crashed with missing price: {e}")


class T05_ManualClose(unittest.TestCase):

    def test_manual_close_works(self):
        ex = make_executor()
        sig = make_signal()
        ex.execute_signal(sig, current_price=65000)
        result = ex.close_position_manual("BTCUSDT", 65500)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

    def test_manual_close_removes_position(self):
        ex = make_executor()
        sig = make_signal()
        ex.execute_signal(sig, current_price=65000)
        ex.close_position_manual("BTCUSDT", 65500)
        trades = ex.get_open_trades()
        self.assertEqual(len(trades), 0)

    def test_manual_close_no_position(self):
        ex = make_executor()
        result = ex.close_position_manual("BTCUSDT", 65000)
        self.assertIsNone(result)


class T06_PnLTracking(unittest.TestCase):

    def test_winning_trade_positive_pnl(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        ex.check_and_manage_positions({"BTCUSDT": 67000})
        self.assertGreater(ex.get_total_pnl(), 0)

    def test_losing_trade_negative_pnl(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        ex.check_and_manage_positions({"BTCUSDT": 64000})
        self.assertLess(ex.get_total_pnl(), 0)

    def test_win_rate_after_win(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        ex.check_and_manage_positions({"BTCUSDT": 67000})
        self.assertGreater(ex.get_win_rate(), 0)

    def test_trade_history_recorded(self):
        ex = make_executor()
        sig = make_signal(direction="BUY", atr=500)
        ex.execute_signal(sig, current_price=65000)
        ex.check_and_manage_positions({"BTCUSDT": 67000})
        history = ex.get_trade_history()
        self.assertEqual(len(history), 1)


class T07_PerformanceSummary(unittest.TestCase):

    def setUp(self):
        self.ex = make_executor(equity=100_000)

    def _run_trade(self, direction, entry, exit_price, atr=500):
        sig = make_signal(
            symbol=f"{direction}USDT",
            direction=direction,
            atr=atr,
        )
        sig.symbol = "BTCUSDT"
        self.ex.rm.account.open_positions.clear()
        self.ex._paper_trades.clear()
        self.ex.execute_signal(sig, current_price=entry)
        self.ex.check_and_manage_positions({"BTCUSDT": exit_price})

    def test_summary_has_all_keys(self):
        summary = self.ex.get_performance_summary()
        for key in [
            "total_trades", "open_trades", "wins", "losses",
            "win_rate", "total_pnl", "avg_win", "avg_loss",
            "profit_factor",
        ]:
            self.assertIn(key, summary)

    def test_summary_empty_initially(self):
        summary = self.ex.get_performance_summary()
        self.assertEqual(summary["total_trades"], 0)
        self.assertEqual(summary["total_pnl"], 0.0)

    def test_win_rate_calculation(self):
        self._run_trade("BUY", 65000, 67000)
        summary = self.ex.get_performance_summary()
        self.assertGreaterEqual(summary["win_rate"], 0)
        self.assertLessEqual(summary["win_rate"], 100)

    def test_order_counter_increments(self):
        ex = make_executor()
        sig1 = make_signal(symbol="BTCUSDT")
        sig2 = make_signal(symbol="ETHUSDT")
        ex.execute_signal(sig1, current_price=65000)
        ex.execute_signal(sig2, current_price=3000)
        self.assertEqual(ex._order_counter, 2)


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 5: Executor Tests")
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