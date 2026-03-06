import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager import (
    RiskManager,
    RiskCheckResult,
    Position,
    AccountState,
)
from config.settings import (
    MAX_SIMULTANEOUS_POSITIONS,
    DAILY_DRAWDOWN_KILL_PCT,
    LEVERAGE_LOW,
    LEVERAGE_MEDIUM,
    LEVERAGE_HIGH,
    CONVICTION_MEDIUM_THRESH,
    CONVICTION_HIGH_THRESH,
    MAX_LOSS_PER_TRADE_USD,
)


def make_rm(equity=10_000.0) -> RiskManager:
    return RiskManager(initial_equity=equity)


def approved_check(rm, symbol="BTCUSDT", direction="BUY",
                   price=65000, atr=500, conviction=0.75):
    return rm.check_trade(symbol, direction, price, atr, conviction)


class T01_PositionSizing(unittest.TestCase):

    def test_approved_trade_returns_true(self):
        rm = make_rm()
        result = approved_check(rm)
        self.assertTrue(result.approved)

    def test_position_size_positive(self):
        rm = make_rm()
        result = approved_check(rm)
        self.assertGreater(result.position_size, 0)

    def test_margin_required_positive(self):
        rm = make_rm()
        result = approved_check(rm)
        self.assertGreater(result.margin_required, 0)

    def test_risk_usd_is_2pct_of_equity(self):
        rm = make_rm(equity=10_000)
        result = approved_check(rm, atr=500, price=65000)
        self.assertLessEqual(result.risk_usd, MAX_LOSS_PER_TRADE_USD)
        self.assertGreater(result.risk_usd, 0)

    def test_risk_reward_at_least_2(self):
        rm = make_rm()
        result = approved_check(rm)
        self.assertGreaterEqual(result.risk_reward, 2.0)

    def test_reward_greater_than_risk(self):
        rm = make_rm()
        result = approved_check(rm)
        self.assertGreater(result.reward_usd, result.risk_usd)


class T02_LeverageTiers(unittest.TestCase):

    def test_low_conviction_gets_10x(self):
        rm = make_rm()
        result = approved_check(rm, conviction=0.50)
        self.assertEqual(result.leverage, LEVERAGE_LOW)

    def test_medium_conviction_gets_20x(self):
        rm = make_rm()
        result = approved_check(rm, conviction=CONVICTION_MEDIUM_THRESH)
        self.assertEqual(result.leverage, LEVERAGE_MEDIUM)

    def test_high_conviction_gets_50x(self):
        rm = make_rm()
        result = approved_check(rm, conviction=CONVICTION_HIGH_THRESH)
        self.assertEqual(result.leverage, LEVERAGE_HIGH)


class T03_TradeRejections(unittest.TestCase):

    def test_rejects_invalid_price(self):
        rm = make_rm()
        result = rm.check_trade("BTCUSDT", "BUY", price=0, atr=500, conviction=0.75)
        self.assertFalse(result.approved)
        self.assertIn("Invalid", result.reason)

    def test_rejects_invalid_atr(self):
        rm = make_rm()
        result = rm.check_trade("BTCUSDT", "BUY", price=65000, atr=0, conviction=0.75)
        self.assertFalse(result.approved)

    def test_rejects_insufficient_margin(self):
        rm = make_rm(equity=1.0)
        rm.account.daily_start_equity = 1.0
        result = rm.check_trade("BTCUSDT", "BUY", price=65000, atr=1, conviction=0.75)
        self.assertFalse(result.approved)
        self.assertIn("margin", result.reason.lower())

    def test_rejects_duplicate_symbol(self):
        rm = make_rm()
        check = approved_check(rm)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )
        result = approved_check(rm)
        self.assertFalse(result.approved)
        self.assertIn("already open", result.reason)

    def test_rejects_when_max_positions_reached(self):
        rm = make_rm(equity=1_000_000)
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "PEPEUSDT", "DOGEUSDT"]
        for sym in symbols:
            check = rm.check_trade(sym, "BUY", 100, 1, 0.75)
            if check.approved:
                rm.open_position(
                    symbol=sym, check=check,
                    entry_price=100, stop_loss=99,
                    take_profit=102, partial_close=101,
                    direction="BUY",
                )
        result = rm.check_trade("SHIBUSDT", "BUY", 100, 1, 0.75)
        self.assertFalse(result.approved)
        self.assertIn("Max positions", result.reason)


class T04_KillSwitch(unittest.TestCase):

    def test_kill_switch_activates_on_drawdown(self):
        rm = make_rm(equity=10_000)
        rm.account.daily_start_equity = 10_000
        rm.update_equity(9_100)
        self.assertTrue(rm.account.kill_switch)

    def test_kill_switch_rejects_trades(self):
        rm = make_rm(equity=10_000)
        rm.account.kill_switch = True
        rm._last_reset_day = time.gmtime().tm_yday
        result = rm.check_trade("BTCUSDT", "BUY", 65000, 500, 0.75)
        self.assertFalse(result.approved)
        self.assertIn("Kill switch", result.reason)

    def test_kill_switch_can_be_reset(self):
        rm = make_rm(equity=10_000)
        rm.account.kill_switch = True
        rm.reset_kill_switch()
        self.assertFalse(rm.account.kill_switch)

    def test_kill_switch_threshold(self):
        rm = make_rm(equity=10_000)
        rm.account.daily_start_equity = 10_000
        rm.update_equity(9_201)
        self.assertFalse(rm.account.kill_switch)
        rm.update_equity(9_199)
        self.assertTrue(rm.account.kill_switch)


class T05_PositionManagement(unittest.TestCase):

    def setUp(self):
        self.rm = make_rm(equity=10_000)
        check = approved_check(self.rm)
        self.check = check
        self.pos = self.rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )

    def test_position_is_open(self):
        self.assertTrue(self.pos.is_open)

    def test_position_count_increases(self):
        self.assertEqual(self.rm.account.position_count, 1)

    def test_position_direction(self):
        self.assertEqual(self.pos.direction, "BUY")

    def test_position_entry_price(self):
        self.assertEqual(self.pos.entry_price, 65000)

    def test_unrealised_pnl_profit(self):
        pnl = self.pos.unrealised_pnl(66000)
        self.assertGreater(pnl, 0)

    def test_unrealised_pnl_loss(self):
        pnl = self.pos.unrealised_pnl(64000)
        self.assertLess(pnl, 0)

    def test_close_position_returns_pnl(self):
        pnl = self.rm.close_position("BTCUSDT", exit_price=66000)
        self.assertGreater(pnl, 0)

    def test_close_position_removes_from_tracking(self):
        self.rm.close_position("BTCUSDT", exit_price=66000)
        self.assertEqual(self.rm.account.position_count, 0)

    def test_partial_close(self):
        pnl = self.rm.close_position("BTCUSDT", exit_price=65750, partial=True)
        self.assertGreater(pnl, 0)
        pos = self.rm.account.open_positions.get("BTCUSDT")
        self.assertIsNotNone(pos)
        self.assertTrue(pos.partial_closed)

    def test_notional_value(self):
        self.assertGreater(self.pos.notional, 0)


class T06_AntiLiquidation(unittest.TestCase):

    def test_anti_liq_not_triggered_normal(self):
        rm = make_rm(equity=10_000)
        check = approved_check(rm)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )
        result = rm.check_anti_liquidation("BTCUSDT", current_price=65000)
        self.assertFalse(result)

    def test_anti_liq_triggered_on_big_loss(self):
        rm = make_rm(equity=10_000)
        check = approved_check(rm, conviction=0.9)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=60000,
            take_profit=70000, partial_close=67500,
            direction="BUY",
        )
        result = rm.check_anti_liquidation("BTCUSDT", current_price=1)
        self.assertTrue(result)

    def test_anti_liq_no_position(self):
        rm = make_rm()
        result = rm.check_anti_liquidation("XYZUSDT", current_price=100)
        self.assertFalse(result)


class T07_AccountState(unittest.TestCase):

    def test_initial_equity(self):
        rm = make_rm(equity=50_000)
        self.assertEqual(rm.account.equity, 50_000)

    def test_daily_drawdown_zero_at_start(self):
        rm = make_rm()
        self.assertEqual(rm.account.daily_drawdown_pct, 0.0)

    def test_free_margin_decreases_on_open(self):
        rm = make_rm(equity=10_000)
        free_before = rm.account.free_margin
        check = approved_check(rm)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )
        self.assertLess(rm.account.free_margin, free_before)

    def test_trades_today_increments(self):
        rm = make_rm(equity=10_000)
        check = approved_check(rm)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )
        self.assertEqual(rm.account.trades_today, 1)

    def test_get_open_positions(self):
        rm = make_rm(equity=10_000)
        check = approved_check(rm)
        rm.open_position(
            symbol="BTCUSDT", check=check,
            entry_price=65000, stop_loss=64500,
            take_profit=66000, partial_close=65750,
            direction="BUY",
        )
        positions = rm.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "BTCUSDT")

    def test_sell_direction_pnl(self):
        rm = make_rm(equity=10_000)
        check = rm.check_trade("BTCUSDT", "SELL", 65000, 500, 0.75)
        if check.approved:
            rm.open_position(
                symbol="BTCUSDT", check=check,
                entry_price=65000, stop_loss=65500,
                take_profit=64000, partial_close=64500,
                direction="SELL",
            )
            pnl = rm.close_position("BTCUSDT", exit_price=64000)
            self.assertGreater(pnl, 0)


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 4: Risk Manager Tests")
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
