import sys
import os
import time
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import MockDataFeed, Candle, DataFeed
from modules.meme_scanner import (
    MemeCoinScanner,
    MemeOpportunity,
    calc_volume_spike_score,
    calc_price_momentum_score,
    calc_breakout_score,
    calc_taker_aggression,
    calc_candle_strength,
)
from config.settings import KLINE_BUFFER_SIZE


def make_candle(close, high=None, low=None, volume=10.0,
                taker_buy_vol=None, i=0):
    return Candle(
        symbol="PEPEUSDT", timeframe="1m",
        open_time=i * 60_000,
        open=close,
        high=high or close * 1.001,
        low=low or close * 0.999,
        close=close,
        volume=volume,
        quote_vol=close * volume,
        trades=100,
        taker_buy_vol=taker_buy_vol if taker_buy_vol is not None else volume * 0.5,
        taker_buy_quote_vol=close * (taker_buy_vol or volume * 0.5),
        is_closed=True,
    )


def flat_candles(price=1.0, n=30, volume=10.0):
    return [make_candle(price, volume=volume, i=i) for i in range(n)]


def trending_candles(start, end, n=30, volume=10.0):
    prices = [start + (end - start) * i / n for i in range(n)]
    return [make_candle(p, volume=volume, i=i) for i, p in enumerate(prices)]


def spike_candles(base_price=1.0, base_vol=10.0, spike_vol=100.0, n=30):
    candles = flat_candles(base_price, n=n - 1, volume=base_vol)
    candles.append(make_candle(
        base_price * 1.02,
        volume=spike_vol,
        taker_buy_vol=spike_vol * 0.8,
        i=n,
    ))
    return candles


class T01_VolumeSpikeScore(unittest.TestCase):

    def test_no_spike_low_score(self):
        candles = flat_candles(1.0, n=30, volume=10.0)
        result = calc_volume_spike_score(candles)
        self.assertLess(result["score"], 0.6)

    def test_spike_high_score(self):
        candles = spike_candles(spike_vol=100.0)
        result = calc_volume_spike_score(candles)
        self.assertGreater(result["score"], 0.4)

    def test_multiplier_correct(self):
        candles = flat_candles(1.0, n=25, volume=10.0)
        candles.append(make_candle(1.0, volume=50.0, i=25))
        result = calc_volume_spike_score(candles)
        self.assertAlmostEqual(result["multiplier"], 5.0, delta=0.5)

    def test_insufficient_data(self):
        candles = [make_candle(1.0, i=i) for i in range(5)]
        result = calc_volume_spike_score(candles)
        self.assertEqual(result["score"], 0.0)

    def test_extreme_spike_capped_at_1(self):
        candles = spike_candles(spike_vol=10000.0)
        result = calc_volume_spike_score(candles)
        self.assertLessEqual(result["score"], 1.0)


class T02_PriceMomentumScore(unittest.TestCase):

    def test_strong_upward_move(self):
        candles = trending_candles(1.0, 1.1, n=30)
        result = calc_price_momentum_score(candles)
        self.assertGreater(result["score"], 0.3)
        self.assertEqual(result["direction"], "UP")

    def test_strong_downward_move(self):
        candles = trending_candles(1.1, 1.0, n=30)
        result = calc_price_momentum_score(candles)
        self.assertGreater(result["score"], 0.3)
        self.assertEqual(result["direction"], "DOWN")

    def test_flat_low_score(self):
        candles = flat_candles(1.0, n=30)
        result = calc_price_momentum_score(candles)
        self.assertLess(result["score"], 0.4)

    def test_insufficient_data(self):
        candles = [make_candle(1.0, i=i) for i in range(3)]
        result = calc_price_momentum_score(candles)
        self.assertEqual(result["score"], 0.0)

    def test_score_capped_at_1(self):
        candles = trending_candles(1.0, 2.0, n=30)
        result = calc_price_momentum_score(candles)
        self.assertLessEqual(result["score"], 1.0)

    def test_direction_neutral_flat(self):
        candles = flat_candles(1.0, n=30)
        result = calc_price_momentum_score(candles)
        self.assertIn(result["direction"], ["UP", "DOWN", "NEUTRAL"])


class T03_BreakoutScore(unittest.TestCase):

    def test_upside_breakout(self):
        baseline = flat_candles(1.0, n=25)
        breakout = [make_candle(1.5, i=25)]
        candles  = baseline + breakout
        result   = calc_breakout_score(candles)
        self.assertEqual(result["type"], "UPSIDE_BREAKOUT")
        self.assertGreater(result["score"], 0.5)

    def test_downside_breakout(self):
        baseline = flat_candles(1.0, n=25)
        breakout = [make_candle(0.5, i=25)]
        candles  = baseline + breakout
        result   = calc_breakout_score(candles)
        self.assertEqual(result["type"], "DOWNSIDE_BREAKOUT")
        self.assertGreater(result["score"], 0.5)

    def test_no_breakout(self):
        candles = flat_candles(1.0, n=30)
        result  = calc_breakout_score(candles)
        self.assertEqual(result["type"], "NONE")

    def test_score_range(self):
        candles = trending_candles(1.0, 1.5, n=30)
        result  = calc_breakout_score(candles)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_insufficient_data(self):
        candles = [make_candle(1.0, i=i) for i in range(5)]
        result  = calc_breakout_score(candles)
        self.assertEqual(result["score"], 0.0)


class T04_TakerAggression(unittest.TestCase):

    def test_strong_buy_aggression(self):
        candles = [make_candle(1.0, volume=10.0,
                               taker_buy_vol=8.0, i=i) for i in range(10)]
        result = calc_taker_aggression(candles)
        self.assertIn(result["bias"], ["STRONG_BUY", "BUY"])
        self.assertGreater(result["ratio"], 0.6)

    def test_strong_sell_aggression(self):
        candles = [make_candle(1.0, volume=10.0,
                               taker_buy_vol=2.0, i=i) for i in range(10)]
        result = calc_taker_aggression(candles)
        self.assertIn(result["bias"], ["STRONG_SELL", "SELL"])
        self.assertLess(result["ratio"], 0.4)

    def test_neutral_aggression(self):
        candles = [make_candle(1.0, volume=10.0,
                               taker_buy_vol=5.0, i=i) for i in range(10)]
        result = calc_taker_aggression(candles)
        self.assertEqual(result["bias"], "NEUTRAL")

    def test_ratio_range(self):
        candles = flat_candles(1.0, n=10)
        result = calc_taker_aggression(candles)
        self.assertGreaterEqual(result["ratio"], 0.0)
        self.assertLessEqual(result["ratio"], 1.0)


class T05_CandleStrength(unittest.TestCase):

    def test_strong_bullish_candles(self):
        candles = [make_candle(
            close=1.0 + i * 0.01,
            high=1.0 + i * 0.01 + 0.005,
            low=1.0 + i * 0.01 - 0.001,
            i=i
        ) for i in range(5)]
        result = calc_candle_strength(candles)
        self.assertGreater(result["score"], 0.3)
        self.assertGreater(result["bullish_count"], 0)

    def test_mixed_candles_low_score(self):
        candles = []
        for i in range(6):
            if i % 2 == 0:
                candles.append(make_candle(1.01, i=i))
            else:
                candles.append(make_candle(0.99, i=i))
        result = calc_candle_strength(candles)
        self.assertLessEqual(result["score"], 0.7)

    def test_insufficient_data(self):
        candles = [make_candle(1.0, i=i) for i in range(1)]
        result = calc_candle_strength(candles)
        self.assertGreaterEqual(result["score"], 0.0)


class T06_MemeCoinScanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feed = MockDataFeed(
            symbols=["PEPEUSDT", "DOGEUSDT", "SHIBUSDT"],
            seed_price=0.001,
            volatility=0.005,
        )
        cls.feed.start()
        time.sleep(0.5)
        cls.scanner = MemeCoinScanner(cls.feed, scan_interval=60)

    @classmethod
    def tearDownClass(cls):
        cls.feed.stop()

    def test_scan_symbol_returns_opportunity(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertIsInstance(result, MemeOpportunity)

    def test_opportunity_has_symbol(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertEqual(result.symbol, "PEPEUSDT")

    def test_signal_is_valid(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertIn(result.signal, [
            "STRONG_BUY", "BUY", "STRONG_SELL",
            "SELL", "WATCH", "NEUTRAL"
        ])

    def test_total_score_range(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 1.0)

    def test_price_is_positive(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertGreater(result.price, 0.0)

    def test_scan_all_returns_all_symbols(self):
        results = self.scanner.scan_all()
        self.assertEqual(len(results), len(self.feed.symbols))

    def test_scan_all_sorted_by_score(self):
        results = self.scanner.scan_all()
        scores = [r.total_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_reasons_is_list(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertIsInstance(result.reasons, list)

    def test_is_actionable_property(self):
        result = self.scanner.scan_symbol("PEPEUSDT")
        self.assertIsInstance(result.is_actionable, bool)

    def test_volume_spike_detected(self):
        from config.settings import KLINE_BUFFER_SIZE
        sym = "PEPEUSDT"
        spike_vol = 9999.0
        with self.feed._lock:
            buf = self.feed._candles[sym]["1m"]
            last = list(buf)[-1]
            spike = Candle(
                symbol=sym, timeframe="1m",
                open_time=last.open_time + 60_000,
                open=last.close,
                high=last.close * 1.05,
                low=last.close * 0.99,
                close=last.close * 1.03,
                volume=spike_vol,
                quote_vol=last.close * spike_vol,
                trades=5000,
                taker_buy_vol=spike_vol * 0.85,
                taker_buy_quote_vol=last.close * spike_vol * 0.85,
                is_closed=True,
            )
            buf.append(spike)

        result = self.scanner.scan_symbol(sym)
        self.assertGreater(result.volume_spike, 3.0)
        self.assertGreater(result.volume_score, 0.4)


class T07_ScannerBackground(unittest.TestCase):

    def test_scanner_starts_and_stops(self):
        feed    = MockDataFeed(symbols=["PEPEUSDT"])
        feed.start()
        scanner = MemeCoinScanner(feed, scan_interval=1)
        scanner.start()
        time.sleep(2)
        scanner.stop()
        feed.stop()
        self.assertFalse(scanner._running)

    def test_get_latest_after_scan(self):
        feed    = MockDataFeed(symbols=["DOGEUSDT"])
        feed.start()
        scanner = MemeCoinScanner(feed, scan_interval=1)
        scanner.start()
        time.sleep(2)
        result = scanner.get_latest("DOGEUSDT")
        scanner.stop()
        feed.stop()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MemeOpportunity)

    def test_get_top_opportunities(self):
        feed    = MockDataFeed(symbols=["PEPEUSDT", "DOGEUSDT", "SHIBUSDT"])
        feed.start()
        scanner = MemeCoinScanner(feed, scan_interval=1)
        scanner.start()
        time.sleep(2)
        tops = scanner.get_top_opportunities(n=3)
        scanner.stop()
        feed.stop()
        self.assertIsInstance(tops, list)
        self.assertLessEqual(len(tops), 3)


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 3: Meme Coin Scanner Tests")
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