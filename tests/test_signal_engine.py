import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import MockDataFeed, Candle
from modules.signal_engine import (
    SignalEngine, SignalResult,
    calc_rsi, calc_ema, calc_ema_crossover,
    calc_macd, calc_bollinger_bands,
    calc_stochastic, calc_atr,
    calc_vwap_signal, calc_order_flow_signal,
    calc_volume_profile, calc_momentum_divergence,
    calc_funding_sentiment,
)


def make_candle(close, high=None, low=None, open_=None, volume=10.0, i=0):
    return Candle(
        symbol="BTCUSDT", timeframe="1m",
        open_time=i * 60_000,
        open=open_ or close,
        high=high or close * 1.001,
        low=low or close * 0.999,
        close=close,
        volume=volume,
        quote_vol=close * volume,
        trades=100,
        taker_buy_vol=volume * 0.6,
        taker_buy_quote_vol=close * volume * 0.6,
        is_closed=True,
    )


def trending_candles(start, end, n=50):
    prices = [start + (end - start) * i / n for i in range(n)]
    return [make_candle(p, i=i) for i, p in enumerate(prices)]


def flat_candles(price=100.0, n=50):
    return [make_candle(price, i=i) for i in range(n)]


class T01_RSI(unittest.TestCase):

    def test_oversold(self):
        candles = trending_candles(100, 70, n=50)
        rsi = calc_rsi(candles)
        self.assertLess(rsi, 40)

    def test_overbought(self):
        candles = trending_candles(70, 100, n=50)
        rsi = calc_rsi(candles)
        self.assertGreater(rsi, 60)

    def test_neutral(self):
        candles = flat_candles(100.0, n=50)
        rsi = calc_rsi(candles)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    def test_range(self):
        candles = trending_candles(100, 200, n=50)
        rsi = calc_rsi(candles)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    def test_insufficient_data(self):
        candles = [make_candle(100, i=i) for i in range(3)]
        rsi = calc_rsi(candles)
        self.assertEqual(rsi, 50.0)


class T02_EMA(unittest.TestCase):

    def test_ema_length(self):
        import numpy as np
        values = np.array([float(i) for i in range(50)])
        ema = calc_ema(values, 9)
        self.assertEqual(len(ema), 50)

    def test_ema_follows_trend(self):
        import numpy as np
        values = np.array([float(i) for i in range(50)])
        ema = calc_ema(values, 9)
        self.assertGreater(ema[-1], ema[20])

    def test_crossover_bullish(self):
        candles = trending_candles(100, 200, n=50)
        result = calc_ema_crossover(candles)
        self.assertIn(result["signal"], ["BULLISH", "GOLDEN_CROSS"])

    def test_crossover_bearish(self):
        candles = trending_candles(200, 100, n=50)
        result = calc_ema_crossover(candles)
        self.assertIn(result["signal"], ["BEARISH", "DEATH_CROSS"])


class T03_MACD(unittest.TestCase):

    def test_bullish_trend(self):
        candles = trending_candles(100, 200, n=60)
        result = calc_macd(candles)
        self.assertIn(result["trend"], ["BULLISH", "BULLISH_CROSS"])

    def test_bearish_trend(self):
        candles = trending_candles(200, 100, n=60)
        result = calc_macd(candles)
        self.assertIn(result["trend"], ["BEARISH", "BEARISH_CROSS"])

    def test_has_all_keys(self):
        candles = trending_candles(100, 150, n=60)
        result = calc_macd(candles)
        for key in ["macd", "signal", "histogram", "trend"]:
            self.assertIn(key, result)

    def test_insufficient_data(self):
        candles = [make_candle(100, i=i) for i in range(5)]
        result = calc_macd(candles)
        self.assertEqual(result["macd"], 0)


class T04_BollingerBands(unittest.TestCase):

    def test_has_all_keys(self):
        candles = flat_candles(100.0, n=30)
        result = calc_bollinger_bands(candles)
        for key in ["upper", "middle", "lower", "pct_b", "bandwidth"]:
            self.assertIn(key, result)

    def test_upper_greater_than_lower(self):
        candles = trending_candles(95, 105, n=30)
        result = calc_bollinger_bands(candles)
        self.assertGreater(result["upper"], result["lower"])

    def test_pct_b_range(self):
        candles = flat_candles(100.0, n=30)
        result = calc_bollinger_bands(candles)
        self.assertGreaterEqual(result["pct_b"], 0.0)
        self.assertLessEqual(result["pct_b"], 1.0)

    def test_price_at_lower_band(self):
        candles = trending_candles(200, 100, n=30)
        result = calc_bollinger_bands(candles)
        self.assertLess(result["pct_b"], 0.5)


class T05_Stochastic(unittest.TestCase):

    def test_oversold(self):
        candles = trending_candles(100, 70, n=30)
        result = calc_stochastic(candles)
        self.assertIn(result["signal"], ["OVERSOLD", "BEARISH"])

    def test_overbought(self):
        candles = trending_candles(70, 100, n=30)
        result = calc_stochastic(candles)
        self.assertIn(result["signal"], ["OVERBOUGHT", "BULLISH"])

    def test_k_range(self):
        candles = flat_candles(100.0, n=30)
        result = calc_stochastic(candles)
        self.assertGreaterEqual(result["k"], 0)
        self.assertLessEqual(result["k"], 100)


class T06_ATR(unittest.TestCase):

    def test_atr_positive(self):
        candles = trending_candles(100, 200, n=30)
        atr = calc_atr(candles)
        self.assertGreater(atr, 0)

    def test_high_volatility_higher_atr(self):
        low_vol  = [make_candle(100, high=100.1, low=99.9, i=i) for i in range(30)]
        high_vol = [make_candle(100, high=105.0, low=95.0, i=i) for i in range(30)]
        self.assertGreater(calc_atr(high_vol), calc_atr(low_vol))

    def test_insufficient_data(self):
        candles = [make_candle(100, i=i) for i in range(3)]
        atr = calc_atr(candles)
        self.assertGreaterEqual(atr, 0)


class T07_VWAPSignal(unittest.TestCase):

    def test_above_vwap(self):
        snap = {"last_close": 102.0, "vwap": 100.0, "price_vs_vwap": 0.02}
        result = calc_vwap_signal(snap)
        self.assertEqual(result["signal"], "ABOVE_VWAP")

    def test_below_vwap(self):
        snap = {"last_close": 98.0, "vwap": 100.0, "price_vs_vwap": -0.02}
        result = calc_vwap_signal(snap)
        self.assertEqual(result["signal"], "BELOW_VWAP")

    def test_at_vwap(self):
        snap = {"last_close": 100.0, "vwap": 100.0, "price_vs_vwap": 0.0}
        result = calc_vwap_signal(snap)
        self.assertEqual(result["signal"], "AT_VWAP")

    def test_zero_vwap(self):
        snap = {"last_close": 100.0, "vwap": 0.0, "price_vs_vwap": 0.0}
        result = calc_vwap_signal(snap)
        self.assertEqual(result["signal"], "NEUTRAL")


class T08_OrderFlow(unittest.TestCase):

    def test_buy_pressure(self):
        snap = {"ofi_100": 0.8, "ofi_500": 0.6, "ob_imbalance": 0.5}
        result = calc_order_flow_signal(snap)
        self.assertEqual(result["signal"], "BUY_PRESSURE")

    def test_sell_pressure(self):
        snap = {"ofi_100": -0.8, "ofi_500": -0.6, "ob_imbalance": -0.5}
        result = calc_order_flow_signal(snap)
        self.assertEqual(result["signal"], "SELL_PRESSURE")

    def test_neutral(self):
        snap = {"ofi_100": 0.0, "ofi_500": 0.0, "ob_imbalance": 0.0}
        result = calc_order_flow_signal(snap)
        self.assertEqual(result["signal"], "NEUTRAL")


class T09_VolumeProfile(unittest.TestCase):

    def test_high_buy_volume(self):
        candles = flat_candles(100.0, n=30)
        snap = {"relative_volume": 4.0, "taker_buy_vol": 8.0, "volume_1m": 10.0}
        result = calc_volume_profile(candles, snap)
        self.assertEqual(result["signal"], "HIGH_BUY_VOLUME")

    def test_high_sell_volume(self):
        candles = flat_candles(100.0, n=30)
        snap = {"relative_volume": 4.0, "taker_buy_vol": 2.0, "volume_1m": 10.0}
        result = calc_volume_profile(candles, snap)
        self.assertEqual(result["signal"], "HIGH_SELL_VOLUME")

    def test_normal_volume(self):
        candles = flat_candles(100.0, n=30)
        snap = {"relative_volume": 1.0, "taker_buy_vol": 5.0, "volume_1m": 10.0}
        result = calc_volume_profile(candles, snap)
        self.assertEqual(result["signal"], "NORMAL_VOLUME")


class T10_MomentumDivergence(unittest.TestCase):

    def test_confirmed_bullish(self):
        candles = trending_candles(100, 150, n=30)
        result = calc_momentum_divergence(candles)
        self.assertIn(result["signal"], ["CONFIRMED_BULLISH", "NEUTRAL"])

    def test_confirmed_bearish(self):
        candles = trending_candles(150, 100, n=30)
        result = calc_momentum_divergence(candles)
        self.assertIn(result["signal"], ["CONFIRMED_BEARISH", "NEUTRAL"])

    def test_insufficient_data(self):
        candles = [make_candle(100, i=i) for i in range(5)]
        result = calc_momentum_divergence(candles)
        self.assertEqual(result["signal"], "NEUTRAL")


class T11_FundingSentiment(unittest.TestCase):

    def test_crowded_long(self):
        snap = {"funding_rate": 0.002}
        result = calc_funding_sentiment(snap)
        self.assertEqual(result["signal"], "CROWDED_LONG")

    def test_crowded_short(self):
        snap = {"funding_rate": -0.002}
        result = calc_funding_sentiment(snap)
        self.assertEqual(result["signal"], "CROWDED_SHORT")

    def test_neutral(self):
        snap = {"funding_rate": 0.0}
        result = calc_funding_sentiment(snap)
        self.assertEqual(result["signal"], "NEUTRAL")


class T12_SignalEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feed = MockDataFeed(
            symbols=["BTCUSDT", "ETHUSDT"],
            seed_price=65_000,
        )
        cls.feed.start()
        time.sleep(0.5)
        cls.engine = SignalEngine(cls.feed)

    @classmethod
    def tearDownClass(cls):
        cls.feed.stop()

    def test_analyse_returns_signal_result(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertIsInstance(result, SignalResult)

    def test_direction_is_valid(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertIn(result.direction, ["BUY", "SELL", "NEUTRAL"])

    def test_conviction_range(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertGreaterEqual(result.conviction, 0.0)
        self.assertLessEqual(result.conviction, 1.0)

    def test_leverage_tier_valid(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertIn(result.leverage_tier, [10, 20, 50])

    def test_atr_positive(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertGreater(result.atr, 0)

    def test_signal_scores_has_all_indicators(self):
        result = self.engine.analyse("BTCUSDT")
        for key in ["rsi", "ema", "macd", "bb", "stoch", "vwap",
                    "order_flow", "volume", "momentum", "funding"]:
            self.assertIn(key, result.signal_scores)

    def test_stop_loss_correct_side_buy(self):
        result = self.engine.analyse("BTCUSDT")
        if result.direction == "BUY":
            snap = self.feed.get_snapshot("BTCUSDT")
            self.assertLess(result.stop_loss, snap["last_close"])

    def test_stop_loss_correct_side_sell(self):
        result = self.engine.analyse("BTCUSDT")
        if result.direction == "SELL":
            snap = self.feed.get_snapshot("BTCUSDT")
            self.assertGreater(result.stop_loss, snap["last_close"])

    def test_scan_all_returns_list(self):
        results = self.engine.scan_all()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.feed.symbols))

    def test_scan_all_sorted_by_conviction(self):
        results = self.engine.scan_all()
        convictions = [r.conviction for r in results]
        self.assertEqual(convictions, sorted(convictions, reverse=True))

    def test_is_tradeable(self):
        result = self.engine.analyse("BTCUSDT")
        if result.direction != "NEUTRAL":
            self.assertIsInstance(result.is_tradeable, bool)

    def test_notes_is_list(self):
        result = self.engine.analyse("BTCUSDT")
        self.assertIsInstance(result.notes, list)


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 2: Signal Engine Tests")
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