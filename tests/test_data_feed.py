import sys
import os
import time
import threading
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_feed import (
    Candle,
    OrderBook,
    FundingRate,
    AggrTrade,
    VWAPCalculator,
    MockDataFeed,
    DataFeed,
)
from config.settings import KLINE_BUFFER_SIZE, PRIMARY_TF, VOLUME_LOOKBACK


def make_candle(
    symbol="BTCUSDT", tf="1m",
    open_=100.0, high=105.0, low=98.0, close=103.0,
    volume=10.0, quote_vol=1030.0, trades=100,
    taker_buy_vol=6.0, taker_buy_quote_vol=618.0,
    open_time=None, is_closed=True,
) -> Candle:
    if open_time is None:
        open_time = int(time.time() * 1000)
    return Candle(
        symbol=symbol, timeframe=tf,
        open_time=open_time,
        open=open_, high=high, low=low, close=close,
        volume=volume, quote_vol=quote_vol,
        trades=trades,
        taker_buy_vol=taker_buy_vol,
        taker_buy_quote_vol=taker_buy_quote_vol,
        is_closed=is_closed,
    )


def make_order_book(
    symbol="BTCUSDT",
    bid_price=100.0, ask_price=100.1,
    bid_qty=10.0, ask_qty=5.0, levels=5,
) -> OrderBook:
    bids = [(bid_price - i * 0.1, bid_qty) for i in range(levels)]
    asks = [(ask_price + i * 0.1, ask_qty) for i in range(levels)]
    return OrderBook(
        symbol=symbol,
        timestamp=int(time.time() * 1000),
        bids=bids, asks=asks,
    )


class T01_CandleProperties(unittest.TestCase):

    def setUp(self):
        self.bull = make_candle(open_=100, high=110, low=95, close=108)
        self.bear = make_candle(open_=100, high=105, low=90, close=91)
        self.doji = make_candle(open_=100, high=105, low=95, close=100)

    def test_is_bullish(self):
        self.assertTrue(self.bull.is_bullish)
        self.assertFalse(self.bear.is_bullish)
        self.assertTrue(self.doji.is_bullish)

    def test_body_pct_bullish(self):
        self.assertAlmostEqual(self.bull.body_pct, 8 / 15, places=3)

    def test_body_pct_doji(self):
        self.assertEqual(self.doji.body_pct, 0.0)

    def test_body_pct_zero_range(self):
        flat = make_candle(open_=100, high=100, low=100, close=100)
        self.assertEqual(flat.body_pct, 0.0)

    def test_close_time_1m(self):
        c = make_candle(tf="1m", open_time=0)
        self.assertEqual(c.close_time, 60_000)

    def test_close_time_5m(self):
        c = make_candle(tf="5m", open_time=0)
        self.assertEqual(c.close_time, 300_000)


class T02_OrderBookProperties(unittest.TestCase):

    def test_best_bid_ask(self):
        ob = make_order_book(bid_price=100.0, ask_price=100.5)
        self.assertAlmostEqual(ob.best_bid, 100.0)
        self.assertAlmostEqual(ob.best_ask, 100.5)

    def test_mid_price(self):
        ob = make_order_book(bid_price=100.0, ask_price=101.0)
        self.assertAlmostEqual(ob.mid_price, 100.5)

    def test_spread(self):
        ob = make_order_book(bid_price=100.0, ask_price=100.2)
        self.assertAlmostEqual(ob.spread, 0.2, places=5)

    def test_imbalance_equal(self):
        ob = make_order_book(bid_qty=10.0, ask_qty=10.0, levels=5)
        self.assertAlmostEqual(ob.imbalance, 0.0, places=5)

    def test_imbalance_fully_bid(self):
        ob = make_order_book(bid_qty=100.0, ask_qty=0.0, levels=5)
        self.assertAlmostEqual(ob.imbalance, 1.0, places=5)

    def test_imbalance_fully_ask(self):
        ob = make_order_book(bid_qty=0.0, ask_qty=100.0, levels=5)
        self.assertAlmostEqual(ob.imbalance, -1.0, places=5)

    def test_empty_order_book(self):
        ob = OrderBook(symbol="BTCUSDT", timestamp=0, bids=[], asks=[])
        self.assertEqual(ob.best_bid, 0.0)
        self.assertEqual(ob.best_ask, 0.0)
        self.assertEqual(ob.imbalance, 0.0)


class T03_VWAPCalculator(unittest.TestCase):

    def test_single_candle_vwap(self):
        calc = VWAPCalculator()
        c = make_candle(high=110, low=90, close=100, volume=10)
        vwap = calc.update(c)
        self.assertAlmostEqual(vwap, 100.0, places=5)

    def test_two_candle_vwap(self):
        calc = VWAPCalculator()
        c1 = make_candle(open_time=1_000_000_000, high=110, low=90,  close=100, volume=10)
        c2 = make_candle(open_time=1_000_060_000, high=120, low=100, close=110, volume=20)
        calc.update(c1)
        vwap = calc.update(c2)
        tp1 = (110 + 90  + 100) / 3
        tp2 = (120 + 100 + 110) / 3
        expected = (tp1 * 10 + tp2 * 20) / 30
        self.assertAlmostEqual(vwap, expected, places=5)

    def test_vwap_resets_on_new_session(self):
        calc = VWAPCalculator()
        day1_ms = int(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        c1 = make_candle(open_time=day1_ms, high=110, low=90, close=100, volume=10)
        calc.update(c1)
        day2_ms = int(datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        c2 = make_candle(open_time=day2_ms, high=200, low=180, close=190, volume=5)
        vwap_day2 = calc.update(c2)
        tp2 = (200 + 180 + 190) / 3
        self.assertAlmostEqual(vwap_day2, tp2, places=5)

    def test_vwap_zero_volume(self):
        calc = VWAPCalculator()
        c = make_candle(volume=0)
        calc.update(c)
        self.assertEqual(calc.value, 0.0)


class T04_MockDataFeedHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feed = MockDataFeed(symbols=["BTCUSDT"], seed_price=65_000)
        cls.feed.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.feed.stop()

    def test_candle_buffer_filled(self):
        candles = self.feed.get_candles("BTCUSDT", "1m")
        self.assertGreater(len(candles), 0)

    def test_order_book_present(self):
        ob = self.feed.get_order_book("BTCUSDT")
        self.assertIsNotNone(ob)
        self.assertGreater(len(ob.bids), 0)

    def test_funding_rate_present(self):
        fr = self.feed.get_funding_rate("BTCUSDT")
        self.assertIsNotNone(fr)

    def test_aggr_trades_present(self):
        trades = self.feed.get_aggr_trades("BTCUSDT")
        self.assertGreater(len(trades), 0)


class T05_BufferAndOrdering(unittest.TestCase):

    def test_buffer_does_not_exceed_max(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        feed.start()
        candles = feed.get_candles("BTCUSDT", "1m")
        self.assertLessEqual(len(candles), KLINE_BUFFER_SIZE)
        feed.stop()

    def test_timestamps_ascending(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        feed.start()
        candles = feed.get_candles("BTCUSDT", "1m", n=50)
        times = [c.open_time for c in candles]
        self.assertEqual(times, sorted(times))
        feed.stop()


class T06_DataFrame(unittest.TestCase):

    def setUp(self):
        self.feed = MockDataFeed(symbols=["BTCUSDT"])
        self.feed.start()

    def tearDown(self):
        self.feed.stop()

    def test_columns_present(self):
        df = self.feed.get_dataframe("BTCUSDT")
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, df.columns)

    def test_high_gte_low(self):
        df = self.feed.get_dataframe("BTCUSDT")
        self.assertTrue((df["high"] >= df["low"]).all())

    def test_no_nulls(self):
        df = self.feed.get_dataframe("BTCUSDT")
        self.assertFalse(df.isnull().any().any())


class T07_RelativeVolume(unittest.TestCase):

    def test_returns_positive_float(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        feed.start()
        rv = feed.get_relative_volume("BTCUSDT")
        self.assertGreater(rv, 0.0)
        feed.stop()

    def test_sparse_history_returns_1(self):
        fresh = MockDataFeed.__new__(MockDataFeed)
        DataFeed.__init__(fresh, symbols=["XYZUSDT"], auto_start=False)
        rv = fresh.get_relative_volume("XYZUSDT")
        self.assertEqual(rv, 1.0)


class T08_OrderFlowImbalance(unittest.TestCase):

    def _feed_with_trades(self, buy_ratio):
        from collections import deque
        feed = MockDataFeed(symbols=["BTCUSDT"])
        feed.start()
        with feed._lock:
            feed._aggr_trades["BTCUSDT"] = deque(maxlen=1000)
            for i in range(200):
                is_buy = i < int(200 * buy_ratio)
                feed._aggr_trades["BTCUSDT"].append(AggrTrade(
                    symbol="BTCUSDT", price=65000,
                    quantity=1.0, is_buy=is_buy,
                    timestamp=int(time.time() * 1000),
                ))
        return feed

    def test_all_buys(self):
        feed = self._feed_with_trades(1.0)
        self.assertAlmostEqual(feed.get_order_flow_imbalance("BTCUSDT", 200), 1.0, places=5)
        feed.stop()

    def test_all_sells(self):
        feed = self._feed_with_trades(0.0)
        self.assertAlmostEqual(feed.get_order_flow_imbalance("BTCUSDT", 200), -1.0, places=5)
        feed.stop()

    def test_balanced(self):
        feed = self._feed_with_trades(0.5)
        self.assertAlmostEqual(feed.get_order_flow_imbalance("BTCUSDT", 200), 0.0, places=3)
        feed.stop()


class T09_Snapshot(unittest.TestCase):

    def setUp(self):
        self.feed = MockDataFeed(symbols=["BTCUSDT"])
        self.feed.start()

    def tearDown(self):
        self.feed.stop()

    def test_minimum_data_points(self):
        import numbers
        snap = self.feed.get_snapshot("BTCUSDT")
        count = sum(1 for v in snap.values() if isinstance(v, numbers.Number))
        self.assertGreaterEqual(count, 15)

    def test_required_fields(self):
        snap = self.feed.get_snapshot("BTCUSDT")
        for field in ["symbol", "last_close", "vwap", "ob_imbalance", "ofi_100", "funding_rate"]:
            self.assertIn(field, snap)

    def test_ask_greater_than_bid(self):
        snap = self.feed.get_snapshot("BTCUSDT")
        self.assertGreater(snap["best_ask"], snap["best_bid"])


class T10_Callbacks(unittest.TestCase):

    def test_callback_fires(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        received = []
        feed.subscribe_candle_close(lambda c: received.append(c))
        feed.start()
        deadline = time.time() + 3
        while not received and time.time() < deadline:
            time.sleep(0.1)
        feed.stop()
        self.assertGreater(len(received), 0)

    def test_bad_callback_doesnt_crash(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        good = []
        feed.subscribe_candle_close(lambda c: (_ for _ in ()).throw(RuntimeError("oops")))
        feed.subscribe_candle_close(lambda c: good.append(c))
        feed.start()
        time.sleep(1.5)
        feed.stop()
        self.assertGreater(len(good), 0)


class T11_ThreadSafety(unittest.TestCase):

    def test_concurrent_reads(self):
        feed = MockDataFeed(symbols=["BTCUSDT", "ETHUSDT"])
        feed.start()
        errors = []
        stop = threading.Event()

        def reader(sym):
            while not stop.is_set():
                try:
                    feed.get_candles(sym, "1m")
                    feed.get_order_book(sym)
                    feed.get_snapshot(sym)
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=reader, args=(s,), daemon=True)
                   for s in ["BTCUSDT", "ETHUSDT", "BTCUSDT"]]
        for t in threads:
            t.start()
        time.sleep(2)
        stop.set()
        for t in threads:
            t.join(timeout=3)
        feed.stop()
        self.assertEqual(errors, [])


class T12_HealthCheck(unittest.TestCase):

    def test_health_structure(self):
        feed = MockDataFeed(symbols=["BTCUSDT"])
        feed.start()
        health = feed.health_check()
        self.assertIn("BTCUSDT", health)
        self.assertIn("1m", health["BTCUSDT"])
        self.assertIn("buffer_pct", health["BTCUSDT"]["1m"])
        self.assertGreater(health["BTCUSDT"]["vwap"], 0)
        feed.stop()


class T13_Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feed = MockDataFeed(
            symbols=["BTCUSDT", "ETHUSDT", "PEPEUSDT"],
            seed_price=65_000,
        )
        cls.feed.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        cls.feed.stop()

    def _check(self, sym):
        self.assertGreater(len(self.feed.get_candles(sym, "1m")), 0)
        self.assertFalse(self.feed.get_dataframe(sym).empty)
        ob = self.feed.get_order_book(sym)
        self.assertIsNotNone(ob)
        self.assertGreater(ob.mid_price, 0)
        self.assertGreater(self.feed.get_vwap(sym), 0)

    def test_btcusdt(self):
        self._check("BTCUSDT")

    def test_ethusdt(self):
        self._check("ETHUSDT")

    def test_pepeusdt(self):
        self._check("PEPEUSDT")


if __name__ == "__main__":
    print("=" * 60)
    print("  QuantBot — Module 1: Data Feed Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
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