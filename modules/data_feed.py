from __future__ import annotations

import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    AGGR_TRADE_BUFFER,
    ALL_SYMBOLS,
    BINANCE_FUTURES_BASE_URL,
    BINANCE_FUTURES_WS_BASE,
    KLINE_BUFFER_SIZE,
    MAX_RECONNECT_RETRIES,
    ORDER_BOOK_DEPTH,
    PING_INTERVAL_S,
    PRIMARY_TF,
    RECONNECT_DELAY_S,
    SECONDARY_TF,
    TERTIARY_TF,
    VOLUME_LOOKBACK,
    WS_STREAM_TIMEOUT_S,
    LOG_DIR,
    LOG_LEVEL,
)
from config.logging_config import setup_logging

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="data_feed")


@dataclass
class Candle:
    symbol:     str
    timeframe:  str
    open_time:  int
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    quote_vol:  float
    trades:     int
    taker_buy_vol:  float
    taker_buy_quote_vol: float
    is_closed:  bool = False

    @property
    def close_time(self) -> int:
        _map = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
        return self.open_time + _map.get(self.timeframe, 60_000)

    @property
    def body_pct(self) -> float:
        rng = self.high - self.low
        if rng == 0:
            return 0.0
        return abs(self.close - self.open) / rng

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class OrderBook:
    symbol:     str
    timestamp:  int
    bids:       List[Tuple[float, float]] = field(default_factory=list)
    asks:       List[Tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if self.bids and self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0

    @property
    def bid_depth(self) -> float:
        return sum(q for _, q in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(q for _, q in self.asks)

    @property
    def imbalance(self) -> float:
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass
class FundingRate:
    symbol:         str
    rate:           float
    next_funding_ms: int


@dataclass
class AggrTrade:
    symbol:      str
    price:       float
    quantity:    float
    is_buy:      bool
    timestamp:   int


def _rest_fetch_klines(
    symbol: str,
    interval: str,
    limit: int = KLINE_BUFFER_SIZE,
    base_url: str = BINANCE_FUTURES_BASE_URL,
) -> List[Candle]:
    try:
        import requests
        url = f"{base_url}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        candles = []
        for r in raw:
            candles.append(Candle(
                symbol    = symbol,
                timeframe = interval,
                open_time = int(r[0]),
                open      = float(r[1]),
                high      = float(r[2]),
                low       = float(r[3]),
                close     = float(r[4]),
                volume    = float(r[5]),
                quote_vol = float(r[7]),
                trades    = int(r[8]),
                taker_buy_vol       = float(r[9]),
                taker_buy_quote_vol = float(r[10]),
                is_closed = True,
            ))
        logger.info(f"[REST] Loaded {len(candles)} {interval} candles for {symbol}")
        return candles
    except Exception as exc:
        logger.warning(f"[REST] kline fetch failed for {symbol}/{interval}: {exc}")
        return []


def _rest_fetch_order_book(
    symbol: str,
    limit: int = ORDER_BOOK_DEPTH,
    base_url: str = BINANCE_FUTURES_BASE_URL,
) -> Optional[OrderBook]:
    try:
        import requests
        url = f"{base_url}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": limit}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return OrderBook(
            symbol    = symbol,
            timestamp = int(time.time() * 1000),
            bids      = [(float(p), float(q)) for p, q in data["bids"]],
            asks      = [(float(p), float(q)) for p, q in data["asks"]],
        )
    except Exception as exc:
        logger.warning(f"[REST] order book fetch failed for {symbol}: {exc}")
        return None


def _rest_fetch_funding_rate(
    symbol: str,
    base_url: str = BINANCE_FUTURES_BASE_URL,
) -> Optional[FundingRate]:
    try:
        import requests
        url = f"{base_url}/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return FundingRate(
            symbol          = symbol,
            rate            = float(data.get("lastFundingRate", 0)),
            next_funding_ms = int(data.get("nextFundingTime", 0)),
        )
    except Exception as exc:
        logger.warning(f"[REST] funding rate fetch failed for {symbol}: {exc}")
        return None


class VWAPCalculator:
    def __init__(self):
        self._cum_tp_vol: float = 0.0
        self._cum_vol:    float = 0.0
        self._session_date: Optional[str] = None

    def update(self, candle: Candle) -> float:
        today = datetime.fromtimestamp(candle.open_time / 1000, tz=timezone.utc).date().isoformat()
        if today != self._session_date:
            self.reset()
            self._session_date = today
        tp = (candle.high + candle.low + candle.close) / 3
        self._cum_tp_vol += tp * candle.volume
        self._cum_vol    += candle.volume
        return self.value

    def reset(self):
        self._cum_tp_vol = 0.0
        self._cum_vol    = 0.0

    @property
    def value(self) -> float:
        return self._cum_tp_vol / self._cum_vol if self._cum_vol > 0 else 0.0


class DataFeed:
    def __init__(
        self,
        symbols: List[str] = ALL_SYMBOLS,
        timeframes: List[str] = None,
        base_url: str = BINANCE_FUTURES_BASE_URL,
        ws_base: str = BINANCE_FUTURES_WS_BASE,
        auto_start: bool = False,
    ):
        self.symbols    = [s.upper() for s in symbols]
        self.timeframes = timeframes or [PRIMARY_TF, SECONDARY_TF, TERTIARY_TF]
        self.base_url   = base_url
        self.ws_base    = ws_base
        self._lock = threading.RLock()

        self._candles: Dict[str, Dict[str, deque]] = {
            s: {tf: deque(maxlen=KLINE_BUFFER_SIZE) for tf in self.timeframes}
            for s in self.symbols
        }
        self._live_candle: Dict[str, Dict[str, Optional[Candle]]] = {
            s: {tf: None for tf in self.timeframes}
            for s in self.symbols
        }
        self._order_books: Dict[str, Optional[OrderBook]] = {s: None for s in self.symbols}
        self._funding: Dict[str, Optional[FundingRate]] = {s: None for s in self.symbols}
        self._aggr_trades: Dict[str, deque] = {
            s: deque(maxlen=AGGR_TRADE_BUFFER) for s in self.symbols
        }
        self._vwap: Dict[str, VWAPCalculator] = {s: VWAPCalculator() for s in self.symbols}
        self._ws_threads: List[threading.Thread] = []
        self._ws_connections = {}
        self._last_heartbeat: Dict[str, float] = {}
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._candle_close_callbacks: List[callable] = []

        if auto_start:
            self.start()

    def start(self):
        if self._running:
            logger.warning("DataFeed already running.")
            return
        self._running = True
        logger.info(f"DataFeed starting for {len(self.symbols)} symbols")
        self._bootstrap_rest()
        self._start_ws_streams()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="ws_watchdog", daemon=True
        )
        self._watchdog_thread.start()
        logger.info("DataFeed started")

    def stop(self):
        self._running = False
        for name, ws in self._ws_connections.items():
            try:
                ws.close()
            except Exception:
                pass
        logger.info("DataFeed stopped.")

    def subscribe_candle_close(self, callback: callable):
        self._candle_close_callbacks.append(callback)

    def _bootstrap_rest(self):
        logger.info("Bootstrapping historical data via REST")
        for symbol in self.symbols:
            for tf in self.timeframes:
                candles = _rest_fetch_klines(symbol, tf, limit=KLINE_BUFFER_SIZE,
                                             base_url=self.base_url)
                with self._lock:
                    for c in candles:
                        self._candles[symbol][tf].append(c)
                        if c.is_closed:
                            self._vwap[symbol].update(c)
            ob = _rest_fetch_order_book(symbol, base_url=self.base_url)
            with self._lock:
                self._order_books[symbol] = ob
            fr = _rest_fetch_funding_rate(symbol, base_url=self.base_url)
            with self._lock:
                self._funding[symbol] = fr

    def _start_ws_streams(self):
        streams = []
        for sym in self.symbols:
            for tf in self.timeframes:
                streams.append(f"{sym.lower()}@kline_{tf}")
        for sym in self.symbols:
            streams.append(f"{sym.lower()}@depth{ORDER_BOOK_DEPTH}@100ms")
        for sym in self.symbols:
            streams.append(f"{sym.lower()}@aggTrade")
        for sym in self.symbols:
            streams.append(f"{sym.lower()}@markPrice@1s")

        chunk_size = 100
        chunks = [streams[i:i+chunk_size] for i in range(0, len(streams), chunk_size)]
        for idx, chunk in enumerate(chunks):
            t = threading.Thread(
                target=self._run_ws_connection,
                args=(chunk, f"ws_conn_{idx}"),
                name=f"ws_conn_{idx}",
                daemon=True,
            )
            self._ws_threads.append(t)
            t.start()

    def _run_ws_connection(self, streams: List[str], conn_name: str):
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed.")
            return

        retry = 0
        while self._running and retry < MAX_RECONNECT_RETRIES:
            url = f"{self.ws_base}/stream?streams=" + "/".join(streams)
            logger.info(f"[{conn_name}] Connecting...")

            def on_message(ws, raw):
                self._last_heartbeat[conn_name] = time.time()
                try:
                    msg = json.loads(raw)
                    data = msg.get("data", msg)
                    stream = msg.get("stream", "")
                    self._dispatch(stream, data)
                except Exception as exc:
                    logger.error(f"[{conn_name}] dispatch error: {exc}")

            def on_error(ws, err):
                logger.warning(f"[{conn_name}] WS error: {err}")

            def on_close(ws, code, reason):
                logger.warning(f"[{conn_name}] WS closed")

            def on_open(ws):
                logger.info(f"[{conn_name}] WS connected")
                self._last_heartbeat[conn_name] = time.time()

            ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            self._ws_connections[conn_name] = ws
            ws.run_forever(ping_interval=PING_INTERVAL_S, ping_timeout=10, reconnect=0)

            if not self._running:
                break
            retry += 1
            delay = min(RECONNECT_DELAY_S * (2 ** retry), 60)
            logger.warning(f"[{conn_name}] Reconnecting in {delay}s")
            time.sleep(delay)

    def _dispatch(self, stream: str, data: dict):
        if "@kline_" in stream:
            self._handle_kline(data)
        elif "@depth" in stream:
            self._handle_depth(data)
        elif "@aggTrade" in stream:
            self._handle_aggr_trade(data)
        elif "@markPrice" in stream:
            self._handle_mark_price(data)

    def _handle_kline(self, data: dict):
        k = data.get("k", {})
        symbol = k.get("s", "").upper()
        tf     = k.get("i", "")
        if symbol not in self._candles or tf not in self._candles[symbol]:
            return
        candle = Candle(
            symbol=symbol, timeframe=tf,
            open_time=int(k["t"]),
            open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]),  close=float(k["c"]),
            volume=float(k["v"]), quote_vol=float(k["q"]),
            trades=int(k["n"]),
            taker_buy_vol=float(k["V"]),
            taker_buy_quote_vol=float(k["Q"]),
            is_closed=bool(k["x"]),
        )
        with self._lock:
            self._live_candle[symbol][tf] = candle
            if candle.is_closed:
                self._candles[symbol][tf].append(candle)
                if tf == PRIMARY_TF:
                    self._vwap[symbol].update(candle)
                for cb in self._candle_close_callbacks:
                    try:
                        cb(candle)
                    except Exception as exc:
                        logger.error(f"Callback error: {exc}")

    def _handle_depth(self, data: dict):
        symbol = data.get("s", "").upper()
        if symbol not in self._order_books:
            return
        ob = OrderBook(
            symbol=symbol,
            timestamp=int(data.get("T", time.time() * 1000)),
            bids=[(float(p), float(q)) for p, q in data.get("b", [])],
            asks=[(float(p), float(q)) for p, q in data.get("a", [])],
        )
        with self._lock:
            self._order_books[symbol] = ob

    def _handle_aggr_trade(self, data: dict):
        symbol = data.get("s", "").upper()
        if symbol not in self._aggr_trades:
            return
        trade = AggrTrade(
            symbol=symbol,
            price=float(data["p"]),
            quantity=float(data["q"]),
            is_buy=not data.get("m", True),
            timestamp=int(data["T"]),
        )
        with self._lock:
            self._aggr_trades[symbol].append(trade)

    def _handle_mark_price(self, data: dict):
        symbol = data.get("s", "").upper()
        if symbol not in self._funding:
            return
        fr = FundingRate(
            symbol=symbol,
            rate=float(data.get("r", 0)),
            next_funding_ms=int(data.get("T", 0)),
        )
        with self._lock:
            self._funding[symbol] = fr

    def _watchdog_loop(self):
        while self._running:
            time.sleep(30)
            now = time.time()
            for name, last_hb in list(self._last_heartbeat.items()):
                if now - last_hb > WS_STREAM_TIMEOUT_S:
                    logger.warning(f"[watchdog] Stream {name} stale")
                    ws = self._ws_connections.get(name)
                    if ws:
                        try:
                            ws.close()
                        except Exception:
                            pass

    def get_candles(self, symbol: str, timeframe: str = PRIMARY_TF, n: int = KLINE_BUFFER_SIZE) -> List[Candle]:
        symbol = symbol.upper()
        with self._lock:
            buf = self._candles.get(symbol, {}).get(timeframe)
            if buf is None:
                return []
            result = list(buf)
        return result[-n:] if len(result) > n else result

    def get_latest_candle(self, symbol: str, timeframe: str = PRIMARY_TF) -> Optional[Candle]:
        candles = self.get_candles(symbol, timeframe, n=1)
        return candles[-1] if candles else None

    def get_live_candle(self, symbol: str, timeframe: str = PRIMARY_TF) -> Optional[Candle]:
        symbol = symbol.upper()
        with self._lock:
            return self._live_candle.get(symbol, {}).get(timeframe)

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        symbol = symbol.upper()
        with self._lock:
            return self._order_books.get(symbol)

    def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        symbol = symbol.upper()
        with self._lock:
            return self._funding.get(symbol)

    def get_aggr_trades(self, symbol: str, n: int = 100) -> List[AggrTrade]:
        symbol = symbol.upper()
        with self._lock:
            buf = self._aggr_trades.get(symbol, deque())
            trades = list(buf)
        return trades[-n:] if len(trades) > n else trades

    def get_vwap(self, symbol: str) -> float:
        symbol = symbol.upper()
        with self._lock:
            return self._vwap.get(symbol, VWAPCalculator()).value

    def get_relative_volume(self, symbol: str, timeframe: str = PRIMARY_TF) -> float:
        candles = self.get_candles(symbol, timeframe, n=VOLUME_LOOKBACK + 1)
        if len(candles) < 2:
            return 1.0
        closed = candles[:-1][-VOLUME_LOOKBACK:]
        live   = candles[-1]
        avg_vol = np.mean([c.volume for c in closed]) if closed else 1.0
        if avg_vol == 0:
            return 1.0
        return live.volume / avg_vol

    def get_order_flow_imbalance(self, symbol: str, n: int = 100) -> float:
        trades = self.get_aggr_trades(symbol, n=n)
        if not trades:
            return 0.0
        buy_vol  = sum(t.quantity for t in trades if t.is_buy)
        sell_vol = sum(t.quantity for t in trades if not t.is_buy)
        total    = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def get_dataframe(self, symbol: str, timeframe: str = PRIMARY_TF, n: int = KLINE_BUFFER_SIZE) -> pd.DataFrame:
        candles = self.get_candles(symbol, timeframe, n=n)
        if not candles:
            return pd.DataFrame()
        rows = [{
            "open_time":           c.open_time,
            "open":                c.open,
            "high":                c.high,
            "low":                 c.low,
            "close":               c.close,
            "volume":              c.volume,
            "quote_vol":           c.quote_vol,
            "trades":              c.trades,
            "taker_buy_vol":       c.taker_buy_vol,
            "taker_buy_quote_vol": c.taker_buy_quote_vol,
        } for c in candles]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df

    def get_snapshot(self, symbol: str) -> dict:
        symbol = symbol.upper()
        candles_1m  = self.get_candles(symbol, "1m",  n=50)
        candles_5m  = self.get_candles(symbol, "5m",  n=20)
        candles_15m = self.get_candles(symbol, "15m", n=10)
        live        = self.get_live_candle(symbol)
        ob          = self.get_order_book(symbol)
        fr          = self.get_funding_rate(symbol)
        last_close  = candles_1m[-1].close if candles_1m else 0.0

        return {
            "symbol":           symbol,
            "timestamp":        int(time.time() * 1000),
            "last_close":       last_close,
            "last_high":        candles_1m[-1].high  if candles_1m else 0.0,
            "last_low":         candles_1m[-1].low   if candles_1m else 0.0,
            "last_open":        candles_1m[-1].open  if candles_1m else 0.0,
            "volume_1m":        candles_1m[-1].volume       if candles_1m else 0.0,
            "quote_volume_1m":  candles_1m[-1].quote_vol    if candles_1m else 0.0,
            "taker_buy_vol":    candles_1m[-1].taker_buy_vol if candles_1m else 0.0,
            "relative_volume":  self.get_relative_volume(symbol, "1m"),
            "best_bid":         ob.best_bid    if ob else 0.0,
            "best_ask":         ob.best_ask    if ob else 0.0,
            "mid_price":        ob.mid_price   if ob else last_close,
            "spread":           ob.spread      if ob else 0.0,
            "ob_imbalance":     ob.imbalance   if ob else 0.0,
            "bid_depth":        ob.bid_depth   if ob else 0.0,
            "ask_depth":        ob.ask_depth   if ob else 0.0,
            "ofi_100":          self.get_order_flow_imbalance(symbol, n=100),
            "ofi_500":          self.get_order_flow_imbalance(symbol, n=500),
            "vwap":             self.get_vwap(symbol),
            "price_vs_vwap":    (last_close - self.get_vwap(symbol)) / max(self.get_vwap(symbol), 1),
            "funding_rate":     fr.rate            if fr else 0.0,
            "next_funding_ms":  fr.next_funding_ms if fr else 0,
            "candles_1m":       candles_1m,
            "candles_5m":       candles_5m,
            "candles_15m":      candles_15m,
            "live_candle":      live,
        }

    def health_check(self) -> dict:
        report = {}
        with self._lock:
            for sym in self.symbols:
                sym_report = {}
                for tf in self.timeframes:
                    buf = self._candles[sym][tf]
                    sym_report[tf] = {
                        "candles": len(buf),
                        "buffer_pct": round(len(buf) / KLINE_BUFFER_SIZE * 100, 1),
                    }
                sym_report["has_order_book"] = self._order_books[sym] is not None
                sym_report["has_funding"]    = self._funding[sym] is not None
                sym_report["aggr_trades"]    = len(self._aggr_trades[sym])
                sym_report["vwap"]           = round(self._vwap[sym].value, 4)
                report[sym] = sym_report
        now = time.time()
        ws_health = {}
        for name, last in self._last_heartbeat.items():
            age = round(now - last, 1)
            ws_health[name] = {"age_s": age, "ok": age < WS_STREAM_TIMEOUT_S}
        report["_ws_health"] = ws_health
        return report


class MockDataFeed(DataFeed):
    def __init__(
        self,
        symbols: List[str] = None,
        seed_price: float = 65_000.0,
        volatility: float = 0.001,
        **kwargs,
    ):
        symbols = symbols or ["BTCUSDT"]
        super().__init__(symbols=symbols, auto_start=False, **kwargs)
        self._seed_prices: Dict[str, float] = {s: seed_price for s in self.symbols}
        self._volatility = volatility
        self._mock_thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._inject_history()
        self._mock_thread = threading.Thread(
            target=self._mock_tick_loop, daemon=True, name="mock_tick"
        )
        self._mock_thread.start()
        logger.info(f"MockDataFeed started for {self.symbols}")

    def _inject_history(self):
        now_ms = int(time.time() * 1000)
        _tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
        for sym in self.symbols:
            price = self._seed_prices[sym]
            for tf in self.timeframes:
                tf_dur = _tf_ms.get(tf, 60_000)
                for i in range(KLINE_BUFFER_SIZE):
                    ret   = random.gauss(0, self._volatility)
                    open_ = price
                    close = price * (1 + ret)
                    high  = max(open_, close) * (1 + abs(random.gauss(0, self._volatility / 2)))
                    low   = min(open_, close) * (1 - abs(random.gauss(0, self._volatility / 2)))
                    vol   = random.uniform(1, 50) * (price / 65000)
                    taker_buy = vol * random.uniform(0.3, 0.7)
                    t     = now_ms - (KLINE_BUFFER_SIZE - i) * tf_dur
                    c = Candle(
                        symbol=sym, timeframe=tf,
                        open_time=t, open=open_, high=high, low=low, close=close,
                        volume=vol, quote_vol=vol * close,
                        trades=random.randint(50, 500),
                        taker_buy_vol=taker_buy,
                        taker_buy_quote_vol=taker_buy * close,
                        is_closed=True,
                    )
                    self._candles[sym][tf].append(c)
                    if tf == PRIMARY_TF:
                        self._vwap[sym].update(c)
                    price = close
                self._seed_prices[sym] = price

            mid = self._seed_prices[sym]
            self._order_books[sym] = OrderBook(
                symbol=sym,
                timestamp=now_ms,
                bids=[(round(mid - i * 0.5, 2), round(random.uniform(0.1, 5), 4))
                      for i in range(1, ORDER_BOOK_DEPTH + 1)],
                asks=[(round(mid + i * 0.5, 2), round(random.uniform(0.1, 5), 4))
                      for i in range(1, ORDER_BOOK_DEPTH + 1)],
            )
            self._funding[sym] = FundingRate(
                symbol=sym,
                rate=random.uniform(-0.001, 0.001),
                next_funding_ms=now_ms + 8 * 3600 * 1000,
            )
            for _ in range(200):
                price_jitter = mid * random.gauss(0, 0.0005)
                is_buy = random.random() > 0.48
                self._aggr_trades[sym].append(AggrTrade(
                    symbol=sym,
                    price=mid + price_jitter,
                    quantity=random.uniform(0.001, 0.5),
                    is_buy=is_buy,
                    timestamp=now_ms - random.randint(0, 60_000),
                ))

    def _mock_tick_loop(self):
        _tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
        while self._running:
            time.sleep(1)
            now_ms = int(time.time() * 1000)
            for sym in self.symbols:
                price = self._seed_prices[sym]
                for tf in self.timeframes:
                    ret   = random.gauss(0, self._volatility)
                    open_ = price
                    close = price * (1 + ret)
                    high  = max(open_, close) * (1 + abs(random.gauss(0, self._volatility / 2)))
                    low   = min(open_, close) * (1 - abs(random.gauss(0, self._volatility / 2)))
                    vol   = random.uniform(1, 50) * (price / 65_000)
                    taker_buy = vol * random.uniform(0.3, 0.7)
                    c = Candle(
                        symbol=sym, timeframe=tf,
                        open_time=now_ms,
                        open=open_, high=high, low=low, close=close,
                        volume=vol, quote_vol=vol * close,
                        trades=random.randint(50, 500),
                        taker_buy_vol=taker_buy,
                        taker_buy_quote_vol=taker_buy * close,
                        is_closed=True,
                    )
                    with self._lock:
                        self._candles[sym][tf].append(c)
                        if tf == PRIMARY_TF:
                            self._vwap[sym].update(c)
                    for cb in self._candle_close_callbacks:
                        try:
                            cb(c)
                        except Exception:
                            pass
                self._seed_prices[sym] = close
                mid = close
                with self._lock:
                    self._order_books[sym] = OrderBook(
                        symbol=sym,
                        timestamp=now_ms,
                        bids=[(round(mid - i * 0.5, 2), round(random.uniform(0.1, 5), 4))
                              for i in range(1, ORDER_BOOK_DEPTH + 1)],
                        asks=[(round(mid + i * 0.5, 2), round(random.uniform(0.1, 5), 4))
                              for i in range(1, ORDER_BOOK_DEPTH + 1)],
                    )
                is_buy = random.random() > 0.48
                with self._lock:
                    self._aggr_trades[sym].append(AggrTrade(
                        symbol=sym,
                        price=close * random.gauss(1, 0.0002),
                        quantity=random.uniform(0.001, 1.0),
                        is_buy=is_buy,
                        timestamp=now_ms,
                    ))