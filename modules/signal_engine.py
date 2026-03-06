from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    RSI_PERIOD, EMA_FAST, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD_DEV,
    STOCH_K_PERIOD, STOCH_D_PERIOD,
    ATR_PERIOD, MIN_SIGNAL_SCORE,
    LOG_DIR, LOG_LEVEL,
)
from config.logging_config import setup_logging
from modules.data_feed import Candle, DataFeed, MockDataFeed

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="signal_engine")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    symbol:          str
    timestamp:       int
    direction:       str        # BUY, SELL, NEUTRAL
    conviction:      float      # 0.0 to 1.0
    leverage_tier:   int        # 10, 20, or 50
    signal_scores:   Dict[str, float]  # individual indicator scores
    atr:             float      # current ATR value
    stop_loss:       float      # suggested stop loss price
    take_profit:     float      # suggested take profit price
    partial_close:   float      # suggested partial close price
    notes:           List[str]  # human readable reasoning

    @property
    def is_tradeable(self) -> bool:
        return self.direction != "NEUTRAL" and self.conviction >= MIN_SIGNAL_SCORE


# ── Indicator Functions ────────────────────────────────────────────────────────

def _closes(candles: List[Candle]) -> np.ndarray:
    return np.array([c.close for c in candles], dtype=float)

def _highs(candles: List[Candle]) -> np.ndarray:
    return np.array([c.high for c in candles], dtype=float)

def _lows(candles: List[Candle]) -> np.ndarray:
    return np.array([c.low for c in candles], dtype=float)

def _volumes(candles: List[Candle]) -> np.ndarray:
    return np.array([c.volume for c in candles], dtype=float)


def calc_rsi(candles: List[Candle], period: int = RSI_PERIOD) -> float:
    """
    Relative Strength Index.
    Returns 0-100. Oversold < 30, Overbought > 70.
    """
    closes = _closes(candles)
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    ema = np.zeros(len(values))
    if len(values) < period:
        return ema
    k = 2 / (period + 1)
    ema[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_ema_crossover(candles: List[Candle]) -> dict:
    """
    EMA fast/slow crossover.
    Returns current fast EMA, slow EMA, and crossover direction.
    """
    closes = _closes(candles)
    fast = calc_ema(closes, EMA_FAST)
    slow = calc_ema(closes, EMA_SLOW)

    if fast[-1] == 0 or slow[-1] == 0:
        return {"fast": 0, "slow": 0, "signal": "NEUTRAL", "gap_pct": 0}

    gap_pct = (fast[-1] - slow[-1]) / slow[-1] * 100
    prev_gap = (fast[-2] - slow[-2]) / slow[-2] * 100 if len(fast) > 1 else 0

    if fast[-1] > slow[-1] and fast[-2] <= slow[-2]:
        signal = "GOLDEN_CROSS"
    elif fast[-1] < slow[-1] and fast[-2] >= slow[-2]:
        signal = "DEATH_CROSS"
    elif fast[-1] > slow[-1]:
        signal = "BULLISH"
    else:
        signal = "BEARISH"

    return {
        "fast":    round(fast[-1], 4),
        "slow":    round(slow[-1], 4),
        "signal":  signal,
        "gap_pct": round(gap_pct, 4),
    }


def calc_macd(candles: List[Candle]) -> dict:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal
    """
    closes = _closes(candles)
    fast_ema = calc_ema(closes, MACD_FAST)
    slow_ema = calc_ema(closes, MACD_SLOW)
    macd_line = fast_ema - slow_ema

    valid = macd_line[MACD_SLOW - 1:]
    if len(valid) < MACD_SIGNAL:
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}

    signal_line = calc_ema(valid, MACD_SIGNAL)
    histogram   = valid - signal_line

    trend = "BULLISH" if histogram[-1] > 0 else "BEARISH"
    if histogram[-1] > 0 and histogram[-2] <= 0:
        trend = "BULLISH_CROSS"
    elif histogram[-1] < 0 and histogram[-2] >= 0:
        trend = "BEARISH_CROSS"

    return {
        "macd":      round(float(valid[-1]), 6),
        "signal":    round(float(signal_line[-1]), 6),
        "histogram": round(float(histogram[-1]), 6),
        "trend":     trend,
    }


def calc_bollinger_bands(candles: List[Candle]) -> dict:
    """
    Bollinger Bands: middle = SMA(20), upper/lower = middle ± 2σ
    %B = (price - lower) / (upper - lower)
    """
    closes = _closes(candles)
    if len(closes) < BB_PERIOD:
        return {"upper": 0, "middle": 0, "lower": 0, "pct_b": 0.5, "bandwidth": 0}

    middle = np.mean(closes[-BB_PERIOD:])
    std    = np.std(closes[-BB_PERIOD:])
    upper  = middle + BB_STD_DEV * std
    lower  = middle - BB_STD_DEV * std
    price  = closes[-1]

    pct_b = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    bandwidth = (upper - lower) / middle * 100

    return {
        "upper":     round(upper, 4),
        "middle":    round(middle, 4),
        "lower":     round(lower, 4),
        "pct_b":     round(pct_b, 4),
        "bandwidth": round(bandwidth, 4),
        "price":     round(price, 4),
    }


def calc_stochastic(candles: List[Candle]) -> dict:
    """
    Stochastic Oscillator.
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)
    """
    if len(candles) < STOCH_K_PERIOD + STOCH_D_PERIOD:
        return {"k": 50.0, "d": 50.0, "signal": "NEUTRAL"}

    closes = _closes(candles)
    highs  = _highs(candles)
    lows   = _lows(candles)

    k_values = []
    for i in range(STOCH_K_PERIOD - 1, len(candles)):
        highest = np.max(highs[i - STOCH_K_PERIOD + 1: i + 1])
        lowest  = np.min(lows[i  - STOCH_K_PERIOD + 1: i + 1])
        if highest == lowest:
            k_values.append(50.0)
        else:
            k = (closes[i] - lowest) / (highest - lowest) * 100
            k_values.append(k)

    k = k_values[-1]
    d = np.mean(k_values[-STOCH_D_PERIOD:]) if len(k_values) >= STOCH_D_PERIOD else k

    if k < 20 and d < 20:
        signal = "OVERSOLD"
    elif k > 80 and d > 80:
        signal = "OVERBOUGHT"
    elif k > d:
        signal = "BULLISH"
    else:
        signal = "BEARISH"

    return {
        "k":      round(k, 2),
        "d":      round(d, 2),
        "signal": signal,
    }


def calc_atr(candles: List[Candle], period: int = ATR_PERIOD) -> float:
    """
    Average True Range.
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = EMA(TR, period)
    """
    if len(candles) < period + 1:
        highs  = _highs(candles)
        lows   = _lows(candles)
        return float(np.mean(highs - lows)) if len(candles) > 0 else 0.0

    highs  = _highs(candles)
    lows   = _lows(candles)
    closes = _closes(candles)

    tr_list = []
    for i in range(1, len(candles)):
        hl  = highs[i]  - lows[i]
        hpc = abs(highs[i]  - closes[i - 1])
        lpc = abs(lows[i]   - closes[i - 1])
        tr_list.append(max(hl, hpc, lpc))

    tr = np.array(tr_list)
    atr_vals = calc_ema(tr, period)
    return float(atr_vals[-1]) if atr_vals[-1] > 0 else float(np.mean(tr[-period:]))


def calc_vwap_signal(snapshot: dict) -> dict:
    """
    VWAP signal: where is price relative to VWAP?
    Above VWAP = bullish bias, below = bearish.
    """
    price     = snapshot.get("last_close", 0)
    vwap      = snapshot.get("vwap", 0)
    pct_diff  = snapshot.get("price_vs_vwap", 0)

    if vwap == 0:
        return {"signal": "NEUTRAL", "pct_diff": 0}

    if pct_diff > 0.002:
        signal = "ABOVE_VWAP"
    elif pct_diff < -0.002:
        signal = "BELOW_VWAP"
    else:
        signal = "AT_VWAP"

    return {
        "price":    round(price, 4),
        "vwap":     round(vwap, 4),
        "pct_diff": round(pct_diff * 100, 4),
        "signal":   signal,
    }


def calc_order_flow_signal(snapshot: dict) -> dict:
    """
    Order flow imbalance signal.
    Uses both short (100 trade) and long (500 trade) windows.
    """
    ofi_100 = snapshot.get("ofi_100", 0)
    ofi_500 = snapshot.get("ofi_500", 0)
    ob_imb  = snapshot.get("ob_imbalance", 0)

    combined = (ofi_100 * 0.4 + ofi_500 * 0.3 + ob_imb * 0.3)

    if combined > 0.2:
        signal = "BUY_PRESSURE"
    elif combined < -0.2:
        signal = "SELL_PRESSURE"
    else:
        signal = "NEUTRAL"

    return {
        "ofi_100":  round(ofi_100, 4),
        "ofi_500":  round(ofi_500, 4),
        "ob_imb":   round(ob_imb, 4),
        "combined": round(combined, 4),
        "signal":   signal,
    }


def calc_volume_profile(candles: List[Candle], snapshot: dict) -> dict:
    """
    Volume profile signal.
    Looks at relative volume and taker buy ratio.
    """
    rel_vol       = snapshot.get("relative_volume", 1.0)
    taker_buy     = snapshot.get("taker_buy_vol", 0)
    total_vol     = snapshot.get("volume_1m", 1)
    taker_ratio   = taker_buy / total_vol if total_vol > 0 else 0.5

    if rel_vol > 3.0 and taker_ratio > 0.6:
        signal = "HIGH_BUY_VOLUME"
    elif rel_vol > 3.0 and taker_ratio < 0.4:
        signal = "HIGH_SELL_VOLUME"
    elif rel_vol > 1.5:
        signal = "ELEVATED_VOLUME"
    else:
        signal = "NORMAL_VOLUME"

    return {
        "relative_volume": round(rel_vol, 2),
        "taker_ratio":     round(taker_ratio, 4),
        "signal":          signal,
    }


def calc_momentum_divergence(candles: List[Candle]) -> dict:
    """
    Momentum divergence: compare price momentum vs RSI momentum.
    Bullish divergence = price making lower lows but RSI making higher lows.
    Bearish divergence = price making higher highs but RSI making lower highs.
    """
    if len(candles) < 20:
        return {"signal": "NEUTRAL", "price_momentum": 0, "rsi_momentum": 0}

    recent   = candles[-10:]
    previous = candles[-20:-10]

    price_now  = np.mean([c.close for c in recent])
    price_prev = np.mean([c.close for c in previous])
    price_momentum = (price_now - price_prev) / price_prev if price_prev > 0 else 0

    rsi_now  = calc_rsi(recent,   period=min(RSI_PERIOD, len(recent) - 1))
    rsi_prev = calc_rsi(previous, period=min(RSI_PERIOD, len(previous) - 1))
    rsi_momentum = (rsi_now - rsi_prev) / 100

    if price_momentum < -0.001 and rsi_momentum > 0.02:
        signal = "BULLISH_DIVERGENCE"
    elif price_momentum > 0.001 and rsi_momentum < -0.02:
        signal = "BEARISH_DIVERGENCE"
    elif price_momentum > 0 and rsi_momentum > 0:
        signal = "CONFIRMED_BULLISH"
    elif price_momentum < 0 and rsi_momentum < 0:
        signal = "CONFIRMED_BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "signal":          signal,
        "price_momentum":  round(price_momentum * 100, 4),
        "rsi_momentum":    round(rsi_momentum * 100, 4),
    }


def calc_funding_sentiment(snapshot: dict) -> dict:
    """
    Funding rate sentiment.
    Positive funding = longs paying shorts = bullish but crowded.
    Negative funding = shorts paying longs = bearish but crowded.
    Extreme funding = contrarian signal.
    """
    rate = snapshot.get("funding_rate", 0)

    if rate > 0.001:
        signal = "CROWDED_LONG"
    elif rate < -0.001:
        signal = "CROWDED_SHORT"
    elif rate > 0.0003:
        signal = "MILD_BULLISH"
    elif rate < -0.0003:
        signal = "MILD_BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "rate":   round(rate, 6),
        "signal": signal,
    }


# ── Scoring Engine ─────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Combines all 11 indicators into a single conviction score.

    Each indicator returns a score from -1.0 (strong sell) to +1.0 (strong buy).
    Scores are weighted by reliability and combined into a final score.
    Final score is then mapped to BUY/SELL/NEUTRAL with a conviction level.

    Indicator weights (must sum to 1.0):
        RSI              0.10
        EMA Crossover    0.12
        MACD             0.12
        Bollinger Bands  0.08
        Stochastic       0.08
        ATR              0.05  (volatility filter only)
        VWAP             0.10
        Order Flow       0.15
        Volume Profile   0.08
        Momentum Div     0.07
        Funding Rate     0.05
    """

    WEIGHTS = {
        "rsi":         0.10,
        "ema":         0.12,
        "macd":        0.12,
        "bb":          0.08,
        "stoch":       0.08,
        "vwap":        0.10,
        "order_flow":  0.15,
        "volume":      0.08,
        "momentum":    0.07,
        "funding":     0.05,
        "atr_filter":  0.05,
    }

    def __init__(self, feed: DataFeed):
        self.feed = feed

    def analyse(self, symbol: str) -> SignalResult:
        """
        Run all indicators on a symbol and return a SignalResult.
        This is the main method called by the Executor and Dashboard.
        """
        snap     = self.feed.get_snapshot(symbol)
        candles  = snap.get("candles_1m", [])
        candles5 = snap.get("candles_5m", [])

        if len(candles) < 30:
            return self._neutral(symbol, snap, "Insufficient candle history")

        scores = {}
        notes  = []

        # ── 1. RSI ─────────────────────────────────────────────────────────────
        rsi = calc_rsi(candles)
        if rsi < 30:
            scores["rsi"] = 0.8
            notes.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            scores["rsi"] = -0.8
            notes.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:
            scores["rsi"] = 0.3
        elif rsi > 55:
            scores["rsi"] = -0.3
        else:
            scores["rsi"] = 0.0

        # ── 2. EMA Crossover ───────────────────────────────────────────────────
        ema = calc_ema_crossover(candles)
        ema_map = {
            "GOLDEN_CROSS": 1.0,
            "BULLISH":       0.5,
            "NEUTRAL":       0.0,
            "BEARISH":      -0.5,
            "DEATH_CROSS":  -1.0,
        }
        scores["ema"] = ema_map.get(ema["signal"], 0.0)
        if ema["signal"] in ("GOLDEN_CROSS", "DEATH_CROSS"):
            notes.append(f"EMA {ema['signal']}")

        # ── 3. MACD ────────────────────────────────────────────────────────────
        macd = calc_macd(candles)
        macd_map = {
            "BULLISH_CROSS":  1.0,
            "BULLISH":        0.5,
            "NEUTRAL":        0.0,
            "BEARISH":       -0.5,
            "BEARISH_CROSS": -1.0,
        }
        scores["macd"] = macd_map.get(macd["trend"], 0.0)
        if macd["trend"] in ("BULLISH_CROSS", "BEARISH_CROSS"):
            notes.append(f"MACD {macd['trend']}")

        # ── 4. Bollinger Bands ─────────────────────────────────────────────────
        bb = calc_bollinger_bands(candles)
        pct_b = bb["pct_b"]
        if pct_b < 0.05:
            scores["bb"] = 0.9
            notes.append("Price at lower Bollinger Band")
        elif pct_b > 0.95:
            scores["bb"] = -0.9
            notes.append("Price at upper Bollinger Band")
        elif pct_b < 0.2:
            scores["bb"] = 0.4
        elif pct_b > 0.8:
            scores["bb"] = -0.4
        else:
            scores["bb"] = 0.0

        # ── 5. Stochastic ──────────────────────────────────────────────────────
        stoch = calc_stochastic(candles)
        stoch_map = {
            "OVERSOLD":   0.9,
            "BULLISH":    0.4,
            "NEUTRAL":    0.0,
            "BEARISH":   -0.4,
            "OVERBOUGHT":-0.9,
        }
        scores["stoch"] = stoch_map.get(stoch["signal"], 0.0)

        # ── 6. ATR Filter ──────────────────────────────────────────────────────
        atr = calc_atr(candles)
        price = snap.get("last_close", 1)
        atr_pct = atr / price if price > 0 else 0
        if atr_pct < 0.0005:
            scores["atr_filter"] = -0.5
            notes.append("Low volatility — ATR filter reducing score")
        elif atr_pct > 0.005:
            scores["atr_filter"] = 0.5
            notes.append(f"High volatility ATR={atr:.4f}")
        else:
            scores["atr_filter"] = 0.2

        # ── 7. VWAP ────────────────────────────────────────────────────────────
        vwap_sig = calc_vwap_signal(snap)
        vwap_map = {
            "ABOVE_VWAP": 0.7,
            "AT_VWAP":    0.0,
            "BELOW_VWAP":-0.7,
        }
        scores["vwap"] = vwap_map.get(vwap_sig["signal"], 0.0)

        # ── 8. Order Flow ──────────────────────────────────────────────────────
        of = calc_order_flow_signal(snap)
        of_map = {
            "BUY_PRESSURE":  0.9,
            "NEUTRAL":       0.0,
            "SELL_PRESSURE": -0.9,
        }
        scores["order_flow"] = of_map.get(of["signal"], 0.0)
        if of["signal"] != "NEUTRAL":
            notes.append(f"Order flow: {of['signal']} ({of['combined']:.2f})")

        # ── 9. Volume Profile ──────────────────────────────────────────────────
        vol = calc_volume_profile(candles, snap)
        vol_map = {
            "HIGH_BUY_VOLUME":  0.9,
            "ELEVATED_VOLUME":  0.3,
            "NORMAL_VOLUME":    0.0,
            "HIGH_SELL_VOLUME": -0.9,
        }
        scores["volume"] = vol_map.get(vol["signal"], 0.0)
        if vol["signal"] in ("HIGH_BUY_VOLUME", "HIGH_SELL_VOLUME"):
            notes.append(f"Volume spike: {vol['signal']} (relVol={vol['relative_volume']}x)")

        # ── 10. Momentum Divergence ────────────────────────────────────────────
        mom = calc_momentum_divergence(candles)
        mom_map = {
            "BULLISH_DIVERGENCE":  0.9,
            "CONFIRMED_BULLISH":   0.5,
            "NEUTRAL":             0.0,
            "CONFIRMED_BEARISH":  -0.5,
            "BEARISH_DIVERGENCE": -0.9,
        }
        scores["momentum"] = mom_map.get(mom["signal"], 0.0)
        if "DIVERGENCE" in mom["signal"]:
            notes.append(f"Momentum: {mom['signal']}")

        # ── 11. Funding Rate Sentiment ─────────────────────────────────────────
        fund = calc_funding_sentiment(snap)
        fund_map = {
            "CROWDED_SHORT":  0.6,
            "MILD_BEARISH":   0.2,
            "NEUTRAL":        0.0,
            "MILD_BULLISH":  -0.2,
            "CROWDED_LONG":  -0.6,
        }
        scores["funding"] = fund_map.get(fund["signal"], 0.0)

        # ── Weighted Conviction Score ──────────────────────────────────────────
        raw_score = sum(
            scores[k] * self.WEIGHTS[k]
            for k in scores
            if k in self.WEIGHTS
        )

        # Normalise to 0-1 range
        # raw_score is in [-1, +1], map to [0, 1]
        conviction = (raw_score + 1) / 2

        # ── Direction ──────────────────────────────────────────────────────────
        if raw_score >= 0.15:
            direction = "BUY"
        elif raw_score <= -0.15:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        # ── Leverage Tier ──────────────────────────────────────────────────────
        abs_conv = abs(raw_score)
        if abs_conv >= 0.80:
            leverage_tier = 50
        elif abs_conv >= 0.65:
            leverage_tier = 20
        else:
            leverage_tier = 10

        # ── Stop Loss / Take Profit ────────────────────────────────────────────
        from config.settings import ATR_STOP_MULT, ATR_TP_MULT, ATR_PARTIAL_CLOSE_MULT
        if direction == "BUY":
            stop_loss     = price - atr * ATR_STOP_MULT
            take_profit   = price + atr * ATR_TP_MULT
            partial_close = price + atr * ATR_PARTIAL_CLOSE_MULT
        elif direction == "SELL":
            stop_loss     = price + atr * ATR_STOP_MULT
            take_profit   = price - atr * ATR_TP_MULT
            partial_close = price - atr * ATR_PARTIAL_CLOSE_MULT
        else:
            stop_loss     = 0.0
            take_profit   = 0.0
            partial_close = 0.0

        return SignalResult(
            symbol        = symbol,
            timestamp     = int(time.time() * 1000),
            direction     = direction,
            conviction    = round(conviction, 4),
            leverage_tier = leverage_tier,
            signal_scores = {k: round(v, 4) for k, v in scores.items()},
            atr           = round(atr, 6),
            stop_loss     = round(stop_loss, 4),
            take_profit   = round(take_profit, 4),
            partial_close = round(partial_close, 4),
            notes         = notes,
        )

    def scan_all(self) -> List[SignalResult]:
        """
        Run analyse() on every symbol in the feed.
        Returns results sorted by conviction, highest first.
        """
        results = []
        for sym in self.feed.symbols:
            try:
                result = self.analyse(sym)
                results.append(result)
            except Exception as exc:
                logger.error(f"Signal engine error for {sym}: {exc}")
        results.sort(key=lambda r: r.conviction, reverse=True)
        return results

    def _neutral(self, symbol: str, snap: dict, reason: str) -> SignalResult:
        price = snap.get("last_close", 0)
        return SignalResult(
            symbol        = symbol,
            timestamp     = int(time.time() * 1000),
            direction     = "NEUTRAL",
            conviction    = 0.5,
            leverage_tier = 10,
            signal_scores = {},
            atr           = 0.0,
            stop_loss     = 0.0,
            take_profit   = 0.0,
            partial_close = 0.0,
            notes         = [reason],
        )