from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    MEME_SYMBOLS,
    VOLUME_SPIKE_MULT,
    MIN_PRICE_MOVE_PCT,
    LOG_DIR,
    LOG_LEVEL,
)
from config.logging_config import setup_logging
from modules.data_feed import DataFeed, MockDataFeed, Candle

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="meme_scanner")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class MemeOpportunity:
    symbol:           str
    timestamp:        int
    price:            float
    volume_spike:     float    # how many times above average volume
    price_move_pct:   float    # % price move in last candle
    momentum_score:   float    # 0 to 1
    volume_score:     float    # 0 to 1
    breakout_score:   float    # 0 to 1
    total_score:      float    # combined 0 to 1
    signal:           str      # STRONG_BUY, BUY, WATCH, NEUTRAL
    reasons:          List[str]

    @property
    def is_actionable(self) -> bool:
        return self.signal in ("STRONG_BUY", "BUY")


# ── Scanner Functions ──────────────────────────────────────────────────────────

def calc_volume_spike_score(candles: List[Candle], lookback: int = 20) -> dict:
    """
    Detects unusual volume spikes.
    Compares current bar volume to rolling average.
    Returns spike multiplier and score 0 to 1.
    """
    if len(candles) < lookback + 1:
        return {"multiplier": 1.0, "score": 0.0}

    recent   = candles[-1]
    baseline = candles[-(lookback + 1):-1]
    avg_vol  = np.mean([c.volume for c in baseline])

    if avg_vol == 0:
        return {"multiplier": 1.0, "score": 0.0}

    multiplier = recent.volume / avg_vol

    if multiplier >= 10.0:
        score = 1.0
    elif multiplier >= VOLUME_SPIKE_MULT:
        score = 0.5 + (multiplier - VOLUME_SPIKE_MULT) / (10.0 - VOLUME_SPIKE_MULT) * 0.5
    else:
        score = max(0.0, multiplier / VOLUME_SPIKE_MULT * 0.5)

    return {
        "multiplier": round(multiplier, 2),
        "score":      round(min(score, 1.0), 4),
    }


def calc_price_momentum_score(candles: List[Candle], lookback: int = 5) -> dict:
    """
    Measures recent price momentum.
    Looks at price change over last N candles and acceleration.
    Returns direction, pct change, and score 0 to 1.
    """
    if len(candles) < lookback + 1:
        return {"pct_change": 0.0, "score": 0.0, "direction": "NEUTRAL"}

    price_now  = candles[-1].close
    price_prev = candles[-(lookback + 1)].close

    if price_prev == 0:
        return {"pct_change": 0.0, "score": 0.0, "direction": "NEUTRAL"}

    pct_change = (price_now - price_prev) / price_prev

    abs_change = abs(pct_change)
    if abs_change >= 0.05:
        score = 1.0
    elif abs_change >= 0.02:
        score = 0.7 + (abs_change - 0.02) / 0.03 * 0.3
    elif abs_change >= MIN_PRICE_MOVE_PCT:
        score = 0.3 + (abs_change - MIN_PRICE_MOVE_PCT) / (0.02 - MIN_PRICE_MOVE_PCT) * 0.4
    else:
        score = abs_change / MIN_PRICE_MOVE_PCT * 0.3

    direction = "UP" if pct_change > 0 else "DOWN" if pct_change < 0 else "NEUTRAL"

    return {
        "pct_change": round(pct_change * 100, 4),
        "score":      round(min(score, 1.0), 4),
        "direction":  direction,
    }


def calc_breakout_score(candles: List[Candle], lookback: int = 20) -> dict:
    """
    Detects price breakouts above recent highs or below recent lows.
    A breakout with volume confirmation is a strong meme coin signal.
    """
    if len(candles) < lookback + 1:
        return {"score": 0.0, "type": "NONE", "level": 0.0}

    recent   = candles[-1]
    baseline = candles[-(lookback + 1):-1]

    highest_high = max(c.high  for c in baseline)
    lowest_low   = min(c.low   for c in baseline)
    avg_range    = np.mean([c.high - c.low for c in baseline])

    if recent.close > highest_high:
        breakout_size = (recent.close - highest_high) / avg_range if avg_range > 0 else 0
        score = min(0.5 + breakout_size * 0.5, 1.0)
        btype = "UPSIDE_BREAKOUT"
        level = highest_high
    elif recent.close < lowest_low:
        breakout_size = (lowest_low - recent.close) / avg_range if avg_range > 0 else 0
        score = min(0.5 + breakout_size * 0.5, 1.0)
        btype = "DOWNSIDE_BREAKOUT"
        level = lowest_low
    else:
        distance_to_high = (highest_high - recent.close) / avg_range if avg_range > 0 else 1
        score = max(0.0, 0.3 - distance_to_high * 0.1)
        btype = "NONE"
        level = highest_high

    return {
        "score": round(min(score, 1.0), 4),
        "type":  btype,
        "level": round(level, 6),
    }


def calc_taker_aggression(candles: List[Candle], lookback: int = 5) -> dict:
    """
    Measures how aggressively takers are buying vs selling.
    High taker buy ratio = aggressive buyers entering = bullish.
    """
    if len(candles) < lookback:
        return {"ratio": 0.5, "score": 0.0, "bias": "NEUTRAL"}

    recent = candles[-lookback:]
    total_vol     = sum(c.volume for c in recent)
    taker_buy_vol = sum(c.taker_buy_vol for c in recent)

    if total_vol == 0:
        return {"ratio": 0.5, "score": 0.0, "bias": "NEUTRAL"}

    ratio = taker_buy_vol / total_vol

    if ratio >= 0.7:
        score = 1.0
        bias  = "STRONG_BUY"
    elif ratio >= 0.6:
        score = 0.7
        bias  = "BUY"
    elif ratio <= 0.3:
        score = 1.0
        bias  = "STRONG_SELL"
    elif ratio <= 0.4:
        score = 0.7
        bias  = "SELL"
    else:
        score = 0.0
        bias  = "NEUTRAL"

    return {
        "ratio": round(ratio, 4),
        "score": round(score, 4),
        "bias":  bias,
    }


def calc_candle_strength(candles: List[Candle], lookback: int = 3) -> dict:
    """
    Measures candle body strength over recent bars.
    Strong consecutive bullish candles = momentum continuation signal.
    """
    if len(candles) < lookback:
        return {"score": 0.0, "bullish_count": 0, "avg_body_pct": 0.0}

    recent = candles[-lookback:]
    bullish_count = sum(1 for c in recent if c.is_bullish)
    avg_body_pct  = np.mean([c.body_pct for c in recent])

    if bullish_count == lookback and avg_body_pct > 0.6:
        score = 1.0
    elif bullish_count == lookback:
        score = 0.7
    elif bullish_count >= lookback - 1:
        score = 0.4
    else:
        score = 0.0

    return {
        "score":         round(score, 4),
        "bullish_count": bullish_count,
        "avg_body_pct":  round(avg_body_pct, 4),
    }


# ── Main Scanner Class ─────────────────────────────────────────────────────────

class MemeCoinScanner:
    """
    Continuously scans meme coins for high probability setups.

    Scoring breakdown:
        Volume spike      30%  — is unusual volume present?
        Price momentum    25%  — is price moving aggressively?
        Breakout          20%  — is price breaking key levels?
        Taker aggression  15%  — are buyers/sellers piling in?
        Candle strength   10%  — are candles strong and directional?

    Signals:
        STRONG_BUY  total_score >= 0.75 and direction UP
        BUY         total_score >= 0.55 and direction UP
        WATCH       total_score >= 0.40
        NEUTRAL     everything else
    """

    WEIGHTS = {
        "volume":     0.30,
        "momentum":   0.25,
        "breakout":   0.20,
        "aggression": 0.15,
        "candle":     0.10,
    }

    def __init__(self, feed: DataFeed, scan_interval: int = 30):
        self.feed          = feed
        self.scan_interval = scan_interval
        self._results:  Dict[str, MemeOpportunity] = {}
        self._lock      = threading.RLock()
        self._running   = False
        self._thread:   Optional[threading.Thread] = None

    def start(self):
        """Start background scanning loop."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._scan_loop,
            name="meme_scanner",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"MemeCoinScanner started — scanning every {self.scan_interval}s")

    def stop(self):
        self._running = False
        logger.info("MemeCoinScanner stopped.")

    def scan_symbol(self, symbol: str) -> MemeOpportunity:
        """
        Run full scan on a single symbol.
        Returns a MemeOpportunity with scores and signal.
        """
        candles = self.feed.get_candles(symbol, "1m", n=50)
        snap    = self.feed.get_snapshot(symbol)
        price   = snap.get("last_close", 0.0)

        if len(candles) < 25:
            return self._neutral_opportunity(symbol, price, "Insufficient data")

        reasons = []

        # ── Component scores ──────────────────────────────────────────────────
        vol_data  = calc_volume_spike_score(candles)
        mom_data  = calc_price_momentum_score(candles)
        brk_data  = calc_breakout_score(candles)
        agg_data  = calc_taker_aggression(candles)
        cnd_data  = calc_candle_strength(candles)

        vol_score = vol_data["score"]
        mom_score = mom_data["score"]
        brk_score = brk_data["score"]
        agg_score = agg_data["score"]
        cnd_score = cnd_data["score"]

        # ── Combined score ────────────────────────────────────────────────────
        total = (
            vol_score * self.WEIGHTS["volume"]   +
            mom_score * self.WEIGHTS["momentum"] +
            brk_score * self.WEIGHTS["breakout"] +
            agg_score * self.WEIGHTS["aggression"] +
            cnd_score * self.WEIGHTS["candle"]
        )

        # ── Build reasons ─────────────────────────────────────────────────────
        if vol_data["multiplier"] >= VOLUME_SPIKE_MULT:
            reasons.append(
                f"Volume spike {vol_data['multiplier']}x above average"
            )
        if abs(mom_data["pct_change"]) >= MIN_PRICE_MOVE_PCT * 100:
            reasons.append(
                f"Price move {mom_data['pct_change']:+.2f}% in last 5 bars"
            )
        if brk_data["type"] != "NONE":
            reasons.append(f"Breakout detected: {brk_data['type']}")
        if agg_data["bias"] in ("STRONG_BUY", "STRONG_SELL"):
            reasons.append(f"Taker aggression: {agg_data['bias']} ratio={agg_data['ratio']}")
        if cnd_data["bullish_count"] >= 3:
            reasons.append(
                f"Strong candles: {cnd_data['bullish_count']} bullish bars"
            )

        # ── Signal ────────────────────────────────────────────────────────────
        direction = mom_data["direction"]
        if total >= 0.75 and direction == "UP":
            signal = "STRONG_BUY"
        elif total >= 0.55 and direction == "UP":
            signal = "BUY"
        elif total >= 0.75 and direction == "DOWN":
            signal = "STRONG_SELL"
        elif total >= 0.55 and direction == "DOWN":
            signal = "SELL"
        elif total >= 0.40:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        return MemeOpportunity(
            symbol         = symbol,
            timestamp      = int(time.time() * 1000),
            price          = round(price, 8),
            volume_spike   = vol_data["multiplier"],
            price_move_pct = mom_data["pct_change"],
            momentum_score = round(mom_score, 4),
            volume_score   = round(vol_score, 4),
            breakout_score = round(brk_score, 4),
            total_score    = round(total, 4),
            signal         = signal,
            reasons        = reasons,
        )

    def scan_all(self) -> List[MemeOpportunity]:
        """
        Scan all symbols in the feed.
        Returns list sorted by total score, highest first.
        """
        results = []
        for sym in self.feed.symbols:
            try:
                opp = self.scan_symbol(sym)
                results.append(opp)
                with self._lock:
                    self._results[sym] = opp
            except Exception as exc:
                logger.error(f"Scan error for {sym}: {exc}")

        results.sort(key=lambda x: x.total_score, reverse=True)
        return results

    def get_top_opportunities(self, n: int = 3) -> List[MemeOpportunity]:
        """Returns top N actionable opportunities."""
        with self._lock:
            all_results = list(self._results.values())
        actionable = [r for r in all_results if r.is_actionable]
        actionable.sort(key=lambda x: x.total_score, reverse=True)
        return actionable[:n]

    def get_latest(self, symbol: str) -> Optional[MemeOpportunity]:
        """Returns the most recent scan result for a symbol."""
        with self._lock:
            return self._results.get(symbol.upper())

    def _scan_loop(self):
        """Background loop that scans all symbols periodically."""
        while self._running:
            try:
                results = self.scan_all()
                actionable = [r for r in results if r.is_actionable]
                if actionable:
                    logger.info(
                        f"Meme scan complete — "
                        f"{len(actionable)} actionable opportunities found"
                    )
                    for opp in actionable[:3]:
                        logger.info(
                            f"  {opp.symbol}: {opp.signal} "
                            f"score={opp.total_score:.2f} "
                            f"volSpike={opp.volume_spike:.1f}x "
                            f"move={opp.price_move_pct:+.2f}%"
                        )
            except Exception as exc:
                logger.error(f"Scan loop error: {exc}")
            time.sleep(self.scan_interval)

    def _neutral_opportunity(
        self, symbol: str, price: float, reason: str
    ) -> MemeOpportunity:
        return MemeOpportunity(
            symbol         = symbol,
            timestamp      = int(time.time() * 1000),
            price          = price,
            volume_spike   = 1.0,
            price_move_pct = 0.0,
            momentum_score = 0.0,
            volume_score   = 0.0,
            breakout_score = 0.0,
            total_score    = 0.0,
            signal         = "NEUTRAL",
            reasons        = [reason],
        )