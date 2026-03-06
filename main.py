from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    ALL_SYMBOLS,
    MEME_SYMBOLS,
    SCALP_SYMBOLS,
    SWING_SYMBOLS,
    SCALP_ATR_STOP_MULT,
    SCALP_ATR_TP_MULT,
    SCALP_MIN_SIGNAL_SCORE,
    SWING_ATR_STOP_MULT,
    SWING_ATR_TP_MULT,
    SWING_MIN_SIGNAL_SCORE,
    LOG_DIR,
    LOG_LEVEL,
)
from config.logging_config import setup_logging
from modules.data_feed import DataFeed
from modules.signal_engine import SignalEngine, SignalResult
from modules.meme_scanner import MemeCoinScanner
from modules.risk_manager import RiskManager
from modules.executor import PaperExecutor
from modules.portfolio_tracker import PortfolioTracker
from modules.dashboard import Dashboard

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="main")


def should_take_scalp(signal: SignalResult) -> bool:
    """Fast scalp — meme coins, tight SL, quick TP."""
    return (
        signal.symbol in SCALP_SYMBOLS and
        signal.is_tradeable and
        signal.conviction >= SCALP_MIN_SIGNAL_SCORE
    )


def should_take_swing(signal: SignalResult) -> bool:
    """Swing trade — core coins, wider SL, bigger TP."""
    return (
        signal.symbol in SWING_SYMBOLS and
        signal.is_tradeable and
        signal.conviction >= SWING_MIN_SIGNAL_SCORE
    )


def main():
    logger.info("=" * 60)
    logger.info("  QuantBot Starting — Scalp + Swing Mode")
    logger.info("=" * 60)

    # ── Step 1: Data Feed ──────────────────────────────────────────
    logger.info("Starting data feed...")
    feed = DataFeed(symbols=ALL_SYMBOLS)
    feed.start()
    logger.info(f"Data feed running for {len(ALL_SYMBOLS)} symbols")

    logger.info("Waiting for data to load...")
    time.sleep(5)

    # ── Step 2: Signal Engine ──────────────────────────────────────
    logger.info("Initialising signal engine...")
    engine = SignalEngine(feed=feed)
    logger.info("Signal engine ready — 11 indicators active")

    # ── Step 3: Meme Scanner ───────────────────────────────────────
    logger.info("Starting meme coin scanner...")
    scanner = MemeCoinScanner(feed=feed, scan_interval=30)
    scanner.start()

    # ── Step 4: Risk Manager ───────────────────────────────────────
    logger.info("Initialising risk manager...")
    rm = RiskManager(initial_equity=100_000.0)

    # ── Step 5: Executor ───────────────────────────────────────────
    executor = PaperExecutor(risk_manager=rm)
    logger.info("Paper executor ready")

    # ── Step 6: Portfolio Tracker ──────────────────────────────────
    tracker = PortfolioTracker(
        executor=executor,
        risk_manager=rm,
        update_interval=5,
    )
    tracker.start()

    # ── Step 7: Dashboard ──────────────────────────────────────────
    dashboard = Dashboard(
        feed=feed, engine=engine, scanner=scanner,
        executor=executor, tracker=tracker, rm=rm,
        host="0.0.0.0", port=5000,
    )
    dashboard.start()

    logger.info("=" * 60)
    logger.info("  All systems running — open http://localhost:5000")
    logger.info("  SCALP mode: " + str(SCALP_SYMBOLS))
    logger.info("  SWING mode: " + str(SWING_SYMBOLS))
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    scan_interval  = 3
    last_scan_time = 0

    try:
        while True:
            now = time.time()

            if now - last_scan_time >= scan_interval:
                last_scan_time = now

                prices = {
                    sym: feed.get_snapshot(sym).get("last_close", 0)
                    for sym in feed.symbols
                }

                # Always manage open positions first
                executor.check_and_manage_positions(prices)

                if dashboard.autopilot:
                    signals = engine.scan_all()

                    for signal in signals:
                        price = prices.get(signal.symbol, 0)
                        if price <= 0:
                            continue

                        # ── Scalp trade ────────────────────────────
                        if should_take_scalp(signal):
                            atr = signal.atr
                            signal.stop_loss = (
                                price - atr * SCALP_ATR_STOP_MULT
                                if signal.direction == "BUY"
                                else price + atr * SCALP_ATR_STOP_MULT
                            )
                            signal.take_profit = (
                                price + atr * SCALP_ATR_TP_MULT
                                if signal.direction == "BUY"
                                else price - atr * SCALP_ATR_TP_MULT
                            )
                            result = executor.execute_signal(
                                signal, current_price=price
                            )
                            if result:
                                logger.info(
                                    f"SCALP: {signal.symbol} "
                                    f"{signal.direction} @ {price:.6f} "
                                    f"conviction={signal.conviction:.2f}"
                                )

                        # ── Swing trade ────────────────────────────
                        elif should_take_swing(signal):
                            atr = signal.atr
                            signal.stop_loss = (
                                price - atr * SWING_ATR_STOP_MULT
                                if signal.direction == "BUY"
                                else price + atr * SWING_ATR_STOP_MULT
                            )
                            signal.take_profit = (
                                price + atr * SWING_ATR_TP_MULT
                                if signal.direction == "BUY"
                                else price - atr * SWING_ATR_TP_MULT
                            )
                            result = executor.execute_signal(
                                signal, current_price=price
                            )
                            if result:
                                logger.info(
                                    f"SWING: {signal.symbol} "
                                    f"{signal.direction} @ {price:.6f} "
                                    f"conviction={signal.conviction:.2f}"
                                )

                perf = executor.get_performance_summary()
                if perf["total_trades"] > 0:
                    logger.info(
                        f"Portfolio: trades={perf['total_trades']} "
                        f"winrate={perf['win_rate']}% "
                        f"pnl=${perf['total_pnl']:.2f} "
                        f"open={perf['open_trades']}"
                    )

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")

    finally:
        logger.info("Shutting down...")
        tracker.stop()
        scanner.stop()
        feed.stop()
        dashboard.stop()

        perf = executor.get_performance_summary()
        logger.info("=" * 60)
        logger.info("  Final Performance Summary")
        logger.info("=" * 60)
        logger.info(f"  Total trades:  {perf['total_trades']}")
        logger.info(f"  Win rate:      {perf['win_rate']}%")
        logger.info(f"  Total PnL:     ${perf['total_pnl']:.2f}")
        logger.info(f"  Profit factor: {perf['profit_factor']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
