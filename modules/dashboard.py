from __future__ import annotations

import sys
import os
import time
import threading
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import LOG_DIR, LOG_LEVEL, ALL_SYMBOLS
from config.logging_config import setup_logging
from modules.data_feed import DataFeed, MockDataFeed
from modules.signal_engine import SignalEngine
from modules.meme_scanner import MemeCoinScanner
from modules.risk_manager import RiskManager
from modules.executor import PaperExecutor
from modules.portfolio_tracker import PortfolioTracker

logger = setup_logging(log_dir=LOG_DIR, level=LOG_LEVEL, module_name="dashboard")

try:
    from flask import Flask, jsonify, request, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.error("Flask not installed — run: pip install flask")


# ── HTML Template ──────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantBot Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: #0a0e1a;
            color: #e0e6f0;
            font-family: 'Segoe UI', monospace;
            font-size: 13px;
        }

        /* ── Header ── */
        .header {
            background: #0d1525;
            border-bottom: 1px solid #1e2d45;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-title {
            font-size: 18px;
            font-weight: 700;
            color: #00d4ff;
            letter-spacing: 2px;
        }

        .header-subtitle {
            font-size: 11px;
            color: #5a7a9a;
            margin-top: 2px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }

        .status-dot.danger { background: #ff4444; animation: none; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* ── Layout ── */
        .main {
            display: grid;
            grid-template-columns: 1fr 340px;
            grid-template-rows: auto auto auto;
            gap: 12px;
            padding: 12px;
            height: calc(100vh - 60px);
        }

        /* ── Panels ── */
        .panel {
            background: #0d1525;
            border: 1px solid #1e2d45;
            border-radius: 8px;
            overflow: hidden;
        }

        .panel-header {
            background: #111d30;
            padding: 8px 14px;
            font-size: 11px;
            font-weight: 600;
            color: #5a9fd4;
            letter-spacing: 1px;
            text-transform: uppercase;
            border-bottom: 1px solid #1e2d45;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-body { padding: 12px; }

        /* ── Stats Bar ── */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
            grid-column: 1 / -1;
        }

        .stat-card {
            background: #0d1525;
            border: 1px solid #1e2d45;
            border-radius: 8px;
            padding: 12px 14px;
        }

        .stat-label {
            font-size: 10px;
            color: #5a7a9a;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 700;
            margin-top: 4px;
            color: #e0e6f0;
        }

        .stat-value.positive { color: #00ff88; }
        .stat-value.negative { color: #ff4444; }
        .stat-value.warning  { color: #ffaa00; }
        .stat-value.danger   { color: #ff4444; }

        /* ── Chart ── */
        .chart-container {
            grid-column: 1;
            grid-row: 2 / 4;
        }

        #chart {
            width: 100%;
            height: 380px;
        }

        /* ── Symbol selector ── */
        .symbol-tabs {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }

        .symbol-tab {
            padding: 3px 10px;
            border-radius: 4px;
            border: 1px solid #1e2d45;
            background: transparent;
            color: #5a7a9a;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }

        .symbol-tab:hover, .symbol-tab.active {
            background: #00d4ff22;
            border-color: #00d4ff;
            color: #00d4ff;
        }

        /* ── Right panel ── */
        .right-panel {
            grid-column: 2;
            grid-row: 2 / 4;
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
        }

        /* ── Tables ── */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        th {
            text-align: left;
            padding: 6px 8px;
            color: #5a7a9a;
            font-weight: 600;
            border-bottom: 1px solid #1e2d45;
            font-size: 10px;
            text-transform: uppercase;
        }

        td {
            padding: 6px 8px;
            border-bottom: 1px solid #111d30;
        }

        tr:hover td { background: #111d30; }

        .badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }

        .badge-buy    { background: #00ff8822; color: #00ff88; }
        .badge-sell   { background: #ff444422; color: #ff4444; }
        .badge-neutral{ background: #ffaa0022; color: #ffaa00; }
        .badge-watch  { background: #00d4ff22; color: #00d4ff; }

        /* ── Controls ── */
        .controls {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            padding: 10px 14px;
        }

        .btn {
            padding: 7px 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }

        .btn-primary {
            background: #00d4ff22;
            color: #00d4ff;
            border: 1px solid #00d4ff44;
        }

        .btn-primary:hover { background: #00d4ff44; }

        .btn-success {
            background: #00ff8822;
            color: #00ff88;
            border: 1px solid #00ff8844;
        }

        .btn-success:hover { background: #00ff8844; }

        .btn-danger {
            background: #ff444422;
            color: #ff4444;
            border: 1px solid #ff444444;
        }

        .btn-danger:hover { background: #ff444444; }

        .btn-warning {
            background: #ffaa0022;
            color: #ffaa00;
            border: 1px solid #ffaa0044;
        }

        /* ── Toggle ── */
        .toggle-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .toggle {
            width: 44px;
            height: 24px;
            background: #1e2d45;
            border-radius: 12px;
            cursor: pointer;
            position: relative;
            transition: background 0.3s;
        }

        .toggle.on { background: #00ff8866; }

        .toggle-knob {
            width: 18px;
            height: 18px;
            background: #5a7a9a;
            border-radius: 50%;
            position: absolute;
            top: 3px;
            left: 3px;
            transition: all 0.3s;
        }

        .toggle.on .toggle-knob {
            left: 23px;
            background: #00ff88;
        }

        /* ── Signal bars ── */
        .signal-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 0;
            border-bottom: 1px solid #111d30;
        }

        .signal-symbol {
            width: 80px;
            font-weight: 600;
            font-size: 12px;
        }

        .signal-score-bar {
            flex: 1;
            height: 6px;
            background: #1e2d45;
            border-radius: 3px;
            overflow: hidden;
        }

        .signal-score-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s;
        }

        .signal-score-value {
            width: 36px;
            text-align: right;
            font-size: 11px;
            color: #5a7a9a;
        }

        /* ── Log ── */
        .log-container {
            height: 120px;
            overflow-y: auto;
            font-size: 11px;
            font-family: monospace;
        }

        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #111d30;
            color: #5a7a9a;
        }

        .log-entry.buy   { color: #00ff88; }
        .log-entry.sell  { color: #ff4444; }
        .log-entry.warn  { color: #ffaa00; }
        .log-entry.close { color: #00d4ff; }

        /* ── Kill switch banner ── */
        .kill-banner {
            display: none;
            background: #ff444422;
            border: 1px solid #ff4444;
            color: #ff4444;
            text-align: center;
            padding: 8px;
            font-weight: 700;
            letter-spacing: 2px;
            grid-column: 1 / -1;
            border-radius: 6px;
        }

        .kill-banner.active { display: block; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 2px; }
    </style>
</head>
<body>

<!-- Header -->
<div class="header">
    <div>
        <div class="header-title">⚡ QUANTBOT</div>
        <div class="header-subtitle">Institutional-Grade Crypto Scalping System</div>
    </div>
    <div class="header-right">
        <div class="toggle-wrap">
            <span style="font-size:11px;color:#5a7a9a;">AUTO PILOT</span>
            <div class="toggle" id="autopilotToggle" onclick="toggleAutopilot()">
                <div class="toggle-knob"></div>
            </div>
            <span id="autopilotLabel" style="font-size:11px;color:#5a7a9a;">OFF</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText" style="font-size:11px;color:#5a7a9a;">LIVE</span>
        </div>
        <div id="clockDisplay" style="font-size:11px;color:#5a7a9a;"></div>
    </div>
</div>

<!-- Main Grid -->
<div class="main">

    <!-- Kill Switch Banner -->
    <div class="kill-banner" id="killBanner">
        ⚠ KILL SWITCH ACTIVE — DAILY DRAWDOWN LIMIT HIT — ALL TRADING HALTED
    </div>

    <!-- Stats Bar -->
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-label">Equity</div>
            <div class="stat-value" id="statEquity">$0.00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Daily PnL</div>
            <div class="stat-value" id="statDailyPnl">$0.00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total PnL</div>
            <div class="stat-value" id="statTotalPnl">$0.00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div class="stat-value" id="statWinRate">0%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Open Positions</div>
            <div class="stat-value" id="statPositions">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Daily Drawdown</div>
            <div class="stat-value" id="statDrawdown">0%</div>
        </div>
    </div>

    <!-- Chart Panel -->
    <div class="panel chart-container">
        <div class="panel-header">
            <span>CANDLESTICK CHART</span>
            <div class="symbol-tabs" id="symbolTabs"></div>
        </div>
        <div class="panel-body" style="padding:8px;">
            <div id="chart"></div>
        </div>
    </div>

    <!-- Right Panel -->
    <div class="right-panel">

        <!-- Open Positions -->
        <div class="panel">
            <div class="panel-header">Open Positions</div>
            <div class="panel-body" style="padding:0;">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry</th>
                            <th>PnL</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="positionsTable">
                        <tr><td colspan="5" style="color:#5a7a9a;text-align:center;padding:16px;">
                            No open positions
                        </td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Signal Scanner -->
        <div class="panel">
            <div class="panel-header">Signal Scanner</div>
            <div class="panel-body" id="signalScanner">
                <div style="color:#5a7a9a;text-align:center;">Loading signals...</div>
            </div>
        </div>

        <!-- Controls -->
        <div class="panel">
            <div class="panel-header">Controls</div>
            <div class="controls">
                <button class="btn btn-success" onclick="runScan()">
                    🔍 Scan Now
                </button>
                <button class="btn btn-warning" onclick="closeAll()">
                    ⏹ Close All
                </button>
                <button class="btn btn-danger" onclick="resetKillSwitch()">
                    🔄 Reset Kill Switch
                </button>
            </div>
        </div>

        <!-- Trade Log -->
        <div class="panel">
            <div class="panel-header">Trade Log</div>
            <div class="panel-body">
                <div class="log-container" id="tradeLog">
                    <div class="log-entry">System started — paper trading mode active</div>
                </div>
            </div>
        </div>

    </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
let autopilotOn   = false;
let currentSymbol = 'BTCUSDT';
let chart         = null;
let candleSeries  = null;
let updateTimer   = null;

// ── Chart Setup ────────────────────────────────────────────────────────────────
function initChart() {
    const container = document.getElementById('chart');
    chart = LightweightCharts.createChart(container, {
        width:  container.clientWidth,
        height: 380,
        layout: {
            background: { color: '#0d1525' },
            textColor:  '#5a7a9a',
        },
        grid: {
            vertLines:   { color: '#1e2d45' },
            horzLines:   { color: '#1e2d45' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#1e2d45' },
        timeScale: {
            borderColor:     '#1e2d45',
            timeVisible:     true,
            secondsVisible:  false,
        },
    });

    candleSeries = chart.addCandlestickSeries({
        upColor:       '#00ff88',
        downColor:     '#ff4444',
        borderVisible: false,
        wickUpColor:   '#00ff88',
        wickDownColor: '#ff4444',
    });

    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });
}

// ── Symbol Tabs ────────────────────────────────────────────────────────────────
function initSymbolTabs(symbols) {
    const tabs = document.getElementById('symbolTabs');
    tabs.innerHTML = '';
    symbols.forEach(sym => {
        const btn = document.createElement('button');
        btn.className = 'symbol-tab' + (sym === currentSymbol ? ' active' : '');
        btn.textContent = sym.replace('USDT', '');
        btn.onclick = () => {
            currentSymbol = sym;
            document.querySelectorAll('.symbol-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadCandles();
        };
        tabs.appendChild(btn);
    });
}

// ── Load Candles ───────────────────────────────────────────────────────────────
async function loadCandles() {
    try {
        const res  = await fetch(`/api/candles/${currentSymbol}`);
        const data = await res.json();
        if (data.candles && data.candles.length > 0) {
            candleSeries.setData(data.candles);
        }
    } catch(e) {
        console.error('Candle load error:', e);
    }
}

// ── Update Dashboard ───────────────────────────────────────────────────────────
async function updateDashboard() {
    try {
        const res  = await fetch('/api/snapshot');
        const data = await res.json();

        // Stats
        setStatValue('statEquity',    '$' + fmt(data.equity), data.equity);
        setStatValue('statDailyPnl',  '$' + fmt(data.daily_pnl), data.daily_pnl);
        setStatValue('statTotalPnl',  '$' + fmt(data.total_pnl), data.total_pnl);
        setStatValue('statWinRate',   data.win_rate + '%', data.win_rate - 50);
        setStatValue('statPositions', data.open_positions, 0);
        setStatValue('statDrawdown',  data.daily_drawdown + '%', -data.daily_drawdown);

        // Kill switch
        const banner = document.getElementById('killBanner');
        const dot    = document.getElementById('statusDot');
        if (data.kill_switch) {
            banner.classList.add('active');
            dot.classList.add('danger');
        } else {
            banner.classList.remove('active');
            dot.classList.remove('danger');
        }

        // Positions table
        updatePositionsTable(data.open_trade_list || []);

        // Candles (live update)
        await loadCandles();

    } catch(e) {
        console.error('Dashboard update error:', e);
    }
}

async function updateSignals() {
    try {
        const res  = await fetch('/api/signals');
        const data = await res.json();
        renderSignals(data.signals || []);
    } catch(e) {
        console.error('Signal update error:', e);
    }
}

// ── Render Positions ───────────────────────────────────────────────────────────
function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsTable');
    if (positions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="color:#5a7a9a;text-align:center;padding:16px;">
            No open positions</td></tr>`;
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnlClass = p.unrealised_pnl >= 0 ? 'positive' : 'negative';
        const side     = p.direction === 'BUY' ? 'badge-buy' : 'badge-sell';
        return `
        <tr>
            <td style="font-weight:600;">${p.symbol.replace('USDT','')}</td>
            <td><span class="badge ${side}">${p.direction}</span></td>
            <td>${p.entry_price.toFixed(2)}</td>
            <td class="${pnlClass}">$${fmt(p.unrealised_pnl)}</td>
            <td>
                <button class="btn btn-danger"
                    style="padding:2px 8px;font-size:10px;"
                    onclick="closePosition('${p.symbol}')">
                    Close
                </button>
            </td>
        </tr>`;
    }).join('');
}

// ── Render Signals ─────────────────────────────────────────────────────────────
function renderSignals(signals) {
    const container = document.getElementById('signalScanner');
    if (signals.length === 0) {
        container.innerHTML = '<div style="color:#5a7a9a;text-align:center;">No signals</div>';
        return;
    }
    container.innerHTML = signals.map(s => {
        const score   = Math.round(s.conviction * 100);
        const color   = s.direction === 'BUY' ? '#00ff88' :
                        s.direction === 'SELL' ? '#ff4444' : '#5a7a9a';
        const badgeClass = s.direction === 'BUY'  ? 'badge-buy'  :
                           s.direction === 'SELL' ? 'badge-sell' :
                           'badge-neutral';
        return `
        <div class="signal-bar">
            <div class="signal-symbol">${s.symbol.replace('USDT','')}</div>
            <span class="badge ${badgeClass}" style="width:48px;text-align:center;">
                ${s.direction}
            </span>
            <div class="signal-score-bar">
                <div class="signal-score-fill"
                    style="width:${score}%;background:${color};">
                </div>
            </div>
            <div class="signal-score-value">${score}%</div>
        </div>`;
    }).join('');
}

// ── Controls ───────────────────────────────────────────────────────────────────
function toggleAutopilot() {
    autopilotOn = !autopilotOn;
    const toggle = document.getElementById('autopilotToggle');
    const label  = document.getElementById('autopilotLabel');
    toggle.classList.toggle('on', autopilotOn);
    label.textContent = autopilotOn ? 'ON' : 'OFF';
    label.style.color = autopilotOn ? '#00ff88' : '#5a7a9a';
    fetch('/api/autopilot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: autopilotOn})
    });
    addLog(autopilotOn ? 'Auto pilot enabled' : 'Auto pilot disabled',
           autopilotOn ? 'buy' : 'warn');
}

async function runScan() {
    addLog('Running signal scan...', '');
    await updateSignals();
    addLog('Scan complete', 'close');
}

async function closeAll() {
    if (!confirm('Close ALL open positions?')) return;
    const res  = await fetch('/api/close_all', {method: 'POST'});
    const data = await res.json();
    addLog('All positions closed', 'close');
    updateDashboard();
}

async function closePosition(symbol) {
    const res  = await fetch(`/api/close/${symbol}`, {method: 'POST'});
    const data = await res.json();
    addLog(`Closed ${symbol}`, 'close');
    updateDashboard();
}

async function resetKillSwitch() {
    await fetch('/api/reset_kill_switch', {method: 'POST'});
    addLog('Kill switch reset', 'warn');
    updateDashboard();
}

// ── Log ────────────────────────────────────────────────────────────────────────
function addLog(message, type) {
    const log  = document.getElementById('tradeLog');
    const time = new Date().toLocaleTimeString();
    const div  = document.createElement('div');
    div.className = 'log-entry ' + (type || '');
    div.textContent = `[${time}] ${message}`;
    log.insertBefore(div, log.firstChild);
    if (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function fmt(n) {
    return parseFloat(n).toFixed(2);
}

function setStatValue(id, text, numVal) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className   = 'stat-value';
    if      (numVal > 0)  el.classList.add('positive');
    else if (numVal < 0)  el.classList.add('negative');
}

function updateClock() {
    const now = new Date();
    document.getElementById('clockDisplay').textContent =
        now.toUTCString().slice(17, 25) + ' UTC';
}

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
    initChart();

    // Load symbols
    try {
        const res  = await fetch('/api/symbols');
        const data = await res.json();
        initSymbolTabs(data.symbols || []);
    } catch(e) {}

    await loadCandles();
    await updateDashboard();
    await updateSignals();

    // Update every 3 seconds
    setInterval(updateDashboard, 3000);
    setInterval(updateSignals,   10000);
    setInterval(updateClock,     1000);
    updateClock();

    addLog('QuantBot dashboard loaded', 'close');
}

init();
</script>
</body>
</html>
"""


# ── Flask App ──────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Flask web dashboard for QuantBot.

    Endpoints:
        GET  /                      — main dashboard UI
        GET  /api/snapshot          — full portfolio snapshot
        GET  /api/signals           — all signal scores
        GET  /api/candles/<symbol>  — candle data for chart
        GET  /api/symbols           — list of tracked symbols
        POST /api/autopilot         — toggle auto trading
        POST /api/close/<symbol>    — close a position
        POST /api/close_all         — close all positions
        POST /api/reset_kill_switch — reset kill switch
    """

    def __init__(
        self,
        feed:      DataFeed,
        engine:    SignalEngine,
        scanner:   MemeCoinScanner,
        executor:  PaperExecutor,
        tracker:   PortfolioTracker,
        rm:        RiskManager,
        host:      str = "0.0.0.0",
        port:      int = 5000,
    ):
        self.feed     = feed
        self.engine   = engine
        self.scanner  = scanner
        self.executor = executor
        self.tracker  = tracker
        self.rm       = rm
        self.host     = host
        self.port     = port
        self.autopilot = False
        self.app      = Flask(__name__) if FLASK_AVAILABLE else None
        self._thread: Optional[threading.Thread] = None

        if self.app:
            self._register_routes()

    def start(self):
        if not FLASK_AVAILABLE:
            logger.error("Flask not available")
            return
        self._thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="dashboard",
        )
        self._thread.start()
        logger.info(f"Dashboard started at http://{self.host}:{self.port}")

    def stop(self):
        logger.info("Dashboard stopped")

    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        @app.route("/api/snapshot")
        def api_snapshot():
            prices = {
                sym: self.feed.get_snapshot(sym).get("last_close", 0)
                for sym in self.feed.symbols
            }
            snap = self.tracker.get_snapshot(current_prices=prices)
            return jsonify({
                "equity":          round(snap.equity, 2),
                "balance":         round(snap.balance, 2),
                "unrealised_pnl":  snap.unrealised_pnl,
                "daily_pnl":       snap.daily_pnl,
                "total_pnl":       snap.total_pnl,
                "open_positions":  snap.open_positions,
                "total_trades":    snap.total_trades,
                "win_rate":        snap.win_rate,
                "profit_factor":   snap.profit_factor,
                "daily_drawdown":  snap.daily_drawdown,
                "kill_switch":     snap.kill_switch,
                "open_trade_list": snap.open_trade_list,
            })

        @app.route("/api/signals")
        def api_signals():
            results = self.engine.scan_all()
            signals = [{
                "symbol":     r.symbol,
                "direction":  r.direction,
                "conviction": r.conviction,
                "leverage":   r.leverage_tier,
                "atr":        r.atr,
                "notes":      r.notes[:2],
            } for r in results]
            return jsonify({"signals": signals})

        @app.route("/api/candles/<symbol>")
        def api_candles(symbol):
            candles = self.feed.get_candles(symbol.upper(), "1m", n=100)
            data = [{
                "time":  c.open_time // 1000,
                "open":  c.open,
                "high":  c.high,
                "low":   c.low,
                "close": c.close,
            } for c in candles]
            return jsonify({"candles": data, "symbol": symbol})

        @app.route("/api/symbols")
        def api_symbols():
            return jsonify({"symbols": self.feed.symbols})

        @app.route("/api/autopilot", methods=["POST"])
        def api_autopilot():
            data = request.get_json()
            self.autopilot = data.get("enabled", False)
            logger.info(f"Autopilot: {'ON' if self.autopilot else 'OFF'}")
            return jsonify({"autopilot": self.autopilot})

        @app.route("/api/close/<symbol>", methods=["POST"])
        def api_close(symbol):
            price = self.feed.get_snapshot(symbol.upper()).get("last_close", 0)
            result = self.executor.close_position_manual(symbol.upper(), price)
            return jsonify({
                "success": result is not None,
                "symbol":  symbol,
            })

        @app.route("/api/close_all", methods=["POST"])
        def api_close_all():
            closed = []
            for trade in self.executor.get_open_trades():
                price = self.feed.get_snapshot(trade.symbol).get("last_close", 0)
                self.executor.close_position_manual(trade.symbol, price)
                closed.append(trade.symbol)
            return jsonify({"closed": closed})

        @app.route("/api/reset_kill_switch", methods=["POST"])
        def api_reset_kill_switch():
            self.rm.reset_kill_switch()
            return jsonify({"success": True})