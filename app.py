import time
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Suppress background logs and warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & ULTRA-LIGHT MOBILE CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SMC Live Quant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Mobile-first ultra responsive container */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.3rem !important;
        padding-right: 0.3rem !important;
    }
    header, footer {visibility: hidden !important;}
    .table-container {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px dashed #30363d;
        border-radius: 6px;
        background-color: #090c10;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
        color: #c9d1d9;
        font-size: 11px;
        white-space: nowrap;
    }
    th {
        background: #161b22;
        color: #8b949e;
        text-transform: uppercase;
        font-size: 9.5px;
        padding: 8px 6px;
        border-bottom: 1px dashed #30363d;
        border-right: 1px dashed #30363d;
        text-align: center;
    }
    td {
        padding: 8px 6px;
        border-bottom: 1px dashed #30363d;
        border-right: 1px dashed #21262d;
        text-align: center;
    }
    .header-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        background-color: #090c10;
        border: 1.5px solid #30363d;
        border-radius: 6px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. HARD CONFIGURATION & UNIVERSE
# -----------------------------------------------------------------------------
MIN_PRICE = 300.0
MAX_PRICE = 600.0
HIGH_CONVICTION_TCS_THRESHOLD = 65

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BPCL.NS", 
    "NAVA.NS", "CONCOR.NS", "POONAWALLA.NS", "GMDCLTD.NS", "LICHSGFIN.NS", 
    "JSWINFRA.NS", "REDINGTON.NS", "ASHOKLEY.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", 
    "CROMPTON.NS", "NATIONALUM.NS", "RBLBANK.NS", "NUVOCO.NS", "COALINDIA.NS", 
    "VBL.NS", "SYNGENE.NS", "ELECON.NS", "JINDALSAW.NS", "TATAPOWER.NS", 
    "JSWENERGY.NS", "USHAMART.NS", "NTPC.NS", "ICICIPRULI.NS", "BATAINDIA.NS", 
    "CANBK.NS", "DLF.NS", "HINDPETRO.NS", "IOC.NS", "AADHARHFC.NS", "IGIL.NS"
]

# -----------------------------------------------------------------------------
# 2. NUMPY QUANT MATH & TRUE SMC ENGINE
# -----------------------------------------------------------------------------
def fast_atr_1d(high, low, close, period=14):
    if len(close) < 2: return 5.0
    tr0 = high[1:] - low[1:]
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    if len(tr) < period: return float(np.mean(tr)) if len(tr) > 0 else 5.0
    return float(np.mean(tr[-period:]))

def fast_ema(arr, span):
    if len(arr) < span: return arr[-1] if len(arr) > 0 else 0.0
    alpha = 2.0 / (span + 1.0)
    ema = arr[0]
    for x in arr[1:]:
        ema = alpha * x + (1.0 - alpha) * ema
    return ema

def fast_rsi(close, period=14):
    if len(close) < period + 2: return 50.0
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def extract_true_smc_zones(df: pd.DataFrame, tf_name: str, pivot_len: int = 2):
    """
    True Institutional Order Block Engine with FVG (Fair Value Gap) and Displacement Validation.
    """
    if df.empty or len(df) < 20: return []
    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    o = df['Open'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)
    v = df['Volume'].to_numpy(dtype=float)
    v_mean = np.mean(v) if len(v) > 0 else 1.0
    atr_arr = np.abs(h - l)
    
    zones = []
    for i in range(pivot_len, len(c) - pivot_len - 1):
        mid = i
        current_atr = atr_arr[mid] if atr_arr[mid] > 0 else 1.0
        is_vol = v[mid] > (1.2 * v_mean)
        
        # Institutional Supply Block: Bearish Displacement + FVG Formation
        if h[mid] == np.max(h[mid - pivot_len : mid + pivot_len + 1]):
            disp_down = (o[mid + 1] - c[mid + 1]) > (0.5 * current_atr) if (mid + 1 < len(c)) else False
            has_fvg = (l[mid] > h[mid + 2]) if (mid + 2 < len(c)) else False
            if is_vol or disp_down or has_fvg:
                zones.append({
                    'type': 'SUPPLY', 
                    'tf': tf_name, 
                    'top': float(h[mid]), 
                    'bot': float(max(o[mid], c[mid])),
                    'origin_idx': mid
                })
                
        # Institutional Demand Block: Bullish Displacement + FVG Formation
        if l[mid] == np.min(l[mid - pivot_len : mid + pivot_len + 1]):
            disp_up = (c[mid + 1] - o[mid + 1]) > (0.5 * current_atr) if (mid + 1 < len(c)) else False
            has_fvg = (h[mid] < l[mid + 2]) if (mid + 2 < len(c)) else False
            if is_vol or disp_up or has_fvg:
                zones.append({
                    'type': 'DEMAND', 
                    'tf': tf_name, 
                    'top': float(min(o[mid], c[mid])), 
                    'bot': float(l[mid]),
                    'origin_idx': mid
                })
                
    return zones[-2:]

# -----------------------------------------------------------------------------
# 3. UNIVERSE INITIALIZATION & CONFLICT-FREE FILTERING (CACHED FOR ZERO LAG)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def init_universe_mtf(tickers):
    hist_1d = yf.download(tickers=tickers, period="30d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=True)
    hist_1h = yf.download(tickers=tickers, period="10d", interval="60m", group_by="ticker", threads=True, progress=False, auto_adjust=True)
    hist_15m = yf.download(tickers=tickers, period="5d", interval="15m", group_by="ticker", threads=True, progress=False, auto_adjust=True)

    universe = {}
    for t in tickers:
        try:
            df_1d = hist_1d[t].dropna() if isinstance(hist_1d.columns, pd.MultiIndex) else hist_1d.dropna()
            if df_1d.empty or len(df_1d) < 10: continue
            
            open_today = float(df_1d['Open'].iloc[-1])
            # Strict Filter 1: ₹300 <= Price <= ₹600
            if not (MIN_PRICE <= open_today <= MAX_PRICE): continue
            
            df_1h = hist_1h[t].dropna() if isinstance(hist_1h.columns, pd.MultiIndex) else hist_1h.dropna()
            df_15m = hist_15m[t].dropna() if isinstance(hist_15m.columns, pd.MultiIndex) else hist_15m.dropna()
            
            z_1d = extract_true_smc_zones(df_1d, '1D')
            z_1h = extract_true_smc_zones(df_1h, '1H')
            z_15m = extract_true_smc_zones(df_15m, '15m')
            
            all_zones = z_15m + z_1h + z_1d
            if not all_zones: continue

            # Strict Filter 2: Open Price must lie inside valid zone
            matched_zones = [z for z in all_zones if (z['bot'] * 0.998 <= open_today <= z['top'] * 1.002)]
            if not matched_zones: continue
            
            has_supply = any(z['type'] == 'SUPPLY' for z in matched_zones)
            has_demand = any(z['type'] == 'DEMAND' for z in matched_zones)
            
            # Strict Filter 3: Reject Dual-Zone Conflicts
            if has_supply and has_demand: continue

            atr_val = fast_atr_1d(df_1d['High'].to_numpy(float), df_1d['Low'].to_numpy(float), df_1d['Close'].to_numpy(float), 14)

            # Extract Opposing Targets
            opposing_zones = [z for z in all_zones if z['type'] != ('SUPPLY' if has_supply else 'DEMAND')]

            universe[t] = {
                "symbol": t.replace(".NS", ""),
                "open": open_today,
                "atr": atr_val,
                "zones": matched_zones,
                "opposing_zones": opposing_zones,
                "direction": "SUPPLY" if has_supply else "DEMAND"
            }
        except Exception:
            continue
            
    return universe

# -----------------------------------------------------------------------------
# 4. DASHBOARD RENDERER & BROKEN-LINE BOX UI
# -----------------------------------------------------------------------------
def build_html_view(rows, timestamp, latency_ms, total_scanned):
    rows_str = ""
    if not rows:
        rows_str = """
        <tr>
            <td colspan="10" style="padding: 24px; color: #8b949e; text-align: center; font-style: italic;">
                ⏳ Scanning in background... No high-conviction pure-zone setups (₹300-₹600) right now.
            </td>
        </tr>
        """
    else:
        for r in rows:
            def dot(b): return "<span style='color:#00e676;'>🟢</span>" if b else "<span style='color:#ff5252;'>🔴</span>"
            emas_html = f"{dot(r['e1'])} {dot(r['e3'])} {dot(r['e5'])} {dot(r['e15'])}"
            
            rows_str += f"""
            <tr style='border-bottom: 1px dashed #30363d;'>
                <td style='font-weight: 900; text-align: left; color: #ffffff; padding: 10px 8px; border-right: 1px dashed #21262d;'>{r['symbol']}</td>
                <td style='text-align: left; font-size: 10px; border-right: 1px dashed #21262d;'>{r['zone_html']}</td>
                <td style='border-right: 1px dashed #21262d;'>₹{r['open']:.2f}</td>
                <td style='font-weight: 700; border-right: 1px dashed #21262d;'>₹{r['ltp']:.2f}</td>
                <td style='color: {'#00e676' if r['pnl'] >= 0 else '#ff5252'}; font-weight: 800; border-right: 1px dashed #21262d;'>{r['pnl']:+.2f}%</td>
                <td style='border-right: 1px dashed #21262d;'>{emas_html}</td>
                <td style='padding: 6px; border-right: 1px dashed #21262d;'>{r['pressure_box']}</td>
                <td style='color: #00e676; font-weight: 800; border-right: 1px dashed #21262d;'>₹{r['target']:.2f}</td>
                <td style='border-right: 1px dashed #21262d;'><span style='color: #00e676; font-weight: 900; font-size: 12px;'>{r['tcs']}/100</span></td>
                <td style='color: {'#00e676' if r['imbalance'] >= 0 else '#ff5252'}; font-weight: 700;'>{r['cobi_html']}</td>
            </tr>
            """
        
    return f"""
    <div class="header-bar">
        <div style="color: #00e676; font-size: 12px; font-weight: 900; letter-spacing: 0.5px;">⚡ FREE QUANT ENGINE | SMC LIVE PRESSURE DASHBOARD</div>
        <div style="color: #8b949e; font-size: 10px; background: #161b22; padding: 3px 8px; border-radius: 4px; border: 1px dashed #30363d; margin-top: 4px;">
            LIVE STREAM: {timestamp} IST | Active: {len(rows)}/{total_scanned} | Latency: {latency_ms}ms
        </div>
    </div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Symbol</th>
                    <th style="text-align: left;">Zone Alignments</th>
                    <th>Open</th>
                    <th>LTP</th>
                    <th>Change</th>
                    <th>EMAS (1m|3m|5m|15m)</th>
                    <th style="min-width: 140px;">Supply/Demand Delta Box</th>
                    <th>Target (₹)</th>
                    <th>TCS Score</th>
                    <th>Buyer/Seller (COBI)</th>
                </tr>
            </thead>
            <tbody>
                {rows_str}
            </tbody>
        </table>
    </div>
    """

# -----------------------------------------------------------------------------
# 5. STREAMLIT REAL-TIME MULTI-FACTOR ENGINE
# -----------------------------------------------------------------------------
placeholder = st.empty()

universe = init_universe_mtf(TICKERS)
tickers_list = list(universe.keys())

if not tickers_list:
    placeholder.markdown("""
        <div style='color: #ff5252; background: #161b22; border: 1px dashed #ff5252; padding: 15px; border-radius: 6px; font-family: monospace;'>
            ⚠️ No stocks qualified in ₹300-₹600 range with clean SMC zones.
        </div>
    """, unsafe_allow_html=True)
else:
    t0 = time.time()
    try:
        batch = yf.download(
            tickers=tickers_list, period="1d", interval="1m",
            group_by="ticker", threads=True, progress=False, auto_adjust=True
        )
        
        high_conviction_rows = []
        
        for ticker, info in universe.items():
            try:
                df = batch[ticker].dropna() if isinstance(batch.columns, pd.MultiIndex) else batch.dropna()
                if df.empty or len(df) < 5: continue
                
                close = df['Close'].to_numpy(dtype=float)
                high = df['High'].to_numpy(dtype=float)
                low = df['Low'].to_numpy(dtype=float)
                open_arr = df['Open'].to_numpy(dtype=float)
                vol = df['Volume'].to_numpy(dtype=float)
                
                ltp = float(close[-1])
                open_p = info['open']
                atr_val = info['atr']
                pnl_pct = ((ltp - open_p) / open_p) * 100.0
                
                # 1. Background VWAP & Momentum RSI
                tp = (high + low + close) / 3.0
                cum_vol = np.sum(vol) + 1e-6
                vwap = float(np.sum(tp * vol) / cum_vol)
                rsi = fast_rsi(close, 14)
                
                # 2. Accurate 1m, 3m, 5m, 15m EMAs (Fixed Sampling)
                ema1 = fast_ema(close[-15:], 13)
                ema3 = fast_ema(close[-45::3], 13) if len(close) >= 40 else ema1
                ema5 = fast_ema(close[-75::5], 13) if len(close) >= 65 else ema1
                ema15 = fast_ema(close[-225::15], 13) if len(close) >= 150 else ema1
                
                bull_cnt = sum([ltp > ema1, ltp > ema3, ltp > ema5, ltp > ema15])
                bear_cnt = 4 - bull_cnt
                
                # 3. Clean Zone Badges
                matched_zones = []
                for z in info['zones']:
                    if z['type'] == 'SUPPLY':
                        matched_zones.append(f"<span style='color:#ff5252; font-weight:700;'>SUPPLY ({z['tf']})</span>")
                    else:
                        matched_zones.append(f"<span style='color:#00e676; font-weight:700;'>DEMAND ({z['tf']})</span>")
                            
                zone_html = " | ".join(matched_zones)
                
                # 4. Intra-Bar Price-Volume Imbalance (Refined COBI)
                bar_range = (high - low) + 1e-6
                buy_power = np.sum(vol * ((close - low) / bar_range))
                sell_power = np.sum(vol * ((high - close) / bar_range))
                tot_power = buy_power + sell_power + 1e-6
                
                buy_pct = (buy_power / tot_power) * 100.0
                imbalance_pct = ((buy_power - sell_power) / tot_power) * 100.0
                cobi_html = f"{buy_pct:.0f}% Buy ({imbalance_pct:+.1f}%)"
                
                # 5. True Directional Institutional Pressure Index
                directional_move = (ltp - open_p) if info['direction'] == 'DEMAND' else (open_p - ltp)
                pressure_pct = min(100.0, max(0.0, (directional_move / atr_val) * 100.0))
                
                # 6. Advanced Target Engine (Structure & Opposing Zones)
                if info['direction'] == "SUPPLY":
                    border_c = "#ff3838"
                    bg_c = "rgba(255, 56, 56, 0.12)"
                    status_t = "SUPPLY ACCUMULATION"
                    # Next Opposing Demand Target or ATR Projected
                    opp_targets = [z['top'] for z in info['opposing_zones'] if z['top'] < ltp]
                    target_price = max(opp_targets) if opp_targets else (ltp - (atr_val * 1.618))
                else:
                    border_c = "#00e676"
                    bg_c = "rgba(0, 230, 118, 0.12)"
                    status_t = "DEMAND ABSORPTION"
                    # Next Opposing Supply Target or ATR Projected
                    opp_targets = [z['bot'] for z in info['opposing_zones'] if z['bot'] > ltp]
                    target_price = min(opp_targets) if opp_targets else (ltp + (atr_val * 1.618))
                    
                retest_html = "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST REJECTION</div>" if pressure_pct > 75.0 else ""
                
                pressure_box_html = f"""
                <div style='border: 1px dashed {border_c}; background-color: {bg_c}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
                    <div style='font-size: 9px; font-weight: 800; color: {border_c};'>{status_t}</div>
                    <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{pressure_pct:.1f}%</div>
                    {retest_html}
                </div>
                """
                
                # 7. Weighted Institutional Conviction Score (TCS)
                mtf_score = (max(bull_cnt, bear_cnt) / 4.0) * 25.0
                vwap_score = 20.0 if ((info['direction'] == 'DEMAND' and ltp > vwap) or (info['direction'] == 'SUPPLY' and ltp < vwap)) else 0.0
                rsi_score = 15.0 if ((info['direction'] == 'DEMAND' and 50 <= rsi <= 70) or (info['direction'] == 'SUPPLY' and 30 <= rsi <= 50)) else 5.0
                zone_confluence_score = min(20.0, len(matched_zones) * 10.0)
                pressure_component = min(20.0, pressure_pct * 0.2)
                
                tcs = int(min(100.0, max(0.0, mtf_score + vwap_score + rsi_score + zone_confluence_score + pressure_component)))
                
                # 8. High-Conviction Screen Filter
                if tcs >= HIGH_CONVICTION_TCS_THRESHOLD:
                    high_conviction_rows.append({
                        "symbol": info['symbol'],
                        "zone_html": zone_html,
                        "open": open_p,
                        "ltp": ltp,
                        "pnl": pnl_pct,
                        "pressure_box": pressure_box_html,
                        "target": target_price,
                        "tcs": tcs,
                        "cobi_html": cobi_html,
                        "imbalance": imbalance_pct,
                        "e1": ltp > ema1, "e3": ltp > ema3, "e5": ltp > ema5, "e15": ltp > ema15
                    })
            except Exception:
                continue
                
        high_conviction_rows.sort(key=lambda x: x['tcs'], reverse=True)
        elapsed_ms = int((time.time() - t0) * 1000)
        now_time = datetime.datetime.now().strftime('%H:%M:%S')
        
        placeholder.markdown(build_html_view(high_conviction_rows, now_time, elapsed_ms, len(tickers_list)), unsafe_allow_html=True)
        
    except Exception:
        pass

    # Ultra-low latency auto-refresh for zero-lag mobile streaming
    time.sleep(1)
    st.rerun()
