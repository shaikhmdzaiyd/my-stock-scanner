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

st.set_page_config(
    page_title="FREE QUANT ENGINE | SMC LIVE PRESSURE DASHBOARD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 1. HARD CONFIGURATION & UNIVERSE
# -----------------------------------------------------------------------------
MIN_PRICE = 300.0
MAX_PRICE = 600.0
HIGH_CONVICTION_TCS_THRESHOLD = 65

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BPCL.NS", 
    "NAVA.NS", "CONCOR.NS", "POONAWALLA.NS", "GMDCLTD.NS", "LICHSGFIN.NS", 
    "JSWINFRA.NS", "REDINGTON.NS", "PRECWIRE.NS", "HINDCOPPER.NS", "RELAXO.NS",
    "ASHOKLEY.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", 
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

def compute_bidirectional_cri(df_1m, ltp, target_zone_price, atr_14, current_direction="SUPPLY", span_bars=5):
    """
    df_1m: Realtime 1-minute OHLCV
    target_zone_price: Opposing HTF level (Demand top if currently in Short/Supply, Supply bot if in Long/Demand)
    atr_14: Daily ATR
    current_direction: 'SUPPLY' (तलाश बुलिश रिवर्सल की) या 'DEMAND' (तलाश बेयरिश रिवर्सल की)
    """
    if len(df_1m) < span_bars:
        return 0.0, "HOLD", "NO_ACTION"
    
    sub = df_1m.iloc[-span_bars:]
    c = sub['Close'].to_numpy(dtype=float)
    h = sub['High'].to_numpy(dtype=float)
    l = sub['Low'].to_numpy(dtype=float)
    o = sub['Open'].to_numpy(dtype=float)
    v = sub['Volume'].to_numpy(dtype=float)
    bar_range = (h - l) + 1e-6
    
    # 1. Exponential Structural Proximity (Sp)
    if current_direction == "SUPPLY":
        dist = max(0.0, ltp - target_zone_price)
    else:
        dist = max(0.0, target_zone_price - ltp)
    s_p = np.exp(-2.5 * (dist / (atr_14 + 1e-6)))
    
    # 2. Linearly Weighted Volume Absorption (Av)
    weights = np.arange(1, span_bars + 1)
    if current_direction == "SUPPLY":
        power = (c - l) / bar_range
    else:
        power = (h - c) / bar_range
    weighted_vol = v * weights
    a_v = float(np.sum(power * weighted_vol) / (np.sum(weighted_vol) + 1e-6))
    
    # 3. Volume-Scaled Wick Sweep (Wr)
    v_mean = np.mean(v) if len(v) > 0 else 1.0
    vol_mult = np.sqrt(min(2.0, max(0.5, v[-1] / (v_mean + 1e-6))))
    if current_direction == "SUPPLY":
        raw_wick = max(0.0, min(o[-1], c[-1]) - l[-1]) / bar_range[-1]
    else:
        raw_wick = max(0.0, h[-1] - max(o[-1], c[-1])) / bar_range[-1]
    w_r = min(1.0, float(raw_wick * vol_mult))
    
    # 4. Micro-Structure Dynamic Shift (Ms)
    tot_v = np.sum(v) + 1e-6
    micro_vwap = np.sum(((h + l + c) / 3.0) * v) / tot_v
    alpha = 2.0 / (13.0 + 1.0)
    ema1 = c[0]
    for x in c[1:]: ema1 = alpha * x + (1.0 - alpha) * ema1
    
    norm_factor = 0.1 * atr_14 + 1e-6
    if current_direction == "SUPPLY":
        diff_ema = (ltp - ema1) / norm_factor
        diff_vwap = (ltp - micro_vwap) / norm_factor
    else:
        diff_ema = (ema1 - ltp) / norm_factor
        diff_vwap = (micro_vwap - ltp) / norm_factor
        
    sig_ema = 1.0 / (1.0 + np.exp(-np.clip(diff_ema, -10, 10)))
    sig_vwap = 1.0 / (1.0 + np.exp(-np.clip(diff_vwap, -10, 10)))
    m_s = float(0.5 * sig_ema + 0.5 * sig_vwap)
    
    # Composite Score
    cri = float((0.35 * s_p + 0.25 * a_v + 0.25 * w_r + 0.15 * m_s) * 100.0)
    cri = min(100.0, max(0.0, cri))
    
    # Action Logic
    new_side = "BUY_CALL" if current_direction == "SUPPLY" else "SELL_PUT"
    if cri >= 80.0:
        status, action = "CRITICAL_REVERSAL", f"ENTER_{new_side}_NOW"
    elif cri >= 65.0:
        status, action = "EARLY_TRIGGER", f"READY_{new_side}"
    elif cri >= 40.0:
        status, action = "PULLBACK_TRAIL", "TIGHTEN_SL"
    else:
        status, action = "TREND_STABLE", f"HOLD_{'SHORT' if current_direction == 'SUPPLY' else 'LONG'}"
        
    return round(cri, 2), status, action

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
# 3. UNIVERSE INITIALIZATION & CONFLICT-FREE FILTERING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def init_universe_mtf_cached(tickers):
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
# 4. EXACT IMAGE-2 TABLE VIEW RENDERER (BROKEN-LINE BOX UI)
# -----------------------------------------------------------------------------
def build_html_view(rows, timestamp, latency_ms, total_scanned):
    rows_str = ""
    if not rows:
        rows_str = """
        <tr>
            <td colspan="11" style="padding: 24px; color: #8b949e; text-align: center; font-style: italic;">
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
                <td style='padding: 6px; border-right: 1px dashed #21262d;'>{r['cri_box']}</td>
                <td style='color: #00e676; font-weight: 800; border-right: 1px dashed #21262d;'>₹{r['target']:.2f}</td>
                <td style='border-right: 1px dashed #21262d;'><span style='color: #00e676; font-weight: 900; font-size: 12px;'>{r['tcs']}/100</span></td>
                <td style='color: {'#00e676' if r['imbalance'] >= 0 else '#ff5252'}; font-weight: 700;'>{r['cobi_html']}</td>
            </tr>
            """
        
    return f"""
    <div style="background-color: #090c10; border: 1.5px solid #30363d; border-radius: 8px; padding: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; color: #c9d1d9; overflow-x: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #30363d; padding-bottom: 8px; margin-bottom: 10px;">
            <div style="color: #00e676; font-size: 13px; font-weight: 900; letter-spacing: 0.5px;">⚡ FREE QUANT ENGINE | SMC LIVE PRESSURE DASHBOARD</div>
            <div style="color: #8b949e; font-size: 11px; background: #161b22; padding: 4px 10px; border-radius: 4px; border: 1px dashed #30363d;">
                LIVE STREAM: {timestamp} IST | Active: {len(rows)}/{total_scanned} | Latency: {latency_ms}ms
            </div>
        </div>
        <table style="width: 100%; min-width: 900px; border-collapse: collapse; font-size: 11px; text-align: center; border: 1px dashed #30363d;">
            <thead>
                <tr style="background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 9.5px; border-bottom: 1px dashed #30363d;">
                    <th style="text-align: left; padding: 8px 6px; border-right: 1px dashed #30363d;">Symbol</th>
                    <th style="text-align: left; padding: 8px 6px; border-right: 1px dashed #30363d;">Zone Alignments</th>
                    <th style="border-right: 1px dashed #30363d;">Open</th>
                    <th style="border-right: 1px dashed #30363d;">LTP</th>
                    <th style="border-right: 1px dashed #30363d;">Change</th>
                    <th style="border-right: 1px dashed #30363d;">EMAS (1m|3m|5m|15m)</th>
                    <th style="border-right: 1px dashed #30363d; min-width: 145px;">Supply/Demand Delta Box</th>
                    <th style="border-right: 1px dashed #30363d; min-width: 145px;">Reversal Engine (CRI)</th>
                    <th style="border-right: 1px dashed #30363d;">Target (₹)</th>
                    <th style="border-right: 1px dashed #30363d;">TCS Score</th>
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
# 5. ZERO-BLINK LIVE STREAMLIT ENGINE
# -----------------------------------------------------------------------------
# CSS to remove padding and default streamlit headers
st.markdown("""
<style>
    .block-container { padding: 8px 12px 12px 12px !important; }
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

universe = init_universe_mtf_cached(TICKERS)
tickers_list = list(universe.keys())

if not tickers_list:
    st.warning("⚠️ No stocks qualified in ₹300-₹600 range with clean SMC zones.")
    st.stop()

# Streamlit Single UI Container for 0-Lag Smooth In-Place Updates
dashboard_placeholder = st.empty()

while True:
    try:
        t0 = time.time()
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
                    opp_targets = [z['top'] for z in info['opposing_zones'] if z['top'] < ltp]
                    target_price = max(opp_targets) if opp_targets else (ltp - (atr_val * 1.618))
                else:
                    border_c = "#00e676"
                    bg_c = "rgba(0, 230, 118, 0.12)"
                    status_t = "DEMAND ABSORPTION"
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
                
                # 7. Compute Bidirectional CRI (Reversal Engine)
                cri_val, cri_status, cri_action = compute_bidirectional_cri(
                    df, ltp, target_price, atr_val, current_direction=info['direction'], span_bars=5
                )
                
                # Style CRI Box (Broken-line Box UI with Direct Trade Signals)
                if cri_val >= 80.0:
                    cri_border = "#00e676" if "BUY" in cri_action else "#ff3838"
                    cri_bg = "rgba(0, 230, 118, 0.15)" if "BUY" in cri_action else "rgba(255, 56, 56, 0.15)"
                    action_color = "#00e676" if "BUY" in cri_action else "#ff5252"
                elif cri_val >= 65.0:
                    cri_border = "#ffaa00"
                    cri_bg = "rgba(255, 170, 0, 0.12)"
                    action_color = "#ffaa00"
                elif cri_val >= 40.0:
                    cri_border = "#ffc107"
                    cri_bg = "rgba(255, 193, 7, 0.08)"
                    action_color = "#ffc107"
                else:
                    cri_border = "#30363d"
                    cri_bg = "#161b22"
                    action_color = "#8b949e"
                    
                cri_box_html = f"""
                <div style='border: 1px dashed {cri_border}; background-color: {cri_bg}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
                    <div style='font-size: 9px; font-weight: 800; color: {action_color};'>{cri_status}</div>
                    <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{cri_val:.1f}%</div>
                    <div style='color: {action_color}; font-size: 8.5px; font-weight: 900; margin-top: 1px;'>⚡ {cri_action}</div>
                </div>
                """
                
                # 8. Weighted Institutional Conviction Score (TCS)
                mtf_score = (max(bull_cnt, bear_cnt) / 4.0) * 25.0
                vwap_score = 20.0 if ((info['direction'] == 'DEMAND' and ltp > vwap) or (info['direction'] == 'SUPPLY' and ltp < vwap)) else 0.0
                rsi_score = 15.0 if ((info['direction'] == 'DEMAND' and 50 <= rsi <= 70) or (info['direction'] == 'SUPPLY' and 30 <= rsi <= 50)) else 5.0
                zone_confluence_score = min(20.0, len(matched_zones) * 10.0)
                pressure_component = min(20.0, pressure_pct * 0.2)
                
                tcs = int(min(100.0, max(0.0, mtf_score + vwap_score + rsi_score + zone_confluence_score + pressure_component)))
                
                # 9. High-Conviction Screen Filter
                if tcs >= HIGH_CONVICTION_TCS_THRESHOLD:
                    high_conviction_rows.append({
                        "symbol": info['symbol'],
                        "zone_html": zone_html,
                        "open": open_p,
                        "ltp": ltp,
                        "pnl": pnl_pct,
                        "pressure_box": pressure_box_html,
                        "cri_box": cri_box_html,
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
        
        # In-place clean render (Zero Blink / Zero Screen Flash)
        clean_html = build_html_view(high_conviction_rows, now_time, elapsed_ms, len(tickers_list))
        # Minify HTML string to prevent Markdown codeblock trigger
        compact_html = "".join([line.strip() for line in clean_html.split("\n")])
        dashboard_placeholder.markdown(compact_html, unsafe_allow_html=True)
        
        time.sleep(1)
        
    except Exception:
        time.sleep(1)
        continue
