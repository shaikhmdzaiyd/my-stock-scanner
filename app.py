import os
import sys
import time
import math
import warnings
import logging
import requests
from io import StringIO
from datetime import datetime
import concurrent.futures

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Suppress Warnings & Logging
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =====================================================================
# STREAMLIT PAGE CONFIGURATION & EXACT TERMINAL STYLING
# =====================================================================
st.set_page_config(
    page_title="Institutional Quant Terminal V2",
    page_icon="⚡",
    layout="wide"
)

# Strict UI Styles matching Google Colab Output
st.markdown("""
<style>
    .stApp { background-color: #090d16 !important; color: #e1e7ed; }
    header, footer, #MainMenu { visibility: hidden; }
    
    div.stButton > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        width: 100%;
        margin-bottom: 15px;
    }
    div.stButton > button:hover {
        background-color: #0369a1 !important;
    }

    .terminal-container {
        background-color: #060911;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        overflow-x: auto;
    }
    
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 12px;
        margin-bottom: 15px;
    }
    .terminal-title {
        font-size: 18px;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: 0.5px;
    }
    .terminal-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }
    
    .regime-box {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-neutral {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 11px;
    }
    .adx-val {
        font-size: 11px;
        color: #94a3b8;
        font-weight: bold;
    }

    .quant-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        text-align: left;
    }
    .quant-table th {
        color: #94a3b8;
        font-weight: 700;
        padding: 10px 8px;
        border-bottom: 1px solid #1e293b;
        font-size: 12px;
        white-space: nowrap;
    }
    .quant-table td {
        padding: 12px 8px;
        border-bottom: 1px solid #0f172a;
        color: #cbd5e1;
        white-space: nowrap;
        font-family: monospace;
    }
    .quant-table tr:hover {
        background-color: #0f172a;
    }
    
    .buy-tag {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 11px;
    }
    .sell-tag {
        background: #881337;
        color: #fda4af;
        border: 1px solid #e11d48;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 11px;
    }
    
    .tcs-badge-green {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 12px;
    }
    .tcs-badge-orange {
        background: #451a03;
        color: #fb923c;
        border: 1px solid #d97706;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 12px;
    }
    
    .symbol-text { color: #ffffff; font-weight: bold; font-family: sans-serif; }
    .tvf-text { color: #f59e0b; font-weight: bold; }
    .demand-text { color: #34d399; font-weight: bold; }
    .supply-text { color: #fb7185; font-weight: bold; }
    
    .dot-green { height: 7px; width: 7px; background-color: #22c55e; border-radius: 50%; display: inline-block; margin-right: 2px; }
    .dot-red { height: 7px; width: 7px; background-color: #ef4444; border-radius: 50%; display: inline-block; margin-right: 2px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONFIGURATION
# =====================================================================
CONFIG = {
    "universe_name": "NIFTY 500",
    "history_period": "2y",
    "benchmark": "^NSEI",
    "download_threads": 8,
    "request_timeout": 15,
    "use_price_filter": True,
    "min_price": 300.0,
    "max_price": 600.0,
    "min_avg_turnover_cr": 3.0,
    "min_avg_volume": 50000,
    "ema_fast": 13, "ema_mid1": 21, "ema_mid2": 50, "ema_slow": 200,
    "atr_len": 14, "min_atr_pct": 0.50, "max_atr_pct": 8.00,
    "rsi_len": 14, "adx_len": 14,
    "momentum_lookback": 20,
    "swing_lookback": 20,
    "volume_expansion": 1.10,
    "minimum_candidate_score": 45,
    "top_n": 20
}

# =====================================================================
# UTILITIES & CALCULATIONS
# =====================================================================
def safe_float(x, default=np.nan):
    try: return default if pd.isna(x) else float(x)
    except Exception: return default

def clamp(x, lo=0.0, hi=100.0):
    return lo if pd.isna(x) else max(lo, min(hi, float(x)))

def normalize_ohlcv(df):
    if df is None or df.empty: return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if all(c in ["Open","High","Low","Close","Adj Close","Volume"] for c in out.columns.get_level_values(0)):
            out.columns = out.columns.get_level_values(0)
        elif all(c in ["Open","High","Low","Close","Adj Close","Volume"] for c in out.columns.get_level_values(1)):
            out.columns = out.columns.get_level_values(1)
    
    rename = {c: str(c).strip().capitalize() for c in out.columns if str(c).strip().lower() in ["open","high","low","close","volume"]}
    out = out.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in needed): return None
    out = out[needed].copy().apply(pd.to_numeric, errors="coerce").dropna(subset=["Open","High","Low","Close"])
    return out[~out.index.duplicated(keep="last")]

def completed_daily_bars(df):
    if df is None or df.empty: return df
    out = df.copy()
    try:
        idx = pd.to_datetime(out.index)
        idx_ist = idx.tz_convert("Asia/Kolkata") if idx.tz is not None else idx.tz_localize("Asia/Kolkata")
        today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
        out = out.loc[idx_ist.normalize() < today_ist]
    except Exception: pass
    return out

@st.cache_data(ttl=3600)
def get_nifty500_universe():
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://raw.githubusercontent.com/anirbanc/indian-stock-market/master/ind_nifty500list.csv",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=CONFIG["request_timeout"])
            if r.status_code == 200 and len(r.text) > 1000:
                df = pd.read_csv(StringIO(r.text))
                if "Symbol" in df.columns:
                    syms = [s if str(s).endswith(".NS") else str(s) + ".NS" for s in df["Symbol"].dropna().str.strip().str.upper()]
                    if len(syms) >= 400: return sorted(list(dict.fromkeys(syms)))
        except Exception: continue
    return []

def atr_series(df, length=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def rsi_series(close, length=14):
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100).where(~((avg_gain == 0) & (avg_loss > 0)), 0)

def adx_di(df, length=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean(), plus_di, minus_di

def calculate_vwap_and_imbalance(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)
    
    cl_change = df["Close"].diff()
    buy_vol = np.where(cl_change >= 0, df["Volume"], 0)
    sell_vol = np.where(cl_change < 0, df["Volume"], 0)
    
    obi = (buy_vol - sell_vol) / (df["Volume"].replace(0, np.nan)) * 100.0
    cobi = pd.Series(obi, index=df.index).ewm(span=10, adjust=False).mean()
    
    pms = ((df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)) * 100.0
    rovl = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)
    
    return safe_float(vwap.iloc[-1]), safe_float(obi.iloc[-1]), safe_float(cobi.iloc[-1]), safe_float(pms.iloc[-1]), safe_float(rovl.iloc[-1])

def calculate_demand_supply_pct(df, close):
    sh = df["High"].iloc[-30:].max()
    sl = df["Low"].iloc[-30:].min()
    
    supply_pct = ((sh - close) / close) * 100.0
    demand_pct = ((close - sl) / close) * 100.0
    return round(demand_pct, 1), round(supply_pct, 1)

def simulate_multi_timeframe_ema(close, df):
    tf_1m = 1 if close > df["Close"].ewm(span=5, adjust=False).mean().iloc[-1] else 0
    tf_3m = 1 if close > df["Close"].ewm(span=9, adjust=False).mean().iloc[-1] else 0
    tf_5m = 1 if close > df["Close"].ewm(span=13, adjust=False).mean().iloc[-1] else 0
    tf_15m = 1 if close > df["Close"].ewm(span=21, adjust=False).mean().iloc[-1] else 0
    return {"1m": tf_1m, "3m": tf_3m, "5m": tf_5m, "15m": tf_15m}

def compute_tcs(df, close, atr, rovl, supply_pct, mtf_dict, market):
    open_p = safe_float(df["Open"].iloc[-1])
    prev_close = safe_float(df["Close"].iloc[-2])
    
    s_gap = 100.0 * max(0.0, 1.0 - (abs(open_p - prev_close) / atr)) if atr > 0 else 50.0
    s_supply = 100.0 * min(1.0, supply_pct / 3.0) if supply_pct > 0 else 0.0
    s_vol = min(100.0, rovl * 50.0)
    
    mtf_count = sum(mtf_dict.values())
    s_mtf = 25.0 * mtf_count
    s_regime = 100.0 if market["score"] >= 0 else 0.0
    
    tcs = (0.25 * s_gap) + (0.25 * s_supply) + (0.20 * s_vol) + (0.15 * s_mtf) + (0.15 * s_regime)
    return round(clamp(tcs), 1)

def build_benchmark_regime():
    raw = yf.download(CONFIG["benchmark"], period=CONFIG["history_period"], interval="1d", auto_adjust=False, progress=False, threads=False)
    df = completed_daily_bars(normalize_ohlcv(raw))
    if df is None or len(df) < 220: raise RuntimeError("Insufficient benchmark data.")
    for span in [20, 50, 200]: df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    df["ATR"], df["RSI"] = atr_series(df, CONFIG["atr_len"]), rsi_series(df["Close"], CONFIG["rsi_len"])
    adx, pdi, mdi = adx_di(df, CONFIG["adx_len"])
    df["ADX"], df["+DI"], df["-DI"] = adx, pdi, mdi
    l = df.iloc[-1]
    s = (1 if l["Close"] > l["EMA20"] else -1) + (1 if l["EMA20"] > l["EMA50"] else -1) + \
        (1 if l["EMA50"] > l["EMA200"] else -1) + (1 if l["+DI"] > l["-DI"] else -1) + (1 if l["ADX"] >= 20 else 0)
    regime = "BULLISH" if s >= 3 else ("BEARISH" if s <= -3 else "NEUTRAL")
    return {"regime": regime, "score": s, "close": safe_float(l["Close"]), "adx": safe_float(l["ADX"]), "rsi": safe_float(l["RSI"]), "df": df}

def structure_context(df):
    look = CONFIG["swing_lookback"]
    if len(df) < look + 5: return {"structure": "UNKNOWN"}
    recent, cur = df.iloc[-look-1:-1], df.iloc[-1]
    sh, sl = recent["High"].max(), recent["Low"].min()
    half = max(5, look // 2)
    a, b = recent.iloc[:half], recent.iloc[half:]
    if b["High"].max() > a["High"].max() and b["Low"].min() > a["Low"].min(): struct = "HH-HL"
    elif b["High"].max() < a["High"].max() and b["Low"].min() < a["Low"].min(): struct = "LH-LL"
    elif cur["Close"] > sh: struct = "BULLISH BOS"
    elif cur["Close"] < sl: struct = "BEARISH BOS"
    else: struct = "RANGE"
    return {"structure": struct}

def analyze_stock(symbol, raw_df, benchmark_df, market):
    try:
        df = completed_daily_bars(normalize_ohlcv(raw_df))
        if df is None or len(df) < 220: return None, "INSUFFICIENT_HISTORY"
        cur = df.iloc[-1]
        close = safe_float(cur["Close"])
        if not np.isfinite(close) or close <= 0: return None, "BAD_PRICE"
        
        if CONFIG["use_price_filter"]:
            if close < CONFIG["min_price"] or close > CONFIG["max_price"]:
                return None, "STRICT_PRICE_LIMIT"
            
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        avg_turn = (df["Close"].iloc[-21:-1] * df["Volume"].iloc[-21:-1]).mean() / 1e7
        if avg_vol < CONFIG["min_avg_volume"] or avg_turn < CONFIG["min_avg_turnover_cr"]: return None, "LIQUIDITY"

        df["ATR"] = atr_series(df, CONFIG["atr_len"])
        for e in [13, 21, 50, 200]: df[f"EMA{e}"] = df["Close"].ewm(span=e, adjust=False).mean()
        df["RSI"] = rsi_series(df["Close"], CONFIG["rsi_len"])
        adx, pdi, mdi = adx_di(df, CONFIG["adx_len"])
        df["ADX"], df["+DI"], df["-DI"] = adx, pdi, mdi

        atr = safe_float(df["ATR"].iloc[-1])
        atr_pct = atr / close * 100 if atr > 0 else np.nan
        if not np.isfinite(atr_pct) or atr_pct < CONFIG["min_atr_pct"] or atr_pct > CONFIG["max_atr_pct"]: return None, "VOLATILITY"

        vwap, obi, cobi, pms, rovl = calculate_vwap_and_imbalance(df)
        demand_pct, supply_pct = calculate_demand_supply_pct(df, close)
        mtf_ema = simulate_multi_timeframe_ema(close, df)

        e13, e21, e50, e200 = df["EMA13"].iloc[-1], df["EMA21"].iloc[-1], df["EMA50"].iloc[-1], df["EMA200"].iloc[-1]
        rsi, adxv = df["RSI"].iloc[-1], df["ADX"].iloc[-1]

        vol_ratio = safe_float(cur["Volume"]) / safe_float(df["Volume"].iloc[-21:-1].mean(), 1.0)
        lb = CONFIG["momentum_lookback"]
        stock_mom = (close / df["Close"].iloc[-lb-1] - 1.0) * 100.0
        bench_mom = (market["close"] / benchmark_df["Close"].iloc[-lb-1] - 1.0) * 100.0
        rs_val = stock_mom - bench_mom
        struct = structure_context(df)

        buy, sell = 0.0, 0.0
        if close > e13: buy += 10
        if e13 > e21: buy += 10
        if e21 > e50: buy += 10
        if e50 > e200: buy += 10
        if struct["structure"] in ["HH-HL", "BULLISH BOS"]: buy += 20
        if vol_ratio >= CONFIG["volume_expansion"]: buy += 15
        if rs_val >= 1.5: buy += 15

        if close < e13: sell += 10
        if e13 < e21: sell += 10
        if e21 < e50: sell += 10
        if struct["structure"] in ["LH-LL", "BEARISH BOS"]: sell += 20

        buy, sell = clamp(buy), clamp(sell)
        direction, raw_score = ("BUY", buy) if buy >= sell else ("SELL", sell)

        if raw_score < CONFIG["minimum_candidate_score"]: return None, "SCORE"

        tcs_score = compute_tcs(df, close, atr, rovl, supply_pct, mtf_ema, market)
        tvf_code = f"{'T+' if close > e13 else 'T-'}{'S+' if 'BOS' in struct['structure'] or 'HH' in struct['structure'] else 'S-'}"

        return {
            "Symbol": symbol.replace(".NS", ""),
            "Direction": direction, 
            "TVF": tvf_code,
            "Score": round(raw_score, 1), 
            "CMP": round(close, 2), 
            "VWAP": round(vwap, 2), 
            "OBI": round(obi, 1), 
            "COBI": round(cobi, 1), 
            "PMS": round(pms, 1),
            "rOVL": round(rovl, 2), 
            "Demand_Pct": demand_pct, 
            "Supply_Pct": supply_pct,
            "MTF_EMA": mtf_ema, 
            "TCS": tcs_score,
            "RSI": round(rsi, 1), 
            "ADX": round(adxv, 1)
        }, "PASS"
    except Exception: return None, "OTHER"

def download_market_data(symbols):
    raw = yf.download(tickers=symbols, period=CONFIG["history_period"], interval="1d", auto_adjust=False, group_by="ticker", threads=True, progress=False)
    data = {}
    if raw is None or raw.empty: return data
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(dict.fromkeys(raw.columns.get_level_values(0)))
        for sym in symbols:
            if sym in level0:
                try: data[sym] = normalize_ohlcv(raw[sym])
                except Exception: pass
    return {k: v for k, v in data.items() if v is not None and not v.empty}

def run_master_scan():
    symbols = get_nifty500_universe()
    if not symbols:
        return pd.DataFrame(), None
        
    market = build_benchmark_regime()
    market_data = download_market_data(symbols)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["download_threads"]) as ex:
        futures = {ex.submit(analyze_stock, sym, market_data[sym], market["df"], market): sym for sym in sorted(market_data.keys())}
        for fut in concurrent.futures.as_completed(futures):
            res, status = fut.result()
            if res: results.append(res)

    top_df = pd.DataFrame(results)
    if not top_df.empty:
        # Deterministic sorting (TCS -> Score -> Symbol alphabetical tie breaker)
        top_df = top_df.sort_values(by=["TCS", "Score", "Symbol"], ascending=[False, False, True]).head(CONFIG["top_n"]).reset_index(drop=True)
    return top_df, market

# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    if st.button("🚀 EXECUTE QUANT SCAN"):
        with st.spinner("Fetching Market Data & Executing Engine..."):
            top_df, market = run_master_scan()
            st.session_state['top_df'] = top_df
            st.session_state['market'] = market

    if 'top_df' in st.session_state and st.session_state['top_df'] is not None:
        top_df = st.session_state['top_df']
        market = st.session_state['market']
        
        if top_df.empty:
            st.warning("No stocks met the filter criteria.")
            return

        rows_list = []
        for idx, r in top_df.iterrows():
            mtf = r['MTF_EMA']
            d1m = f'<span class="{"dot-green" if mtf["1m"] else "dot-red"}"></span>'
            d3m = f'<span class="{"dot-green" if mtf["3m"] else "dot-red"}"></span>'
            d5m = f'<span class="{"dot-green" if mtf["5m"] else "dot-red"}"></span>'
            d15m = f'<span class="{"dot-green" if mtf["15m"] else "dot-red"}"></span>'
            mtf_dots = f"{d1m}{d3m}{d5m}{d15m}"

            tcs_class = "tcs-badge-green" if r['TCS'] >= 75.0 else "tcs-badge-orange"
            side_class = "buy-tag" if r['Direction'] == "BUY" else "sell-tag"

            row = (
                f'<tr>'
                f'<td class="symbol-text">{r["Symbol"]}</td>'
                f'<td><span class="{side_class}">{r["Direction"]}</span></td>'
                f'<td class="tvf-text">{r["TVF"]}</td>'
                f'<td>{r["Score"]}</td>'
                f'<td><span class="{tcs_class}">{r["TCS"]}%</span></td>'
                f'<td>₹{r["CMP"]}</td>'
                f'<td>₹{r["VWAP"]}</td>'
                f'<td>{r["OBI"]}%</td>'
                f'<td>{r["COBI"]}%</td>'
                f'<td>{r["PMS"]}</td>'
                f'<td>{r["rOVL"]}x</td>'
                f'<td class="demand-text">+{r["Demand_Pct"]}%</td>'
                f'<td class="supply-text">-{r["Supply_Pct"]}%</td>'
                f'<td>{mtf_dots}</td>'
                f'<td>{r["RSI"]}</td>'
                f'<td>{r["ADX"]}</td>'
                f'</tr>'
            )
            rows_list.append(row)

        all_rows_html = "".join(rows_list)

        full_html = (
            f'<div class="terminal-container">'
            f'<div class="terminal-header">'
            f'<div>'
            f'<div class="terminal-title">INSTITUTIONAL QUANT TERMINAL V2 (OPTIMIZED)</div>'
            f'<div class="terminal-sub">RANGE: RS 300 TO RS 600 | TCS SYSTEM ACTIVE</div>'
            f'</div>'
            f'<div class="regime-box">'
            f'<span class="badge-neutral">REGIME: {market["regime"]}</span>'
            f'<span class="adx-val">ADX: {market["adx"]:.1f}</span>'
            f'</div>'
            f'</div>'
            f'<table class="quant-table">'
            f'<thead>'
            f'<tr>'
            f'<th>Symbol</th>'
            f'<th>Side</th>'
            f'<th>TVF</th>'
            f'<th>Score</th>'
            f'<th>TCS %</th>'
            f'<th>CMP</th>'
            f'<th>VWAP</th>'
            f'<th>OBI</th>'
            f'<th>COBI</th>'
            f'<th>PMS</th>'
            f'<th>rOVL</th>'
            f'<th>Demand %</th>'
            f'<th>Supply %</th>'
            f'<th>MTF EMA13<br><span style="font-size:9px; color:#64748b;">(1m, 3m, 5m, 15m)</span></th>'
            f'<th>RSI</th>'
            f'<th>ADX</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>'
            f'{all_rows_html}'
            f'</tbody>'
            f'</table>'
            f'</div>'
        )
        
        st.markdown(full_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
