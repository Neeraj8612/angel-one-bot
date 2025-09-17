#finly nifty me 20:20 ka ratio win rate 81% jab market up trande ho 
import os
import time
import threading
import requests
import streamlit as st
import pandas as pd
import pyotp
from dotenv import load_dotenv
from datetime import datetime, timedelta, time as dt_time
import json
import random
from time import sleep
from threading import Lock

# Try import SmartConnect and SmartWebSocket
try:
    from SmartApi import SmartConnect, SmartWebSocket
except Exception:
    try:
        from SmartApi import SmartConnect, SmartWebSocket
    except Exception:
        SmartConnect = None
        SmartWebSocket = None

# ==============================================================================
# ===== CENTRAL CONFIGURATION FOR ALL INDICES ==================================
# ==============================================================================
INDEX_CONFIG = {
    "FINNIFTY": {
        "lot_size": 65,
        "exchange": "NFO",
        "strike_step": 50,
        "description": "Expires on Tuesday"
    },
    "NIFTY": {
        "lot_size": 75,
        "exchange": "NFO",
        "strike_step": 50,
        "description": "User requested as nifty50"
    },
    "BANKNIFTY": {
        "lot_size": 35,
        "exchange": "NFO",
        "strike_step": 100,
    },
    "SENSEX": {
        "lot_size": 20,
        "exchange": "BFO",  # Note: BSE F&O Exchange
        "strike_step": 100,
        "description": "Expires on Friday"
    }
}
# ==============================================================================

# Load env
load_dotenv()
API_KEY = os.getenv("API_KEY", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_PWD = os.getenv("CLIENT_PWD", "")
TOTP_SECRET = os.getenv("TOTP_SECRET", "")

# Session defaults
defaults = {
    "obj": None,
    "feed_token": None,
    "logged_in": False,
    "search_raw": {},
    "expiries": [],
    "selected_expiry": None,
    "paper_trades": [],
    "running_monitors": []
}

# Initialize session state on every run
def initialize_session_state():
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
initialize_session_state()

# Helpers: SmartConnect
def ensure_smartconnect():
    if SmartConnect is None:
        raise RuntimeError("SmartConnect import failed. Install smartapi-python.")
def create_smartconnect(api_key):
    ensure_smartconnect()
    try:
        return SmartConnect(api_key=api_key)
    except TypeError:
        return SmartConnect(api_key)

# Instrument Master loader with caching
INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
def load_instrument_master_json(cache_file="instrument_master.json", cache_expiry_hours=24):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=cache_expiry_hours):
            st.info("Loaded instrument master from cache.")
            return cached_data
    try:
        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            with open(cache_file, "w") as f:
                json.dump(data["data"], f)
            return data["data"]
        return None
    except Exception as e:
        st.error(f"Instrument master download failed: {e}")
        return None
def normalize_expiry(expiry_str):
    if not expiry_str:
        return None
    try:
        return pd.to_datetime(expiry_str).strftime("%Y-%m-%d")
    except:
        formats = ["%d%b%Y", "%Y-%m-%d", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(expiry_str, fmt).strftime("%Y-%m-%d")
            except:
                continue
        return None
def build_search_from_master(master_list, symbol_name="NIFTY", exchange="NFO"):
    if not master_list:
        return None
    rows = []
    for item in master_list:
        itm = {k.lower(): v for k, v in item.items()}
        exch = itm.get("exch_seg") or itm.get("exchange") or itm.get("exch")
        name = itm.get("name") or itm.get("symbol") or itm.get("tradingsymbol") or itm.get("instrument")
        inst_type = itm.get("instrumenttype") or itm.get("insttype") or itm.get("instrument_type")
        
        if exch and str(exch).upper() == exchange.upper():
            nm = str(name or "").upper()
            itype = str(inst_type or "").upper()
            
            if (itype in ["OPTIDX", "FUTIDX"] and symbol_name.upper() in nm) or \
               (symbol_name.upper() in nm and itype == ""):
                
                row = {
                    "expiry": normalize_expiry(itm.get("expiry") or itm.get("expirydate") or itm.get("expiry_date")),
                    "strike": itm.get("strike") or itm.get("strikeprice") or itm.get("strike_price"),
                    "tradingsymbol": itm.get("tradingsymbol") or itm.get("symbolname") or itm.get("symbol"),
                    "symboltoken": itm.get("symboltoken") or itm.get("token") or itm.get("instrument_token") or itm.get("id"),
                    "instrumenttype": itype,
                    "name": name
                }
                if row["tradingsymbol"] and (row["expiry"] or itype == "FUTIDX"):
                    rows.append(row)
    if not rows:
        return None
    return {"data": rows}

def try_search_variants(obj, symbol: str, exchange: str = "NFO"):
    last_exc = None
    candidates = []
    try:
        candidates.append(lambda: obj.searchScrip(symbol, exchange))
    except Exception:
        pass
    try:
        candidates.append(lambda: obj.searchScrip(exchange=exchange, searchscrip=symbol))
    except Exception:
        pass
    for fn in candidates:
        try:
            res = fn()
            if not res:
                continue
            if isinstance(res, list):
                return {"data": res}
            if isinstance(res, dict) and "data" in res:
                return res
            if isinstance(res, dict):
                for k in ("data", "Data", "result"):
                    if k in res and isinstance(res[k], list):
                        return {"data": res[k]}
        except Exception as e:
            last_exc = e
            continue
    return None

def try_fetch_candles(obj, symbol_token, interval="5min", days_back=1, exchange="NFO"):
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days_back)
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    iarg = imap.get(interval, interval)
    params = {
        "exchange": exchange,
        "symboltoken": str(symbol_token),
        "interval": iarg,
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M")
    }
    try:
        res = obj.getCandleData(params)
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")
    if not res or "data" not in res or not res["data"]:
        raise RuntimeError("Token ya interval ke liye koi candle data nahi mila")
    rows = []
    for row in res["data"]:
        if len(row) < 6:
            continue
        rows.append({
            "timestamp": row[0], "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]), "volume": int(row[5])
        })
    if not rows:
        raise RuntimeError("Data mila, par parse nahi ho paaya")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def try_fetch_historical_candles(obj, symbol_token, interval, target_date, retries=5, exchange="NFO"):
    if target_date > datetime.now().date():
        st.warning(f"Cannot fetch data for future date {target_date.strftime('%Y-%m-%d')}. Skipping.")
        return None
    for attempt in range(retries):
        try:
            from_dt = target_date.strftime("%Y-%m-%d") + " 09:15"
            to_dt = target_date.strftime("%Y-%m-%d") + " 15:30"
            interval_map = {"1min":"ONE_MINUTE","5min":"FIVE_MINUTE","15min":"FIFTEEN_MINUTE"}
            iarg = interval_map.get(interval, interval)
            params = {
                "exchange": exchange,
                "symboltoken": str(symbol_token),
                "interval": iarg,
                "fromdate": from_dt,
                "todate": to_dt
            }
            res = obj.getCandleData(params)
            if not res or 'data' not in res or not res['data']:
                return None
            rows = []
            for c in res['data']:
                if isinstance(c, (list, tuple)) and len(c) >= 6:
                    rows.append({
                        "timestamp": pd.to_datetime(c[0]), "open": float(c[1]), "high": float(c[2]),
                        "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])
                    })
            if rows:
                df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                return df
            return None
        except Exception as e:
            if attempt < retries - 1:
                sleep(random.uniform(1, 3))
            else:
                return None

def find_token_in_search(search_dict, expiry, strike, option_cepe):
    if not search_dict or "data" not in search_dict:
        return None, None
    for s in search_dict.get("data", []):
        s_exp = s.get("expiry") or s.get("Expiry") or s.get("exp")
        s_strike = s.get("strike") or s.get("Strike") or s.get("strikePrice") or s.get("strike_price")
        try:
            s_strike_val = int(float(s_strike)) if s_strike is not None else None
        except:
            s_strike_val = None
        s_sym = (s.get("tradingsymbol") or s.get("symbolname") or s.get("tradingsymbol"))
        token = s.get("symboltoken") or s.get("token") or s.get("instrument_token") or s.get("id")
        if s_exp == expiry and s_strike_val == int(strike):
            if s_sym and option_cepe and str(s_sym).upper().endswith(option_cepe.upper()):
                return token, s_sym
    return None, None

def get_option_ltp_with_backoff(obj, exchange, tradingsymbol, symboltoken, retries=3):
    for attempt in range(retries):
        try:
            r = obj.ltpData(exchange=exchange, tradingsymbol=tradingsymbol, symboltoken=str(symboltoken))
            if isinstance(r, dict):
                data = r.get("data")
                if isinstance(data, dict) and "ltp" in data:
                    return float(data["ltp"])
                if isinstance(data, list) and len(data)>0 and isinstance(data[0], dict) and "ltp" in data[0]:
                    return float(data[0]["ltp"])
            raise RuntimeError("LTP parse failed")
        except Exception as e:
            if attempt == retries - 1:
                raise e
            sleep(random.uniform(1, 2) * (2 ** attempt))
    return None

def find_index_futures_token(search_dict, obj, index_name="NIFTY", exchange="NFO"):
    """Finds the nearest expiry futures token for the given index."""
    if not search_dict or "data" not in search_dict:
        st.warning(f"Search data is empty. Trying to fetch fresh data for {index_name}...")
        try:
            search = try_search_variants(obj, index_name, exchange)
            if search and "data" in search:
                search_dict = search
                st.session_state["search_raw"] = search
                st.info(f"Fetched fresh {index_name} data via searchScrip.")
            else:
                st.error(f"Fresh searchScrip for {index_name} failed.")
                return None
        except Exception as e:
            st.error(f"searchScrip retry for {index_name} failed: {e}")
            return None

    futures_contracts = []
    for item in search_dict.get("data", []):
        name = item.get("name", "")
        inst_type = item.get("instrumenttype", "")
        if name.upper() == index_name.upper() and inst_type == "FUTIDX":
            futures_contracts.append(item)
    
    if not futures_contracts:
        st.error(f"No {index_name} Futures found in the instrument data. Cannot proceed.")
        return None
    
    futures_contracts.sort(key=lambda x: x.get("expiry", ""))
    selected_token = futures_contracts[0].get("symboltoken")
    st.info(f"Selected {index_name} Futures token: {selected_token}")
    return selected_token

# Strategy Logic (EMA + RSI)
def calculate_rsi(df, period=14):
    if len(df) < period: return pd.Series([50] * len(df), index=df.index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(df, period=25):
    return df['close'].ewm(span=period, adjust=False).mean()

def detect_ema_rsi_crossover_strategy(df, ema_period=25, rsi_period=14, rsi_min=50, rsi_max=52):
    signals = []
    if df is None or len(df) < (ema_period + 2): return signals
    df['ema'] = calculate_ema(df, period=ema_period)
    df['rsi'] = calculate_rsi(df, period=rsi_period)
    for i in range(2, len(df)):
        prev1, prev2, curr = df.iloc[i-1], df.iloc[i-2], df.iloc[i]
        is_below = prev2['high'] < prev2['ema'] and prev1['high'] < prev1['ema']
        is_crossover = prev1['close'] < prev1['ema'] and curr['close'] > curr['ema']
        is_rsi_valid = rsi_min <= curr['rsi'] <= rsi_max
        is_trade_time = dt_time(9, 30) <= curr['timestamp'].time() < dt_time(15, 20)
        if is_below and is_crossover and is_rsi_valid and is_trade_time:
            signals.append({"signal": "CALL", "entry_index": i, "entry_price": float(curr['close']),
                            "timestamp": curr['timestamp'], "rsi": curr['rsi']})
    return signals

# Paper trade helpers
def paper_record(entry_time, expiry, strike, option_type, qty, entry_price, lot_size):
    trade = {
        "id": len(st.session_state["paper_trades"]) + 1, "time": entry_time, "expiry": expiry,
        "strike": strike, "option": option_type, "qty": qty, "entry_price": entry_price,
        "status": "OPEN", "exit_price": None, "pnl": None, "lot_size": lot_size
    }
    st.session_state["paper_trades"].append(trade)
    return trade

def paper_close(trade_id, exit_price):
    for t in st.session_state["paper_trades"]:
        if t["id"] == trade_id and t["status"] == "OPEN":
            t["exit_price"] = exit_price
            t["status"] = "CLOSED"
            t["pnl"] = (exit_price - t["entry_price"]) * t["qty"]
            return t
    return None

# Order placement functions
def place_market_buy(obj, tradingsymbol, token, qty, exchange="NFO"):
    orderparams = {
        "variety": "NORMAL", "tradingsymbol": tradingsymbol, "symboltoken": token,
        "transactiontype": "BUY", "exchange": exchange, "ordertype": "MARKET",
        "producttype": "INTRADAY", "duration": "DAY", "price": 0, "quantity": qty
    }
    return obj.placeOrder(orderparams)

def place_market_sell(obj, tradingsymbol, token, qty, exchange="NFO"):
    orderparams = {
        "variety": "NORMAL", "tradingsymbol": tradingsymbol, "symboltoken": token,
        "transactiontype": "SELL", "exchange": exchange, "ordertype": "MARKET",
        "producttype": "INTRADAY", "duration": "DAY", "price": 0, "quantity": qty
    }
    return obj.placeOrder(orderparams)

# Trade Monitor
monitor_lock = Lock()
def monitor_trade_background(monitor_id, mode, obj, config, token, tradingsymbol, qty, entry_price, sl_initial, tp, paper_trade_id=None):
    with monitor_lock:
        if monitor_id in st.session_state.get("running_monitors", []): return
        st.session_state["running_monitors"].append(monitor_id)
    
    sl = sl_initial
    high_water_mark = entry_price
    sl_trailed = False
    trailing_trigger_offset = 20.0

    def on_exit(ltp, reason=""):
        st.success(f"Exit signal for {tradingsymbol} at ₹{ltp:.2f}. Reason: {reason}")
        if mode == "Paper":
            paper_close(paper_trade_id, ltp)
        else:
            try:
                place_market_sell(obj, tradingsymbol, token, qty, exchange=config["exchange"])
                st.info("Live SELL order placed.")
            except Exception as e:
                st.error(f"Failed to place live SELL order: {e}")
        
        with monitor_lock:
            if monitor_id in st.session_state["running_monitors"]:
                st.session_state["running_monitors"].remove(monitor_id)

    while True:
        with monitor_lock:
            if monitor_id not in st.session_state["running_monitors"]: break
        
        if datetime.now().time() >= dt_time(15, 20):
            try:
                ltp = get_option_ltp_with_backoff(obj, config["exchange"], tradingsymbol, token) or high_water_mark
                on_exit(ltp, "End of Day (3:20 PM)")
            except Exception:
                on_exit(high_water_mark, "End of Day (3:20 PM) - LTP fetch failed")
            break
        
        try:
            ltp = get_option_ltp_with_backoff(obj, config["exchange"], tradingsymbol, token)
            if ltp is None:
                sleep(2)
                continue
            
            high_water_mark = max(high_water_mark, ltp)

            if not sl_trailed and high_water_mark >= (entry_price + trailing_trigger_offset):
                sl = entry_price
                sl_trailed = True
                st.info(f"Trailing SL for {tradingsymbol} activated. New SL: ₹{sl:.2f}")

            if ltp <= sl or ltp >= tp:
                reason = "SL Hit" if ltp <= sl else "TP Hit"
                on_exit(ltp, reason)
                break
        except Exception:
            pass
        sleep(2)

    with monitor_lock:
        if monitor_id in st.session_state["running_monitors"]:
            st.session_state["running_monitors"].remove(monitor_id)

# UI Section
st.set_page_config(page_title="Angel One - Index Strategy Bot", layout="wide")
st.title("📈 Angel One - EMA/RSI Crossover Strategy Bot")

col1, col2 = st.columns([2,1])
with col2:
    mode = st.selectbox("Mode", ["Paper (Safe)", "Live (Real Orders)"])
    st.markdown("- Paper: simulate trades locally (safe)\n- Live: places real market orders (be careful!)")
with col1:
    st.header("1) Login / Logout")
    if st.session_state.get("logged_in"):
        st.success("✅ Logged in")
        if st.button("Logout"):
            try: st.session_state["obj"].terminateSession(CLIENT_ID)
            except: pass
            for key in list(st.session_state.keys()): del st.session_state[key]
            initialize_session_state()
            st.rerun()
    else:
        if st.button("Login (use .env)"):
            try:
                sc = create_smartconnect(API_KEY)
                otp = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else None
                data = sc.generateSession(CLIENT_ID, CLIENT_PWD, otp)
                st.session_state["obj"] = sc
                st.session_state["logged_in"] = True
                try:
                    token_response = sc.getfeedToken()
                    if isinstance(token_response, dict):
                        st.session_state["feed_token"] = token_response.get("data", {}).get("feedToken")
                    elif isinstance(token_response, str):
                        st.session_state["feed_token"] = token_response
                    st.success("Login successful + Feed Token fetched.")
                except Exception as e:
                    st.warning(f"Login successful, but error fetching Feed Token: {e}")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

st.markdown("---")

if st.session_state.get("logged_in") and st.session_state.get("obj"):
    obj = st.session_state["obj"]
    
    st.header("2) Setup")
    selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()))
    config = INDEX_CONFIG[selected_index]
    st.info(f"Selected Index: {selected_index} | Lot Size: {config['lot_size']} | Exchange: {config['exchange']}")

    if st.button(f"📅 Get {selected_index} Options & Expiries"):
        with st.spinner(f"Fetching instruments for {selected_index}..."):
            master = load_instrument_master_json()
            if master:
                built = build_search_from_master(master, symbol_name=selected_index, exchange=config["exchange"])
                if built:
                    st.session_state["search_raw"] = built
                    expiries = sorted(list({i.get("expiry") for i in built.get("data",[]) if i.get("expiry")}))
                    st.session_state["expiries"] = expiries
                    st.success(f"Found {len(expiries)} expiries for {selected_index}")
                else:
                    st.warning(f"Could not find {selected_index} in master file. Falling back to API.")
                    search = try_search_variants(obj, selected_index, config["exchange"])
                    if search and search.get("data"):
                        st.session_state["search_raw"] = search
                        expiries = sorted(list({i.get("expiry") for i in search.get("data",[]) if i.get("expiry")}))
                        st.session_state["expiries"] = expiries
                        st.success(f"Found {len(expiries)} expiries for {selected_index} via API")
    
    expiry = st.selectbox("Select expiry", st.session_state.get("expiries", []))
    
    st.markdown("##### Strategy & Trade Parameters")
    c1,c2,c3 = st.columns(3)
    with c1:
        interval = st.selectbox("Candle interval", ["5min","1min","15min"])
        ema_period = st.number_input("EMA Period", min_value=5, value=25, step=1)
        contracts = st.number_input("Contracts (lots)", min_value=1, value=1, step=1)
    with c2:
        rsi_period = st.number_input("RSI Period", min_value=5, value=14, step=1)
        rsi_range = st.slider("RSI Range", 20, 80, (50, 52))
        sl_offset = st.number_input("SL Offset (₹)", min_value=1.0, value=20.0, step=0.5)
    with c3:
        days_back = st.number_input("Days for Candles", min_value=1, value=1, step=1)
        tp_offset = st.number_input("TP Offset (₹)", min_value=1.0, value=20.0, step=0.5)

    st.header("3) Run Strategy")
    if st.button("▶ Run Strategy Now"):
        if not expiry:
            st.error("Please select an expiry date first.")
            st.stop()
        try:
            search_raw = st.session_state.get("search_raw", {})
            index_token = find_index_futures_token(search_raw, obj, index_name=selected_index, exchange=config["exchange"])
            if not index_token: st.stop()
            
            df = try_fetch_candles(obj, index_token, interval, days_back, config["exchange"])
            signals = detect_ema_rsi_crossover_strategy(df, ema_period, rsi_period, rsi_range[0], rsi_range[1])

            if not signals:
                st.info("No valid CALL signals found in latest data.")
            else:
                sig = signals[-1]
                st.success(f"Found signal at {sig['timestamp']}. Processing...")
                
                spot = float(df['close'].iloc[-1])
                step = config["strike_step"]
                atm = round(spot / step) * step
                
                strike = atm 
                option_cepe = "CE"
                
                token, tradingsymbol = find_token_in_search(search_raw, expiry, strike, option_cepe)
                if not token:
                    raise RuntimeError(f"Could not find {option_cepe} option token for Strike ₹{strike}.")
                
                entry_price = get_option_ltp_with_backoff(obj, config["exchange"], tradingsymbol, token)
                sl = entry_price - sl_offset
                tp = entry_price + tp_offset
                qty = int(contracts * config["lot_size"])

                st.markdown(f"**Trade Plan for {selected_index}:**\n- **Symbol:** `{tradingsymbol}`\n- **Entry:** `₹{entry_price:.2f}`\n- **SL:** `₹{sl:.2f}`\n- **TP:** `₹{tp:.2f}`\n- **Qty:** `{qty}`")

                if mode.startswith("Paper"):
                    trade = paper_record(datetime.now(), expiry, strike, option_cepe, qty, entry_price, config["lot_size"])
                    monitor_id = f"paper-{trade['id']}-{time.time()}"
                    t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Paper", obj, config, token, tradingsymbol, qty, entry_price, sl, tp, trade['id']))
                    t.daemon = True
                    t.start()
                    st.info("Started background monitor (paper).")
                else:
                    resp = place_market_buy(obj, tradingsymbol, token, qty, config["exchange"])
                    st.success(f"Live BUY order placed: {resp}")
                    monitor_id = f"live-{time.time()}"
                    t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Live", obj, config, token, tradingsymbol, qty, entry_price, sl, tp, None))
                    t.daemon = True
                    t.start()
                    st.info("Started background monitor (live).")
        except Exception as e:
            st.error(f"Strategy run failed: {e}")

    # Backtest Section
    st.header("4) Backtest Strategy")
    start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=90))
    end_date = st.date_input("End Date", datetime.now().date())
    hypo_option_price = st.number_input("Hypothetical Option Entry Price (₹)", min_value=1.0, value=120.0, step=0.5)
    trade_cost = st.number_input("Cost per Trade (Brokerage, etc.) ₹", min_value=0.0, value=50.0, step=1.0)

    if st.button("▶ Run Backtest Now"):
        try:
            futures_token = find_index_futures_token(st.session_state.get("search_raw", {}), obj, index_name=selected_index, exchange=config["exchange"])
            if not futures_token: st.stop()
            
            all_trades = []
            date_range = (end_date - start_date).days + 1
            progress_bar = st.progress(0)

            for i, single_date in enumerate((start_date + timedelta(n) for n in range(date_range))):
                if single_date.weekday() >= 5: continue
                df_hist = try_fetch_historical_candles(obj, futures_token, interval, single_date, exchange=config["exchange"])
                if df_hist is not None and not df_hist.empty:
                    signals = detect_ema_rsi_crossover_strategy(df_hist, ema_period, rsi_period, rsi_range[0], rsi_range[1])
                    for sig in signals:
                        # Backtest simulation logic
                        hypo_entry = hypo_option_price
                        hypo_sl_initial = hypo_entry - sl_offset
                        hypo_tp = hypo_entry + tp_offset
                        trailing_trigger = hypo_entry + 20.0
                        
                        hypo_sl_current = hypo_sl_initial
                        sl_was_trailed = False
                        trade_status = "UNKNOWN"
                        exit_price = hypo_entry

                        for j in range(sig['entry_index'] + 1, len(df_hist)):
                            candle = df_hist.iloc[j]
                            price_change = candle['close'] - sig['entry_price']
                            estimated_high = hypo_entry + (candle['high'] - sig['entry_price'])
                            estimated_low = hypo_entry + (candle['low'] - sig['entry_price'])
                            
                            if not sl_was_trailed and estimated_high >= trailing_trigger:
                                hypo_sl_current = hypo_entry
                                sl_was_trailed = True
                            
                            if estimated_high >= hypo_tp:
                                trade_status = "TP_HIT"
                                exit_price = hypo_tp
                                break
                            
                            if estimated_low <= hypo_sl_current:
                                trade_status = "SL_HIT"
                                exit_price = hypo_sl_current
                                break

                            if candle['timestamp'].time() >= dt_time(15, 20):
                                trade_status = "EOD_EXIT"
                                exit_price = hypo_entry + price_change
                                break
                        
                        if trade_status == "UNKNOWN":
                            trade_status = "EOD_NO_EXIT"
                            exit_price = hypo_entry + (df_hist.iloc[-1]['close'] - sig['entry_price'])

                        net_pnl = ((exit_price - hypo_entry) * config["lot_size"]) - trade_cost
                        all_trades.append({
                            "date": sig['timestamp'].date(),
                            "signal_time": sig['timestamp'].time(),
                            "signal_type": sig['signal'],  # यह लाइन जोड़ी गई है
                            "index_entry_price": sig['entry_price'],  # यह लाइन जोड़ी गई है
                            "hypo_option_entry": hypo_entry, # यह लाइन भी जोड़ी गई है
                            "status": trade_status,
                            "net_pnl": net_pnl
                        })

                progress_bar.progress((i + 1) / date_range)
            
            if not all_trades:
                st.info("No signals found in the date range.")
            else:
                trades_df = pd.DataFrame(all_trades)
                st.success(f"Backtest complete! Found {len(all_trades)} trades.")
                
                # Display the main trade dataframe
                st.dataframe(trades_df[['date', 'signal_time', 'signal_type', 'index_entry_price', 'hypo_option_entry', 'status', 'net_pnl']])
                
                # Calculate all summary metrics
                total_pnl = trades_df['net_pnl'].sum()
                wins = trades_df[trades_df['net_pnl'] > 0]
                losses = trades_df[trades_df['net_pnl'] <= 0] # Includes break-even trades
                win_rate = (len(wins) / len(all_trades)) * 100 if all_trades else 0

                # Display the summary in a clean, professional format
                st.markdown("### Backtest Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Net PnL (After Costs)", value=f"₹ {total_pnl:,.2f}")
                with col2:
                    st.metric(label="Total Trades", value=len(all_trades))
                    st.metric(label="Winning Trades", value=len(wins))
                with col3:
                    st.metric(label="Win Rate", value=f"{win_rate:.2f} %")
                    st.metric(label="Losing Trades", value=len(losses))
        except Exception as e:
            st.error(f"Backtest failed: {e}")

# Display Paper Trades and Debug Info
st.subheader("Paper Trades (history)")
if st.session_state.get("paper_trades"):
    st.dataframe(pd.DataFrame(st.session_state.get("paper_trades")))
else:
    st.info("No paper trades have been recorded yet.")