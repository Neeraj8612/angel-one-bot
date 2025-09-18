# app.py (Updated with PE Options, RSI, WebSocket, and KeyError Fix)
import os
import time
import threading
import requests
import streamlit as st
import pandas as pd
import pyotp
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import random
from time import sleep
from threading import Lock
import gzip

# Try import SmartConnect and SmartWebSocket
try:
    from SmartApi import SmartConnect, SmartWebSocket
except Exception:
    try:
        from SmartApi import SmartConnect, SmartWebSocket
    except Exception:
        SmartConnect = None
        SmartWebSocket = None

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

# नया और बेहतर तरीका


def load_instrument_master_json(cache_file="instrument_master.json.gz", cache_expiry_hours=24):
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=cache_expiry_hours):
            st.info("Loaded instrument master from compressed cache (.gz).")
            # Gzip फाइल को पढ़ें
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                return json.load(f)

    # ... (बाकी का डाउनलोड वाला लॉजिक वैसा ही रहेगा) ...
    try:
        st.info("Downloading fresh instrument master...")
        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # डाउनलोड हुए डेटा को सीधे .gz फॉर्मेट में सेव करें
        with gzip.open(cache_file, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        
        return data
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

def build_search_from_master(master_list, symbol_name="NIFTY"):
    """
    Filter master_list for NIFTY OPTIDX and FUTIDX in NFO and return search-like dict.
    """
    if not master_list:
        return None
    rows = []
    for item in master_list:
        itm = {k.lower(): v for k, v in item.items()}
        exch = itm.get("exch_seg") or itm.get("exchange") or itm.get("exch")
        name = itm.get("name") or itm.get("symbol") or itm.get("tradingsymbol") or itm.get("instrument")
        inst_type = itm.get("instrumenttype") or itm.get("insttype") or itm.get("instrument_type")
        
        if exch and str(exch).upper() == "NFO":
            nm = str(name or "").upper()
            itype = str(inst_type or "").upper()
            
            # --- सुधार यहाँ है: allowed types में 'FUTIDX' जोड़ा गया है ---
            if (itype in ["OPTIDX", "FUTIDX"] and symbol_name.upper() in nm) or \
               (symbol_name.upper() in nm and itype == ""):
                
                row = {
                    "expiry": normalize_expiry(itm.get("expiry") or itm.get("expirydate") or itm.get("expiry_date")),
                    "strike": itm.get("strike") or itm.get("strikeprice") or itm.get("strike_price"),
                    "tradingsymbol": itm.get("tradingsymbol") or itm.get("symbolname") or itm.get("symbol"),
                    "symboltoken": itm.get("symboltoken") or itm.get("token") or itm.get("instrument_token") or itm.get("id"),
                    "instrumenttype": itype, # instrument type को स्टोर करना सुनिश्चित करें
                    "name": name
                }
                if row["tradingsymbol"] and (row["expiry"] or itype == "FUTIDX"): # फ्यूचर्स में सभी फॉर्मेट में एक्सपायरी नहीं हो सकती है
                    rows.append(row)
    if not rows:
        return None
    return {"data": rows}

def try_search_variants(obj, symbol: str, exchange: str = None):
    last_exc = None
    candidates = []
    try:
        candidates.append(lambda: obj.searchScrip(symbol))
    except Exception:
        pass
    try:
        candidates.append(lambda: obj.searchScrip(searchscrip=symbol))
    except Exception:
        pass
    try:
        candidates.append(lambda: obj.searchScrip(keyword=symbol))
    except Exception:
        pass
    try:
        if exchange:
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

def try_fetch_candles(obj, symbol_token, interval="5min", days_back=1):
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days_back)
    imap = {
        "1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE",
        "1minute": "ONE_MINUTE", "5minute": "FIVE_MINUTE", "15minute": "FIFTEEN_MINUTE",
    }
    iarg = imap.get(interval, interval)

    params = {
        "exchange": "NFO",
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
            "timestamp": row[0],
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(row[5])
        })

    if not rows:
        raise RuntimeError("Data mila, par parse nahi ho paaya")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def try_fetch_historical_candles(obj, symbol_token, interval, target_date, retries=5):
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
                "exchange": "NFO",
                "symboltoken": str(symbol_token),
                "interval": iarg,
                "fromdate": from_dt,
                "todate": to_dt
            }
            
            res = obj.getCandleData(params)

            if not res or 'data' not in res or not res['data']:
                st.warning(f"No data received for {target_date.strftime('%Y-%m-%d')}. Response: {res}. Possibly holiday or invalid token.")
                return None

            rows = []
            for c in res['data']:
                if isinstance(c, (list, tuple)) and len(c) >= 6:
                    rows.append({
                        "timestamp": pd.to_datetime(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5])
                    })
            
            if rows:
                df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                return df
            return None
        except Exception as e:
            st.warning(f"Error fetching historical candles for {target_date.strftime('%Y-%m-%d')} (attempt {attempt+1}): {e}")
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
            if s_sym and option_cepe and option_cepe in str(s_sym):
                return token, s_sym
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
            if isinstance(r, dict):
                for v in (r.get("result") or r.get("data") or []):
                    if isinstance(v, dict) and "ltp" in v:
                        return float(v["ltp"])
            raise RuntimeError("LTP parse failed")
        except Exception as e:
            if attempt == retries - 1:
                raise e
            sleep(random.uniform(1, 2) * (2 ** attempt))
    return None

def find_nifty_futures_token(search_dict, obj):
    if not search_dict or "data" not in search_dict:
        st.warning("Search data is empty. Please click 'Get NIFTY Options' first.")
        try:
            search = try_search_variants(obj, "NIFTY", "NFO")
            if search and "data" in search:
                search_dict = search
                st.session_state["search_raw"] = search
                st.info("Fetched fresh NIFTY data via searchScrip.")
            else:
                st.warning("Fresh searchScrip failed. Using fallback token '26009'.")
                return "26009"
        except Exception as e:
            st.error(f"searchScrip retry failed: {e}. Using fallback token '26009'.")
            return "26009"

    nifty_futures = []
    for item in search_dict.get("data", []):
        name = item.get("name", "")
        inst_type = item.get("instrumenttype", "")
        if name.upper() == "NIFTY" and inst_type == "FUTIDX":
            nifty_futures.append(item)
    
    if not nifty_futures:
        st.warning("No NIFTY Futures found in search data. Trying fresh searchScrip...")
        try:
            search = try_search_variants(obj, "NIFTY", "NFO")
            if search and "data" in search:
                search_dict = search
                st.session_state["search_raw"] = search
                for item in search_dict.get("data", []):
                    name = item.get("name", "")
                    inst_type = item.get("instrumenttype", "")
                    if name.upper() == "NIFTY" and inst_type == "FUTIDX":
                        nifty_futures.append(item)
                if not nifty_futures:
                    st.error("Still no NIFTY Futures found. Using fallback token '26009'.")
                    return "26009"
            else:
                st.error("Fresh searchScrip failed. Using fallback token '26009'.")
                return "26009"
        except Exception as e:
            st.error(f"searchScrip retry failed: {e}. Using fallback token '26009'.")
            return "26009"
    
    nifty_futures.sort(key=lambda x: x.get("expiry", ""))
    selected_token = nifty_futures[0].get("symboltoken")
    st.info(f"Selected NIFTY Futures token: {selected_token}")
    return selected_token

# Strategy helpers with RSI and PE
def calculate_rsi(df, period=14):
    if len(df) < period:
        return pd.Series([50] * len(df), index=df.index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_doji_row(row, tol=0.25):
    rng = row["high"] - row["low"]
    if rng <= 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / rng) <= tol

def is_downtrend(prev_candles):
    if prev_candles is None or len(prev_candles) < 3:
        return False
    closes = list(prev_candles['close'])
    return all(closes[i] > closes[i+1] for i in range(len(closes)-1))

def is_uptrend(prev_candles):
    if prev_candles is None or len(prev_candles) < 3:
        return False
    closes = list(prev_candles['close'])
    return all(closes[i] < closes[i+1] for i in range(len(closes)-1))

def detect_doji_and_entry(df, lookback_prev=3, lookahead=5, rsi_period=14, rsi_call_threshold=30, rsi_pe_threshold=70, enable_pe=False):
    signals = []
    if df is None or len(df) < (lookback_prev + 2):
        return signals
    df['rsi'] = calculate_rsi(df, rsi_period)
    n = len(df)
    for i in range(lookback_prev, n - lookahead):
        prev = df.iloc[i - lookback_prev:i]
        row = df.iloc[i]
        rsi_val = df['rsi'].iloc[i]
        if is_downtrend(prev) and is_doji_row(row):
            doji_high = row['high']
            for j in range(1, lookahead+1):
                nxt = df.iloc[i+j]
                if nxt['high'] > doji_high and rsi_val < rsi_call_threshold:
                    signals.append({
                        "signal": "CALL",
                        "doji_index": i,
                        "entry_index": i+j,
                        "entry_price": float(nxt['close']),
                        "doji_high": float(doji_high),
                        "rsi": rsi_val
                    })
                    break
        if enable_pe and is_uptrend(prev) and is_doji_row(row):
            doji_low = row['low']
            for j in range(1, lookahead+1):
                nxt = df.iloc[i+j]
                if nxt['low'] < doji_low and rsi_val > rsi_pe_threshold:
                    signals.append({
                        "signal": "PUT",
                        "doji_index": i,
                        "entry_index": i+j,
                        "entry_price": float(nxt['close']),
                        "doji_low": float(doji_low),
                        "rsi": rsi_val
                    })
                    break
    return signals

# Paper trade helpers
LOT_SIZE = 50
def paper_record(entry_time, expiry, strike, option_type, qty, entry_price):
    trade = {
        "id": len(st.session_state["paper_trades"]) + 1,
        "time": entry_time,
        "expiry": expiry,
        "strike": strike,
        "option": option_type,
        "qty": qty,
        "entry_price": entry_price,
        "status": "OPEN",
        "exit_price": None,
        "pnl": None
    }
    st.session_state["paper_trades"].append(trade)
    return trade

def paper_close(trade_id, exit_price):
    for t in st.session_state["paper_trades"]:
        if t["id"] == trade_id and t["status"] == "OPEN":
            t["exit_price"] = exit_price
            t["status"] = "CLOSED"
            t["pnl"] = (exit_price - t["entry_price"]) * t["qty"] if t["option"] == "CE" else (t["entry_price"] - exit_price) * t["qty"]
            return t
    return None

# Orders
def place_market_buy(obj, tradingsymbol, token, qty):
    orderparams = {
        "variety": "NORMAL",
        "tradingsymbol": tradingsymbol,
        "symboltoken": token,
        "transactiontype": "BUY",
        "exchange": "NFO",
        "ordertype": "MARKET",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": 0,
        "quantity": qty
    }
    return obj.placeOrder(orderparams)

def place_market_sell(obj, tradingsymbol, token, qty):
    orderparams = {
        "variety": "NORMAL",
        "tradingsymbol": tradingsymbol,
        "symboltoken": token,
        "transactiontype": "SELL",
        "exchange": "NFO",
        "ordertype": "MARKET",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": 0,
        "quantity": qty
    }
    return obj.placeOrder(orderparams)

# WebSocket Monitor
class RealtimeLTPMonitor:
    def __init__(self, feed_token, client_code, token, tradingsymbol, sl, tp, mode, paper_trade_id=None, on_exit_callback=None):
        self.feed_token = feed_token
        self.client_code = client_code
        self.token = token
        self.tradingsymbol = tradingsymbol
        self.sl = sl
        self.tp = tp
        self.mode = mode
        self.paper_trade_id = paper_trade_id
        self.on_exit_callback = on_exit_callback
        self.obj = st.session_state["obj"]
        self.ss = None
        self.running = True

    def on_message(self, ws, message):
        if not self.running:
            return
        try:
            data = json.loads(message)
            if data and "ltp" in data:
                ltp = float(data["ltp"])
                if ltp <= self.sl or ltp >= self.tp:
                    if self.mode == "Paper":
                        paper_close(self.paper_trade_id, ltp)
                    else:
                        place_market_sell(self.obj, self.tradingsymbol, self.token, LOT_SIZE)
                    self.on_exit_callback(ltp)
                    self.close()
        except Exception as e:
            st.error(f"WebSocket message error: {e}")

    def on_open(self, ws):
        st.info("WebSocket connected for LTP monitoring.")
        task = "mw"
        subscribe_token = f"NFO|{self.token}"
        self.ss.subscribe(task, subscribe_token)

    def on_error(self, ws, error):
        st.error(f"WebSocket error: {error}")

    def on_close(self, ws):
        st.info("WebSocket closed.")
        self.running = False

    def start(self):
        if SmartWebSocket is None:
            st.error("SmartWebSocket not available. Falling back to polling.")
            return False
        self.ss = SmartWebSocket(self.feed_token, self.client_code)
        self.ss._on_open = self.on_open
        self.ss._on_message = self.on_message
        self.ss._on_error = self.on_error
        self.ss._on_close = self.on_close
        self.ss.connect()
        return True

    def close(self):
        self.running = False
        if self.ss:
            self.ss.close()

monitor_lock = Lock()

def monitor_trade_background(monitor_id, mode, obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, paper_trade_id=None, use_websocket=True):
    with monitor_lock:
        st.session_state["running_monitors"].append(monitor_id)
    def on_exit(ltp):
        with monitor_lock:
            if monitor_id in st.session_state["running_monitors"]:
                st.session_state["running_monitors"].remove(monitor_id)
    try:
        if use_websocket and st.session_state.get("feed_token"):
            monitor = RealtimeLTPMonitor(st.session_state["feed_token"], CLIENT_ID, token, tradingsymbol, sl, tp, mode, paper_trade_id, on_exit)
            monitor.start()
            while monitor.running:
                time.sleep(1)
        else:
            while True:
                with monitor_lock:
                    if monitor_id not in st.session_state["running_monitors"]:
                        break
                try:
                    ltp = get_option_ltp_with_backoff(obj, "NFO", tradingsymbol, token) or entry_price
                    if ltp <= sl or ltp >= tp:
                        if mode == "Paper":
                            paper_close(paper_trade_id, ltp)
                        else:
                            place_market_sell(obj, tradingsymbol, token, qty)
                        on_exit(ltp)
                        break
                except Exception:
                    pass
                time.sleep(2)
    finally:
        with monitor_lock:
            if monitor_id in st.session_state["running_monitors"]:
                st.session_state["running_monitors"].remove(monitor_id)

# UI
st.set_page_config(page_title="Angel One — Doji Bot (Enhanced)", layout="wide")
st.title("📈 Angel One — Doji Strategy (CE/PE) + RSI + WebSocket Monitor")

col1, col2 = st.columns([2,1])

with col2:
    mode = st.selectbox("Mode", ["Paper (Safe)", "Live (Real Orders)"])
    st.markdown("- Paper: simulate trades locally (safe)\n- Live: places real market orders (be careful!)")

with col1:
    st.header("1) Login / Logout")
    if st.session_state["logged_in"]:
        st.success("✅ Logged in")
        if st.button("Logout"):
            try:
                st.session_state["obj"].terminateSession(CLIENT_ID)
            except Exception:
                try:
                    st.session_state["obj"].logout()
                except:
                    pass
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["obj"] = None
            st.session_state["logged_in"] = False
            initialize_session_state()
            st.success("Logged out and session reinitialized")
    else:
        if st.button("Login (use .env)"):
            try:
                sc = create_smartconnect(API_KEY)
                otp = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else None
                data = sc.generateSession(CLIENT_ID, CLIENT_PWD, otp)
                st.session_state["obj"] = sc
                st.session_state["logged_in"] = True
                if data and isinstance(data, dict):
                    st.session_state["session_data"] = data
                try:
                    token_response = sc.getfeedToken()
                   # st.write("Debug - API Response for Feed Token:", token_response)

                    # जांचें कि रेस्पॉन्स एक डिक्शनरी है या नहीं
                    if isinstance(token_response, dict):
                        feed_token_value = token_response.get("data", {}).get("feedToken")
                        if feed_token_value:
                            st.session_state["feed_token"] = feed_token_value
                            st.success("Login successful + Feed Token fetched.")
                        else:
                           st.warning(f"Login successful, but could not find Feed Token in response: {token_response}")
                    
                    # जांचें कि रेस्पॉन्स सिर्फ एक स्ट्रिंग है या नहीं
                    elif isinstance(token_response, str):
                        st.session_state["feed_token"] = token_response
                        st.success("Login successful + Feed Token fetched.")
                    
                    else:
                        st.warning(f"Login successful, but Feed Token response was in an unexpected format: {token_response}")

                except Exception as e:
                    st.warning(f"Login successful, but an error occurred while fetching Feed Token: {e}")
                # --- फीड टोकन का कोड यहाँ खत्म होता है ---
            except Exception as e:
                st.error(f"Login failed: {e}")

st.markdown("---")

if st.session_state["logged_in"] and st.session_state["obj"]:
    obj = st.session_state["obj"]

    st.subheader("2) Fetch Expiries / Setup")
    if st.button("📅 Get NIFTY Options (Instrument Master)"):
        master = load_instrument_master_json()
        if master:
            built = build_search_from_master(master, symbol_name="NIFTY")
            if built:
                st.session_state["search_raw"] = built
                expiries = sorted(list({item.get("expiry") for item in built.get("data", []) if item.get("expiry")}))
                st.session_state["expiries"] = expiries
                st.success(f"Found {len(expiries)} expiries (from instrument master)")
            else:
                st.warning("Instrument master downloaded but no NIFTY options rows found. Falling back to API search...")
                search = try_search_variants(obj, "NIFTY", "NFO")
                if search:
                    st.session_state["search_raw"] = search
                    expiries = sorted(list({item.get("expiry") for item in search.get("data", []) if item.get("expiry")}))
                    st.session_state["expiries"] = expiries
                    st.success(f"Found {len(expiries)} expiries (from searchScrip)")
                else:
                    st.error("Fallback searchScrip also returned no usable data.")
        else:
            st.info("Instrument master download failed or blocked. Trying API searchScrip fallback...")
            search = try_search_variants(obj, "NIFTY", "NFO")
            if search:
                st.session_state["search_raw"] = search
                expiries = sorted(list({item.get("expiry") for item in search.get("data", []) if item.get("expiry")}))
                st.session_state["expiries"] = expiries
                st.success(f"Found {len(expiries)} expiries (from searchScrip)")
            else:
                st.error("No instrument data available from instrument master or API. Expand debug below.")
                st.write("API fallback response:", search)

    expiry = None
    if st.session_state.get("expiries"):
        expiry = st.selectbox("Select expiry", st.session_state["expiries"])
        st.session_state["selected_expiry"] = expiry

    interval = st.selectbox("Candle interval", ["5min","1min","15min"])
    strike_mode = st.radio("Strike Mode", ["ATM","ITM100","OTM100"])
    contracts = st.number_input("Contracts (lots)", min_value=1, value=1, step=1)
    sl_offset = st.number_input("SL Offset (₹)", min_value=0.0, value=20.0, step=0.5)
    tp_offset = st.number_input("TP Offset (₹)", min_value=0.0, value=20.0, step=0.5)
    days_back = st.number_input("Days Back for Candles", min_value=1, value=1, step=1)
    enable_pe = st.checkbox("Enable PE Signals (Bearish)")
    rsi_period = st.number_input("RSI Period", min_value=5, value=14, step=1)
    rsi_call_threshold = st.number_input("RSI Threshold for CALL (Oversold < )", min_value=20, value=30, step=5)
    rsi_pe_threshold = st.number_input("RSI Threshold for PE (Overbought > )", min_value=70, value=70, step=5)
    use_websocket = st.checkbox("Use WebSocket for Real-time Monitoring")

    st.subheader("3) Run Doji Strategy (one-shot)")
    if st.button("▶ Run Strategy Now"):
        if contracts <= 0:
            st.error("Contracts must be >0")
            st.stop()
        if not st.session_state.get("search_raw"):
            st.error("No instrument data available. Click 'Get NIFTY Options' first.")
            st.stop()
        try:
            search_raw = st.session_state.get("search_raw", {})
            index_token = None
            for item in search_raw.get("data", []):
                sym = item.get("tradingsymbol") or item.get("symbol") or item.get("symbolname")
                if sym and "NIFTY" in str(sym).upper():
                    index_token = item.get("symboltoken") or item.get("token") or item.get("id")
                    break
            if not index_token:
                st.warning("Index token not found; using fallback '26000'")
                index_token = "26000"

            df = try_fetch_candles(obj, index_token, interval=interval, days_back=days_back)
            st.write("Candles tail (latest):")
            st.dataframe(df.tail(10))

            signals = detect_doji_and_entry(df, lookback_prev=3, lookahead=5, rsi_period=rsi_period, rsi_call_threshold=rsi_call_threshold, rsi_pe_threshold=rsi_pe_threshold, enable_pe=enable_pe)
            if not signals:
                st.info("No valid doji+breakout signals found in latest data.")
            else:
                st.success(f"Found {len(signals)} signal(s). Processing first signal...")
                sig = signals[0]
                entry_price = sig["entry_price"]
                futures_token = find_nifty_futures_token(search_raw, obj)
                spot = float(df['close'].iloc[-1])
                if futures_token:
                    try:
                        futures_ltp = get_option_ltp_with_backoff(obj, "NFO", "NIFTY-FUT", futures_token)
                        atm = round(futures_ltp / 50) * 50
                    except:
                        atm = round(spot / 50) * 50
                else:
                    atm = round(spot / 50) * 50
                if strike_mode == "ATM":
                    strike = atm
                elif strike_mode == "ITM100":
                    strike = atm - 100 if sig["signal"] == "CALL" else atm + 100
                else:
                    strike = atm + 100 if sig["signal"] == "CALL" else atm - 100
                option_cepe = sig["signal"]
                token, tradingsymbol = find_token_in_search(search_raw, expiry, strike, option_cepe)
                if not token:
                    st.error("Option token not found for selected expiry/strike.")
                    st.write(search_raw.get("data", [])[:20])
                else:
                    opt_ltp = get_option_ltp_with_backoff(obj, "NFO", tradingsymbol, token) or entry_price
                    entry_price = opt_ltp
                    sl = entry_price - sl_offset
                    tp = entry_price + tp_offset
                    qty = int(contracts * LOT_SIZE)
                    st.write({"signal": sig["signal"], "entry_price": entry_price, "SL": sl, "TP": tp, "qty": qty, "RSI": sig.get("rsi", "N/A")})

                    if mode.startswith("Paper"):
                        trade = paper_record(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expiry, strike, option_cepe, qty, entry_price)
                        st.success(f"Paper trade recorded: id={trade['id']}, {option_cepe}")
                        monitor_id = f"paper-{trade['id']}-{time.time()}"
                        t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Paper", obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, trade['id'], use_websocket))
                        t.daemon = True
                        t.start()
                        st.info("Started background SL/TP monitor (paper).")
                    else:
                        resp = place_market_buy(obj, tradingsymbol, token, qty)
                        st.success(f"Live BUY placed: {resp}, {option_cepe}")
                        monitor_id = f"live-{time.time()}"
                        t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Live", obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, None, use_websocket))
                        t.daemon = True
                        t.start()
                        st.info("Started background SL/TP monitor (live).")

        except Exception as e:
            st.error(f"Strategy run failed: {e}")
            st.write("Check debug section for raw search data.")

# Backtest
    st.markdown("---")
    st.header("4) Backtest Strategy on Historical Data (using NIFTY Futures)")
    today = datetime.now().date()
    default_start = today - timedelta(days=90)
    start_date = st.date_input("Start Date for Backtesting", default_start, max_value=today)
    end_date = st.date_input("End Date for Backtesting", today, max_value=today)
    hypothetical_option_price = st.number_input("Hypothetical Option Entry Price (₹):", min_value=1.0, value=120.0, step=0.5)
    trade_cost = st.number_input("Cost per Trade (Brokerage, Taxes etc.) ₹:", min_value=0.0, value=50.0, step=1.0) # <-- ब्रोकरेज के लिए नया इनपुट

    if hypothetical_option_price <= 0:
        st.error("Hypothetical price must be >0")
        st.stop()
    
    if st.button("▶ Run Backtest Now"):
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        elif not st.session_state.get("search_raw"):
            st.error("No instrument data available. Click 'Get NIFTY Options' first.")
        else:
            try:
                futures_token = find_nifty_futures_token(st.session_state.get("search_raw", {}), obj)
                st.info(f"Using NIFTY Futures (Token: {futures_token}) for backtesting.")
                
                all_trades = []
                current_date = start_date
                
                while current_date <= end_date:
                    if current_date.weekday() >= 5:  # Skip weekends
                        current_date += timedelta(days=1)
                        continue
                    
                    with st.spinner(f"Fetching data for {current_date.strftime('%Y-%m-%d')}..."):
                        df_historical = try_fetch_historical_candles(obj, futures_token, interval, current_date, retries=3)

                    if df_historical is not None and not df_historical.empty:
                        signals = detect_doji_and_entry(df_historical)
                        
                        for sig in signals:
                            entry_time = df_historical.iloc[sig['entry_index']]['timestamp']
                            index_entry_price = sig['entry_price']
                            
                            # --- YAHAN SE BADE BADLAV SHURU HOTE HAIN ---
                            
                            hypo_entry_option_price = hypothetical_option_price
                            hypo_sl_option_price = hypo_entry_option_price - sl_offset
                            hypo_tp_option_price = hypo_entry_option_price + tp_offset
                            
                            # Simulate if SL or TP was hit first based on NIFTY FUTURES movement
                            # We will use option prices for this simulation
                            trade_pnl = 0
                            trade_status = "UNKNOWN"
                            
                            # Let's find the outcome based on the option's hypothetical prices
                            # This requires checking future candles after the entry signal
                            temp_df = df_historical.copy()
                            temp_df['option_price_est'] = hypo_entry_option_price + (temp_df['close'] - index_entry_price)

                            for i in range(sig['entry_index'] + 1, len(temp_df)):
                                candle = temp_df.iloc[i]
                                # Check if Stoploss is hit
                                if candle['option_price_est'] <= hypo_sl_option_price:
                                    trade_pnl = (hypo_sl_option_price - hypo_entry_option_price) * LOT_SIZE
                                    trade_status = "SL_HIT"
                                    break
                                # Check if Target is hit
                                if candle['option_price_est'] >= hypo_tp_option_price:
                                    trade_pnl = (hypo_tp_option_price - hypo_entry_option_price) * LOT_SIZE
                                    trade_status = "TP_HIT"
                                    break
                            
                            if trade_status == "UNKNOWN": # Neither SL nor TP hit by EOD
                                final_option_price = temp_df.iloc[-1]['option_price_est']
                                trade_pnl = (final_option_price - hypo_entry_option_price) * LOT_SIZE
                                trade_status = "EOD_EXIT"
                            
                            net_pnl = trade_pnl - trade_cost # <-- ब्रोकरेज घटाया गया

                            all_trades.append({
                                "date": entry_time.date(),
                                "signal_time": entry_time.time(),
                                "signal_type": "CALL",
                                "index_entry_price": index_entry_price,
                                "hypo_option_entry": hypo_entry_option_price,
                                "hypo_option_sl": hypo_sl_option_price,
                                "hypo_option_tp": hypo_tp_option_price,
                                "status": trade_status,
                                "net_pnl": net_pnl
                            })

                    else:
                        st.warning(f"No data for {current_date.strftime('%Y-%m-%d')} (possibly holiday). Skipping.")
                    
                    current_date += timedelta(days=1)
                    time.sleep(0.5)
                
                if not all_trades:
                    st.info("No valid signals found in the date range.")
                else:
                    st.success(f"Found {len(all_trades)} trade(s) over {(end_date - start_date).days} days!")
                    trades_df = pd.DataFrame(all_trades)
                    
                    # Display results
                    st.dataframe(trades_df[[
                        'date', 'signal_time', 'signal_type', 'index_entry_price', 
                        'hypo_option_entry', 'status', 'net_pnl'
                    ]])
                    
                    total_pnl = trades_df['net_pnl'].sum()
                    wins = trades_df[trades_df['net_pnl'] > 0]
                    losses = trades_df[trades_df['net_pnl'] <= 0]
                    win_rate = (len(wins) / len(all_trades)) * 100 if all_trades else 0
                    
                    st.write(f"### Backtest Summary")
                    st.write(f"**Total Net PnL (After Costs): ₹{total_pnl:,.2f}**")
                    st.write(f"**Win Rate:** {win_rate:.2f}% ({len(wins)} wins / {len(losses)} losses)")
                    st.write(f"**Total Trades:** {len(all_trades)}")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
            
st.markdown("---")
st.subheader("Paper Trades (history)")

if "paper_trades" not in st.session_state:
    st.session_state["paper_trades"] = []
    st.warning("Paper trades not initialized. Set to empty list.")

if st.session_state["paper_trades"]:
    df_tr = pd.DataFrame(st.session_state["paper_trades"])
    df_tr['time'] = pd.to_datetime(df_tr['time'])

    st.write("Filter trades by date:")
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input("Select Date", datetime.now().date())
    
    with col2:
        show_all = st.checkbox("Show all trades", value=False)

    if show_all:
        st.dataframe(df_tr)
    else:
        filtered_df = df_tr[df_tr['time'].dt.date == selected_date]
        if filtered_df.empty:
            st.info(f"No trades found for {selected_date.strftime('%d-%b-%Y')}.")
        else:
            st.dataframe(filtered_df)
else:
    st.info("No paper trades have been recorded yet.")

st.markdown("Debug: raw instrument master / search response")
if st.session_state.get("search_raw"):
    with st.expander("Raw search response (filterable)"):
        filter_symbol = st.text_input("Filter by tradingsymbol (e.g., NIFTY25SEP25 or NIFTY-FUT):")
        data = st.session_state["search_raw"].get("data", [])
        if filter_symbol:
            data = [item for item in data if filter_symbol.upper() in str(item.get("tradingsymbol", "")).upper()]
        st.write(data[:50])
        st.write(f"Total rows: {len(data)}")
        futures_count = sum(1 for item in data if item.get("name", "").upper() == "NIFTY" and item.get("instrumenttype", "") == "FUTIDX")
        st.write(f"NIFTY Futures entries found: {futures_count}")
else:
    st.info("Please login and click 'Get NIFTY Options' to fetch instrument data.")