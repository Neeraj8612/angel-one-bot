# app.py (Updated — keeps your original logic, adds Instrument Master based expiry/token loader)
import os
import time
import threading
import requests
import streamlit as st
import pandas as pd
import pyotp
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Try import SmartConnect (support both package name variants)
try:
    from SmartApi import SmartConnect
except Exception:
    try:
        from SmartApi import SmartConnect
    except Exception:
        SmartConnect = None

# ---------------- Load env ----------------
load_dotenv()
API_KEY = os.getenv("API_KEY", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_PWD = os.getenv("CLIENT_PWD", "")
TOTP_SECRET = os.getenv("TOTP_SECRET", "")

# ---------------- Session defaults ----------------
defaults = {
    "obj": None,
    "logged_in": False,
    "search_raw": {},
    "expiries": [],
    "selected_expiry": None,
    "paper_trades": [],
    "running_monitors": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Helpers: SmartConnect ----------------
def ensure_smartconnect():
    if SmartConnect is None:
        raise RuntimeError("SmartConnect import failed. Install smartapi-python or SmartApi package.")

def create_smartconnect(api_key):
    ensure_smartconnect()
    try:
        return SmartConnect(api_key=api_key)
    except TypeError:
        return SmartConnect(api_key)

# ---------------- Instrument Master loader ----------------
INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def load_instrument_master_json():
    """
    Download instrument master JSON from Angel (OpenAPI). Returns list of dicts or None.
    """
    try:
        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # data usually is a list of dicts
        if isinstance(data, list):
            return data
        # sometimes nested
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        return None
    except Exception as e:
        print("Instrument master download failed:", e)
        return None

def build_search_from_master(master_list, symbol_name="NIFTY"):
    """
    Filter master_list for NIFTY OPTIDX in NFO and return search-like dict: {"data":[...]}
    Each item kept minimal keys: expiry, strike, tradingsymbol, symboltoken, instrumenttype, name
    """
    if not master_list:
        return None
    rows = []
    for item in master_list:
        # normalize keys to lower for robust access
        itm = {k.lower(): v for k, v in item.items()}
        # check exchange field
        exch = itm.get("exch_seg") or itm.get("exchange") or itm.get("exch")
        name = itm.get("name") or itm.get("symbol") or itm.get("tradingsymbol") or itm.get("instrument")
        inst_type = itm.get("instrumenttype") or itm.get("insttype") or itm.get("instrument_type")
        # try to match NFO + NIFTY + option index
        if exch and str(exch).upper() == "NFO":
            nm = str(name or "").upper()
            itype = str(inst_type or "").upper()
            # prefer instrumenttype OPTIDX; if missing, check name contains NIFTY
            if (itype == "OPTIDX" and symbol_name.upper() in nm) or (symbol_name.upper() in nm and itype == ""):
                # build normalized dict
                row = {
                    "expiry": itm.get("expiry") or itm.get("expirydate") or itm.get("expiry_date"),
                    "strike": itm.get("strike") or itm.get("strikeprice") or itm.get("strike_price"),
                    "tradingsymbol": itm.get("tradingsymbol") or itm.get("symbolname") or itm.get("symbol"),
                    "symboltoken": itm.get("symboltoken") or itm.get("token") or itm.get("instrument_token") or itm.get("id"),
                    "instrumenttype": itype,
                    "name": name
                }
                # only include if expiry present or tradingsymbol present
                if row["tradingsymbol"]:
                    rows.append(row)
    if not rows:
        return None
    return {"data": rows}

# ---------------- robust searchScrip wrapper (fallback) ----------------
def try_search_variants(obj, symbol: str, exchange: str = None):
    """
    Attempts multiple signatures of searchScrip; returns dict with 'data' or None
    (But primary source will be instrument master if available)
    """
    last_exc = None
    candidates = []
    # create candidate lambdas that may exist
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
            # try find nested list
            if isinstance(res, dict):
                for k in ("data", "Data", "result"):
                    if k in res and isinstance(res[k], list):
                        return {"data": res[k]}
        except Exception as e:
            last_exc = e
            continue
    return None

# ---------------- existing helper functions kept as-is ----------------
def try_fetch_candles(obj, symbol_token, interval="5min", days_back=1):
    to_dt   = datetime.now()
    from_dt = to_dt - timedelta(days=days_back)
    imap    = {
        "1min":"ONE_MINUTE",  "5min":"FIVE_MINUTE",  "15min":"FIFTEEN_MINUTE",
        "1minute":"ONE_MINUTE","5minute":"FIVE_MINUTE","15minute":"FIFTEEN_MINUTE",
    }
    iarg = imap.get(interval, interval)

    params = {
        "exchange":   "NFO",
        "symboltoken": str(symbol_token),
        "interval":    iarg,
        "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_dt.strftime("%Y-%m-%d %H:%M")
    }

    try:
        res = obj.getCandleData(params)  # API से data लाओ
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")  # अगर API ही fail हो जाए

    if not res or "data" not in res or not res["data"]:
        raise RuntimeError("Token ya interval ke liye koi candle data nahi mila")  # Data missing

    # अब data को line-by-line पढ़ो और DataFrame बनाओ
    try:
        rows = []
        for row in res["data"]:
            if len(row) < 6:
                continue  # अगर row अधूरी है तो छोड़ दो
            rows.append({
                "timestamp": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": int(row[5])
            })

        if not rows:
            raise RuntimeError("Data mila, par parse nahi ho paaya")  # Parsing fail

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        raise RuntimeError(f"Parsing mein error: {e}")

def try_fetch_historical_candles(obj, symbol_token, interval, target_date):
    """Fetches candle data for a specific historical date."""
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
            print(f"No data received for {target_date.strftime('%Y-%m-%d')}. Response: {res}")
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
        print(f"Error fetching historical candles for {target_date.strftime('%Y-%m-%d')}: {e}")
        st.error(f"Failed to fetch historical data: {e}")
        return None
    
def find_token_in_search(search_dict, expiry, strike, option_cepe):
    """Scan instrument master search response for a matching token + tradingsymbol"""
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

def get_option_ltp(obj, exchange, tradingsymbol, symboltoken):
    try:
        r = obj.ltpData(exchange=exchange, tradingsymbol=tradingsymbol, symboltoken=str(symboltoken))
        if isinstance(r, dict):
            data = r.get("data")
            if isinstance(data, dict) and "ltp" in data:
                return float(data["ltp"])
            if isinstance(data, list) and len(data)>0 and isinstance(data[0], dict) and "ltp" in data[0]:
                return float(data[0]["ltp"])
        # try nested
        if isinstance(r, dict):
            for v in (r.get("result") or r.get("data") or []):
                if isinstance(v, dict) and "ltp" in v:
                    return float(v["ltp"])
        raise RuntimeError("LTP parse failed")
    except Exception as e:
        raise

def find_nifty_futures_token(search_dict):
    """Finds the symbol token for the nearest NIFTY futures contract."""
    if not search_dict or "data" not in search_dict:
        return None
    
    nifty_futures = []
    for item in search_dict.get("data", []):
        # normalize keys
        itm = {k.lower(): v for k, v in item.items()}
        name = itm.get("name", "")
        inst_type = itm.get("instrumenttype", "")
        
        if name == "NIFTY" and inst_type == "FUTIDX":
            nifty_futures.append(item)
    
    if not nifty_futures:
        return None
    
    # Sort by expiry to find the nearest contract
    nifty_futures.sort(key=lambda x: x.get("expiry", ""))
    
    # Return the token of the first (nearest) future
    return nifty_futures[0].get("symboltoken")

# ---------------- Strategy helpers (same as before) ----------------
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

def detect_doji_and_entry(df, lookback_prev=3, lookahead=5):
    signals = []
    if df is None or len(df) < (lookback_prev + 2):
        return signals
    n = len(df)
    for i in range(lookback_prev, n - lookahead):
        prev = df.iloc[i - lookback_prev:i]
        if not is_downtrend(prev):
            continue
        row = df.iloc[i]
        if is_doji_row(row):
            doji_high = row['high']
            for j in range(1, lookahead+1):
                nxt = df.iloc[i+j]
                if nxt['high'] > doji_high:
                    signals.append({
                        "signal": "CALL",
                        "doji_index": i,
                        "entry_index": i+j,
                        "entry_price": float(nxt['close']),
                        "doji_high": float(doji_high)
                    })
                    break
    return signals

# ---------------- Paper trade helpers (same) ----------------
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
            t["pnl"] = (exit_price - t["entry_price"]) * t["qty"]
            return t
    return None

# ---------------- Orders & monitor (same) ----------------
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

def monitor_trade_background(monitor_id, mode, obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, paper_trade_id=None):
    st.session_state["running_monitors"].append(monitor_id)
    try:
        while True:
            if monitor_id not in st.session_state["running_monitors"]:
                break
            try:
                ltp = None
                try:
                    ltp = get_option_ltp(obj, "NFO", tradingsymbol, token)
                except Exception:
                    ltp = entry_price
                if ltp <= sl or ltp >= tp:
                    if mode == "Paper":
                        paper_close(paper_trade_id, ltp)
                    else:
                        try:
                            resp = place_market_sell(obj, tradingsymbol, token, qty)
                        except Exception:
                            pass
                    break
            except Exception:
                pass
            time.sleep(2)
    finally:
        if monitor_id in st.session_state["running_monitors"]:
            st.session_state["running_monitors"].remove(monitor_id)

# ---------------- UI / Streamlit (keeps same flow) ----------------
st.set_page_config(page_title="Angel One — Doji Bot (Final)", layout="wide")
st.title("📈 Angel One — Doji Strategy (Buy-side) + Paper/Live + SL/TP Monitor")

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
                try:
                    st.session_state["obj"].terminateSession(CLIENT_ID)
                except Exception:
                    try:
                        st.session_state["obj"].logout()
                    except:
                        pass
                st.session_state["obj"] = None
                st.session_state["logged_in"] = False
                st.success("Logged out")
            except Exception as e:
                st.error(f"Logout failed: {e}")
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
                st.success("Login successful")
            except Exception as e:
                st.error(f"Login failed: {e}")

st.markdown("---")

# ------------------ EXPIRY FETCH (instrument master preferred) ------------------
if st.session_state["logged_in"] and st.session_state["obj"]:
    obj = st.session_state["obj"]

    st.subheader("2) Fetch Expiries / Setup")
    if st.button("📅 Get NIFTY Options (Instrument Master)"):
        # 1) try instrument master
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
                # fallback to API searchScrip
                search = try_search_variants(obj, "NIFTY", "NFO")
                if search:
                    st.session_state["search_raw"] = search
                    expiries = sorted(list({item.get("expiry") for item in search.get("data", []) if item.get("expiry")}))
                    st.session_state["expiries"] = expiries
                    st.success(f"Found {len(expiries)} expiries (from searchScrip)")
                else:
                    st.error("Fallback searchScrip also returned no usable data.")
        else:
            # instrument master download failed -> fallback to searchScrip
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

    # expiry select
    expiry = None
    if st.session_state.get("expiries"):
        expiry = st.selectbox("Select expiry", st.session_state["expiries"])
        st.session_state["selected_expiry"] = expiry

    interval = st.selectbox("Candle interval", ["5min","1min","15min"])
    strike_mode = st.radio("Strike Mode", ["ATM","ITM100","OTM100"])
    contracts = st.number_input("Contracts (lots)", min_value=1, value=1, step=1)

    st.subheader("3) Run Doji Strategy (one-shot)")
    st.caption("Click 'Run Strategy Now' to analyze latest candles and place trades (Paper/Live).")
    if st.button("▶ Run Strategy Now"):
        try:
            search_raw = st.session_state.get("search_raw", {})
            # Find NIFTY index token if present (fallbacks)
            index_token = None
            for item in search_raw.get("data", []):
                sym = item.get("tradingsymbol") or item.get("symbol") or item.get("symbolname")
                if sym and "NIFTY" in str(sym).upper():
                    index_token = item.get("symboltoken") or item.get("token") or item.get("id")
                    break
            if not index_token:
                st.warning("Index token not found in instrument master; using fallback token '26000'")
                index_token = "26000"

            df = try_fetch_candles(obj, index_token, interval=interval, days_back=1)
            st.write("Candles tail (latest):")
            st.dataframe(df.tail(10))

            signals = detect_doji_and_entry(df, lookback_prev=3, lookahead=5)
            if not signals:
                st.info("No valid doji+breakout signals found in latest data.")
            else:
                st.success(f"Found {len(signals)} signal(s). Processing first signal...")
                sig = signals[0]
                entry_price = sig["entry_price"]
                # choose strike based on spot (approx)
                spot = float(df['close'].iloc[-1])
                atm = round(spot/50)*50
                if strike_mode == "ATM":
                    strike = atm
                elif strike_mode == "ITM100":
                    strike = atm - 100
                else:
                    strike = atm + 100
                option_cepe = "CE"
                token, tradingsymbol = find_token_in_search(search_raw, expiry, strike, option_cepe)
                if not token:
                    st.error("Option token not found for selected expiry/strike. Expand raw search to debug.")
                    if search_raw:
                        with st.expander("Raw search snippet"):
                            st.write(search_raw.get("data", [])[:20])
                else:
                    try:
                        opt_ltp = get_option_ltp(obj, "NFO", tradingsymbol, token)
                    except Exception:
                        opt_ltp = entry_price
                    entry_price = opt_ltp if opt_ltp else entry_price
                    sl = entry_price - 10
                    tp = entry_price + 20
                    qty = int(contracts * LOT_SIZE)
                    st.write({"entry_price": entry_price, "SL": sl, "TP": tp, "qty": qty})

                    if mode.startswith("Paper"):
                        trade = paper_record(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), expiry, strike, option_cepe, qty, entry_price)
                        st.success(f"Paper trade recorded: id={trade['id']}")
                        monitor_id = f"paper-{trade['id']}-{time.time()}"
                        t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Paper", obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, trade['id']))
                        t.daemon = True
                        t.start()
                        st.info("Started background SL/TP monitor (paper).")
                    else:
                        try:
                            resp = place_market_buy(obj, tradingsymbol, token, qty)
                            st.success(f"Live BUY placed: {resp}")
                            monitor_id = f"live-{time.time()}"
                            t = threading.Thread(target=monitor_trade_background, args=(monitor_id, "Live", obj, expiry, strike, option_cepe, token, tradingsymbol, qty, entry_price, sl, tp, None))
                            t.daemon = True
                            t.start()
                            st.info("Started background SL/TP monitor (live).")
                        except Exception as e:
                            st.error(f"Live order failed: {e}")

        except Exception as e:
            st.error(f"Strategy run failed: {e}")

# --------- 4) Backtest Strategy ---------
    st.markdown("---")
    st.header("4) Backtest Strategy on Historical Data (using NIFTY Futures)")
    backtest_date = st.date_input("Select a date for backtesting", datetime.now().date() - timedelta(days=1))
    hypothetical_option_price = st.number_input("Enter a hypothetical option entry price (e.g., 120):", value=120)
    
    if st.button("▶ Run Backtest Now"):
        if backtest_date.weekday() in [5, 6]:
            st.error("The selected date is a Saturday or Sunday. Please select a trading day.")
        elif backtest_date >= datetime.now().date():
            st.warning("Please select a date in the past for backtesting.")
        else:
            try:
                # NIFTY फ्यूचर्स का टोकन खोजना
                futures_token = find_nifty_futures_token(st.session_state.get("search_raw", {}))
                
                if not futures_token:
                    st.error("NIFTY Futures token not found. Please click 'Get NIFTY Options' button first.")
                else:
                    st.info(f"Using NIFTY Futures (Token: {futures_token}) for backtesting.")
                    with st.spinner(f"Fetching futures data for {backtest_date.strftime('%Y-%m-%d')}..."):
                        # फ्यूचर्स टोकन का उपयोग करके ऐतिहासिक डेटा लाना
                        df_historical = try_fetch_historical_candles(obj, futures_token, interval, backtest_date)
                    
                    if df_historical is not None and not df_historical.empty:
                        st.write("Running strategy on Futures data to find signals...")
                        signals = detect_doji_and_entry(df_historical)
                        if not signals:
                            st.info(f"No valid signals found for {backtest_date.strftime('%Y-%m-%d')}.")
                        else:
                            st.success(f"Found {len(signals)} signal(s)!")
                            # ... (बाकी का कोड वही रहेगा)
                            signals_display = []
                            for sig in signals:
                                entry_time = df_historical.iloc[sig['entry_index']]['timestamp']
                                index_entry_price = sig['entry_price']
                                atm_strike = round(index_entry_price / 50) * 50
                                signals_display.append({
                                    "Signal Time": entry_time.strftime('%H:%M:%S'), "Futures Entry Price": index_entry_price,
                                    "ATM Strike": f"{atm_strike} CE", "Hypothetical Option Entry": hypothetical_option_price,
                                    "Calculated SL": hypothetical_option_price - 10, "Calculated TP": hypothetical_option_price + 20
                                })
                            st.table(pd.DataFrame(signals_display))
                    else:
                        st.error(f"Could not fetch futures data. The selected date might be a market holiday.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

st.markdown("---")
st.subheader("Paper Trades (history)")

# अगर कोई पेपर ट्रेड मौजूद है, तभी फिल्टर दिखाएंगे
if st.session_state["paper_trades"]:
    # 1. डेटा को DataFrame में बदलना
    df_tr = pd.DataFrame(st.session_state["paper_trades"])
    # महत्वपूर्ण: 'time' कॉलम को टेक्स्ट से datetime ऑब्जेक्ट में बदलना ताकि फिल्टर हो सके
    df_tr['time'] = pd.to_datetime(df_tr['time'])

    # 2. UI में फिल्टर के लिए ऑप्शन बनाना
    st.write("Filter trades by date:")
    col1, col2 = st.columns(2)

    with col1:
        # तारीख चुनने के लिए कैलेंडर
        selected_date = st.date_input("Select Date", datetime.now().date())
    
    with col2:
        # "Show All" का ऑप्शन देने के लिए चेकबॉक्स
        show_all = st.checkbox("Show all trades", value=False)

    # 3. फिल्टर लॉजिक
    if show_all:
        # अगर चेकबॉक्स टिक है, तो सारे ट्रेड दिखाओ
        st.dataframe(df_tr)
    else:
        # अगर चेकबॉक्स टिक नहीं है, तो चुनी हुई तारीख के हिसाब से फिल्टर करो
        filtered_df = df_tr[df_tr['time'].dt.date == selected_date]
        if filtered_df.empty:
            st.info(f"No trades found for {selected_date.strftime('%d-%b-%Y')}.")
        else:
            st.dataframe(filtered_df)

else:
    st.info("No paper trades have been recorded yet.")

st.markdown("Debug: raw instrument master / search response")
if st.session_state.get("search_raw"):
    with st.expander("Raw search response (first 50 rows)"):
        st.write(st.session_state["search_raw"].get("data", [])[:50])

else:
    st.info("Please login to use strategy and expiry fetch.")
