import os
import time
import threading
import requests
import streamlit as st
import pandas as pd
import pyotp
import json
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta, time as dt_time
import gzip
import retrying
import sqlite3
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from threading import Lock

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(file_handler)

# --- Try to import SmartApi ---
try:
    from SmartApi import SmartConnect
except ImportError:
    st.error("SmartApi library not found. Please install it using 'pip install smartapi-python'")
    st.stop()

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_PWD = os.getenv("CLIENT_PWD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# --- Central Index Configuration ---
INDEX_CONFIG = {
    "NIFTY": {"lot_size": 75, "exchange": "NFO", "strike_step": 50, "instrument_name": "NIFTY"},
    "BANKNIFTY": {"lot_size": 35, "exchange": "NFO", "strike_step": 100, "instrument_name": "NIFTY BANK"},
    "FINNIFTY": {"lot_size": 65, "exchange": "NFO", "strike_step": 50, "instrument_name": "NIFTY FIN SERVICE"},
    "SENSEX": {"lot_size": 20, "exchange": "BFO", "strike_step": 100, "instrument_name": "SENSEX"}
}
ALL_INDICES = list(INDEX_CONFIG.keys())

# --- File path for instrument list cache ---
INSTRUMENT_CACHE_FILE = "instrument_list.json.gz"
STATE_DB = "bot_state.db"

# <editor-fold desc="Helper Functions">

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def fetch_and_cache_instrument_list():
    try:
        if os.path.exists(INSTRUMENT_CACHE_FILE):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(INSTRUMENT_CACHE_FILE))
            if datetime.now() - file_mod_time < timedelta(hours=24):
                with gzip.open(INSTRUMENT_CACHE_FILE, 'rt', encoding='utf-8') as f: return json.load(f)
        logging.info("Downloading fresh instrument list.")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        instrument_data = response.json()
        with gzip.open(INSTRUMENT_CACHE_FILE, 'wt', encoding='utf-8') as f: json.dump(instrument_data, f)
        return instrument_data
    except Exception as e:
        st.error(f"Fatal Error: Could not download instrument list. Details: {e}")
        return None

def find_index_futures_token(instrument_list, index_name, exchange):
    if not instrument_list: return None
    config = INDEX_CONFIG[index_name]
    instrument_name = config.get("instrument_name", index_name)
    futures = []
    today = datetime.now().date()
    for item in instrument_list:
        if (item.get("instrumenttype") == "FUTIDX" and item.get("name") == instrument_name and item.get("exch_seg") == exchange):
            # यह जाँच सुनिश्चित करती है कि NIFTY के लिए NIFTYNXT50 का टोकन न चुना जाए
            if instrument_name == "NIFTY" and "NIFTYNXT50" in item.get("symbol", ""):
                continue # यह NIFTYNXT50 है, इसे छोड़ दें और अगले आइटम पर जाएं
            try:
                expiry_date = datetime.strptime(item["expiry"], '%d%b%Y').date()
                if expiry_date >= today:
                    futures.append((expiry_date, item))
            except (ValueError, KeyError): continue
    if not futures: return None
    futures.sort(key=lambda x: x[0])
    return futures[0][1].get("token")


def find_option_token_from_list(instrument_list, index_name, strike, expiry, option_cepe):
    if not instrument_list: raise RuntimeError("Instrument list not loaded.")
    config = INDEX_CONFIG[index_name]
    instrument_name = config.get("instrument_name", index_name)
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d%b%Y").upper()
    for item in instrument_list:
        if (item.get("name") == instrument_name and item.get("exch_seg") == config['exchange'] and
            item.get("instrumenttype") == "OPTIDX" and item.get("strike") and
            float(item.get("strike")) / 100 == float(strike) and item.get("expiry") == expiry_fmt and
            item.get("symbol", "").endswith(option_cepe)):
            return item.get("token"), item.get("symbol")
    raise RuntimeError(f"Could not find {option_cepe} for strike {strike} and expiry {expiry} in instrument list.")

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def try_fetch_candles(obj, symbol_token, interval, days_back, exchange):
    to_dt, from_dt = datetime.now(), datetime.now() - timedelta(days=days_back)
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    params = {"exchange": exchange, "symboltoken": str(symbol_token), "interval": imap.get(interval),
              "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"), "todate": to_dt.strftime("%Y-%m-%d %H:%M")}
    try:
        res = obj.getCandleData(params)
        if not res or "data" not in res or not res["data"]: return None
        rows = [{"timestamp": pd.to_datetime(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in res["data"]]
        return pd.DataFrame(rows)
    except Exception as e:
        logging.error(f"Error fetching candles for {symbol_token}: {e}")
        return None

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def try_fetch_historical_candles(obj, symbol_token, interval, target_date, exchange):
    from_dt_str = target_date.strftime("%Y-%m-%d") + " 09:15"
    to_dt_str = target_date.strftime("%Y-%m-%d") + " 15:30"
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    params = {"exchange": exchange, "symboltoken": str(symbol_token), "interval": imap.get(interval), "fromdate": from_dt_str, "todate": to_dt_str}
    try:
        res = obj.getCandleData(params)
        if not res or 'data' not in res or not res['data']: return None
        rows = [{"timestamp": pd.to_datetime(c[0]), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in res['data']]
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Error fetching historical candles for {target_date.strftime('%Y-%m-%d')}: {e}")
        return None
        
def calculate_ema(df, period): return df['close'].ewm(span=period, adjust=False).mean()
def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    df['macd'] = short_ema - long_ema
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    return df

def detect_strategy_signals(df, params):
    signals = []
    if df is None or len(df) < 26: return signals
    df['ema'] = calculate_ema(df, period=params['ema_period'])
    df['rsi'] = calculate_rsi(df, period=params['rsi_period'])
    df = calculate_macd(df)
    for i in range(1, len(df)):
        current_candle, prev_candle = df.iloc[i], df.iloc[i - 1]
        if not (dt_time(9, 20) <= current_candle['timestamp'].time() < dt_time(15, 20)): continue
        is_ema_bullish_cross = prev_candle['close'] < prev_candle['ema'] and current_candle['close'] > current_candle['ema']
        is_macd_bullish_cross = prev_candle['macd'] < prev_candle['macd_signal'] and current_candle['macd'] > current_candle['macd_signal']
        is_rsi_bullish = current_candle['rsi'] > 50
        if is_ema_bullish_cross and is_macd_bullish_cross and is_rsi_bullish:
            signals.append({"signal": "CALL", "entry_price": float(current_candle['close']), "timestamp": current_candle['timestamp'], "entry_index": i})
        is_ema_bearish_cross = prev_candle['close'] > prev_candle['ema'] and current_candle['close'] < current_candle['ema']
        is_macd_bearish_cross = prev_candle['macd'] > prev_candle['macd_signal'] and current_candle['macd'] < current_candle['macd_signal']
        is_rsi_bearish = current_candle['rsi'] < 50
        if is_ema_bearish_cross and is_macd_bearish_cross and is_rsi_bearish:
            signals.append({"signal": "PUT", "entry_price": float(current_candle['close']), "timestamp": current_candle['timestamp'], "entry_index": i})
    return signals

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def get_option_ltp(obj, exchange, tradingsymbol, symboltoken):
    try:
        r = obj.ltpData(exchange=exchange, tradingsymbol=tradingsymbol, symboltoken=str(symboltoken))
        if r and r.get("data") and "ltp" in r["data"]: return float(r["data"]["ltp"])
    except Exception: return None

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def place_order(obj, symbol, token, qty, exchange, transaction_type, is_paper_trading=False):
    if is_paper_trading:
        logging.info(f"PAPER TRADING: Simulating {transaction_type} order for {symbol} of {qty} quantity.")
        return f"PAPER_{int(time.time())}"
    
    params = {"variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": token, "transactiontype": transaction_type, "exchange": exchange, "ordertype": "MARKET", "producttype": "INTRADAY", "duration": "DAY", "quantity": qty}
    try:
        orderId = obj.placeOrder(params)
        logging.info(f"LIVE order placed for {symbol}. ID: {orderId}")
        return orderId
    except Exception as e:
        logging.error(f"Failed to place LIVE order for {symbol}: {e}")
        st.error(f"Order Placement Failed for {symbol}: {e}")
        return None

# </editor-fold>

class TradingBot:
    def __init__(self):
          self.obj, self.running, self.thread, self.status, self.active_trade, self.params, self.instrument_list, self.last_checked = [None]*8
          self.status = "Idle"
          self.lock = Lock()
          self.daily_pnl = 0
          self.paper_pnl = 0
          self.paper_trades_log = []
          self.load_state()

    def login(self):
        self.status = "Logging in..."
        try:
            self.obj = SmartConnect(api_key=API_KEY)
            otp = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else None
            data = self.obj.generateSession(CLIENT_ID, CLIENT_PWD, otp)
            if data.get('status'):
                self.status = "Logged in."
                self.instrument_list = fetch_and_cache_instrument_list()
                return True
            self.status = f"Login Error: {data.get('message')}"
        except Exception as e: self.status = f"Login Exception: {e}"
        return False
        
    def save_state(self):
        conn = sqlite3.connect(STATE_DB)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)''')
        c.execute("REPLACE INTO state (key, value) VALUES (?, ?)", ("active_trade", json.dumps(self.active_trade)))
        c.execute("REPLACE INTO state (key, value) VALUES (?, ?)", ("daily_pnl", str(self.daily_pnl)))
        c.execute("REPLACE INTO state (key, value) VALUES (?, ?)", ("paper_pnl", str(self.paper_pnl)))
        c.execute("REPLACE INTO state (key, value) VALUES (?, ?)", ("paper_trades_log", json.dumps(self.paper_trades_log)))
        conn.commit()
        conn.close()

    def load_state(self):
        if not os.path.exists(STATE_DB): self.active_trade, self.daily_pnl, self.paper_pnl, self.paper_trades_log = None, 0, 0, []; return
        conn = sqlite3.connect(STATE_DB)
        c = conn.cursor()
        c.execute("SELECT value FROM state WHERE key = 'active_trade'"); row = c.fetchone(); self.active_trade = json.loads(row[0]) if row else None
        c.execute("SELECT value FROM state WHERE key = 'daily_pnl'"); row = c.fetchone(); self.daily_pnl = float(row[0]) if row else 0
        c.execute("SELECT value FROM state WHERE key = 'paper_pnl'"); row = c.fetchone(); self.paper_pnl = float(row[0]) if row else 0
        c.execute("SELECT value FROM state WHERE key = 'paper_trades_log'"); row = c.fetchone(); self.paper_trades_log = json.loads(row[0]) if row else []
        conn.close()

    def start(self, params):
        if not self.running:
            self.params, self.running = params, True
            self.thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
            self.thread.start()
            mode = "Paper Trading" if params.get('is_paper_trading') else "Live Trading"
            self.status = f"Bot started in {mode} mode."

    def stop(self): self.running, self.status = False, "Bot stopped by user."
    
    def monitor_active_trade(self):
        with self.lock:
            if not self.active_trade: return

            is_paper = self.active_trade.get("is_paper_trading", False)
            
            if datetime.now().time() >= dt_time(15, 20):
                self.status = "Exiting position (EOD)..."
                place_order(self.obj, self.active_trade['symbol'], self.active_trade['token'], self.active_trade['qty'], self.active_trade['exchange'], "SELL", is_paper_trading=is_paper)
                self.active_trade = None; self.save_state(); return

            ltp = get_option_ltp(self.obj, self.active_trade['exchange'], self.active_trade['symbol'], self.active_trade['token'])
            if not ltp: return
            
            # Trailing SL logic
            use_trailing = self.params.get('start_trailing_after_points', 0) > 0 and self.params.get('trailing_sl_gap_points', 0) > 0
            if use_trailing:
                self.active_trade['high_water_mark'] = max(self.active_trade.get('high_water_mark', self.active_trade['entry_price']), ltp)
                trailing_start_price = self.active_trade['entry_price'] + self.params['start_trailing_after_points']
                if self.active_trade['high_water_mark'] >= trailing_start_price:
                    potential_new_sl = self.active_trade['high_water_mark'] - self.params['trailing_sl_gap_points']
                    if potential_new_sl > self.active_trade['sl']:
                        self.active_trade['sl'] = potential_new_sl
                        self.status = f"SL Trailed to {potential_new_sl:.2f}"; self.save_state()
            
            exit_reason = None
            exit_on_gain_points = self.params.get('exit_on_points_gain', 0)
            if ltp <= self.active_trade['sl']: exit_reason = f"SL hit at {ltp:.2f}"
            elif exit_on_gain_points > 0 and ltp >= self.active_trade['entry_price'] + exit_on_gain_points: exit_reason = f"Exit on Gain at {ltp:.2f}"
            elif ltp >= self.active_trade['tp']: exit_reason = f"TP hit at {ltp:.2f}"

            if exit_reason:
                pnl = (ltp - self.active_trade['entry_price']) * self.active_trade['qty']
                
                if is_paper:
                    self.paper_pnl += pnl
                    log_entry = self.active_trade.copy()
                    log_entry.update({"exit_price": ltp, "pnl": pnl, "exit_reason": exit_reason, "exit_time": datetime.now().strftime('%H:%M:%S')})
                    self.paper_trades_log.append(log_entry)
                else: 
                    self.daily_pnl += pnl
                    if self.daily_pnl <= -self.params.get('max_daily_loss', 10000):
                        self.status = "Max daily loss reached. Stopping bot."; self.running = False

                self.status = f"Exiting {self.active_trade['symbol']}: {exit_reason}"
                place_order(self.obj, self.active_trade['symbol'], self.active_trade['token'], self.active_trade['qty'], self.active_trade['exchange'], "SELL", is_paper_trading=is_paper)
                self.active_trade = None
                self.save_state()
            
    def run_strategy_loop(self):
    if not self.obj and not self.login(): self.running = False; return
    if not self.instrument_list: self.status, self.running = "Error: Instrument list not loaded.", False; return
    
    indices_to_monitor = self.params.get('indices_to_monitor', [])
    is_paper = self.params.get('is_paper_trading', False)
    trade_direction = self.params.get('trade_direction', 'दोनों (CALL और PUT)')
    min_option_price = self.params.get('min_option_price', 20.0)

    while self.running:
        now = datetime.now()
        try:
            with self.lock:
                if self.active_trade:
                    self.monitor_active_trade()
                elif dt_time(9, 30) <= now.time() < dt_time(15, 20):
                    for index_name in indices_to_monitor:
                        self.status = f"Checking {index_name}..."
                        config = INDEX_CONFIG[index_name]
                        futures_token = find_index_futures_token(self.instrument_list, index_name, config['exchange'])
                        if not futures_token: continue
                        
                        df = try_fetch_candles(self.obj, futures_token, self.params['interval'], 1, config['exchange'])
                        if df is None: continue
                        
                        signals = detect_strategy_signals(df, self.params)

                        if trade_direction == 'केवल CALL':
                            signals = [s for s in signals if s['signal'] == 'CALL']
                        elif trade_direction == 'केवल PUT':
                            signals = [s for s in signals if s['signal'] == 'PUT']

                        if signals:
                            signal = signals[-1]
                            self.status = f"{signal['signal']} Signal in {index_name}! Processing..."
                            
                            spot = signal['entry_price']
                            atm = round(spot / config['strike_step']) * config['strike_step']
                            trade_type = "CE" if signal['signal'] == "CALL" else "PE"
                            token, symbol = find_option_token_from_list(self.instrument_list, index_name, atm, self.params['expiry'], trade_type)
                            entry_price = get_option_ltp(self.obj, config['exchange'], symbol, token)
                            
                            if entry_price and entry_price > min_option_price: 
                                qty = max(config['lot_size'], int((self.params['capital'] * self.params['risk_per_trade']) / self.params['sl_offset']) // config['lot_size'] * config['lot_size'])
                                
                                order_id = place_order(self.obj, symbol, token, qty, config['exchange'], "BUY", is_paper_trading=is_paper)
                                if order_id:
                                    self.active_trade = {
                                        "symbol": symbol, "token": token, "qty": qty, "exchange": config['exchange'], "index": index_name,
                                        "entry_price": entry_price, "sl": entry_price - self.params['sl_offset'],
                                        "tp": entry_price + self.params['tp_offset'], "high_water_mark": entry_price,
                                        "is_paper_trading": is_paper, "entry_time": datetime.now().strftime('%H:%M:%S')
                                    }
                                    self.save_state()
                                    self.status = f"Trade {'Simulated' if is_paper else 'Placed'} for {symbol}."
                                    break
                            else:
                                logging.warning(f"Trade skipped for {symbol}. Entry price {entry_price} is below minimum {min_option_price}.")
                    
                    if not self.active_trade: self.status = "Monitoring..."
        
        except Exception as e:
            self.status = f"Error in loop: {e}"
            logging.error(f"Critical error in strategy loop: {e}", exc_info=True)
        
        self.last_checked = now.strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(10)



# --- Streamlit UI ---
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
if 'bot' not in st.session_state: st.session_state.bot = TradingBot()
bot = st.session_state.bot

# Sidebar for Mode Settings and Tools
st.sidebar.title("⚙️ मोड और टूल्स")
is_paper_trading = st.sidebar.toggle("पेपर ट्रेडिंग मोड", value=True, help="चालू होने पर, कोई वास्तविक ऑर्डर नहीं दिया जाएगा।")
st.session_state.strategy_params = getattr(st.session_state, 'strategy_params', {})
st.session_state.strategy_params['is_paper_trading'] = is_paper_trading

# <<<--- FIXED: "Clear Instrument Cache" button is now in the sidebar ---<<<
st.sidebar.markdown("---")
if st.sidebar.button("🔄 इंस्ट्रूमेंट कैश साफ़ करें"):
    if os.path.exists(INSTRUMENT_CACHE_FILE):
        os.remove(INSTRUMENT_CACHE_FILE)
        st.sidebar.success("कैश साफ़ हो गया!")
    else:
        st.sidebar.info("साफ़ करने के लिए कोई कैश नहीं है।")
# <<<--- यहाँ नया बटन जोड़ा गया है ---<<<
if st.sidebar.button("❌ सक्रिय ट्रेड और PnL रीसेट करें"):
    if os.path.exists(STATE_DB):
        os.remove(STATE_DB)
        st.sidebar.success("स्टेट फ़ाइल डिलीट हो गई! बॉट को फिर से शुरू करें।")
        # Force the bot object to reload its state from the (now non-existent) file
        bot.load_state() 
        st.rerun()
    else:
        st.sidebar.info("कोई स्टेट फ़ाइल मौजूद नहीं है।")
        
if is_paper_trading:
    st.title("📈 ट्रेडिंग बॉट डैशबोर्ड (पेपर ट्रेडिंग)")
    st.info("आप वर्तमान में पेपर ट्रेडिंग मोड में हैं। कोई वास्तविक ट्रेड नहीं किया जाएगा।")
else:
    st.title("💰 ट्रेडिंग बॉट डैशबोर्ड (लाइव ट्रेडिंग)")
    st.warning("आप लाइव ट्रेडिंग मोड में हैं। वास्तविक धन का उपयोग किया जाएगा!")

st.header("1) लॉगिन और बॉट कंट्रोल")
c1, c2, c3 = st.columns([1.5, 1, 3])
with c1:
    if not bot.obj:
        if st.button("Angel One में लॉगिन करें"):
            if bot.login(): st.rerun()
            else: st.error(bot.status)
    else:
        st.success("सफलतापूर्वक लॉगिन हो गया।")
        if st.button("लॉगआउट करें"):
            bot.obj.terminateSession(CLIENT_ID); bot.obj = None; st.rerun()
with c2:
    if not bot.running:
        if st.button("🚀 बॉट शुरू करें"):
            if not st.session_state.strategy_params.get('expiry'):
                st.error("कृपया पहले एक एक्सपायरी चुनें।")
            else:
                trade_mode = st.session_state.strategy_params.get('trade_mode')
                if trade_mode == 'स्वचालित (सभी इंडेक्स)':
                    st.session_state.strategy_params['indices_to_monitor'] = ALL_INDICES
                else:
                    selected_index = trade_mode.split(" ")[1]
                    st.session_state.strategy_params['indices_to_monitor'] = [selected_index]
                bot.start(st.session_state.strategy_params); st.rerun()
    else:
        if st.button("🛑 बॉट रोकें"): bot.stop(); st.rerun()
with c3:
    st.markdown(f"**बॉट स्थिति:** <span style='color:{'green' if bot.running else 'red'};'>{bot.status}</span>", unsafe_allow_html=True)
    
    if bot.running and bot.last_checked:
        try: # <<<--- यह 'try:' लाइन शायद आपके कोड में मिसिंग है, इसे जोड़ें
            last_check_time = datetime.strptime(bot.last_checked, "%Y-%m-%d %H:%M:%S")
            time_diff = (datetime.now() - last_check_time).total_seconds()
            
            color = "green"
            if time_diff > 60:
                color = "red"
                st.warning("बॉट अटक गया है! कृपया रीसेट करें।")
            elif time_diff > 30:
                color = "orange"
                
            st.markdown(f"**आखिरी जाँच:** <span style='color:{color};'>{bot.last_checked} ({int(time_diff)}s ago)</span>", unsafe_allow_html=True)
        except Exception as e:
            logging.error(f"Error rendering heartbeat: {e}")


    # <<<--- नया कोड ब्लॉक यहाँ खत्म होता है ---<<<
pnl_c1, pnl_c2 = st.columns(2)
pnl_c1.metric("आज का लाइव PnL", f"₹ {bot.daily_pnl:,.2f}")
pnl_c2.metric("आज का पेपर PnL", f"₹ {bot.paper_pnl:,.2f}")

st.markdown("---")
if bot.active_trade:
    mode = "पेपर ट्रेड" if bot.active_trade.get('is_paper_trading') else "लाइव ट्रेड"
    st.success(f"सक्रिय {mode}:")
    st.json(bot.active_trade)

st.header("2) स्ट्रेटेजी और ट्रेड पैरामीटर")
st.session_state.strategy_params['trade_direction'] = st.radio(
    "ट्रेड की दिशा चुनें:",
    ['दोनों (CALL और PUT)', 'केवल CALL', 'केवल PUT'],
    horizontal=True, key='trade_direction'
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("मुख्य सेटिंग्स")
    st.session_state.strategy_params['trade_mode'] = st.radio( "ट्रेडिंग इंडेक्स चुनें", [f'केवल {idx}' for idx in ALL_INDICES] + ['स्वचालित (सभी इंडेक्स)'])
    st.session_state.strategy_params['interval'] = st.selectbox("कैंडल इंटरवल", ["5min", "1min", "15min"])
    st.session_state.strategy_params['capital'] = st.number_input("कैपिटल (₹)", value=100000)
    st.session_state.strategy_params['risk_per_trade'] = st.number_input("प्रति ट्रेड रिस्क (%)", 0.01, 10.0, value=1.0, step=0.1) / 100
with c2:
    st.subheader("तकनीकी संकेतक")
    st.session_state.strategy_params['ema_period'] = st.number_input("EMA पीरियड", value=25)
    st.session_state.strategy_params['rsi_period'] = st.number_input("RSI पीरियड", value=14)
    st.info("MACD पैरामीटर (12, 26, 9) मानक हैं।")
with c3:
    st.subheader("SL, TP और ट्रेलिंग")
    # <<<--- FIXED: All trailing options are now back ---<<<
    st.session_state.strategy_params['sl_offset'] = st.number_input("प्रारंभिक SL ऑफ़सेट (₹)", 1.0, value=20.0, step=0.5)
    st.session_state.strategy_params['tp_offset'] = st.number_input("TP ऑफ़सेट (₹)", 1.0, value=20.0, step=0.5)
    st.session_state.strategy_params['start_trailing_after_points'] = st.number_input("इसके बाद ट्रेलिंग शुरू करें (पॉइंट्स)", 0.0, value=0.0, step=1.0)
    st.session_state.strategy_params['trailing_sl_gap_points'] = st.number_input("ट्रेलिंग SL गैप (पॉइंट्स)", 0.0, value=20.0, step=1.0)
    st.session_state.strategy_params['exit_on_points_gain'] = st.number_input("इतने पॉइंट्स लाभ पर बाहर निकलें", 0.0, value=0.0, step=1.0)
    st.session_state.strategy_params['max_daily_loss'] = st.number_input("मैक्स डेली लॉस (₹) (Live Only)", value=10000)
    
st.header("3) एक्सपायरी चुनें")
exp_c1, exp_c2 = st.columns([1,2])
with exp_c1:
    expiry_index_choice = st.selectbox("एक्सपायरी के लिए इंडेक्स", ALL_INDICES)
if exp_c2.button(f"📅 {expiry_index_choice} की एक्सपायरी प्राप्त करें"):
    if bot.obj:
        with st.spinner("..."):
            instrument_list = fetch_and_cache_instrument_list()
            if instrument_list:
                cfg = INDEX_CONFIG[expiry_index_choice]
                expiries = {datetime.strptime(i["expiry"], '%d%b%Y').date() for i in instrument_list if i.get("name") == cfg["instrument_name"] and i.get("exch_seg") == cfg["exchange"] and i.get("instrumenttype") == "OPTIDX" and i.get("expiry")}
                st.session_state["expiries"] = [d.strftime("%Y-%m-%d") for d in sorted(list(expiries)) if d >= datetime.now().date()]
    else: st.warning("पहले लॉगिन करें।")
st.session_state.strategy_params['expiry'] = st.selectbox("एक्सपायरी चुनें", st.session_state.get("expiries", []))

if is_paper_trading and bot.paper_trades_log:
    st.header("आज के पेपर ट्रेड्स का लॉग")
    st.dataframe(pd.DataFrame(bot.paper_trades_log).iloc[::-1])

st.markdown("---")
st.header("4) स्ट्रेटेजी का बैकटेस्ट करें")
st.warning("बैकटेस्टिंग एक समय में केवल एक ही इंडेक्स पर काम करेगा। कृपया नीचे से इंडेक्स चुनें।")

backtest_index = st.selectbox("बैकटेस्ट के लिए इंडेक्स चुनें", ALL_INDICES)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: start_date = st.date_input("प्रारंभ तिथि", datetime.now().date() - timedelta(days=30))
with c2: end_date = st.date_input("अंतिम तिथि", datetime.now().date() - timedelta(days=1))
with c3: hypo_opt_price = st.number_input("काल्पनिक ऑप्शन एंट्री (₹)", 1.0, value=120.0, step=0.5)
with c4: trade_cost = st.number_input("प्रति ट्रेड लागत (₹)", 0.0, value=50.0, step=1.0)
with c5: trade_type_backtest = st.selectbox("बैकटेस्ट ट्रेड प्रकार", ["CALL", "PUT"])
with c6: slippage = st.number_input("स्लिपेज (₹)", 0.0, value=0.5, step=0.1)

if st.button("▶ अभी बैकटेस्ट चलाएं"):
    if not bot.obj: st.error("कृपया पहले लॉगिन करें।")
    elif start_date > end_date: st.error("प्रारंभ तिथि अंतिम तिथि से पहले होनी चाहिए।")
    else:
        with st.spinner(f"{backtest_index} पर बैकटेस्ट चल रहा है..."):
            config = INDEX_CONFIG[backtest_index]
            params = st.session_state.strategy_params
            
            sl_points = params['sl_offset']
            capital = params['capital']
            risk_per_trade = params['risk_per_trade']
            contracts_qty = int((capital * risk_per_trade) / sl_points) if sl_points > 0 else config['lot_size']
            
            all_trades = []
            instrument_list = fetch_and_cache_instrument_list()
            futures_token = find_index_futures_token(instrument_list, backtest_index, config['exchange']) if instrument_list else None
            if not futures_token: st.error(f"{backtest_index} के लिए फ्यूचर्स टोकन नहीं मिला।"); st.stop()
            
            date_range = pd.date_range(start_date, end_date)
            progress_bar = st.progress(0, text="बैकटेस्ट प्रगति पर है...")
            for i, single_date in enumerate(date_range):
                if single_date.weekday() < 5:
                    df_hist = try_fetch_historical_candles(bot.obj, futures_token, params['interval'], single_date, config["exchange"])
                    if df_hist is not None and not df_hist.empty:
                        signals = [s for s in detect_strategy_signals(df_hist, params) if s['signal'] == trade_type_backtest]
                        if signals:
                            sig = signals[0]
                            hypo_entry = hypo_opt_price + slippage
                            current_sl = hypo_entry - params['sl_offset']
                            target_price = hypo_entry + params['tp_offset']
                            status, exit_price = "UNKNOWN", hypo_entry
                            exit_timestamp = None
                            for j in range(sig['entry_index'] + 1, len(df_hist)):
                                c = df_hist.iloc[j]
                                est_high, est_low = hypo_entry + (c['high'] - sig['entry_price']), hypo_entry + (c['low'] - sig['entry_price'])
                                if est_high >= target_price: status, exit_price, exit_timestamp = "TP_HIT", target_price - slippage, c['timestamp']; break
                                if est_low <= current_sl: status, exit_price, exit_timestamp = "SL_HIT", current_sl, c['timestamp']; break
                                if c['timestamp'].time() >= dt_time(15, 20): status, exit_price, exit_timestamp = "EOD_EXIT", hypo_entry + (c['close'] - sig['entry_price']), c['timestamp']; break
                            if status == "UNKNOWN": status, exit_price, exit_timestamp = "EOD_NO_EXIT", hypo_entry + (df_hist.iloc[-1]['close'] - sig['entry_price']), df_hist.iloc[-1]['timestamp']
                            pnl = ((exit_price - hypo_entry) * contracts_qty) - trade_cost
                            all_trades.append({"date": sig['timestamp'].date(), "entry_time": sig['timestamp'].strftime('%H:%M:%S'),"exit_time": exit_timestamp.strftime('%H:%M:%S') if exit_timestamp else 'N/A',"trade_type": sig['signal'], "index_entry": f"{sig['entry_price']:.2f}", "option_entry": f"{hypo_entry:.2f}", "option_exit": f"{exit_price:.2f}", "status": status, "net_pnl": pnl})
                progress_bar.progress((i + 1) / len(date_range), text=f"प्रोसेस हो रहा है: {single_date.strftime('%Y-%m-%d')}")
            
            if not all_trades: st.info(f"{backtest_index} में चयनित तिथि सीमा में कोई ट्रेड नहीं मिला।")
            else:
                trades_df = pd.DataFrame(all_trades)
                st.success(f"बैकटेस्ट पूरा हुआ! {len(all_trades)} ट्रेड मिले।")
                st.dataframe(trades_df)
                total_pnl = trades_df['net_pnl'].sum()
                wins = trades_df[trades_df['net_pnl'] > 0]
                st.markdown("### बैकटेस्ट सारांश")
                k1, k2, k3 = st.columns(3)
                k1.metric("कुल नेट PnL", f"₹ {total_pnl:,.2f}")
                k2.metric("कुल ट्रेड्स", len(trades_df)); k2.metric("लाभ वाले ट्रेड्स", len(wins))
                k3.metric("जीत दर (Win Rate)", f"{(len(wins)/len(trades_df))*100:.2f} %" if len(trades_df) > 0 else "0.00 %"); k3.metric("नुकसान वाले ट्रेड्स", len(trades_df) - len(wins))
                st.subheader("इक्विटी कर्व")
                trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()
                fig, ax = plt.subplots()
                ax.plot(pd.to_datetime(trades_df['date']), trades_df['cum_pnl'])
                ax.set_xlabel('Date'); ax.set_ylabel('Total PnL'); fig.autofmt_xdate()
                st.pyplot(fig)

                












