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
from threading import Lock
import pytz

# --- Timezone Configuration ---
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current Indian Standard Time"""
    return datetime.now(IST)

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(file_handler)

# --- Streamlit Cache Configuration ---
@st.cache_resource(show_spinner=False, ttl=300)
def get_bot_instance():
    """Get cached bot instance"""
    return TradingBot()

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
                with gzip.open(INSTRUMENT_CACHE_FILE, 'rt', encoding='utf-8') as f: 
                    return json.load(f)
        logging.info("Downloading fresh instrument list.")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        instrument_data = response.json()
        with gzip.open(INSTRUMENT_CACHE_FILE, 'wt', encoding='utf-8') as f: 
            json.dump(instrument_data, f)
        return instrument_data
    except Exception as e:
        st.error(f"Fatal Error: Could not download instrument list. Details: {e}")
        return None

def find_index_futures_token(instrument_list, index_name, exchange):
    if not instrument_list: 
        return None
    config = INDEX_CONFIG[index_name]
    instrument_name = config.get("instrument_name", index_name)
    futures = []
    today = datetime.now().date()
    
    for item in instrument_list:
        if (item.get("instrumenttype") == "FUTIDX" and 
            item.get("name") == instrument_name and 
            item.get("exch_seg") == exchange):
            
            # Block NIFTYNXT50 completely
            if "NIFTYNXT50" in item.get("symbol", ""):
                continue
            if "NEXT" in item.get("symbol", "").upper():
                continue
                
            try:
                expiry_date = datetime.strptime(item["expiry"], '%d%b%Y').date()
                if expiry_date >= today:
                    futures.append((expiry_date, item))
            except (ValueError, KeyError): 
                continue
                
    if not futures: 
        return None
    futures.sort(key=lambda x: x[0])
    return futures[0][1].get("token")

def find_option_token_from_list(instrument_list, index_name, strike, expiry, option_cepe):
    if not instrument_list: 
        raise RuntimeError("Instrument list not loaded.")
    config = INDEX_CONFIG[index_name]
    instrument_name = config.get("instrument_name", index_name)
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d%b%Y").upper()
    
    for item in instrument_list:
        if (item.get("name") == instrument_name and 
            item.get("exch_seg") == config['exchange'] and
            item.get("instrumenttype") == "OPTIDX" and 
            item.get("strike") and
            float(item.get("strike")) / 100 == float(strike) and 
            item.get("expiry") == expiry_fmt and
            item.get("symbol", "").endswith(option_cepe)):
            return item.get("token"), item.get("symbol")
    raise RuntimeError(f"Could not find {option_cepe} for strike {strike} and expiry {expiry} in instrument list.")

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def try_fetch_candles(obj, symbol_token, interval, days_back, exchange):
    to_dt, from_dt = get_ist_time(), get_ist_time() - timedelta(days=days_back)
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    params = {
        "exchange": exchange, 
        "symboltoken": str(symbol_token), 
        "interval": imap.get(interval),
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"), 
        "todate": to_dt.strftime("%Y-%m-%d %H:%M")
    }
    try:
        res = obj.getCandleData(params)
        if not res or "data" not in res or not res["data"]: 
            return None
        rows = [{"timestamp": pd.to_datetime(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in res["data"]]
        df = pd.DataFrame(rows)
        
        # Timezone fix
        if df is not None and not df.empty:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
        return df
    except Exception as e:
        logging.error(f"Error fetching candles for {symbol_token}: {e}")
        return None

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def try_fetch_historical_candles(obj, symbol_token, interval, target_date, exchange):
    from_dt_str = target_date.strftime("%Y-%m-%d") + " 09:15"
    to_dt_str = target_date.strftime("%Y-%m-%d") + " 15:30"
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    params = {
        "exchange": exchange, 
        "symboltoken": str(symbol_token), 
        "interval": imap.get(interval), 
        "fromdate": from_dt_str, 
        "todate": to_dt_str
    }
    try:
        res = obj.getCandleData(params)
        if not res or 'data' not in res or not res['data']: 
            return None
        rows = [{"timestamp": pd.to_datetime(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in res['data']]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        
        # Timezone fix
        if df is not None and not df.empty:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        return df
    except Exception as e:
        logging.warning(f"Error fetching historical candles for {target_date.strftime('%Y-%m-%d')}: {e}")
        return None
        
def calculate_ema(df, period): 
    return df['close'].ewm(span=period, adjust=False).mean()

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

def detect_strategy_signals(df, params, is_backtest=False):
    """
    Detect trading signals
    is_backtest: True if for backtesting
    """
    signals = []
    if df is None or len(df) < 26: 
        return signals
        
    df['ema'] = calculate_ema(df, period=params['ema_period'])
    df['rsi'] = calculate_rsi(df, period=params['rsi_period'])
    df = calculate_macd(df)
    
    # Different logic for backtesting
    if is_backtest:
        for i in range(1, len(df)):
            current_candle, prev_candle = df.iloc[i], df.iloc[i - 1]
            
            # Market time check (in backtest)
            if not (dt_time(9, 20) <= current_candle['timestamp'].time() < dt_time(15, 20)): 
                continue
            
            # Signal logic
            is_ema_bullish_cross = prev_candle['close'] < prev_candle['ema'] and current_candle['close'] > current_candle['ema']
            is_macd_bullish_cross = prev_candle['macd'] < prev_candle['macd_signal'] and current_candle['macd'] > current_candle['macd_signal']
            is_rsi_bullish = current_candle['rsi'] > 50
            
            if is_ema_bullish_cross and is_macd_bullish_cross and is_rsi_bullish:
                signals.append({
                    "signal": "CALL", 
                    "entry_price": float(current_candle['close']), 
                    "timestamp": current_candle['timestamp'], 
                    "entry_index": i
                })
            
            is_ema_bearish_cross = prev_candle['close'] > prev_candle['ema'] and current_candle['close'] < current_candle['ema']
            is_macd_bearish_cross = prev_candle['macd'] > prev_candle['macd_signal'] and current_candle['macd'] < current_candle['macd_signal']
            is_rsi_bearish = current_candle['rsi'] < 50
            
            if is_ema_bearish_cross and is_macd_bearish_cross and is_rsi_bearish:
                signals.append({
                    "signal": "PUT", 
                    "entry_price": float(current_candle['close']), 
                    "timestamp": current_candle['timestamp'], 
                    "entry_index": i
                })
    else:
        # For live trading - only last candle
        if len(df) > 0:
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2] if len(df) > 1 else current_candle
            
            # Timezone fix
            current_time = get_ist_time().replace(tzinfo=None)
            candle_time = current_candle['timestamp']
            
            time_diff = (current_time - candle_time).total_seconds()
            
            # Ignore candle if older than 2 minutes
            if time_diff > 120:
                return signals
                
            # Market time check (9:20 AM - 3:20 PM)
            if not (dt_time(9, 20) <= current_candle['timestamp'].time() < dt_time(15, 20)): 
                return signals
            
            # Signal logic (only for last candle)
            is_ema_bullish_cross = prev_candle['close'] < prev_candle['ema'] and current_candle['close'] > current_candle['ema']
            is_macd_bullish_cross = prev_candle['macd'] < prev_candle['macd_signal'] and current_candle['macd'] > current_candle['macd_signal']
            is_rsi_bullish = current_candle['rsi'] > 50
            
            if is_ema_bullish_cross and is_macd_bullish_cross and is_rsi_bullish:
                signals.append({
                    "signal": "CALL", 
                    "entry_price": float(current_candle['close']), 
                    "timestamp": current_candle['timestamp'], 
                    "entry_index": len(df)-1
                })
            
            is_ema_bearish_cross = prev_candle['close'] > prev_candle['ema'] and current_candle['close'] < current_candle['ema']
            is_macd_bearish_cross = prev_candle['macd'] > prev_candle['macd_signal'] and current_candle['macd'] < current_candle['macd_signal']
            is_rsi_bearish = current_candle['rsi'] < 50
            
            if is_ema_bearish_cross and is_macd_bearish_cross and is_rsi_bearish:
                signals.append({
                    "signal": "PUT", 
                    "entry_price": float(current_candle['close']), 
                    "timestamp": current_candle['timestamp'], 
                    "entry_index": len(df)-1
                })
    
    return signals

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def get_option_ltp(obj, exchange, tradingsymbol, symboltoken):
    try:
        r = obj.ltpData(exchange=exchange, tradingsymbol=tradingsymbol, symboltoken=str(symboltoken))
        if r and r.get("data") and "ltp" in r["data"]: 
            return float(r["data"]["ltp"])
    except Exception: 
        return None

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def place_order(obj, symbol, token, qty, exchange, transaction_type, is_paper_trading=False):
    if is_paper_trading:
        logging.info(f"PAPER TRADING: Simulating {transaction_type} order for {symbol} of {qty} quantity.")
        return f"PAPER_{int(time.time())}"
    
    params = {
        "variety": "NORMAL", 
        "tradingsymbol": symbol, 
        "symboltoken": token, 
        "transactiontype": transaction_type, 
        "exchange": exchange, 
        "ordertype": "MARKET", 
        "producttype": "INTRADAY", 
        "duration": "DAY", 
        "quantity": qty
    }
    try:
        orderId = obj.placeOrder(params)
        logging.info(f"LIVE order placed for {symbol}. ID: {orderId}")
        return orderId
    except Exception as e:
        logging.error(f"Failed to place LIVE order for {symbol}: {e}")
        st.error(f"Order Placement Failed for {symbol}: {e}")
        return None

def get_expiry_list(bot, index_name):
    """Get expiry list correctly"""
    if not bot.obj:
        return []
        
    instrument_list = fetch_and_cache_instrument_list()
    if not instrument_list:
        return []
        
    cfg = INDEX_CONFIG[index_name]
    instrument_name = cfg["instrument_name"]
    
    expiries = set()
    for item in instrument_list:
        if (item.get("name") == instrument_name and 
            item.get("exch_seg") == cfg["exchange"] and 
            item.get("instrumenttype") == "OPTIDX" and 
            item.get("expiry")):
            
            # Avoid NIFTYNXT50
            if "NIFTYNXT50" in item.get("symbol", ""):
                continue
            if "NEXT" in item.get("symbol", "").upper():
                continue
                
            try:
                expiry_date = datetime.strptime(item["expiry"], '%d%b%Y').date()
                if expiry_date >= datetime.now().date():
                    expiries.add(expiry_date)
            except ValueError:
                continue
                
    return [d.strftime("%Y-%m-%d") for d in sorted(list(expiries))]

def run_backtest_with_trailing_sl(bot, backtest_index, start_date, end_date, params, 
                                 hypo_opt_price, trade_cost, trade_type_backtest, slippage):
    """Run backtest with Trailing SL"""
    config = INDEX_CONFIG[backtest_index]
    
    sl_points = params['sl_offset']
    capital = params['capital']
    risk_per_trade = params['risk_per_trade']
    contracts_qty = int((capital * risk_per_trade) / sl_points) if sl_points > 0 else config['lot_size']
    
    all_trades = []
    instrument_list = fetch_and_cache_instrument_list()
    futures_token = find_index_futures_token(instrument_list, backtest_index, config['exchange']) 
    
    if not futures_token: 
        return None
    
    date_range = pd.date_range(start_date, end_date)
    
    for single_date in date_range:
        if single_date.weekday() < 5:
            df_hist = try_fetch_historical_candles(bot.obj, futures_token, params['interval'], single_date, config["exchange"])
            if df_hist is not None and not df_hist.empty:
                signals = [s for s in detect_strategy_signals(df_hist, params, is_backtest=True) 
                          if s['signal'] == trade_type_backtest]
                
                if signals:
                    sig = signals[0]
                    hypo_entry = hypo_opt_price + slippage
                    current_sl = hypo_entry - params['sl_offset']
                    target_price = hypo_entry + params['tp_offset']
                    
                    # Trailing SL parameters
                    start_trailing_after = params.get('start_trailing_after_points', 0)
                    trailing_gap = params.get('trailing_sl_gap_points', 0)
                    use_trailing = start_trailing_after > 0 and trailing_gap > 0
                    
                    high_water_mark = hypo_entry  # Initial high water mark
                    status, exit_price = "UNKNOWN", hypo_entry
                    exit_timestamp = None
                    
                    for j in range(sig['entry_index'] + 1, len(df_hist)):
                        c = df_hist.iloc[j]
                        est_price = hypo_entry + (c['close'] - sig['entry_price'])
                        est_high = hypo_entry + (c['high'] - sig['entry_price'])
                        est_low = hypo_entry + (c['low'] - sig['entry_price'])
                        
                        # Trailing SL Logic for Backtesting
                        if use_trailing:
                            high_water_mark = max(high_water_mark, est_price)
                            
                            if high_water_mark >= hypo_entry + start_trailing_after:
                                new_sl = high_water_mark - trailing_gap
                                if new_sl > current_sl:
                                    current_sl = new_sl
                        
                        # Exit Conditions
                        if est_low <= current_sl: 
                            status, exit_price, exit_timestamp = "SL_HIT", current_sl, c['timestamp']
                            break
                        elif est_high >= target_price: 
                            status, exit_price, exit_timestamp = "TP_HIT", target_price - slippage, c['timestamp']
                            break
                        elif c['timestamp'].time() >= dt_time(15, 20): 
                            status, exit_price, exit_timestamp = "EOD_EXIT", est_price, c['timestamp']
                            break
                    
                    if status == "UNKNOWN": 
                        status, exit_price, exit_timestamp = "EOD_NO_EXIT", hypo_entry + (df_hist.iloc[-1]['close'] - sig['entry_price']), df_hist.iloc[-1]['timestamp']
                    
                    pnl = ((exit_price - hypo_entry) * contracts_qty) - trade_cost
                    
                    all_trades.append({
                        "date": sig['timestamp'].date(), 
                        "entry_time": sig['timestamp'].strftime('%H:%M:%S'),
                        "exit_time": exit_timestamp.strftime('%H:%M:%S') if exit_timestamp else 'N/A',
                        "trade_type": sig['signal'], 
                        "index_entry": f"{sig['entry_price']:.2f}", 
                        "option_entry": f"{hypo_entry:.2f}", 
                        "option_exit": f"{exit_price:.2f}", 
                        "sl_used": f"{current_sl:.2f}",
                        "high_water_mark": f"{high_water_mark:.2f}",
                        "status": status, 
                        "net_pnl": pnl
                    })
    
    return all_trades

# </editor-fold>

class TradingBot:
    def __init__(self):
        self.obj = None
        self.running = False
        self.thread = None
        self.status = "Idle"
        self.active_trade = None
        self.params = {}
        self.instrument_list = None
        self.last_checked = None
        self.lock = Lock()
        self.daily_pnl = 0
        self.paper_pnl = 0
        self.paper_trades_log = []
        
        # New heartbeat system with timezone
        self.last_heartbeat = get_ist_time()
        self.heartbeat_interval = 10  # seconds
        self.freeze_threshold = 60    # seconds
        
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
                self.update_heartbeat()
                return True
            self.status = f"Login Error: {data.get('message')}"
        except Exception as e: 
            self.status = f"Login Exception: {e}"
        return False
        
    def update_heartbeat(self):
        """Update heartbeat after every successful loop"""
        self.last_heartbeat = get_ist_time()
        
    def is_bot_frozen(self):
        """Check if bot is frozen"""
        time_since_last_heartbeat = (get_ist_time() - self.last_heartbeat).total_seconds()
        return time_since_last_heartbeat > self.freeze_threshold
        
    def get_bot_status_color(self):
        """Return color based on bot status"""
        if not self.running:
            return "gray"  # Stopped
        
        if self.is_bot_frozen():
            return "red"   # Frozen
        
        time_since_heartbeat = (get_ist_time() - self.last_heartbeat).total_seconds()
        if time_since_heartbeat > 30:
            return "orange"  # Slow/Stuck
        
        return "green"     # Running normally
        
    def get_last_checked_display(self):
        """Return formatted time for display"""
        if self.last_checked:
            try:
                # Timezone conversion
                if isinstance(self.last_checked, str):
                    last_check_naive = datetime.strptime(self.last_checked, "%Y-%m-%d %H:%M:%S")
                    last_check_ist = IST.localize(last_check_naive)
                else:
                    last_check_ist = self.last_checked.astimezone(IST) if self.last_checked.tzinfo else IST.localize(self.last_checked)
                
                current_time_ist = get_ist_time()
                time_diff = (current_time_ist - last_check_ist).total_seconds()
                
                return {
                    'formatted_time': last_check_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                    'seconds_ago': int(time_diff),
                    'is_stale': time_diff > 300  # Older than 5 minutes
                }
            except Exception as e:
                logging.error(f"Time conversion error: {e}")
                return {'formatted_time': 'Error', 'seconds_ago': 999, 'is_stale': True}
        return {'formatted_time': 'Never', 'seconds_ago': 999, 'is_stale': True}

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
        if not os.path.exists(STATE_DB): 
            self.active_trade = None
            self.daily_pnl = 0
            self.paper_pnl = 0
            self.paper_trades_log = []
            return
        conn = sqlite3.connect(STATE_DB)
        c = conn.cursor()
        c.execute("SELECT value FROM state WHERE key = 'active_trade'")
        row = c.fetchone()
        self.active_trade = json.loads(row[0]) if row else None
        c.execute("SELECT value FROM state WHERE key = 'daily_pnl'")
        row = c.fetchone()
        self.daily_pnl = float(row[0]) if row else 0
        c.execute("SELECT value FROM state WHERE key = 'paper_pnl'")
        row = c.fetchone()
        self.paper_pnl = float(row[0]) if row else 0
        c.execute("SELECT value FROM state WHERE key = 'paper_trades_log'")
        row = c.fetchone()
        self.paper_trades_log = json.loads(row[0]) if row else []
        conn.close()

    def start(self, params):
        if not self.running:
            self.params = params
            self.running = True
            self.thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
            self.thread.start()
            mode = "Paper Trading" if params.get('is_paper_trading') else "Live Trading"
            self.status = f"Bot started in {mode} mode."
            self.update_heartbeat()

    def stop(self): 
        self.running = False
        self.status = "Bot stopped by user."
    
    def check_exit_conditions(self, ltp):
        """Check exit conditions"""
        if ltp <= self.active_trade['sl']: 
            return f"SL hit at {ltp:.2f}"
        elif ltp >= self.active_trade['tp']: 
            return f"TP hit at {ltp:.2f}"
        
        # Trailing SL
        if self.params.get('start_trailing_after_points', 0) > 0:
            self.active_trade['high_water_mark'] = max(
                self.active_trade.get('high_water_mark', self.active_trade['entry_price']), 
                ltp
            )
            trailing_start = self.active_trade['entry_price'] + self.params['start_trailing_after_points']
            if self.active_trade['high_water_mark'] >= trailing_start:
                new_sl = self.active_trade['high_water_mark'] - self.params['trailing_sl_gap_points']
                if new_sl > self.active_trade['sl']:
                    self.active_trade['sl'] = new_sl
                    self.status = f"SL Trailed to {new_sl:.2f}"
                    self.save_state()
        
        # Exit on points gain
        exit_on_gain = self.params.get('exit_on_points_gain', 0)
        if exit_on_gain > 0 and ltp >= self.active_trade['entry_price'] + exit_on_gain:
            return f"Exit on Gain at {ltp:.2f}"
        
        return None

    def exit_trade(self, reason, ltp, is_paper):
        """Exit trade"""
        if ltp is None:
            ltp = self.active_trade['entry_price']
        
        pnl = (ltp - self.active_trade['entry_price']) * self.active_trade['qty']
        
        if is_paper:
            self.paper_pnl += pnl
            log_entry = self.active_trade.copy()
            log_entry.update({
                "exit_price": ltp, 
                "pnl": pnl, 
                "exit_reason": reason, 
                "exit_time": get_ist_time().strftime('%H:%M:%S IST')
            })
            self.paper_trades_log.append(log_entry)
        else: 
            self.daily_pnl += pnl
            if self.daily_pnl <= -self.params.get('max_daily_loss', 10000):
                self.status = "Max daily loss reached. Stopping bot."
                self.running = False

        self.status = f"Exiting {self.active_trade['symbol']}: {reason}"
        place_order(self.obj, self.active_trade['symbol'], self.active_trade['token'], 
                    self.active_trade['qty'], self.active_trade['exchange'], "SELL", 
                    is_paper_trading=is_paper)
        self.active_trade = None
        self.save_state()

    def monitor_active_trade(self):
        with self.lock:
            if not self.active_trade: 
                return

            is_paper = self.active_trade.get("is_paper_trading", False)
            
            # First check market time
            if get_ist_time().time() >= dt_time(15, 20):
                self.exit_trade("EOD", None, is_paper)
                return

            try:
                # Get LTP (with retry)
                ltp = None
                for attempt in range(3):
                    ltp = get_option_ltp(self.obj, self.active_trade['exchange'], 
                                       self.active_trade['symbol'], self.active_trade['token'])
                    if ltp is not None:
                        break
                    time.sleep(1)
                
                if ltp is None:
                    self.status = "LTP not available"
                    return
                    
                # SL/TP logic
                exit_reason = self.check_exit_conditions(ltp)
                if exit_reason:
                    self.exit_trade(exit_reason, ltp, is_paper)
                    
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
            
    def run_strategy_loop(self):
        if not self.obj and not self.login(): 
            self.running = False
            return
        if not self.instrument_list: 
            self.status = "Error: Instrument list not loaded."
            self.running = False
            return
        
        indices_to_monitor = self.params.get('indices_to_monitor', [])
        is_paper = self.params.get('is_paper_trading', False)
        trade_direction = self.params.get('trade_direction', 'Both (CALL & PUT)')
        min_option_price = self.params.get('min_option_price', 20.0)

        while self.running:
            now = get_ist_time()
            try:
                # Heartbeat update
                self.update_heartbeat()
                
                with self.lock:
                    if self.active_trade:
                        self.monitor_active_trade()
                    elif dt_time(9, 30) <= now.time() < dt_time(15, 20):
                        for index_name in indices_to_monitor:
                            self.status = f"Checking {index_name}..."
                            config = INDEX_CONFIG[index_name]
                            futures_token = find_index_futures_token(self.instrument_list, index_name, config['exchange'])
                            if not futures_token: 
                                continue
                            
                            df = try_fetch_candles(self.obj, futures_token, self.params['interval'], 1, config['exchange'])
                            if df is None or len(df) == 0: 
                                continue
                            
                            # Check current candle freshness
                            current_candle_time = df.iloc[-1]['timestamp']
                            time_diff = (now - current_candle_time).total_seconds()
                            if time_diff > 120:  # Candle older than 2 minutes
                                continue
                            
                            # Signals for live trading (is_backtest=False)
                            signals = detect_strategy_signals(df, self.params, is_backtest=False)

                            if trade_direction == 'CALL Only':
                                signals = [s for s in signals if s['signal'] == 'CALL']
                            elif trade_direction == 'PUT Only':
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
                                            "symbol": symbol, 
                                            "token": token, 
                                            "qty": qty, 
                                            "exchange": config['exchange'], 
                                            "index": index_name,
                                            "entry_price": entry_price, 
                                            "sl": entry_price - self.params['sl_offset'],
                                            "tp": entry_price + self.params['tp_offset'], 
                                            "high_water_mark": entry_price,
                                            "is_paper_trading": is_paper, 
                                            "entry_time": now.strftime('%H:%M:%S IST')
                                        }
                                        self.save_state()
                                        self.status = f"Trade {'Simulated' if is_paper else 'Placed'} for {symbol}."
                                        break
                                else:
                                    logging.warning(f"Trade skipped for {symbol}. Entry price {entry_price} is below minimum {min_option_price}.")
                        
                        if not self.active_trade: 
                            self.status = "Monitoring..."
            
            except Exception as e:
                self.status = f"Error in loop: {e}"
                logging.error(f"Critical error in strategy loop: {e}", exc_info=True)
                self.update_heartbeat()
            
            self.last_checked = now
            time.sleep(10)

def display_bot_health(bot):
    """Display bot health status"""
    st.subheader("🤖 Bot Health Monitor")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = bot.get_bot_status_color()
        status_text = ""
        if status_color == "green":
            status_text = "🟢 Normal"
        elif status_color == "orange":
            status_text = "🟠 Slow"
        elif status_color == "red":
            status_text = "🔴 Frozen"
        else:
            status_text = "⚫ Stopped"
            
        st.markdown(f"**Status:** <span style='color:{status_color}; font-size:20px;'>{status_text}</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        last_check_info = bot.get_last_checked_display()
        st.metric("Last Check", f"{last_check_info['seconds_ago']} seconds ago")
        
        if last_check_info['is_stale']:
            st.error("⚠️ Data is stale")
            if st.button("🔄 Refresh Data", key="refresh_data"):
                st.rerun()
                
    with col3:
        if bot.running:
            time_since_heartbeat = (get_ist_time() - bot.last_heartbeat).total_seconds()
            st.metric("Heartbeat", f"{int(time_since_heartbeat)}s")
            
            if time_since_heartbeat > bot.freeze_threshold:
                st.error("❌ Freeze Alert!")
            elif time_since_heartbeat > 30:
                st.warning("⚠️ Running Slow")
            else:
                st.success("✅ Normal")
    
    with col4:
        if bot.is_bot_frozen() and bot.running:
            st.error("🔴 Bot is Frozen!")
            if st.button("🔄 Refresh Bot", key="refresh_bot"):
                bot.running = False
                time.sleep(2)
                bot.start(bot.params)
                st.rerun()
        else:
            st.info("ℹ️ System Normal")
    
    # Current server time display
    st.markdown(f"**Server Time:** {get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')}")

# --- Streamlit UI with Auto Refresh ---
st.set_page_config(
    page_title="Trading Bot Dashboard v2.0", 
    layout="wide",
    page_icon="🤖"
)

# Auto-refresh configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [30, 60, 120, 300], index=1)

if auto_refresh:
    st.sidebar.info(f"Auto refreshing every {refresh_interval} seconds")
    time.sleep(refresh_interval)
    st.rerun()

# Bot instance with caching
try:
    bot = get_bot_instance()
except Exception as e:
    st.error(f"Bot initialization failed: {e}")
    bot = TradingBot()

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Sidebar for Mode Settings and Tools
st.sidebar.title("⚙️ Mode and Tools v2.0")
is_paper_trading = st.sidebar.toggle("Paper Trading Mode", value=True, help="When enabled, no real orders will be placed.")
st.session_state.strategy_params = getattr(st.session_state, 'strategy_params', {})
st.session_state.strategy_params['is_paper_trading'] = is_paper_trading

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Instrument Cache"):
    if os.path.exists(INSTRUMENT_CACHE_FILE):
        os.remove(INSTRUMENT_CACHE_FILE)
        st.sidebar.success("Cache cleared!")
    else:
        st.sidebar.info("No cache file to clear.")

if st.sidebar.button("❌ Reset Active Trade and PnL"):
    if os.path.exists(STATE_DB):
        os.remove(STATE_DB)
        st.sidebar.success("State file deleted! Restart the bot.")
        bot.load_state() 
        st.rerun()
    else:
        st.sidebar.info("No state file exists.")
        
if is_paper_trading:
    st.title("📈 Trading Bot Dashboard v2.0 (Paper Trading)")
    st.info("You are currently in Paper Trading mode. No real trades will be executed.")
else:
    st.title("💰 Trading Bot Dashboard v2.0 (Live Trading)")
    st.warning("You are in Live Trading mode. Real money will be used!")

# Bot health display
display_bot_health(bot)

st.header("1) Login and Bot Control")
c1, c2, c3 = st.columns([1.5, 1, 3])
with c1:
    if not bot.obj:
        if st.button("Login to Angel One"):
            if bot.login(): 
                st.rerun()
            else: 
                st.error(bot.status)
    else:
        st.success("Successfully logged in.")
        if st.button("Logout"):
            bot.obj.terminateSession(CLIENT_ID)
            bot.obj = None
            st.rerun()
with c2:
    if not bot.running:
        if st.button("🚀 Start Bot"):
            if not st.session_state.strategy_params.get('expiry'):
                st.error("Please select an expiry first.")
            else:
                trade_mode = st.session_state.strategy_params.get('trade_mode')
                if trade_mode == 'Automatic (All Indices)':
                    st.session_state.strategy_params['indices_to_monitor'] = ALL_INDICES
                else:
                    selected_index = trade_mode.split(" ")[1]
                    st.session_state.strategy_params['indices_to_monitor'] = [selected_index]
                bot.start(st.session_state.strategy_params)
                st.rerun()
    else:
        if st.button("🛑 Stop Bot"): 
            bot.stop()
            st.rerun()
with c3:
    st.markdown(f"**Bot Status:** {bot.status}")
    
    if bot.running:
        last_check_info = bot.get_last_checked_display()
        st.markdown(f"**Last Check:** {last_check_info['formatted_time']}")
        st.markdown(f"**Time Difference:** {last_check_info['seconds_ago']} seconds")

pnl_c1, pnl_c2 = st.columns(2)
pnl_c1.metric("Today's Live PnL", f"₹ {bot.daily_pnl:,.2f}")
pnl_c2.metric("Today's Paper PnL", f"₹ {bot.paper_pnl:,.2f}")

st.markdown("---")
if bot.active_trade:
    mode = "Paper Trade" if bot.active_trade.get('is_paper_trading') else "Live Trade"
    st.success(f"Active {mode}:")
    st.json(bot.active_trade)

st.header("2) Strategy and Trade Parameters")
st.session_state.strategy_params['trade_direction'] = st.radio(
    "Select Trade Direction:",
    ['Both (CALL & PUT)', 'CALL Only', 'PUT Only'],
    horizontal=True, 
    key='trade_direction'
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Main Settings")
    st.session_state.strategy_params['trade_mode'] = st.radio(
        "Select Trading Index", 
        [f'Only {idx}' for idx in ALL_INDICES] + ['Automatic (All Indices)']
    )
    st.session_state.strategy_params['interval'] = st.selectbox("Candle Interval", ["5min", "1min", "15min"])
    st.session_state.strategy_params['capital'] = st.number_input("Capital (₹)", value=100000)
    st.session_state.strategy_params['risk_per_trade'] = st.number_input("Risk Per Trade (%)", 0.01, 10.0, value=1.0, step=0.1) / 100
with c2:
    st.subheader("Technical Indicators")
    st.session_state.strategy_params['ema_period'] = st.number_input("EMA Period", value=25)
    st.session_state.strategy_params['rsi_period'] = st.number_input("RSI Period", value=14)
    st.info("MACD parameters (12, 26, 9) are standard.")
with c3:
    st.subheader("SL, TP and Trailing")
    st.session_state.strategy_params['sl_offset'] = st.number_input("Initial SL Offset (₹)", 1.0, value=20.0, step=0.5)
    st.session_state.strategy_params['tp_offset'] = st.number_input("TP Offset (₹)", 1.0, value=20.0, step=0.5)
    st.session_state.strategy_params['start_trailing_after_points'] = st.number_input("Start Trailing After (Points)", 0.0, value=0.0, step=1.0)
    st.session_state.strategy_params['trailing_sl_gap_points'] = st.number_input("Trailing SL Gap (Points)", 0.0, value=20.0, step=1.0)
    st.session_state.strategy_params['exit_on_points_gain'] = st.number_input("Exit After Points Gain", 0.0, value=0.0, step=1.0)
    st.session_state.strategy_params['max_daily_loss'] = st.number_input("Max Daily Loss (₹) (Live Only)", value=10000)
    
st.header("3) Select Expiry")
exp_c1, exp_c2 = st.columns([1,2])

with exp_c1:
    expiry_index_choice = st.selectbox("Select Index for Expiry", ALL_INDICES)

if exp_c2.button(f"📅 Get {expiry_index_choice} Expiry"):
    if bot.obj:
        with st.spinner("Loading expiry list..."):
            expiries = get_expiry_list(bot, expiry_index_choice)
            if expiries:
                st.session_state["expiries"] = expiries
                st.success(f"Found {len(expiries)} expiries")
            else:
                st.error("No expiries found. Check instrument list.")
    else: 
        st.warning("Please login first.")

# Expiry dropdown
if st.session_state.get("expiries"):
    st.session_state.strategy_params['expiry'] = st.selectbox(
        "Select Expiry", 
        st.session_state.get("expiries", [])
    )
else:
    st.info("Click above button to load expiries")

if is_paper_trading and bot.paper_trades_log:
    st.header("Today's Paper Trades Log")
    st.dataframe(pd.DataFrame(bot.paper_trades_log).iloc[::-1])

st.markdown("---")
st.header("4) Backtest Strategy (With Trailing SL)")
st.warning("Backtesting will work on only one index at a time. Please select index below.")

backtest_index = st.selectbox("Select Index for Backtest", ALL_INDICES)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: 
    start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
with c2: 
    end_date = st.date_input("End Date", datetime.now().date() - timedelta(days=1))
with c3: 
    hypo_opt_price = st.number_input("Hypothetical Option Entry (₹)", 1.0, value=120.0, step=0.5)
with c4: 
    trade_cost = st.number_input("Trade Cost (₹)", 0.0, value=50.0, step=1.0)
with c5: 
    trade_type_backtest = st.selectbox("Backtest Trade Type", ["CALL", "PUT"])
with c6: 
    slippage = st.number_input("Slippage (₹)", 0.0, value=0.5, step=0.1)

if st.button("▶ Run Backtest Now (With Trailing SL)"):
    if not bot.obj: 
        st.error("Please login first.")
    elif start_date > end_date: 
        st.error("Start date must be before end date.")
    else:
        with st.spinner(f"Running backtest for {backtest_index}..."):
            all_trades = run_backtest_with_trailing_sl(
                bot, backtest_index, start_date, end_date, 
                st.session_state.strategy_params,
                hypo_opt_price, trade_cost, trade_type_backtest, slippage
            )
            
            if all_trades is None:
                st.error("Error running backtest.")
            elif not all_trades: 
                st.info(f"No trades found in selected date range for {backtest_index}.")
            else:
                trades_df = pd.DataFrame(all_trades)
                st.success(f"Backtest completed! Found {len(all_trades)} trades.")
                
                # Display trades with Trailing SL
                st.dataframe(trades_df)
                
                # Performance analysis
                total_pnl = trades_df['net_pnl'].sum()
                wins = trades_df[trades_df['net_pnl'] > 0]
                sl_hits = trades_df[trades_df['status'] == 'SL_HIT']
                tp_hits = trades_df[trades_df['status'] == 'TP_HIT']
                trailing_trades = trades_df[trades_df['high_water_mark'].astype(float) > trades_df['option_entry'].astype(float)]
                
                st.markdown("### Backtest Summary (With Trailing SL)")
                k1, k2, k3, k4 = st.columns(4)
                
                k1.metric("Total Net PnL", f"₹ {total_pnl:,.2f}")
                k2.metric("Total Trades", len(trades_df))
                k2.metric("Profitable Trades", len(wins))
                k3.metric("SL Hits", len(sl_hits))
                k3.metric("TP Hits", len(tp_hits))
                k4.metric("Win Rate", f"{(len(wins)/len(trades_df))*100:.2f}%" if len(trades_df) > 0 else "0%")
                k4.metric("Trailing SL Active", f"{len(trailing_trades)} trades")
                
                if len(trailing_trades) > 0:
                    st.info(f"✅ Trailing SL active in {len(trailing_trades)} trades")
                
                # Equity curve
                trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(pd.to_datetime(trades_df['date']), trades_df['cum_pnl'], linewidth=2)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Total PnL (₹)', fontsize=12)
                ax.set_title('Equity Curve with Trailing SL', fontsize=14)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                st.pyplot(fig)

st.markdown("---")
st.success("**v2.0 New Features:** 🟢 Normal | 🟠 Slow | 🔴 Frozen | ✅ Trailing SL Backtesting | 🎯 Real-time Monitoring | 🌏 Timezone Fixed")

# Footer
