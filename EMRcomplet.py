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
from email.mime.multipart import MIMEMultipart
from threading import Lock
import pytz
import warnings
warnings.filterwarnings('ignore')

# --- Global Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(file_handler)

IST = pytz.timezone('Asia/Kolkata')
INSTRUMENT_CACHE_FILE = "instrument_list.json.gz"
STATE_DB = "bot_state.db"

# --- Try to import SmartApi ---
try:
    from SmartApi import SmartConnect
except ImportError:
    st.error("SmartApi library not found. Please install it using 'pip install smartapi-python'")
    # We don't stop here to allow UI to load, but login will fail.

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
    "BANKNIFTY": {"lot_size": 25, "exchange": "NFO", "strike_step": 100, "instrument_name": "BANKNIFTY"},
    "FINNIFTY": {"lot_size": 40, "exchange": "NFO", "strike_step": 50, "instrument_name": "NIFTY FIN SERVICE"},
    "SENSEX": {"lot_size": 10, "exchange": "BFO", "strike_step": 100, "instrument_name": "SENSEX"}
}
ALL_INDICES = list(INDEX_CONFIG.keys())

# <editor-fold desc="Utility Functions">

def get_ist_time():
    """Get current time in IST timezone"""
    return datetime.now(IST)

def make_naive(dt):
    """Convert aware datetime to naive datetime"""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

def format_ist_time(dt=None, format_str="%Y-%m-%d %H:%M:%S"):
    """Format datetime to IST time string"""
    if dt is None:
        dt = get_ist_time()
    elif dt.tzinfo is not None:
        dt = dt.astimezone(IST)
    else:
        dt = IST.localize(dt)
    return dt.strftime(format_str)

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def fetch_and_cache_instrument_list():
    try:
        if os.path.exists(INSTRUMENT_CACHE_FILE):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(INSTRUMENT_CACHE_FILE))
            current_time_naive = make_naive(get_ist_time())
            if current_time_naive - file_mod_time < timedelta(hours=24):
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
    if not instrument_list: return None
    config = INDEX_CONFIG[index_name]
    instrument_name = config.get("instrument_name", index_name)
    
    # Logic to find the nearest month's future token
    futures = []
    today = get_ist_time().date()
    for item in instrument_list:
        if (item.get("instrumenttype") == "FUTIDX" and 
            item.get("name") == instrument_name and 
            item.get("exch_seg") == exchange and
            "NIFTYNXT50" not in item.get("symbol", "") and
            "NEXT" not in item.get("symbol", "").upper()):
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
        if (item.get("name") == instrument_name and 
            item.get("exch_seg") == config['exchange'] and
            item.get("instrumenttype") == "OPTIDX" and 
            item.get("strike") and
            float(item.get("strike")) / 100 == float(strike) and 
            item.get("expiry") == expiry_fmt and
            item.get("symbol", "").endswith(option_cepe)):
            return item.get("token"), item.get("symbol")
            
    raise RuntimeError(f"Could not find {option_cepe} for strike {strike} and expiry {expiry}.")

@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def try_fetch_candles(obj, symbol_token, interval, days_back, exchange):
    to_dt, from_dt = get_ist_time(), get_ist_time() - timedelta(days=days_back)
    imap = {"1min": "ONE_MINUTE", "5min": "FIVE_MINUTE", "15min": "FIFTEEN_MINUTE"}
    params = {
        "exchange": exchange, 
        "symboltoken": str(symbol_token), 
        "interval": imap.get(interval),
        "fromdate": make_naive(from_dt).strftime("%Y-%m-%d %H:%M"), 
        "todate": make_naive(to_dt).strftime("%Y-%m-%d %H:%M")
    }
    try:
        res = obj.getCandleData(params)
        if not res or "data" not in res or not res["data"]: return None
        rows = [{"timestamp": pd.to_datetime(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in res["data"]]
        df = pd.DataFrame(rows)
        if not df.empty:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df
    except Exception as e:
        logging.error(f"Error fetching candles for {symbol_token}: {e}")
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
    signals = []
    if df is None or len(df) < 26: return signals
        
    df['ema'] = calculate_ema(df, period=params['ema_period'])
    df['rsi'] = calculate_rsi(df, period=params['rsi_period'])
    df = calculate_macd(df)
    
    # Check for signals only in the last few candles (for live trading) or all (for backtest)
    start_index = 1 if is_backtest else max(1, len(df) - 2)
    
    for i in range(start_index, len(df)):
        current_candle, prev_candle = df.iloc[i], df.iloc[i - 1]
        
        if not (dt_time(9, 20) <= current_candle['timestamp'].time() < dt_time(15, 20)): 
            continue
        
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
        return None

# </editor-fold>

# <editor-fold desc="Class Definitions">

class MarketCalendar:
    def __init__(self):
        self.holidays_2024 = [
            '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29',
            '2024-04-11', '2024-04-17', '2024-05-01', '2024-06-17',
            '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01',
            '2024-11-15', '2024-12-25'
        ]
    
    def is_trading_day(self, date=None):
        if date is None:
            date = get_ist_time().date()
        if date.weekday() >= 5: return False # Weekend check
        if date.strftime('%Y-%m-%d') in self.holidays_2024: return False # Holiday check
        return True
    
    def is_trading_time(self):
        if not self.is_trading_day(): return False
        now = get_ist_time()
        market_start = dt_time(9, 15)
        market_end = dt_time(15, 30)
        return market_start <= now.time() <= market_end
    
    def next_trading_day(self):
        day = get_ist_time() + timedelta(days=1)
        while not self.is_trading_day(day.date()):
            day += timedelta(days=1)
        return day.date()

class PositionSizer:
    def __init__(self):
        self.min_lots = 1
        self.max_lots = 10
        self.max_capital_risk = 0.1
    
    def calculate_position_size(self, capital, risk_per_trade, sl_points, lot_size, volatility_factor=1.0):
        risk_amount = capital * risk_per_trade
        risk_amount = risk_amount / max(0.5, min(2.0, volatility_factor))
        risk_per_lot = sl_points * lot_size
        
        if risk_per_lot <= 0:
            return self.min_lots * lot_size
        
        optimal_lots = risk_amount / risk_per_lot
        max_lots_based_on_capital = int((capital * self.max_capital_risk) / risk_per_lot)
        max_allowed_lots = min(self.max_lots, max_lots_based_on_capital)
        final_lots = max(self.min_lots, min(max_allowed_lots, int(optimal_lots)))
        
        return final_lots * lot_size

class TradeJournal:
    def __init__(self, db_path="trade_journal.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL,
                commission REAL DEFAULT 0,
                strategy_params TEXT,
                market_condition TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_trade(self, trade_data):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades 
            (timestamp, symbol, trade_type, quantity, entry_price, exit_price, exit_reason, pnl, strategy_params, market_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('timestamp', format_ist_time()),
            trade_data.get('symbol', 'Unknown'),
            trade_data.get('trade_type', 'Unknown'),
            trade_data.get('quantity', 0),
            trade_data.get('entry_price', 0),
            trade_data.get('exit_price'),
            trade_data.get('exit_reason'),
            trade_data.get('pnl', 0),
            json.dumps(trade_data.get('strategy_params', {})),
            trade_data.get('market_condition', 'Normal')
        ))
        conn.commit()
        conn.close()
    
    def get_performance_report(self):
        conn = sqlite3.connect(self.db_path)
        metrics = {}
        total_trades = pd.read_sql('SELECT COUNT(*) as count FROM trades', conn).iloc[0]['count']
        metrics['total_trades'] = total_trades
        
        if total_trades > 0:
            winning_trades = pd.read_sql('SELECT COUNT(*) as count FROM trades WHERE pnl > 0', conn).iloc[0]['count']
            metrics['win_rate'] = (winning_trades / total_trades) * 100
            pnl_data = pd.read_sql('SELECT pnl FROM trades WHERE pnl IS NOT NULL', conn)
            if not pnl_data.empty:
                metrics['total_pnl'] = pnl_data['pnl'].sum()
                metrics['avg_win'] = pnl_data[pnl_data['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                metrics['avg_loss'] = pnl_data[pnl_data['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
                loss_trades = total_trades - winning_trades
                
                # Calculate Profit Factor safely
                if loss_trades > 0 and metrics['avg_loss'] != 0:
                    gross_profit = metrics['avg_win'] * winning_trades
                    gross_loss = abs(metrics['avg_loss'] * loss_trades)
                    metrics['profit_factor'] = gross_profit / gross_loss
                else:
                    metrics['profit_factor'] = 0 if winning_trades == 0 else float('inf')
            
        conn.close()
        return metrics

class AlertSystem:
    def __init__(self):
        # Using global environment variables here
        self.email_sender = EMAIL_SENDER
        self.email_password = EMAIL_PASSWORD
        self.email_receiver = EMAIL_RECEIVER
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        # Ensure all credentials exist before attempting connection
        if not all([self.email_sender, self.email_password, self.email_receiver]):
            logging.warning("Email credentials missing. Skipping email alert.")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = f"Trading Bot Alert: {subject}"
            msg.attach(MIMEText(message, 'plain'))
            
            # Use 'with' statement for safer connection handling
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                text = msg.as_string()
                server.sendmail(self.email_sender, self.email_receiver, text)
            
            logging.info(f"Email alert sent: {subject}")
            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False
    
    def send_trade_alert(self, alert_type, trade_data=None):
        """Send trade-specific alerts with None-safe message formatting."""
        
        # Safe message formatting for trade_executed to prevent AttributeError on None trade_data
        if alert_type == 'trade_executed':
            message = (f"Trade Executed: {trade_data.get('symbol', 'N/A')} | Price: {trade_data.get('entry_price', 'N/A')} | Qty: {trade_data.get('quantity', 'N/A')}" 
                       if trade_data else "Trade Executed: Details not available")
            subject = 'Trade Executed'
        
        # Safe message formatting for sl_hit
        elif alert_type == 'sl_hit':
            message = (f"SL Hit: {trade_data.get('symbol', 'N/A')} | PnL: ₹{trade_data.get('pnl', 0):.2f}" 
                       if trade_data else "Stop Loss Hit: Details not available")
            subject = 'Stop Loss Hit'

        # Safe message formatting for tp_hit
        elif alert_type == 'tp_hit':
            message = (f"Target Hit: {trade_data.get('symbol', 'N/A')} | PnL: ₹{trade_data.get('pnl', 0):.2f}" 
                       if trade_data else "Target Achieved: Details not available")
            subject = 'Target Achieved'
        
        # Bot system alerts
        elif alert_type == 'bot_started':
            message, subject = "Trading Bot has started successfully.", 'Bot Started'
        elif alert_type == 'bot_stopped':
            message, subject = "Trading Bot has been stopped.", 'Bot Stopped'
        
        # Safe message formatting for error
        elif alert_type == 'error':
            message = f"Error occurred: {trade_data.get('error', 'Unknown error')}" if trade_data else "Unknown Bot Error"
            subject = 'Bot Error'
        
        else:
            return False

        return self.send_email_alert(subject, message)

class AdvancedAnalytics:
    def __init__(self, trade_journal):
        self.journal = trade_journal
    
    # ... (other analytics methods remain the same) ...
    def generate_report(self):
        metrics = self.journal.get_performance_report()
        report = {
            'basic_metrics': metrics,
            # 'daily_performance': self.get_daily_performance(), # Remove placeholder/unused methods
            # 'strategy_analysis': self.analyze_strategies(),
            'risk_metrics': self.calculate_risk_metrics(),
            'improvement_suggestions': self.get_suggestions()
        }
        return report
    
    def calculate_risk_metrics(self):
        conn = sqlite3.connect(self.journal.db_path)
        trades = pd.read_sql('''
            SELECT timestamp, pnl FROM trades 
            WHERE pnl IS NOT NULL 
            ORDER BY timestamp
        ''', conn)
        conn.close()
        
        metrics = {}
        if not trades.empty and len(trades) > 1:
            trades['cumulative_pnl'] = trades['pnl'].cumsum()
            trades['running_max'] = trades['cumulative_pnl'].cummax()
            trades['drawdown'] = trades['running_max'] - trades['cumulative_pnl']
            metrics['max_drawdown'] = trades['drawdown'].max()
            metrics['avg_drawdown'] = trades['drawdown'].mean()
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(trades)
        
        return metrics
    
    def calculate_sharpe_ratio(self, trades, risk_free_rate=0.05):
        if len(trades) < 2: return 0
        returns = trades['pnl'].pct_change().dropna()
        if returns.empty or returns.std() == 0: return 0
        
        excess_returns = returns - (risk_free_rate / 252) # Daily risk-free rate estimate
        return (excess_returns.mean() / returns.std()) * (252 ** 0.5)

    def get_suggestions(self):
        metrics = self.journal.get_performance_report()
        suggestions = []
        if metrics.get('total_trades', 0) > 10:
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            if win_rate < 40:
                suggestions.append("Win rate is low. Consider improving entry timing or adding filters.")
            if profit_factor < 1.0:
                suggestions.append("Profit factor below 1.0. Review risk-reward ratios.")
        return suggestions


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
        
        self.market_calendar = MarketCalendar()
        self.position_sizer = PositionSizer()
        self.trade_journal = TradeJournal()
        self.alert_system = AlertSystem()
        self.analytics = AdvancedAnalytics(self.trade_journal)
        
        self.last_heartbeat = get_ist_time()
        self.freeze_threshold = 60
        
        self.load_state()

    def login(self):
        self.status = "Logging in..."
        try:
            # 1. Initialize SmartConnect
            self.obj = SmartConnect(api_key=API_KEY)
            otp = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else None
            
            # 2. Generate Session
            data = self.obj.generateSession(CLIENT_ID, CLIENT_PWD, otp)
            
            # 3. Check for API Response Safely (Fixes 'NoneType' object has no attribute 'get')
            if data is not None:
                if data.get('status'):
                    self.status = "Logged in."
                    self.instrument_list = fetch_and_cache_instrument_list()
                    self.update_heartbeat()
                    
                    # Send login alert
                    self.alert_system.send_trade_alert('bot_started')
                    
                    return True
                
                # If status is False, log the error message safely
                self.status = f"Login Error: {data.get('message', 'API Error: Status False')}"
            else:
                self.status = "Login Error: API returned no data (None)."

        except Exception as e: 
            # 4. Handle exceptions during the process (Network, TOTP, etc.)
            error_msg = str(e)
            self.status = f"Login Exception: {error_msg}"
            self.alert_system.send_trade_alert('error', {'error': error_msg})
            
        return False
        
    def update_heartbeat(self):
        self.last_heartbeat = get_ist_time()
        
    def is_bot_frozen(self):
        time_since_last_heartbeat = (get_ist_time() - self.last_heartbeat).total_seconds()
        return time_since_last_heartbeat > self.freeze_threshold
        
    def get_bot_status_color(self):
        if not self.running: return "gray"
        if self.is_bot_frozen(): return "red"
        time_since_heartbeat = (get_ist_time() - self.last_heartbeat).total_seconds()
        if time_since_heartbeat > 30: return "orange"
        return "green"
        
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
        if not os.path.exists(STATE_DB): return
        conn = sqlite3.connect(STATE_DB)
        c = conn.cursor()
        try:
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
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")
        finally:
            conn.close()

    def start(self, params):
        if not self.running:
            self.params = params
            self.running = True
            
            if not self.market_calendar.is_trading_time():
                next_day = self.market_calendar.next_trading_day()
                self.status = f"Market is closed. Bot will start automatically when market opens on {next_day}."
            
            self.thread = threading.Thread(target=self.run_strategy_loop, daemon=True)
            self.thread.start()
            mode = "Paper Trading" if params.get('is_paper_trading') else "Live Trading"
            self.status = f"Bot started in {mode} mode."
            self.update_heartbeat()
            self.alert_system.send_trade_alert('bot_started')

    def stop(self): 
        self.running = False
        self.status = "Bot stopped by user."
        self.alert_system.send_trade_alert('bot_stopped')
    
    def check_exit_conditions(self, ltp):
        if ltp <= self.active_trade['sl']: return f"SL hit at {ltp:.2f}"
        if ltp >= self.active_trade['tp']: return f"TP hit at {ltp:.2f}"
        
        # Trailing SL logic
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
        
        # Exit on gain logic
        exit_on_gain = self.params.get('exit_on_points_gain', 0)
        if exit_on_gain > 0 and ltp >= self.active_trade['entry_price'] + exit_on_gain:
            return f"Exit on Gain at {ltp:.2f}"
        
        return None

    def exit_trade(self, reason, ltp, is_paper):
        if ltp is None: ltp = self.active_trade['entry_price']
        
        pnl = (ltp - self.active_trade['entry_price']) * self.active_trade['qty']
        trade_data = {
            'timestamp': self.active_trade.get('entry_time', format_ist_time()),
            'symbol': self.active_trade['symbol'],
            'trade_type': self.active_trade.get('index', 'Unknown'),
            'quantity': self.active_trade['qty'],
            'entry_price': self.active_trade['entry_price'],
            'exit_price': ltp,
            'exit_reason': reason,
            'pnl': pnl,
            'strategy_params': self.params
        }
        self.trade_journal.log_trade(trade_data)
        
        if is_paper:
            self.paper_pnl += pnl
            log_entry = self.active_trade.copy()
            log_entry.update({"exit_price": ltp, "pnl": pnl, "exit_reason": reason, "exit_time": format_ist_time()})
            self.paper_trades_log.append(log_entry)
            self.alert_system.send_trade_alert('tp_hit' if 'TP' in reason else 'sl_hit', trade_data)
        else: 
            self.daily_pnl += pnl
            if self.daily_pnl <= -self.params.get('max_daily_loss', 10000):
                self.status = "Max daily loss reached. Stopping bot."
                self.running = False
                self.alert_system.send_trade_alert('bot_stopped')

        self.status = f"Exiting {self.active_trade['symbol']}: {reason}"
        place_order(self.obj, self.active_trade['symbol'], self.active_trade['token'], 
                    self.active_trade['qty'], self.active_trade['exchange'], "SELL", 
                    is_paper_trading=is_paper)
        self.active_trade = None
        self.save_state()

    def monitor_active_trade(self):
        with self.lock:
            if not self.active_trade: return

            is_paper = self.active_trade.get("is_paper_trading", False)
            
            if not self.market_calendar.is_trading_time():
                self.exit_trade("Market Closed", None, is_paper)
                return

            try:
                ltp = get_option_ltp(self.obj, self.active_trade['exchange'], 
                                   self.active_trade['symbol'], self.active_trade['token'])
                
                if ltp is None:
                    self.status = "LTP प्राप्त नहीं हो रहा"
                    return
                    
                exit_reason = self.check_exit_conditions(ltp)
                if exit_reason:
                    self.exit_trade(exit_reason, ltp, is_paper)
                    
            except Exception as e:
                logging.error(f"मॉनिटरिंग एरर: {e}")
                self.alert_system.send_trade_alert('error', {'error': str(e)})
            
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
        trade_direction = self.params.get('trade_direction', 'दोनों (CALL और PUT)')
        min_option_price = self.params.get('min_option_price', 20.0)

        while self.running:
            try:
                self.update_heartbeat()
                
                if not self.market_calendar.is_trading_time():
                    if self.active_trade: self.monitor_active_trade()
                    else: self.status = "Market is closed. Waiting for next trading session."
                    time.sleep(60)
                    continue
                
                with self.lock:
                    if self.active_trade:
                        self.monitor_active_trade()
                    else:
                        for index_name in indices_to_monitor:
                            self.status = f"Checking {index_name}..."
                            config = INDEX_CONFIG[index_name]
                            futures_token = find_index_futures_token(self.instrument_list, index_name, config['exchange'])
                            if not futures_token: continue
                            
                            df = try_fetch_candles(self.obj, futures_token, self.params['interval'], 1, config['exchange'])
                            
                            # Use explicit length check
                            if df is None or len(df) == 0: continue
                            
                            current_candle_time = df.iloc[-1]['timestamp']
                            time_diff = (make_naive(get_ist_time()) - current_candle_time).total_seconds()
                            if time_diff > 120: continue # Skip if data is stale
                            
                            signals = detect_strategy_signals(df, self.params, is_backtest=False)

                            if trade_direction == 'केवल CALL': signals = [s for s in signals if s['signal'] == 'CALL']
                            elif trade_direction == 'केवल PUT': signals = [s for s in signals if s['signal'] == 'PUT']

                            if signals:
                                signal = signals[-1]
                                self.status = f"{signal['signal']} Signal in {index_name}! Processing..."
                                
                                spot = signal['entry_price']
                                atm = round(spot / config['strike_step']) * config['strike_step']
                                trade_type = "CE" if signal['signal'] == "CALL" else "PE"
                                
                                # This needs expiry. Ensure 'expiry' is set in self.params.
                                if 'expiry' not in self.params:
                                    logging.error("Expiry date not set in parameters.")
                                    continue
                                
                                try:
                                    token, symbol = find_option_token_from_list(self.instrument_list, index_name, atm, self.params['expiry'], trade_type)
                                except RuntimeError as re:
                                    logging.warning(f"Skipping trade: {re}")
                                    continue
                                    
                                entry_price = get_option_ltp(self.obj, config['exchange'], symbol, token)
                                
                                if entry_price and entry_price > min_option_price: 
                                    qty = self.position_sizer.calculate_position_size(
                                        self.params['capital'], 
                                        self.params['risk_per_trade'],
                                        self.params['sl_offset'],
                                        config['lot_size']
                                    )
                                    
                                    order_id = place_order(self.obj, symbol, token, qty, config['exchange'], "BUY", is_paper_trading=is_paper)
                                    if order_id:
                                        self.active_trade = {
                                            "symbol": symbol, "token": token, "qty": qty, 
                                            "exchange": config['exchange'], "index": index_name,
                                            "entry_price": entry_price, 
                                            "sl": entry_price - self.params['sl_offset'],
                                            "tp": entry_price + self.params['tp_offset'], 
                                            "high_water_mark": entry_price,
                                            "is_paper_trading": is_paper, 
                                            "entry_time": format_ist_time()
                                        }
                                        self.save_state()
                                        self.status = f"Trade {'Simulated' if is_paper else 'Placed'} for {symbol}."
                                        trade_alert_data = {'symbol': symbol, 'entry_price': entry_price, 'quantity': qty}
                                        self.alert_system.send_trade_alert('trade_executed', trade_alert_data)
                                        break
                                else:
                                    logging.warning(f"Trade skipped for {symbol}. Entry price {entry_price} is below minimum {min_option_price}.")
                        
                        if not self.active_trade: self.status = "Monitoring..."
            
            except Exception as e:
                self.status = f"Error in loop: {e}"
                logging.error(f"Critical error in strategy loop: {e}", exc_info=True)
                self.alert_system.send_trade_alert('error', {'error': str(e)})
            
            self.last_checked = format_ist_time()
            time.sleep(10)

# </editor-fold>

# <editor-fold desc="Streamlit UI Logic and Backtesting">

def get_expiry_list(bot, index_name):
    if not bot.obj: return []
    instrument_list = fetch_and_cache_instrument_list()
    if not instrument_list: return []
        
    cfg = INDEX_CONFIG[index_name]
    instrument_name = cfg["instrument_name"]
    expiries = set()
    today = get_ist_time().date()
    
    for item in instrument_list:
        if (item.get("name") == instrument_name and item.get("exch_seg") == cfg["exchange"] and 
            item.get("instrumenttype") == "OPTIDX" and item.get("expiry")):
            if "NIFTYNXT50" in item.get("symbol", "") or "NEXT" in item.get("symbol", "").upper(): continue
            try:
                expiry_date = datetime.strptime(item["expiry"], '%d%b%Y').date()
                if expiry_date >= today:
                    expiries.add(expiry_date)
            except ValueError:
                continue
                
    return [d.strftime("%Y-%m-%d") for d in sorted(list(expiries))]

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
        if not res or 'data' not in res or not res['data']: return None
        rows = [{"timestamp": pd.to_datetime(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in res['data']]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        if not df.empty:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df
    except Exception as e:
        logging.warning(f"Error fetching historical candles for {target_date.strftime('%Y-%m-%d')}: {e}")
        return None

def run_backtest_with_trailing_sl(bot, backtest_index, start_date, end_date, params, 
                                 hypo_opt_price, trade_cost, trade_type_backtest, slippage):
    config = INDEX_CONFIG[backtest_index]
    sl_points = params['sl_offset']
    capital = params['capital']
    risk_per_trade = params['risk_per_trade']
    
    position_sizer = PositionSizer()
    contracts_qty = position_sizer.calculate_position_size(capital, risk_per_trade, sl_points, config['lot_size'])
    
    all_trades = []
    instrument_list = fetch_and_cache_instrument_list()
    futures_token = find_index_futures_token(instrument_list, backtest_index, config['exchange']) 
    
    if not futures_token: 
        st.error(f"Futures token not found for {backtest_index}")
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
                    
                    start_trailing_after = params.get('start_trailing_after_points', 0)
                    trailing_gap = params.get('trailing_sl_gap_points', 0)
                    use_trailing = start_trailing_after > 0 and trailing_gap > 0
                    
                    high_water_mark = hypo_entry
                    status, exit_price = "UNKNOWN", hypo_entry
                    exit_timestamp = None
                    
                    for j in range(sig['entry_index'] + 1, len(df_hist)):
                        c = df_hist.iloc[j]
                        # Calculate option price movement based on index movement
                        index_move = c['close'] - sig['entry_price']
                        est_price = hypo_entry + index_move # Simplified delta=1 assumption
                        
                        # Use candle High/Low for quicker exits (conservative backtesting)
                        index_high_move = c['high'] - sig['entry_price']
                        index_low_move = c['low'] - sig['entry_price']
                        est_high = hypo_entry + index_high_move
                        est_low = hypo_entry + index_low_move
                        
                        # Trailing SL update
                        if use_trailing:
                            high_water_mark = max(high_water_mark, est_high)
                            if high_water_mark >= hypo_entry + start_trailing_after:
                                new_sl = high_water_mark - trailing_gap
                                if new_sl > current_sl:
                                    current_sl = new_sl
                        
                        # Exit checks
                        if est_low <= current_sl: 
                            status, exit_price, exit_timestamp = "SL_HIT", current_sl, c['timestamp']
                            break
                        elif est_high >= target_price: 
                            # Exit at TP, adjust for slippage on exit
                            status, exit_price, exit_timestamp = "TP_HIT", target_price - slippage, c['timestamp']
                            break
                        elif c['timestamp'].time() >= dt_time(15, 20): 
                            status, exit_price, exit_timestamp = "EOD_EXIT", est_price, c['timestamp']
                            break
                    
                    if status == "UNKNOWN": 
                        status, exit_price, exit_timestamp = "EOD_NO_EXIT", hypo_entry + index_move, df_hist.iloc[-1]['timestamp']
                    
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

def display_bot_health(bot):
    st.subheader("🤖 बॉट हेल्थ मॉनिटर v4.0")
    col1, col2, col3, col4 = st.columns(4)
    status_color = bot.get_bot_status_color()
    status_map = {"green": "🟢 सामान्य", "orange": "🟠 उलझा हुआ", "red": "🔴 फ्रीज", "gray": "⚫ बंद"}
    status_text = status_map.get(status_color, "⚫ बंद")
            
    with col1:
        st.markdown(f"**स्थिति:** <span style='color:{status_color}; font-size:20px;'>{status_text}</span>", unsafe_allow_html=True)
    
    with col2:
        if bot.running and bot.last_checked:
            try:
                last_check_time = datetime.strptime(bot.last_checked, "%Y-%m-%d %H:%M:%S")
                time_diff = (make_naive(get_ist_time()) - last_check_time).total_seconds()
                st.metric("आखिरी जाँच", f"{int(time_diff)} सेकंड पहले")
            except Exception: pass
    
    with col3:
        if bot.running:
            time_since_heartbeat = (get_ist_time() - bot.last_heartbeat).total_seconds()
            st.metric("हार्टबीट", f"{int(time_since_heartbeat)}s")
            if time_since_heartbeat > bot.freeze_threshold:
                st.error("❌ फ्रीज अलर्ट!")
            elif time_since_heartbeat > 30:
                st.warning("⚠️ धीमा चल रहा")
            else:
                st.success("✅ सामान्य")
    
    with col4:
        if bot.is_bot_frozen() and bot.running:
            st.error("🔴 बॉट फ्रीज हो गया है!")
            if st.button("🔄 बॉट रीफ्रेश करें", key="refresh_bot"):
                bot.running = False
                time.sleep(2)
                bot.start(bot.params)
                st.rerun()
        else:
            st.info("ℹ️ सिस्टम सामान्य")

def get_display_time():
    now = get_ist_time()
    return {
        "date": now.strftime("%d-%m-%Y"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "full": now.strftime("%d-%m-%Y %H:%M:%S %Z")
    }
    
def display_current_time():
    current_time = get_display_time()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕐 वर्तमान समय (IST)")
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.sidebar.markdown(f"**{current_time['date']}**")
        st.sidebar.markdown(f"**{current_time['time']}**")
    with col2:
        st.sidebar.markdown(f"*{current_time['day']}*")
    st.sidebar.markdown(f"*{current_time['full']}*")
    return current_time

def display_trade_journal(bot):
    st.header("📊 ट्रेड जर्नल")
    if st.button("🔄 जर्नल रिफ्रेश करें", key="journal_refresh"):
        st.rerun()
    
    try:
        conn = sqlite3.connect(bot.trade_journal.db_path)
        trades_df = pd.read_sql('''
            SELECT timestamp, symbol, trade_type, quantity, entry_price, exit_price, pnl, exit_reason 
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''', conn)
        conn.close()
        
        if len(trades_df) > 0: # Use len() check
            st.dataframe(trades_df)
            metrics = bot.trade_journal.get_performance_report()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("कुल ट्रेड्स", metrics.get('total_trades', 0))
            col2.metric("जीतने वाले ट्रेड्स", metrics.get('total_trades', 0) * metrics.get('win_rate', 0) / 100 if metrics.get('total_trades', 0) > 0 else 0)
            col3.metric("जीत दर", f"{metrics.get('win_rate', 0):.1f}%")
            col4.metric("कुल PnL", f"₹{metrics.get('total_pnl', 0):,.2f}")
        else:
            st.info("अभी तक कोई ट्रेड नहीं हुआ है।")
    except Exception as e:
        st.error(f"जर्नल लोड करने में त्रुटि: {e}")

def display_analytics(bot):
    st.header("📈 एडवांस्ड एनालिटिक्स")
    if st.button("📊 रिपोर्ट जनरेट करें", key="analytics_report"):
        with st.spinner("एनालिटिक्स रिपोर्ट तैयार की जा रही है..."):
            report = bot.analytics.generate_report()
            
            st.subheader("बेसिक मेट्रिक्स")
            metrics = report.get('basic_metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("कुल ट्रेड्स", metrics.get('total_trades', 0))
            col2.metric("जीत दर", f"{metrics.get('win_rate', 0):.1f}%")
            col3.metric("कुल PnL", f"₹{metrics.get('total_pnl', 0):,.2f}")
            col4.metric("प्रॉफिट फैक्टर", f"{metrics.get('profit_factor', 0):.2f}")
            
            st.subheader("रिस्क मेट्रिक्स")
            risk_metrics = report.get('risk_metrics', {})
            col1, col2, col3 = st.columns(3)
            col1.metric("मैक्स ड्रॉडाउन", f"₹{risk_metrics.get('max_drawdown', 0):,.2f}")
            col2.metric("शार्प रेश्यो", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
            col3.metric("औसत ड्रॉडाउन", f"₹{risk_metrics.get('avg_drawdown', 0):,.2f}")
            
            st.subheader("सुधार के सुझाव")
            suggestions = report.get('improvement_suggestions', [])
            if suggestions:
                for suggestion in suggestions: st.info(f"💡 {suggestion}")
            else:
                st.success("🎯 परफॉर्मेंस अच्छी है! कोई major improvements needed नहीं।")


def main():
    st.set_page_config(page_title="Professional Trading Bot v4.0", layout="wide", page_icon="🚀")
    
    if 'bot' not in st.session_state: 
        st.session_state.bot = TradingBot()
    bot = st.session_state.bot
    
    # --- Sidebar ---
    st.sidebar.title("⚙️ प्रोफेशनल ट्रेडिंग बॉट v4.0")
    current_time = display_current_time()
    
    is_paper_trading = st.sidebar.toggle("पेपर ट्रेडिंग मोड", value=True)
    st.session_state.strategy_params = getattr(st.session_state, 'strategy_params', {})
    st.session_state.strategy_params['is_paper_trading'] = is_paper_trading
    
    st.sidebar.subheader("🔔 अलर्ट सेटिंग्स")
    # Email alert checkbox now only controls the UI message, as the system attempts to send alerts only if credentials exist.
    email_alerts_enabled = bool(EMAIL_SENDER) 
    if email_alerts_enabled:
        st.sidebar.success("ईमेल अलर्ट क्रेडेंशियल कॉन्फ़िगर किए गए हैं।")
    else:
        st.sidebar.warning("ईमेल अलर्ट क्रेडेंशियल (EMAIL_SENDER/PASSWORD/RECEIVER) कॉन्फ़िगर नहीं हैं। अलर्ट नहीं भेजे जाएंगे।")

    st.sidebar.markdown("---")
    
    # --- Quick Actions ---
    if st.sidebar.button("🔄 इंस्ट्रूमेंट कैश साफ़ करें", key="clear_cache"):
        if os.path.exists(INSTRUMENT_CACHE_FILE):
            os.remove(INSTRUMENT_CACHE_FILE)
            st.sidebar.success("कैश साफ़ हो गया!")
    
    if st.sidebar.button("📊 ट्रेड जर्नल रीसेट करें", key="reset_journal"):
        if os.path.exists("trade_journal.db"):
            os.remove("trade_journal.db")
            bot.trade_journal = TradeJournal()
            st.session_state.bot.analytics = AdvancedAnalytics(st.session_state.bot.trade_journal)
            st.sidebar.success("जर्नल रीसेट हो गया!")

    # --- Main Content Header ---
    title_text = f"💰 प्रोफेशनल ट्रेडिंग बॉट v4.0 ({'पेपर ट्रेडिंग' if is_paper_trading else 'लाइव ट्रेडिंग'})"
    st.title(title_text)
    st.success("🎯 नई स्थिरता और सुरक्षा के साथ v4.0")
    
    display_bot_health(bot)
    
    # --- 1) Login and Control ---
    st.header("1) लॉगिन और बॉट कंट्रोल")
    c1, c2, c3 = st.columns([1.5, 1, 3])
    
    with c1:
        if not bot.obj:
            if st.button("Angel One में लॉगिन करें", key="login_btn"):
                if bot.login(): st.rerun()
                else: st.error(bot.status)
        else:
            st.success("✅ सफलतापूर्वक लॉगिन हो गया")
            if st.button("लॉगआउट करें", key="logout_btn"):
                bot.obj.terminateSession(CLIENT_ID)
                bot.obj = None
                st.rerun()
    
    with c2:
        if not bot.running:
            if st.button("🚀 बॉट शुरू करें", key="start_btn"):
                if not st.session_state.strategy_params.get('expiry'): st.error("कृपया पहले एक एक्सपायरी चुनें")
                else:
                    trade_mode = st.session_state.strategy_params.get('trade_mode')
                    st.session_state.strategy_params['indices_to_monitor'] = ALL_INDICES if trade_mode == 'स्वचालित (सभी इंडेक्स)' else [trade_mode.split(" ")[1]]
                    bot.start(st.session_state.strategy_params)
                    st.rerun()
        else:
            if st.button("🛑 बॉट रोकें", key="stop_btn"): 
                bot.stop()
                st.rerun()
    
    with c3:
        st.markdown(f"**बॉट स्थिति:** {bot.status}")
        market_status = "🟢 खुला" if bot.market_calendar.is_trading_time() else "🔴 बंद"
        st.markdown(f"**बाजार स्थिति:** {market_status}")
        if bot.running and bot.last_checked:
            try:
                last_check_time = datetime.strptime(bot.last_checked, "%Y-%m-%d %H:%M:%S")
                time_diff = (make_naive(get_ist_time()) - last_check_time).total_seconds()
                st.markdown(f"**आखिरी जाँच:** {int(time_diff)} सेकंड पहले")
            except Exception: pass
    
    pnl_c1, pnl_c2 = st.columns(2)
    pnl_c1.metric("आज का लाइव PnL", f"₹ {bot.daily_pnl:,.2f}")
    pnl_c2.metric("आज का पेपर PnL", f"₹ {bot.paper_pnl:,.2f}")
    
    if bot.active_trade:
        st.success("🎯 सक्रिय ट्रेड चल रहा है:")
        trade_data = {
            "Symbol": bot.active_trade['symbol'], "Quantity": bot.active_trade['qty'],
            "Entry Price": f"₹{bot.active_trade['entry_price']:.2f}", "Current SL": f"₹{bot.active_trade['sl']:.2f}",
            "Current TP": f"₹{bot.active_trade['tp']:.2f}", "Entry Time": bot.active_trade['entry_time']
        }
        st.json(trade_data)
    
    # --- 2) Strategy Parameters ---
    st.header("2) स्ट्रेटेजी और रिस्क मैनेजमेंट")
    st.session_state.strategy_params['trade_direction'] = st.radio("ट्रेड की दिशा चुनें:", ['दोनों (CALL और PUT)', 'केवल CALL', 'केवल PUT'], horizontal=True)
    
    tab1, tab2, tab3 = st.tabs(["मुख्य सेटिंग्स", "रिस्क मैनेजमेंट", "उन्नत सेटिंग्स"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.strategy_params['trade_mode'] = st.radio("ट्रेडिंग इंडेक्स", [f'केवल {idx}' for idx in ALL_INDICES] + ['स्वचालित (सभी इंडेक्स)'])
            st.session_state.strategy_params['interval'] = st.selectbox("कैंडल इंटरवल", ["5min", "1min", "15min"])
            st.session_state.strategy_params['capital'] = st.number_input("कैपिटल (₹)", value=100000, min_value=10000, step=10000)
        with col2:
            st.session_state.strategy_params['ema_period'] = st.number_input("EMA पीरियड", value=25, min_value=5, max_value=50)
            st.session_state.strategy_params['rsi_period'] = st.number_input("RSI पीरियड", value=14, min_value=5, max_value=30)
            st.info("MACD पैरामीटर (12, 26, 9) मानक हैं")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.strategy_params['risk_per_trade'] = st.number_input("प्रति ट्रेड रिस्क (%)", 0.1, 5.0, value=1.0, step=0.1) / 100
            st.session_state.strategy_params['sl_offset'] = st.number_input("SL ऑफ़सेट (₹)", 1.0, 100.0, value=20.0, step=0.5)
            st.session_state.strategy_params['tp_offset'] = st.number_input("TP ऑफ़सेट (₹)", 1.0, 100.0, value=20.0, step=0.5)
        with col2:
            st.session_state.strategy_params['max_daily_loss'] = st.number_input("मैक्स डेली लॉस (₹)", 1000, 50000, value=10000, step=1000)
            st.session_state.strategy_params['min_option_price'] = st.number_input("मिनिमम ऑप्शन प्राइस (₹)", 5.0, 100.0, value=20.0, step=1.0)
            st.info(f"सुझाया गया पोजीशन साइज़ (NIFTY): {bot.position_sizer.calculate_position_size(st.session_state.strategy_params['capital'], st.session_state.strategy_params['risk_per_trade'], st.session_state.strategy_params['sl_offset'], 75)} units")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.strategy_params['start_trailing_after_points'] = st.number_input("ट्रेलिंग SL शुरू करें (पॉइंट्स)", 0.0, 100.0, value=0.0, step=1.0)
            st.session_state.strategy_params['trailing_sl_gap_points'] = st.number_input("ट्रेलिंग SL गैप (पॉइंट्स)", 0.0, 50.0, value=20.0, step=1.0)
        with col2:
            st.session_state.strategy_params['exit_on_points_gain'] = st.number_input("पॉइंट्स गेन पर एक्जिट", 0.0, 200.0, value=0.0, step=1.0)
    
    # --- 3) Expiry Selection ---
    st.header("3) एक्सपायरी चुनें")
    exp_c1, exp_c2 = st.columns([1, 2])
    with exp_c1: expiry_index_choice = st.selectbox("एक्सपायरी के लिए इंडेक्स", ALL_INDICES)
    
    if exp_c2.button(f"📅 {expiry_index_choice} की एक्सपायरी प्राप्त करें", key="get_expiry_btn"):
        if bot.obj:
            with st.spinner("एक्सपायरी लिस्ट लोड हो रही है..."):
                expiries = get_expiry_list(bot, expiry_index_choice)
                if expiries:
                    st.session_state["expiries"] = expiries
                    st.success(f"{len(expiries)} एक्सपायरी मिली")
                else:
                    st.error("कोई एक्सपायरी नहीं मिली")
        else: st.warning("पहले लॉगिन करें")
    
    if st.session_state.get("expiries"):
        st.session_state.strategy_params['expiry'] = st.selectbox("एक्सपायरी चुनें", st.session_state.get("expiries", []))
    else:
        st.session_state.strategy_params['expiry'] = '2025-09-30' # Default value to avoid error if no expiries loaded
        st.info("ऊपर बटन दबाकर एक्सपायरी लोड करें")
    
    # --- 4) Performance Analysis ---
    st.header("4) परफॉर्मेंस एनालिसिस")
    journal_tab, analytics_tab = st.tabs(["📊 ट्रेड जर्नल", "📈 एडवांस्ड एनालिटिक्स"])
    with journal_tab: display_trade_journal(bot)
    with analytics_tab: display_analytics(bot)
    
    # --- 5) Backtesting ---
    st.header("5) बैकटेस्टिंग (Trailing SL के साथ)")
    backtest_col1, backtest_col2 = st.columns([2, 1])
    with backtest_col1:
        backtest_index = st.selectbox("बैकटेस्ट के लिए इंडेक्स", ALL_INDICES, key="bt_index")
        col1, col2, col3 = st.columns(3)
        with col1: start_date = st.date_input("प्रारंभ तिथि", make_naive(get_ist_time()).date() - timedelta(days=30), key="bt_start_date")
        with col2: end_date = st.date_input("अंतिम तिथि", make_naive(get_ist_time()).date() - timedelta(days=1), key="bt_end_date")
        with col3: trade_type_backtest = st.selectbox("ट्रेड प्रकार", ["CALL", "PUT"], key="bt_trade_type")
    
    with backtest_col2:
        hypo_opt_price = st.number_input("ऑप्शन प्राइस (₹)", 1.0, value=120.0, step=0.5, key="bt_opt_price")
        trade_cost = st.number_input("ट्रेड लागत (₹)", 0.0, value=50.0, step=1.0, key="bt_trade_cost")
        slippage = st.number_input("स्लिपेज (₹)", 0.0, value=0.5, step=0.1, key="bt_slippage")
    
    if st.button("▶ बैकटेस्ट चलाएं", type="primary", key="run_backtest"):
        if not bot.obj: st.error("कृपया पहले लॉगिन करें")
        elif start_date > end_date: st.error("प्रारंभ तिथि अंतिम तिथि से पहले होनी चाहिए")
        else:
            with st.spinner(f"{backtest_index} पर बैकटेस्ट चल रहा है..."):
                all_trades = run_backtest_with_trailing_sl(
                    bot, backtest_index, start_date, end_date, st.session_state.strategy_params,
                    hypo_opt_price, trade_cost, trade_type_backtest, slippage
                )
                
                if all_trades is None: st.error("बैकटेस्ट चलाने में त्रुटि हुई")
                elif len(all_trades) == 0: st.info(f"चयनित तिथि सीमा में कोई ट्रेड नहीं मिला")
                else:
                    trades_df = pd.DataFrame(all_trades)
                    st.success(f"बैकटेस्ट पूरा! {len(all_trades)} ट्रेड मिले")
                    st.dataframe(trades_df)
                    
                    # Performance summary
                    total_pnl = trades_df['net_pnl'].sum()
                    wins = len(trades_df[trades_df['net_pnl'] > 0])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("कुल PnL", f"₹{total_pnl:,.2f}")
                    col2.metric("कुल ट्रेड्स", len(trades_df))
                    col3.metric("जीतने वाले ट्रेड्स", wins)
                    # Use explicit length check for safety (Fixes Pandas ValueError)
                    col4.metric("जीत दर", f"{(wins/len(trades_df))*100:.1f}%" if len(trades_df) > 0 else "0%") 
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    ### 🚀 v4.0 नई स्थिरता:
    - **🔐 सुरक्षित लॉगिन:** `NoneType` त्रुटियों से बचने के लिए `login()` फ़ंक्शन को पूरी तरह से सुरक्षित किया गया।
    - **⚙️ मजबूत कोडिंग:** Pandas `ValueError` से बचने के लिए सभी DataFrame चेक्स को `len(df) > 0` का उपयोग करके मजबूत बनाया गया।
    - **🔔 इंटेलिजेंट अलर्ट:** केवल क्रेडेंशियल्स उपलब्ध होने पर ही अलर्ट भेजने का प्रयास।
    """)

if __name__ == "__main__":
    main()

# </editor-fold>
