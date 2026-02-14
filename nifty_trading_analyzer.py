"""
Nifty Option Chain & Technical Analysis for Day Trading
FINAL WORKING VERSION - Using proven NSE API fetch with curl_cffi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import yaml
import os
import logging

warnings.filterwarnings('ignore')

class NiftyAnalyzer:
    def __init__(self, config_path='config.yml'):
        """Initialize analyzer with YAML configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # IST timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        self.nifty_symbol = "^NSEI"
        
    def get_ist_time(self):
        """Get current time in IST"""
        return datetime.now(self.ist)
    
    def format_ist_time(self, dt=None):
        """Format datetime in IST"""
        if dt is None:
            dt = self.get_ist_time()
        elif dt.tzinfo is None:
            dt = self.ist.localize(dt)
        else:
            dt = dt.astimezone(self.ist)
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")
    
    def get_upcoming_expiry_tuesday(self):
        """Calculate nearest Tuesday expiry - EXACT from working code"""
        now = datetime.now(self.ist)
        current_weekday = now.weekday()
        
        # Tuesday = 1
        if current_weekday == 1:  # Today is Tuesday
            if now.hour < 15 or (now.hour == 15 and now.minute < 30):
                expiry_date = now
            else:
                expiry_date = now + timedelta(days=7)
        elif current_weekday == 0:  # Monday
            expiry_date = now + timedelta(days=1)
        else:
            days_ahead = (8 - current_weekday) % 7
            expiry_date = now + timedelta(days=days_ahead)
        
        return expiry_date.strftime('%d-%b-%Y')
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}")
            print("Using default configuration...")
            return self.get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'email': {
                'recipient': 'your_email@gmail.com',
                'sender': 'your_email@gmail.com',
                'app_password': 'your_app_password',
                'subject_prefix': 'Nifty Day Trading Report',
                'send_on_failure': False
            },
            'technical': {
                'timeframe': '1h',
                'period': '6mo',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'ema_short': 20,
                'ema_long': 50,
                'num_support_levels': 2,
                'num_resistance_levels': 2,
                'momentum_threshold_strong': 0.5,
                'momentum_threshold_moderate': 0.2
            },
            'option_chain': {
                'pcr_bullish': 1.0,
                'pcr_very_bullish': 1.2,
                'pcr_bearish': 1.0,
                'pcr_very_bearish': 0.8,
                'strike_range': 500,
                'min_oi': 100000,
                'top_strikes_count': 5
            },
            'recommendation': {
                'strong_buy_threshold': 3,
                'buy_threshold': 1,
                'sell_threshold': -1,
                'strong_sell_threshold': -3,
                'momentum_5h_weight': 2,
                'momentum_1h_weight': 1
            },
            'report': {
                'title': 'NIFTY DAY TRADING ANALYSIS (1H)',
                'save_local': True,
                'local_dir': './reports',
                'filename_format': 'nifty_analysis_%Y%m%d_%H%M%S.html'
            },
            'data_source': {
                'option_chain_source': 'nse',
                'technical_source': 'yahoo',
                'max_retries': 3,
                'retry_delay': 2,
                'timeout': 10,
                'fallback_to_sample': True
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': './logs/nifty_analyzer.log',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'advanced': {
                'verbose': True,
                'debug': False,
                'validate_data': True,
                'min_data_points': 100,
                'use_momentum_filter': True
            }
        }
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        self.logger = logging.getLogger('NiftyAnalyzer')
        self.logger.setLevel(level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', './logs/nifty_analyzer.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def fetch_option_chain(self):
        """Fetch Nifty option chain - EXACT WORKING METHOD from your code"""
        if self.config['data_source']['option_chain_source'] == 'sample':
            self.logger.info("Using sample option chain data")
            return None, None
        
        symbol = "NIFTY"
        selected_expiry = self.get_upcoming_expiry_tuesday()
        
        # EXACT URL from your working code
        api_url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={selected_expiry}"
        base_url = "https://www.nseindia.com/"
        
        # EXACT headers from your working code
        headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
        
        max_retries = self.config['data_source']['max_retries']
        
        # Try curl_cffi first, fallback to requests
        try:
            from curl_cffi import requests as curl_requests
            USE_CURL = True
            self.logger.info("Using curl_cffi for NSE fetch")
        except ImportError:
            USE_CURL = False
            self.logger.warning("curl_cffi not available, using requests (may fail). Install: pip install curl-cffi")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching option chain for {selected_expiry} (attempt {attempt + 1}/{max_retries})...")
                self.logger.info(f"URL: {api_url}")
                
                if USE_CURL:
                    # Your EXACT working method with curl_cffi
                    session = curl_requests.Session()
                    session.get(base_url, headers=headers, impersonate="chrome", timeout=15)
                    import time
                    time.sleep(1)
                    response = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
                else:
                    # Fallback to regular requests
                    import time
                    session = requests.Session()
                    session.get(base_url, headers=headers, timeout=15)
                    time.sleep(1)
                    response = session.get(api_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    json_data = response.json()
                    data = json_data.get('records', {}).get('data', [])
                    
                    if not data:
                        self.logger.warning(f"No data for {selected_expiry}")
                        continue
                    
                    current_price = json_data.get('records', {}).get('underlyingValue', 0)
                    
                    calls_data = []
                    puts_data = []
                    
                    for item in data:
                        strike = item.get('strikePrice', 0)
                        
                        if 'CE' in item:
                            ce = item['CE']
                            calls_data.append({
                                'Strike': strike,
                                'Call_OI': ce.get('openInterest', 0),
                                'Call_Chng_OI': ce.get('changeinOpenInterest', 0),
                                'Call_Volume': ce.get('totalTradedVolume', 0),
                                'Call_IV': ce.get('impliedVolatility', 0),
                                'Call_LTP': ce.get('lastPrice', 0)
                            })
                        
                        if 'PE' in item:
                            pe = item['PE']
                            puts_data.append({
                                'Strike': strike,
                                'Put_OI': pe.get('openInterest', 0),
                                'Put_Chng_OI': pe.get('changeinOpenInterest', 0),
                                'Put_Volume': pe.get('totalTradedVolume', 0),
                                'Put_IV': pe.get('impliedVolatility', 0),
                                'Put_LTP': pe.get('lastPrice', 0)
                            })
                    
                    calls_df = pd.DataFrame(calls_data)
                    puts_df = pd.DataFrame(puts_data)
                    
                    oc_df = pd.merge(calls_df, puts_df, on='Strike', how='outer')
                    oc_df = oc_df.fillna(0)
                    oc_df = oc_df.sort_values('Strike')
                    
                    self.logger.info(f"✅ Option chain fetched: {len(oc_df)} strikes | Spot: ₹{current_price} | Expiry: {selected_expiry}")
                    return oc_df, current_price
                else:
                    self.logger.warning(f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(self.config['data_source'].get('retry_delay', 2))
        
        if self.config['data_source']['fallback_to_sample']:
            self.logger.warning("All attempts failed, using sample data")
        
        return None, None
    
    def get_top_strikes_by_oi(self, oc_df, spot_price):
        """Get top 5 strikes by Open Interest for CE and PE"""
        if oc_df is None or oc_df.empty:
            return {'top_ce_strikes': [], 'top_pe_strikes': []}
        
        top_count = self.config['option_chain'].get('top_strikes_count', 5)
        
        ce_data = oc_df[oc_df['Call_OI'] > 0].copy()
        ce_data = ce_data.sort_values('Call_OI', ascending=False).head(top_count)
        top_ce_strikes = []
        for _, row in ce_data.iterrows():
            strike_type = 'ITM' if row['Strike'] < spot_price else ('ATM' if row['Strike'] == spot_price else 'OTM')
            top_ce_strikes.append({
                'strike': row['Strike'],
                'oi': int(row['Call_OI']),
                'ltp': row['Call_LTP'],
                'iv': row['Call_IV'],
                'type': strike_type
            })
        
        pe_data = oc_df[oc_df['Put_OI'] > 0].copy()
        pe_data = pe_data.sort_values('Put_OI', ascending=False).head(top_count)
        top_pe_strikes = []
        for _, row in pe_data.iterrows():
            strike_type = 'ITM' if row['Strike'] > spot_price else ('ATM' if row['Strike'] == spot_price else 'OTM')
            top_pe_strikes.append({
                'strike': row['Strike'],
                'oi': int(row['Put_OI']),
                'ltp': row['Put_LTP'],
                'iv': row['Put_IV'],
                'type': strike_type
            })
        
        return {'top_ce_strikes': top_ce_strikes, 'top_pe_strikes': top_pe_strikes}
    
    def analyze_option_chain(self, oc_df, spot_price):
        """Analyze option chain for trading signals"""
        if oc_df is None or oc_df.empty:
            self.logger.warning("No option chain data, using sample analysis")
            return self.get_sample_oc_analysis()
        
        config = self.config['option_chain']
        
        total_call_oi = oc_df['Call_OI'].sum()
        total_put_oi = oc_df['Put_OI'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        oc_df['Call_Pain'] = oc_df.apply(
            lambda row: row['Call_OI'] * max(0, spot_price - row['Strike']), axis=1
        )
        oc_df['Put_Pain'] = oc_df.apply(
            lambda row: row['Put_OI'] * max(0, row['Strike'] - spot_price), axis=1
        )
        oc_df['Total_Pain'] = oc_df['Call_Pain'] + oc_df['Put_Pain']
        
        max_pain_strike = oc_df.loc[oc_df['Total_Pain'].idxmax(), 'Strike']
        
        strike_range = config['strike_range']
        nearby_strikes = oc_df[
            (oc_df['Strike'] >= spot_price - strike_range) & 
            (oc_df['Strike'] <= spot_price + strike_range)
        ].copy()
        
        num_resistance = self.config['technical']['num_resistance_levels']
        num_support = self.config['technical']['num_support_levels']
        
        resistance_df = nearby_strikes[nearby_strikes['Strike'] > spot_price].nlargest(num_resistance, 'Call_OI')
        resistances = resistance_df['Strike'].tolist()
        
        support_df = nearby_strikes[nearby_strikes['Strike'] < spot_price].nlargest(num_support, 'Put_OI')
        supports = support_df['Strike'].tolist()
        
        total_call_buildup = oc_df['Call_Chng_OI'].sum()
        total_put_buildup = oc_df['Put_Chng_OI'].sum()
        
        avg_call_iv = oc_df['Call_IV'].mean()
        avg_put_iv = oc_df['Put_IV'].mean()
        
        top_strikes = self.get_top_strikes_by_oi(oc_df, spot_price)
        
        return {
            'pcr': round(pcr, 2),
            'max_pain': max_pain_strike,
            'resistances': sorted(resistances, reverse=True),
            'supports': sorted(supports, reverse=True),
            'call_buildup': total_call_buildup,
            'put_buildup': total_put_buildup,
            'avg_call_iv': round(avg_call_iv, 2),
            'avg_put_iv': round(avg_put_iv, 2),
            'oi_sentiment': 'Bullish' if total_put_buildup > total_call_buildup else 'Bearish',
            'top_ce_strikes': top_strikes['top_ce_strikes'],
            'top_pe_strikes': top_strikes['top_pe_strikes']
        }
    
    def get_sample_oc_analysis(self):
        """Return sample option chain analysis"""
        return {
            'pcr': 1.15,
            'max_pain': 24500,
            'resistances': [24600, 24650],
            'supports': [24400, 24350],
            'call_buildup': 5000000,
            'put_buildup': 6000000,
            'avg_call_iv': 15.5,
            'avg_put_iv': 16.2,
            'oi_sentiment': 'Bullish',
            'top_ce_strikes': [
                {'strike': 24500, 'oi': 5000000, 'ltp': 120, 'iv': 16.5, 'type': 'ATM'},
                {'strike': 24600, 'oi': 4500000, 'ltp': 80, 'iv': 15.8, 'type': 'OTM'},
            ],
            'top_pe_strikes': [
                {'strike': 24500, 'oi': 5500000, 'ltp': 110, 'iv': 16.0, 'type': 'ATM'},
                {'strike': 24400, 'oi': 5000000, 'ltp': 75, 'iv': 15.5, 'type': 'OTM'},
            ]
        }
    
    def run_analysis(self):
        """Run complete analysis"""
        self.logger.info("🚀 Starting Nifty Analysis with proven NSE fetch...")
        self.logger.info("=" * 60)
        
        oc_df, spot_price = self.fetch_option_chain()
        
        if oc_df is not None and spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
            
            # Print Top 5 Call Options
            print("\n" + "="*70)
            print("📞 TOP 5 CALL OPTIONS (by Open Interest)")
            print("="*70)
            print(f"{'#':<4} {'Strike':<10} {'OI':<15} {'LTP':<10} {'IV':<10} {'Type':<8}")
            print("-" * 70)
            for i, strike in enumerate(oc_analysis['top_ce_strikes'], 1):
                print(f"{i:<4} ₹{strike['strike']:<9,.0f} {strike['oi']:<15,} ₹{strike['ltp']:<9.2f} {strike['iv']:<9.1f}% {strike['type']:<8}")
            
            # Print Top 5 Put Options
            print("\n" + "="*70)
            print("📉 TOP 5 PUT OPTIONS (by Open Interest)")
            print("="*70)
            print(f"{'#':<4} {'Strike':<10} {'OI':<15} {'LTP':<10} {'IV':<10} {'Type':<8}")
            print("-" * 70)
            for i, strike in enumerate(oc_analysis['top_pe_strikes'], 1):
                print(f"{i:<4} ₹{strike['strike']:<9,.0f} {strike['oi']:<15,} ₹{strike['ltp']:<9.2f} {strike['iv']:<9.1f}% {strike['type']:<8}")
            
            print("\n" + "="*70)
            print(f"📊 NIFTY Spot: ₹{spot_price:,.2f}")
            print(f"📊 PCR Ratio: {oc_analysis['pcr']}")
            print(f"📊 Max Pain: ₹{oc_analysis['max_pain']:,.0f}")
            print(f"📊 OI Sentiment: {oc_analysis['oi_sentiment']}")
            print("="*70)
        else:
            spot_price = 25796
            oc_analysis = self.get_sample_oc_analysis()
            print("\n⚠️ Using sample data (NSE fetch failed)")
        
        self.logger.info("=" * 60)
        
        return {
            'oc_analysis': oc_analysis,
            'spot_price': spot_price
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 NIFTY OPTION CHAIN ANALYZER - WORKING VERSION")
    print("="*70)
    print("📌 For best results: pip install curl-cffi")
    print("="*70 + "\n")
    
    analyzer = NiftyAnalyzer(config_path='config.yml')
    result = analyzer.run_analysis()
    
    print(f"\n✅ Analysis Complete!\n")
