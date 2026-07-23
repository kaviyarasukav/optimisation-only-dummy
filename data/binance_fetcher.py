import os
import time
import requests
import pandas as pd
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

INTERVAL_MAP = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d"
}

def fetch_binance_klines(symbol: str, interval: str = "1h", total_bars: int = 5000, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance Public API.
    Auto-paginates up to `total_bars` items and caches locally.
    """
    symbol = symbol.upper()
    if not symbol.endswith("USDT") and not symbol.endswith("BUSD"):
        symbol = symbol + "USDT"

    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")
    
    # Return cache if valid and fresh (< 6 hours old)
    if use_cache and os.path.exists(cache_file):
        file_age_hrs = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if file_age_hrs < 6.0:
            df = pd.read_parquet(cache_file)
            if len(df) >= total_bars * 0.8:
                print(f"[Data] Loaded cached data for {symbol} ({interval}) - {len(df)} bars")
                return df

    base_url = "https://api.binance.com/api/v3/klines"
    klines = []
    end_time = None

    limit_per_req = 1000
    needed_requests = (total_bars + limit_per_req - 1) // limit_per_req

    print(f"[Data] Fetching {symbol} ({interval}) from Binance REST API...")

    for _ in range(needed_requests):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit_per_req
        }
        if end_time:
            params["endTime"] = end_time

        try:
            res = requests.get(base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break

            klines = data + klines
            end_time = data[0][0] - 1  # timestamp before oldest returned kline

            if len(data) < limit_per_req:
                break
            time.sleep(0.1)  # Rate limiting courtesy
        except Exception as e:
            print(f"[Data Warning] Error fetching {symbol} batch: {e}")
            break

    if not klines:
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)
        raise ValueError(f"Could not fetch kline data for symbol {symbol}")

    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)

    # Save cache
    try:
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        # Fallback to CSV if parquet engine isn't ready
        df.to_csv(cache_file.replace('.parquet', '.csv'), index=False)

    print(f"[Data] Downloaded {len(df)} bars for {symbol} ({interval})")
    return df
