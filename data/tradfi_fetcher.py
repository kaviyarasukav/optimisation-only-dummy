import os
import time
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOL_MAP = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "NASDAQ": "^NDX",
    "SP500": "^GSPC"
}

def fetch_tradfi_data(asset_key: str, interval: str = "1h", period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch Gold, Silver, NASDAQ, or S&P500 data using yfinance if available.
    Fallback gracefully if network or rate limit fails.
    """
    ticker = SYMBOL_MAP.get(asset_key.upper(), asset_key)
    safe_name = asset_key.upper()
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}_{interval}.parquet")

    if use_cache and os.path.exists(cache_file):
        file_age_hrs = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if file_age_hrs < 12.0:
            print(f"[Data] Loaded cached TradFi data for {safe_name} ({interval})")
            return pd.read_parquet(cache_file)

    try:
        import yfinance as yf
        print(f"[Data] Fetching {safe_name} ({ticker}) via yfinance...")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]

        data = data.reset_index()
        date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
        data = data.rename(columns={date_col: 'timestamp'})

        # Convert to tz-naive timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)

        df = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)

        df = df.dropna().sort_values('timestamp').reset_index(drop=True)
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            df.to_csv(cache_file.replace('.parquet', '.csv'), index=False)
        return df

    except Exception as e:
        print(f"[Data Warning] TradFi fetch failed for {safe_name}: {e}")
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)
        # Return empty DataFrame gracefully
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
