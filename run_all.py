import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Force UTF-8 stdout for Windows command prompt compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from data import fetch_binance_klines, fetch_tradfi_data
from engine import run_parallel_grid_search, run_detailed_single_backtest, split_in_out_of_sample, format_grid_results

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "BIOUSDT"]
TRADFI_SYMBOLS = ["GOLD", "SILVER", "NASDAQ", "SP500"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h"]

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_master_all_optimization(fast_min=5, fast_max=30, slow_min=10, slow_max=100):
    print("=" * 85)
    print("🚀 LAUNCHING MASTER ALL-WAY OPTIMIZATION Across All Assets & Timeframes")
    print("=" * 85)
    print(f"Assets: {len(CRYPTO_SYMBOLS + TRADFI_SYMBOLS)} ({', '.join(CRYPTO_SYMBOLS + TRADFI_SYMBOLS)})")
    print(f"Timeframes: {len(TIMEFRAMES)} ({', '.join(TIMEFRAMES)})")
    print(f"EMA Ranges: Fast [{fast_min}..{fast_max}] x Slow [{slow_min}..{slow_max}]")
    print("=" * 85 + "\n")

    master_leaderboard = []
    t_start_global = time.time()

    fast_range = np.arange(fast_min, fast_max + 1, dtype=np.int64)
    slow_range = np.arange(slow_min, slow_max + 1, dtype=np.int64)

    total_runs = (len(CRYPTO_SYMBOLS) + len(TRADFI_SYMBOLS)) * len(TIMEFRAMES)
    run_count = 0

    all_symbols = CRYPTO_SYMBOLS + TRADFI_SYMBOLS

    for sym in all_symbols:
        for tf in TIMEFRAMES:
            run_count += 1
            print(f"[{run_count}/{total_runs}] Processing {sym} ({tf})...", end=" ", flush=True)
            t0 = time.time()

            try:
                if sym in TRADFI_SYMBOLS:
                    df = fetch_tradfi_data(sym, interval=tf, use_cache=True)
                else:
                    df = fetch_binance_klines(sym, interval=tf, total_bars=3000, use_cache=True)

                if df.empty or len(df) < slow_max:
                    print(f"⚠️ Skipped (Insufficient bars: {len(df)})")
                    continue

                in_sample_df, out_sample_df = split_in_out_of_sample(df, in_sample_ratio=0.7)
                raw_results = run_parallel_grid_search(in_sample_df['close'].values, fast_range, slow_range)
                ranked = format_grid_results(raw_results)

                if not ranked:
                    print("⚠️ No valid strategy results.")
                    continue

                best = ranked[0]

                # Validate top strategy on Unseen Out-of-Sample data
                unseen_raw = run_parallel_grid_search(out_sample_df['close'].values, np.array([best['fast_ema']]), np.array([best['slow_ema']]))
                unseen_res = format_grid_results(unseen_raw)[0] if len(unseen_raw) > 0 else {}

                elapsed = time.time() - t0
                print(f"✓ Done in {elapsed:.2f}s | Top: EMA {best['fast_ema']}/{best['slow_ema']} (+{best['net_profit_pct']}%)")

                # Store in master leaderboard
                master_leaderboard.append({
                    "symbol": sym,
                    "timeframe": tf,
                    "best_fast_ema": best['fast_ema'],
                    "best_slow_ema": best['slow_ema'],
                    "in_sample_profit_pct": best['net_profit_pct'],
                    "in_sample_win_rate_pct": best['win_rate'],
                    "in_sample_profit_factor": best['profit_factor'],
                    "in_sample_max_dd_pct": best['max_drawdown_pct'],
                    "in_sample_avg_hold_bars": best['avg_holding_bars'],
                    "unseen_profit_pct": unseen_res.get('net_profit_pct', 0.0),
                    "unseen_win_rate_pct": unseen_res.get('win_rate', 0.0),
                    "unseen_profit_factor": unseen_res.get('profit_factor', 0.0),
                    "unseen_max_dd_pct": unseen_res.get('max_drawdown_pct', 0.0),
                    "total_candles": len(df)
                })

                # Save asset-specific CSV report
                asset_df = pd.DataFrame(ranked)
                asset_csv = os.path.join(REPORTS_DIR, f"{sym}_{tf}_optimization.csv")
                asset_df.to_csv(asset_csv, index=False)

            except Exception as e:
                print(f"❌ Error: {e}")

    # Generate Master All-Way Summary Reports
    total_elapsed = time.time() - t_start_global
    master_df = pd.DataFrame(master_leaderboard)
    master_df.sort_values(by="in_sample_profit_pct", ascending=False, inplace=True)

    master_csv = os.path.join(REPORTS_DIR, "master_all_assets_optimization.csv")
    master_df.to_csv(master_csv, index=False)

    # Generate Multi-Asset & Multi-Timeframe Interactive Dashboard Visuals
    from engine.master_visualizer import generate_multi_asset_visualizations
    vis_path = generate_multi_asset_visualizations(master_csv)

    print("\n" + "=" * 85)
    print(f"🎉 ALL-WAY MASTER OPTIMIZATION COMPLETE in {total_elapsed:.2f} seconds!")
    print("=" * 85)
    print(f"📁 Master Summary CSV saved to      : {os.path.abspath(master_csv)}")
    print(f"🌐 Interactive Visual Dashboard    : {os.path.abspath(vis_path)}")
    print(f"📁 Individual Asset Reports saved to: {os.path.abspath(REPORTS_DIR)}")
    print("=" * 85 + "\n")

    print(master_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master All-Way Optimization Script")
    parser.add_argument("--fast-min", type=int, default=5)
    parser.add_argument("--fast-max", type=int, default=30)
    parser.add_argument("--slow-min", type=int, default=10)
    parser.add_argument("--slow-max", type=int, default=100)
    args = parser.parse_args()

    run_master_all_optimization(args.fast_min, args.fast_max, args.slow_min, args.slow_max)
