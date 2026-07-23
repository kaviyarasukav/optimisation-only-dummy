import pandas as pd
import numpy as np
from typing import Dict, List, Any
from data import fetch_binance_klines, fetch_tradfi_data
from engine.numba_opt import run_parallel_grid_search
from engine.splitter import split_in_out_of_sample, format_grid_results

def run_cross_asset_testing(
    training_symbol: str = "BTCUSDT",
    test_symbols: List[str] = ["ETHUSDT", "SOLUSDT", "GOLD"],
    interval: str = "1h",
    fast_p: int = 26,
    slow_p: int = 37,
    partial_ratio: float = 1.0
) -> Dict[str, Any]:
    """
    Executes Cross-Asset Testing of optimized parameters across unseen assets.
    Evaluates parameter robustness on both full and partial (e.g. 50%) datasets.
    """
    cross_results = []

    for sym in test_symbols:
        try:
            if sym.upper() in ["GOLD", "SILVER", "NASDAQ", "SP500"]:
                df = fetch_tradfi_data(sym, interval=interval)
            else:
                df = fetch_binance_klines(sym, interval=interval)

            if df.empty:
                continue

            if partial_ratio < 1.0:
                n = int(len(df) * partial_ratio)
                df = df.iloc[:n].reset_index(drop=True)

            prices = df['close'].values
            raw_res = run_parallel_grid_search(prices, np.array([fast_p]), np.array([slow_p]))
            if len(raw_res) > 0:
                formatted = format_grid_results(raw_res)[0]
                cross_results.append({
                    "test_symbol": sym,
                    "interval": interval,
                    "partial_data_ratio": partial_ratio,
                    "net_profit_pct": formatted["net_profit_pct"],
                    "win_rate": formatted["win_rate"],
                    "profit_factor": formatted["profit_factor"],
                    "max_drawdown_pct": formatted["max_drawdown_pct"],
                    "total_trades": formatted["total_trades"],
                    "avg_holding_bars": formatted["avg_holding_bars"],
                    "pass_status": "PASS" if formatted["net_profit_pct"] > 0 else "FAIL"
                })
        except Exception as e:
            print(f"[CrossTest Warning] Failed for symbol {sym}: {e}")

    total_tests = len(cross_results)
    passed_tests = sum(1 for r in cross_results if r["pass_status"] == "PASS")
    pass_rate_pct = (passed_tests / total_tests * 100.0) if total_tests > 0 else 0.0

    return {
        "training_symbol": training_symbol,
        "evaluated_parameters": {"fast_ema": fast_p, "slow_ema": slow_p},
        "pass_rate_pct": round(pass_rate_pct, 2),
        "total_cross_assets_tested": total_tests,
        "passed_assets_count": passed_tests,
        "cross_asset_results": cross_results
    }
