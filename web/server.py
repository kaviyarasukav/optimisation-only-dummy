import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fetch_binance_klines, fetch_tradfi_data
from engine import run_parallel_grid_search, run_detailed_single_backtest, split_in_out_of_sample, format_grid_results
from engine.graphify import generate_strategy_network_graph

app = FastAPI(title="EMA Optimization & Backtesting Suite", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class OptimizeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    fast_min: int = 5
    fast_max: int = 50
    fast_step: int = 1
    slow_min: int = 10
    slow_max: int = 200
    slow_step: int = 2
    in_sample_ratio: float = 0.7
    fee_pct: float = 0.06

@app.get("/")
def get_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/symbols")
def get_symbols():
    return {
        "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "BIOUSDT"],
        "tradfi": ["GOLD", "SILVER", "NASDAQ", "SP500"]
    }

@app.get("/api/timeframes")
def get_timeframes():
    return ["5m", "15m", "30m", "1h", "2h", "4h"]

@app.post("/api/optimize")
def optimize_strategy(req: OptimizeRequest):
    try:
        # 1. Acquire Data
        sym = req.symbol.upper()
        if sym in ["GOLD", "SILVER", "NASDAQ", "SP500"]:
            df = fetch_tradfi_data(sym, interval=req.interval)
        else:
            df = fetch_binance_klines(sym, interval=req.interval)

        if df.empty or len(df) < req.slow_max:
            raise HTTPException(status_code=400, detail=f"Insufficient data for symbol {sym} ({len(df)} bars)")

        # 2. Partition In-Sample (Training) vs Out-of-Sample (Unseen)
        in_sample_df, out_sample_df = split_in_out_of_sample(df, in_sample_ratio=req.in_sample_ratio)

        # 3. Build Parameter Ranges
        fast_range = np.arange(req.fast_min, req.fast_max + 1, req.fast_step, dtype=np.int64)
        slow_range = np.arange(req.slow_min, req.slow_max + 1, req.slow_step, dtype=np.int64)

        fee = req.fee_pct / 100.0

        # 4. Run Parallel Numba Grid Search on In-Sample Data
        in_sample_prices = in_sample_df['close'].values
        raw_results = run_parallel_grid_search(in_sample_prices, fast_range, slow_range, fee=fee)

        if len(raw_results) == 0:
            raise HTTPException(status_code=400, detail="No valid Fast < Slow EMA parameter combinations found.")

        formatted_in_sample = format_grid_results(raw_results)
        top_in_sample = formatted_in_sample[:50]  # Top 50 strategies

        # 5. Validate Top Candidate on Unseen Out-of-Sample Data
        best_candidate = top_in_sample[0]
        best_fast = best_candidate["fast_ema"]
        best_slow = best_candidate["slow_ema"]

        out_sample_prices = out_sample_df['close'].values
        out_sample_raw = run_parallel_grid_search(
            out_sample_prices, 
            np.array([best_fast]), 
            np.array([best_slow]), 
            fee=fee
        )
        out_sample_metrics = format_grid_results(out_sample_raw)[0] if len(out_sample_raw) > 0 else {}

        # 6. Detailed Backtest of Top Candidate on Full Data (for charts & trade log)
        detailed_run = run_detailed_single_backtest(df, best_fast, best_slow, fee=fee)

        # 7. Generate 2D Heatmap Surface Data (Fast vs Slow EMA Net Profit %)
        lookup_dict = {(item["fast_ema"], item["slow_ema"]): item["net_profit_pct"] for item in formatted_in_sample}
        heatmap_matrix = []
        for s_val in slow_range:
            row_vals = []
            for f_val in fast_range:
                if f_val >= s_val:
                    row_vals.append(None)
                else:
                    row_vals.append(lookup_dict.get((int(f_val), int(s_val)), None))
            heatmap_matrix.append(row_vals)

        # 8. Generate Interactive Graphify Strategy Cluster Topology Network
        graph_file = os.path.join(STATIC_DIR, "strategy_graph.html")
        generate_strategy_network_graph(top_in_sample, output_file=graph_file)

        return {
            "symbol": sym,
            "interval": req.interval,
            "total_bars": len(df),
            "in_sample_bars": len(in_sample_df),
            "out_of_sample_bars": len(out_sample_df),
            "best_parameters": {"fast_ema": best_fast, "slow_ema": best_slow},
            "in_sample_top": top_in_sample,
            "out_of_sample_validation": out_sample_metrics,
            "heatmap": {
                "x_fast": fast_range.tolist(),
                "y_slow": slow_range.tolist(),
                "z_profit_pct": heatmap_matrix
            },
            "graph_url": "/static/strategy_graph.html",
            "detailed_run": detailed_run
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
