import os
import sys
import argparse
import uvicorn
import numpy as np

# Force UTF-8 stdout for Windows command prompt compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from data import fetch_binance_klines, fetch_tradfi_data

from engine import run_parallel_grid_search, run_detailed_single_backtest, split_in_out_of_sample, format_grid_results

def main():
    parser = argparse.ArgumentParser(description="EMA Crossover High-Speed Brute-Force Optimization Suite")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Web Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port number")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Crypto symbol or TradFi key (e.g. BTCUSDT, ETHUSDT, GOLD)")
    parser.add_argument("--interval", type=str, default="1h", help="Timeframe interval (5m, 15m, 30m, 1h, 2h, 4h)")
    parser.add_argument("--fast-min", type=int, default=5, help="Min Fast EMA period")
    parser.add_argument("--fast-max", type=int, default=50, help="Max Fast EMA period")
    parser.add_argument("--slow-min", type=int, default=10, help="Min Slow EMA period")
    parser.add_argument("--slow-max", type=int, default=200, help="Max Slow EMA period")
    parser.add_argument("--export-csv", type=str, default=None, help="File path to save full grid results as CSV (e.g. results.csv)")
    parser.add_argument("--export-json", type=str, default=None, help="File path to save full grid results as JSON (e.g. results.json)")
    parser.add_argument("--export-trades", type=str, default=None, help="File path to save detailed trade log of best strategy as CSV (e.g. trades.csv)")
    parser.add_argument("--cascade", action="store_true", help="Run Sequential Multi-EMA Cascade Backtest")
    parser.add_argument("--cross-test", action="store_true", help="Run Cross-Asset & Cross-Timeframe Unseen Testing")
    parser.add_argument("--grid-exponent", action="store_true", help="Run Grid Position Exponentiation Test")
    parser.add_argument("--anomaly-check", action="store_true", help="Run Anomaly Detection & Trading Session Pattern Analysis")
    parser.add_argument("--sync-sheets", action="store_true", help="Sync metrics and daily balance sheet to CSV for Google Sheets")

    args = parser.parse_args()

    if args.ui or len(sys.argv) == 1:
        print("=" * 70)
        print(f"🚀 Launching Web Optimization Dashboard at http://{args.host}:{args.port}")
        print("=" * 70)
        uvicorn.run("web.server:app", host=args.host, port=args.port, reload=False)
        return

    # CLI Terminal Execution
    sym = args.symbol.upper()
    print(f"\n[CLI] Downloading & loading data for {sym} ({args.interval})...")
    if sym in ["GOLD", "SILVER", "NASDAQ", "SP500"]:
        df = fetch_tradfi_data(sym, interval=args.interval)
    else:
        df = fetch_binance_klines(sym, interval=args.interval)

    print(f"[CLI] Loaded {len(df)} historical bars.")
    in_sample_df, out_sample_df = split_in_out_of_sample(df, in_sample_ratio=0.7)
    print(f"[CLI] Split: {len(in_sample_df)} In-Sample bars | {len(out_sample_df)} Unseen Out-of-Sample bars.")

    fast_range = np.arange(args.fast_min, args.fast_max + 1, dtype=np.int64)
    slow_range = np.arange(args.slow_min, args.slow_max + 1, dtype=np.int64)

    total_combos = sum(1 for f in fast_range for s in slow_range if f < s)
    print(f"[CLI] Executing parallel Numba grid search across {total_combos} combinations...")

    raw_results = run_parallel_grid_search(in_sample_df['close'].values, fast_range, slow_range)
    ranked = format_grid_results(raw_results)

    print("\n" + "=" * 70)
    print(f"🏆 TOP 10 OPTIMIZATION CANDIDATES (In-Sample Training for {sym})")
    print("=" * 70)
    print(f"{'Rank':<5} {'Fast':<6} {'Slow':<6} {'Net Profit %':<14} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD %':<10} {'Avg Hold Bars'}")
    print("-" * 70)
    for i, item in enumerate(ranked[:10]):
        print(f"#{i+1:<4} EMA {item['fast_ema']:<2} EMA {item['slow_ema']:<2} {item['net_profit_pct']:>9.2f}%     {item['win_rate']:>6.1f}%    {item['profit_factor']:>12.2f}    {item['max_drawdown_pct']:>8.2f}%    {item['avg_holding_bars']:>8.1f}")

    best = ranked[0]
    detailed_run = run_detailed_single_backtest(df, best['fast_ema'], best['slow_ema'])
    m = detailed_run["metrics"]

    print("\n" + "=" * 90)
    print(f"📊 EXHAUSTIVE STRATEGY METRICS SUMMARY (Best Strategy: EMA {best['fast_ema']} / EMA {best['slow_ema']})")
    print("=" * 90)
    print(f"📅 Dataset Span          : {m['dataset_start_date']}  to  {m['dataset_end_date']} ({m['total_dataset_days']} days, {m['total_candles']} candles)")
    print(f"💰 Initial Capital       : ${m['initial_capital']:,.2f}  --->  Final Equity: ${m['final_equity']:,.2f}")
    print(f"📈 Net Profit & CAGR     : Net Profit: ${m['net_profit']:,.2f} ({m['net_profit_pct']:+.2f}%)  |  CAGR: {m['cagr_pct']:+.2f}%")
    print(f"💵 Gross PnL Breakdown   : Gross Profit: ${m['gross_profit']:,.2f}  |  Gross Loss: ${m['gross_loss']:,.2f}")
    print(f"🎯 Win Rate & Counts     : {m['win_rate_pct']}% ({m['winning_trades']} Wins / {m['losing_trades']} Losses out of {m['total_trades']} Trades)")
    print(f"⚖️ Profit Factor Matrix   : Raw: {m['raw_profit_factor']}  |  Net Fees: {m['net_profit_factor']}  |  Realized Fee+Slip: {m['realized_profit_factor']}")
    print(f"💵 Expectancy per Trade  : ${m['expectancy_dollars']:+.2f} ({m['expectancy_pct']:+.2f}% per trade)  |  Payoff Ratio: {m['payoff_ratio']}")
    print(f"📊 Avg Win / Loss        : Avg Win: ${m['avg_win_dollar']} ({m['avg_win_pct']}%)  |  Avg Loss: ${m['avg_loss_dollar']} ({m['avg_loss_pct']}%)")
    print(f"🚀 Max Win / Loss        : Max Win: ${m['max_win_dollar']} ({m['max_win_pct']}%)  |  Max Loss: ${m['max_loss_dollar']} ({m['max_loss_pct']}%)")
    print(f"🔥 Streaks               : Max Consecutive Wins: {m['max_consecutive_wins']}  |  Max Consecutive Losses: {m['max_consecutive_losses']}")
    print(f"⏱️ Holding Duration      : Avg: {m['avg_holding_bars']} bars | Median: {m['median_holding_bars']} bars | Max: {m['max_holding_bars']} bars")
    print(f"🌐 Market Exposure       : {m['market_exposure_pct']}% of total candles spent in active trade")
    print(f"🛡️ Max Drawdown & Ulcer  : Max DD: -{m['max_drawdown_pct']}% (-${m['max_drawdown_dollars']:,.2f})  |  Ulcer Index: {m['ulcer_index']}")
    print(f"🏛️ Institutional Ratios  : Sharpe: {m['sharpe_ratio']} | Sortino: {m['sortino_ratio']} | Calmar: {m['calmar_ratio']} | Martin: {m['martin_ratio']}")
    print(f"📈 Advanced Ratios       : Gain-to-Pain: {m['gain_to_pain_ratio']} | Tail Ratio: {m['tail_ratio']} | Recovery Factor: {m['recovery_factor']}")
    print(f"⭐ System Quality (SQN)  : {m['system_quality_number_sqn']} (Van Tharp Score)")

    print("\n" + "=" * 90)
    print(f"🔍 TESTING BEST PARAMETERS (EMA {best['fast_ema']} / EMA {best['slow_ema']}) ON UNSEEN OUT-OF-SAMPLE DATA")
    print("=" * 90)
    unseen_raw = run_parallel_grid_search(out_sample_df['close'].values, np.array([best['fast_ema']]), np.array([best['slow_ema']]))
    unseen_res = format_grid_results(unseen_raw)[0]

    print(f"Unseen Net Profit %   : {unseen_res['net_profit_pct']}%")
    print(f"Unseen Win Rate %     : {unseen_res['win_rate']}% ({unseen_res['winning_trades']}W / {unseen_res['losing_trades']}L)")
    print(f"Unseen Profit Factor   : {unseen_res['profit_factor']}")
    print(f"Unseen Max Drawdown %  : {unseen_res['max_drawdown_pct']}%")
    print(f"Unseen Avg Hold Bars   : {unseen_res['avg_holding_bars']} bars")
    print("=" * 85 + "\n")

    # Handle Exports
    import pandas as pd
    import json

    results_df = pd.DataFrame(ranked)

    if args.export_csv:
        csv_path = args.export_csv if args.export_csv.endswith(".csv") else args.export_csv + ".csv"
        results_df.to_csv(csv_path, index=False)
        print(f"💾 Full Grid Results ({len(ranked)} combinations) saved to CSV: {os.path.abspath(csv_path)}")

    if args.export_json:
        json_path = args.export_json if args.export_json.endswith(".json") else args.export_json + ".json"
        with open(json_path, "w") as f:
            json.dump({"symbol": sym, "interval": args.interval, "best_strategy_metrics": m, "results": ranked, "unseen_validation": unseen_res}, f, indent=2)
        print(f"💾 Full Results & Metrics JSON saved to: {os.path.abspath(json_path)}")

    if args.export_trades:
        trades_df = pd.DataFrame(detailed_run["trades"])
        trades_csv_path = args.export_trades if args.export_trades.endswith(".csv") else args.export_trades + ".csv"
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"💾 Trade Log ({len(trades_df)} trades) saved to CSV: {os.path.abspath(trades_csv_path)}")

    # Extended Modules Execution
    if args.cascade:
        from engine.cascade import run_sequential_cascade_backtest
        print("\n" + "=" * 85)
        print("🌊 RUNNING SEQUENTIAL MULTI-EMA CASCADE BACKTEST (Early Warning Exit Protection)")
        print("=" * 85)
        casc_res = run_sequential_cascade_backtest(df, ema_periods=[8, 13, 21, 34, 55, 89, 200])
        print(f"Cascade Net Profit %      : {casc_res['metrics']['net_profit_pct']}%")
        print(f"Cascade Win Rate %        : {casc_res['metrics']['win_rate_pct']}%")
        print(f"Early Warning Exits       : {casc_res['metrics']['early_warning_exits_triggered']} times (Prevented Drawdown)")
        print("=" * 85 + "\n")

    if args.cross_test:
        from engine.cross_testing import run_cross_asset_testing
        print("\n" + "=" * 85)
        print("🔀 RUNNING CROSS-ASSET UNSEEN GENERALIZATION TEST")
        print("=" * 85)
        cross_res = run_cross_asset_testing(training_symbol=sym, test_symbols=["ETHUSDT", "SOLUSDT", "GOLD"], interval=args.interval, fast_p=best['fast_ema'], slow_p=best['slow_ema'])
        print(f"Cross-Asset Pass Rate    : {cross_res['pass_rate_pct']}% ({cross_res['passed_assets_count']}/{cross_res['total_cross_assets_tested']} assets passed)")
        for cr in cross_res["cross_asset_results"]:
            print(f"  • {cr['test_symbol']:<10} Net Profit: {cr['net_profit_pct']:>6.2f}% | Win Rate: {cr['win_rate']:>5.1f}% | Profit Factor: {cr['profit_factor']:>5.2f} [{cr['pass_status']}]")
        print("=" * 85 + "\n")

    if args.grid_exponent:
        from engine.grid_exponent import run_grid_exponentiation_test
        print("\n" + "=" * 85)
        print("📈 RUNNING GRID EXPONENTIATION POSITION SCALING TEST")
        print("=" * 85)
        grid_res = run_grid_exponentiation_test(df, fast_p=best['fast_ema'], slow_p=best['slow_ema'])
        print(f"Grid Net Profit %        : {grid_res['metrics']['net_profit_pct']}%")
        print(f"Grid Win Rate %          : {grid_res['metrics']['win_rate_pct']}%")
        print(f"Grid Transition Prob (%) : {grid_res['grid_level_transition_probabilities_pct']}")
        print("=" * 85 + "\n")

    if args.anomaly_check:
        from engine.anomaly_detector import run_anomaly_and_pattern_detection
        print("\n" + "=" * 85)
        print("🚨 RUNNING ANOMALY DETECTION & TRADING SESSION PATTERN ANALYSIS")
        print("=" * 85)
        anom_res = run_anomaly_and_pattern_detection(detailed_run["trades"], df)
        print(f"Outlier Trades Found     : {anom_res['anomaly_detection']['total_anomalies_found']}")
        print(f"Session Breakdown        : {anom_res['pattern_recognition']['session_performance']}")
        print(f"Value at Risk (VaR 95%)  : {anom_res['statistical_probability_models']['value_at_risk_95_pct']}%")
        print(f"Expected Shortfall (CVaR): {anom_res['statistical_probability_models']['expected_shortfall_cvar_95_pct']}%")
        print("=" * 85 + "\n")

    if args.sync_sheets:
        from portfolio.sheets_sync import sync_balance_sheet_and_metrics
        sheet_path = sync_balance_sheet_and_metrics(m)
        print(f"📊 Metrics and Balance Sheet synced to CSV for Google Sheets: {sheet_path}")

if __name__ == "__main__":
    main()
