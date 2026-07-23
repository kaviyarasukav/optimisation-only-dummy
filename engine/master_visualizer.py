import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "static"))
os.makedirs(STATIC_DIR, exist_ok=True)

def generate_multi_asset_visualizations(master_csv_path: str = "reports/master_all_assets_optimization.csv") -> str:
    """
    Generates rich multi-dimensional interactive charts & visual data representations:
    1. Cross-Asset x Timeframe Return Matrix Heatmap
    2. Multi-Asset Performance Comparison (Profit % & Win Rate)
    3. Risk vs Return Scatter Plot (Net Profit vs Max Drawdown, bubble size = Profit Factor)
    4. In-Sample vs Unseen Out-of-Sample Generalization Comparison
    """
    if not os.path.exists(master_csv_path):
        print(f"[Visualizer Warning] Master CSV not found at {master_csv_path}")
        return ""

    df = pd.read_csv(master_csv_path)

    # 1. Create Subplot Dashboard Layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "🔥 Asset x Timeframe Profitability Heatmap (%)",
            "📊 Multi-Asset Best Net Profit (%) by Timeframe",
            "🎯 Risk vs Return (Net Profit % vs Max Drawdown %)",
            "🛡️ In-Sample vs Unseen Out-of-Sample Return Comparison"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # --- Plot 1: Heatmap Pivot (Asset vs Timeframe) ---
    pivot_df = df.pivot_table(index="symbol", columns="timeframe", values="in_sample_profit_pct", aggfunc="first")
    # Ensure standard timeframe column ordering
    tf_order = [tf for tf in ["5m", "15m", "30m", "1h", "2h", "4h"] if tf in pivot_df.columns]
    pivot_df = pivot_df[tf_order]

    heatmap_trace = go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="Profit %", len=0.45, y=0.8),
        hoverongaps=False
    )
    fig.add_trace(heatmap_trace, row=1, col=1)

    # --- Plot 2: Multi-Asset Bar Chart ---
    symbols = df["symbol"].unique()
    for sym in symbols:
        sub_df = df[df["symbol"] == sym]
        fig.add_trace(
            go.Bar(
                x=sub_df["timeframe"],
                y=sub_df["in_sample_profit_pct"],
                name=sym,
                text=sub_df["in_sample_profit_pct"].apply(lambda v: f"{v:+.1f}%"),
                textposition="auto"
            ),
            row=1, col=2
        )

    # --- Plot 3: Risk vs Return Scatter Bubble Chart ---
    scatter_trace = go.Scatter(
        x=df["in_sample_max_dd_pct"],
        y=df["in_sample_profit_pct"],
        mode="markers+text",
        text=df.apply(lambda r: f"{r['symbol']} ({r['timeframe']})", axis=1),
        textposition="top center",
        marker=dict(
            size=df["in_sample_profit_factor"] * 10,
            color=df["in_sample_win_rate_pct"],
            colorscale="Electric",
            showscale=True,
            colorbar=dict(title="Win Rate %", len=0.45, y=0.2)
        ),
        hovertemplate="<b>%{text}</b><br/>Max DD: %{x}%<br/>Profit: %{y}%<extra></extra>"
    )
    fig.add_trace(scatter_trace, row=2, col=1)

    # --- Plot 4: In-Sample vs Unseen Out-of-Sample Comparison ---
    fig.add_trace(
        go.Bar(x=df.apply(lambda r: f"{r['symbol']} {r['timeframe']}", axis=1), y=df["in_sample_profit_pct"], name="In-Sample Profit %", marker_color="#00c6ff"),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=df.apply(lambda r: f"{r['symbol']} {r['timeframe']}", axis=1), y=df["unseen_profit_pct"], name="Unseen Out-of-Sample %", marker_color="#ff5252"),
        row=2, col=2
    )

    # Layout Styling
    fig.update_layout(
        title=dict(text="🌐 Multi-Asset & Multi-Timeframe Optimization Visual Analytics Hub", font=dict(size=22, color="#ffffff")),
        paper_bgcolor="#0a0e17",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        height=900,
        showlegend=True,
        barmode="group"
    )

    # Axis updates
    fig.update_xaxes(title_text="Timeframe", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Asset Symbol", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(title_text="Timeframe", row=1, col=2, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Net Profit %", row=1, col=2, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(title_text="Max Drawdown % (Risk)", row=2, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Net Profit % (Return)", row=2, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(title_text="Asset & Timeframe", row=2, col=2, gridcolor="rgba(255,255,255,0.05)", tickangle=-45)
    fig.update_yaxes(title_text="Return %", row=2, col=2, gridcolor="rgba(255,255,255,0.05)")

    output_html = os.path.join(STATIC_DIR, "master_visuals.html")
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"[Master Visualizer] Created multi-asset visualization dashboard at {output_html}")
    return output_html
