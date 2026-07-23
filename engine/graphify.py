import os
import json
import numpy as np
from typing import List, Dict, Any

def generate_strategy_network_graph(top_strategies: List[Dict[str, Any]], output_file: str = "web/static/strategy_graph.html") -> str:
    """
    Generates an interactive Network Graph visualizing parameter clusters, 
    profitability topology, and strategy relationships using PyVis & NetworkX.
    """
    try:
        import networkx as nx
        from pyvis.network import Network

        G = nx.Graph()
        
        # Add Central Hub Node
        G.add_node("OPTIMIZATION_HUB", label="Strategy Engine", title="EMA Crossover Optimization Hub", color="#00f2fe", size=30)

        # Add strategy nodes & connections
        for idx, item in enumerate(top_strategies[:30]):
            node_id = f"EMA_{item['fast_ema']}_{item['slow_ema']}"
            profit = item['net_profit_pct']
            win_rate = item['win_rate']
            pf = item['profit_factor']
            holding = item['avg_holding_bars']

            # Node Color based on Profitability
            if profit >= 5.0:
                color = "#00e676" # Bright Green
            elif profit > 0:
                color = "#00c6ff" # Cyan/Blue
            else:
                color = "#ff5252" # Red

            # Node Size based on Profit Factor
            size = max(10, min(40, int(pf * 15)))

            title_hover = (
                f"<b>Strategy Rank #{idx+1}</b><br/>"
                f"Fast EMA: {item['fast_ema']}<br/>"
                f"Slow EMA: {item['slow_ema']}<br/>"
                f"Net Profit: <b>{profit}%</b><br/>"
                f"Win Rate: {win_rate}%<br/>"
                f"Profit Factor: {pf}<br/>"
                f"Avg Hold: {holding} bars"
            )

            G.add_node(
                node_id,
                label=f"EMA ({item['fast_ema']}/{item['slow_ema']})",
                title=title_hover,
                color=color,
                size=size
            )

            # Connect to central hub
            G.add_edge("OPTIMIZATION_HUB", node_id, weight=max(1, profit), color="rgba(255,255,255,0.15)")

        # Connect strategies that are in close parameter neighborhood (parameter clusters)
        nodes_list = [n for n in G.nodes if n != "OPTIMIZATION_HUB"]
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                n1 = nodes_list[i]
                n2 = nodes_list[j]
                f1, s1 = map(int, n1.replace("EMA_", "").split("_"))
                f2, s2 = map(int, n2.replace("EMA_", "").split("_"))

                param_dist = abs(f1 - f2) + abs(s1 - s2)
                if param_dist <= 5:
                    G.add_edge(n1, n2, title=f"Cluster Proximity (Dist={param_dist})", color="rgba(0,242,254,0.3)")

        # Initialize PyVis Network
        net = Network(height="500px", width="100%", bgcolor="#0a0e17", font_color="#ffffff", cdn_resources="remote")
        net.from_nx(G)

        # Physics options for interactive movement
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          }
        }
        """)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        net.save_graph(output_file)
        print(f"[Graphify] Generated interactive strategy network graph at {output_file}")
        return output_file

    except Exception as e:
        print(f"[Graphify Warning] PyVis graph generation fallback: {e}")
        return ""
