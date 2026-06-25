import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set dark aesthetic style to match the terminal
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e222d",
    "figure.facecolor": "#131722",
    "axes.edgecolor": "#2a2e39",
    "grid.color": "#2a2e39",
    "text.color": "#d1d4dc",
    "axes.labelcolor": "#d1d4dc",
    "xtick.color": "#d1d4dc",
    "ytick.color": "#d1d4dc"
})

def generate_backtest_report():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    static_dir = os.path.join(base_dir, "src", "frontend", "static")
    os.makedirs(static_dir, exist_ok=True)
    
    features_csv = os.path.join(data_dir, "processed_features.csv")
    df_features = pd.read_csv(features_csv) if os.path.exists(features_csv) else None
    if df_features is not None:
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        spy_df = df_features[df_features['Ticker'] == 'SPY'].sort_values('Date').copy()
        spy_df.set_index('Date', inplace=True)
        spy_df['50_MA'] = spy_df['Close'].rolling(window=50).mean()
        spy_df['150_MA'] = spy_df['Close'].rolling(window=150).mean()
    
    # 1. Generate the Equity Curves Graph per Regime
    curves_csv = os.path.join(data_dir, "ablation_equity_curves.csv")
    if os.path.exists(curves_csv):
        df_curves = pd.read_csv(curves_csv)
        df_curves['Date'] = pd.to_datetime(df_curves['Date'])
        
        regimes = df_curves['Regime'].unique()
        
        for regime in regimes:
            regime_data = df_curves[df_curves['Regime'] == regime]
            safe_regime_name = regime.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").lower()
            
            # --- EQUITY CURVE ---
            plt.figure(figsize=(14, 6))
            ax = plt.gca()
            
            strat_map = {
                'Buy & Hold': {'color': '#a1a1aa', 'label': 'Buy & Hold', 'lw': 1.5, 'alpha': 0.8},
                'Markowitz MVO (Rolling)': {'color': '#fb923c', 'label': 'Markowitz MVO', 'lw': 1.5, 'alpha': 0.8},
                'Offline CQL Only': {'color': '#60a5fa', 'label': 'Offline CQL Only', 'lw': 2.0, 'alpha': 0.9},
                'Hierarchical LLM+CQL': {'color': '#4ade80', 'label': 'Hierarchical LLM+CQL', 'lw': 2.5, 'alpha': 1.0}
            }
            
            for strat, style in strat_map.items():
                strat_data = regime_data[regime_data['Strategy'] == strat].sort_values('Date')
                if len(strat_data) == 0:
                    continue
                initial_val = strat_data['Value'].iloc[0]
                normalized = strat_data['Value'] / initial_val
                ax.plot(strat_data['Date'], normalized, label=style['label'], color=style['color'], linewidth=style['lw'], alpha=style['alpha'])
                
            ax.set_title(f"Equity Curve: {regime}", fontsize=14, pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', frameon=True, fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Ensure padding so lines/text don't get cut off
            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(static_dir, f"equity_curve_{safe_regime_name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
    
    # 2. Generate the HTML Results Table (Premium Style)
    results_csv = os.path.join(data_dir, "ablation_results.csv")
    if os.path.exists(results_csv):
        df_res = pd.read_csv(results_csv)
        df_test = df_res[df_res['Regime'].str.startswith('Test:')].copy()
        df_test['Regime'] = df_test['Regime'].str.replace('Test: ', '')
        
        html_style = """
        <style scoped>
            .premium-table-container {
                overflow-x: auto;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
                border-radius: 0.5rem;
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }
            .dataframe { 
                font-family: 'Inter', system-ui, sans-serif; 
                border-collapse: collapse; 
                width: 100%; 
                color: #d1d4dc; 
                font-size: 0.875rem; 
                text-align: left;
            }
            .dataframe th { 
                background-color: #131722; 
                padding: 1rem; 
                font-weight: 600; 
                color: #8a93a6;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.75rem;
                border-bottom: 2px solid #2a2e39;
            }
            .dataframe td { 
                padding: 1rem; 
                border-bottom: 1px solid #2a2e39; 
                white-space: nowrap;
            }
            .dataframe tr:last-child td {
                border-bottom: none;
            }
            .dataframe tr:hover { 
                background-color: #2a2e39; 
                transition: background-color 0.15s ease-in-out;
            }
            /* Highlight the main strategy */
            .dataframe tr td:nth-child(2) {
                font-weight: 500;
                color: #e0e3eb;
            }
        </style>
        """
        html_table = df_test.to_html(index=False, classes="dataframe", border=0)
        
        with open(os.path.join(static_dir, "results_table.html"), "w", encoding="utf-8") as f:
            f.write(f'<div class="premium-table-container">\n{html_style}\n{html_table}\n</div>')

if __name__ == "__main__":
    generate_backtest_report()
