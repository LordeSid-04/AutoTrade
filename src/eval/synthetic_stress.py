"""
Synthetic Stress Scenarios Evaluator

Programmatically perturbs historical data to simulate unprecedented market shocks.
Compares the RL policy against Equal-Weight Baseline in these stress scenarios.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import d3rlpy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.rl.custom_encoders import register_custom_encoders
register_custom_encoders()

from src.env.trading_env import MultiAssetTradingEnv
from src.eval.baselines import compute_metrics

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"]
SEEDS = [42, 43, 44, 45, 46]

def load_data():
    processed_data_path = os.path.join(DATA_DIR, "processed_features.csv")
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data not found at {processed_data_path}.")
        sys.exit(1)
    df = pd.read_csv(processed_data_path)
    # We will just take a stable 1-year slice and perturb it
    mask = (df['Date'] >= "2018-01-01") & (df['Date'] <= "2018-12-31")
    return df.loc[mask].copy()

def simulate_flash_crash(df):
    """
    Simulates a sudden, massive drop in equities over a 5-day window, paired with a massive VIX spike.
    """
    perturbed_df = df.copy()
    dates = sorted(perturbed_df['Date'].unique())
    crash_start_idx = len(dates) // 2
    crash_dates = dates[crash_start_idx:crash_start_idx+5]
    
    # Identify equity tickers vs safe havens
    safe_havens = ["GLD", "TLT", "XLP", "XLU"]
    
    for date in crash_dates:
        for ticker in TICKERS:
            mask = (perturbed_df['Date'] == date) & (perturbed_df['Ticker'] == ticker)
            if ticker not in safe_havens:
                perturbed_df.loc[mask, 'Log_Return'] -= 0.05  # -5% daily for 5 days = -25% shock
                perturbed_df.loc[mask, 'Parkinson_Vol'] *= 2.5
            else:
                perturbed_df.loc[mask, 'Log_Return'] += 0.02  # Flight to safety (+10% over 5 days)
                
        # Spike the broad VIX proxy feature (assuming it exists, we'll just spike general vol)
        
    return perturbed_df, "Flash Crash 2.0 (-25% Equities in 5 days)"

def simulate_stagflation_shock(df):
    """
    Simulates a sustained 3-month period of negative equity returns and crashing bonds (rates rising), 
    with only commodities (GLD/Energy) surviving.
    """
    perturbed_df = df.copy()
    dates = sorted(perturbed_df['Date'].unique())
    stag_dates = dates[30:90] # 60 days
    
    survivors = ["GLD", "XLE"]
    
    for date in stag_dates:
        for ticker in TICKERS:
            mask = (perturbed_df['Date'] == date) & (perturbed_df['Ticker'] == ticker)
            if ticker not in survivors:
                # Slow bleed for equities and bonds alike
                perturbed_df.loc[mask, 'Log_Return'] -= 0.003
            else:
                # Commodities/Energy gain
                perturbed_df.loc[mask, 'Log_Return'] += 0.005
                
    return perturbed_df, "Stagflation Shock (Equities & Bonds bleed, Energy/Gold rally)"

def run_rl_multi_seed(env_df, algo_name):
    all_metrics = []
    
    for seed in SEEDS:
        model_path = os.path.join(DATA_DIR, f"{algo_name.lower()}_full_seed_{seed}.d3")
        if not os.path.exists(model_path):
            continue
            
        model = d3rlpy.load_learnable(model_path)
        env = MultiAssetTradingEnv(env_df, TICKERS)
        
        obs, _ = env.reset()
        done = False
        while not done:
            action = model.predict(np.array([obs]))[0]
            obs, reward, done, _, _ = env.step(action)
            
        metrics = compute_metrics(portfolio_history=env.portfolio_history)
        all_metrics.append(metrics)
        
    if not all_metrics:
        return None
        
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if vals:
            agg[key] = np.mean(vals)
            agg[f"{key}_std"] = np.std(vals, ddof=1) if len(vals)>1 else 0.0
    return agg

def run_equal_weight(env_df):
    env = MultiAssetTradingEnv(env_df, TICKERS)
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.ones(len(TICKERS)) / len(TICKERS)
        obs, reward, done, _, _ = env.step(action)
    return compute_metrics(portfolio_history=env.portfolio_history)

def run_stress_tests():
    df = load_data()
    scenarios = [simulate_flash_crash(df), simulate_stagflation_shock(df)]
    
    print("\n" + "="*70)
    print("SYNTHETIC STRESS SCENARIO EVALUATION")
    print("="*70)
    
    results = []
    
    for perturbed_df, scenario_name in scenarios:
        print(f"\nEvaluating: {scenario_name}")
        
        # 1. Equal-Weight Baseline
        ew_metrics = run_equal_weight(perturbed_df)
        results.append({
            "Scenario": scenario_name,
            "Strategy": "Equal-Weight Baseline",
            "Total Return": f"{ew_metrics['total_return']:.4f}",
            "Max Drawdown": f"{ew_metrics['max_drawdown']:.4f}"
        })
        
        # 2. RL Policies
        for algo in ["CQL", "TD3BC"]:
            rl_metrics = run_rl_multi_seed(perturbed_df, algo)
            if rl_metrics:
                results.append({
                    "Scenario": scenario_name,
                    "Strategy": f"{algo} (Multi-Seed)",
                    "Total Return": f"{rl_metrics['total_return']:.4f} ± {rl_metrics.get('total_return_std', 0.0):.4f}",
                    "Max Drawdown": f"{rl_metrics['max_drawdown']:.4f} ± {rl_metrics.get('max_drawdown_std', 0.0):.4f}"
                })
                
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    out_path = os.path.join(DATA_DIR, "stress_test_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved stress test results to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_stress_tests()
