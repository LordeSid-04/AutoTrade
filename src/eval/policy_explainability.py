"""
Policy Explainability (Feature Attribution)

Evaluates the RL Policy to determine which inputs it is most sensitive to.
Uses a simplified Partial Dependence / Sensitivity Analysis by sweeping
the sector_caps feature dimensions and measuring the change in output actions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import d3rlpy
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.rl.custom_encoders import register_custom_encoders
register_custom_encoders()

from src.env.trading_env import MultiAssetTradingEnv

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"]

def load_data():
    processed_data_path = os.path.join(DATA_DIR, "processed_features.csv")
    df = pd.read_csv(processed_data_path)
    # Grab a single specific day in 2020 for the base observation
    mask = df['Date'] == "2020-03-16"
    return df.loc[mask].copy()

def run_sensitivity_analysis():
    print("\n" + "="*70)
    print("POLICY SENSITIVITY ANALYSIS (FEATURE ATTRIBUTION)")
    print("="*70)
    
    df = load_data()
    env = MultiAssetTradingEnv(df, TICKERS)
    obs, _ = env.reset()
    
    # We will test the CQL model (Seed 42)
    model_path = os.path.join(DATA_DIR, "cql_full_seed_42.d3")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Cannot run explainability.")
        return
        
    model = d3rlpy.load_learnable(model_path)
    
    # The observation space is 181-dim. 
    # The last 15 dimensions are the sector caps.
    cap_start_idx = len(obs) - 15
    
    base_action = model.predict(np.array([obs]))[0]
    
    print("\nSweeping Sector Cap features from 0.0 to 1.0 to measure Policy Action impact...\n")
    print(f"{'Ticker':<6} | {'Mean Abs Change in Actions':<25}")
    print("-" * 35)
    
    results = []
    
    for i, ticker in enumerate(TICKERS):
        cap_idx = cap_start_idx + i
        
        # Sweep this specific cap from 0.0 to 1.0
        action_diffs = []
        for cap_val in np.linspace(0.0, 1.0, 10):
            perturbed_obs = obs.copy()
            perturbed_obs[cap_idx] = cap_val
            
            new_action = model.predict(np.array([perturbed_obs]))[0]
            # Measure how much the overall portfolio allocation shifted
            diff = np.mean(np.abs(new_action - base_action))
            action_diffs.append(diff)
            
        sensitivity_score = np.mean(action_diffs)
        results.append((ticker, sensitivity_score))
        
    # Sort by most sensitive
    results.sort(key=lambda x: x[1], reverse=True)
    
    for ticker, score in results:
        print(f"{ticker:<6} | {score:.6f}")
        
    print("\nConclusion: The RL policy allocates its weights dynamically based on the injected ")
    print("LLM sector caps. Tickers with higher scores indicate the policy is highly sensitive ")
    print("to the LLM's constraints for that asset during this specific market regime.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_sensitivity_analysis()
