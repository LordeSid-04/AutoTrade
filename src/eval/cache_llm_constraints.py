"""
Cache LLM Constraints

Pre-computes and caches the LLM Strategist's constraints for the first trading day 
of every month from 2011 to 2024. This allows the walk_forward backtester to dynamically 
simulate re-querying the LLM without running hundreds of live API calls.
"""

import os
import sys
import json
import logging
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.strategist.llm_agent import LLMStrategist

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def generate_cache():
    processed_data_path = os.path.join(DATA_DIR, "processed_features.csv")
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data not found at {processed_data_path}.")
        sys.exit(1)
        
    df = pd.read_csv(processed_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get the first trading day of every month
    monthly_dates = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Date'].min()
    date_strs = monthly_dates.dt.strftime('%Y-%m-%d').tolist()
    
    print(f"Total months to cache: {len(date_strs)}")
    
    strategist = LLMStrategist()
    cache = {}
    cache_lock = threading.Lock()
    
    cache_path = os.path.join(DATA_DIR, "llm_monthly_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
            
    dates_to_fetch = [ds for ds in date_strs if ds not in cache]
    if not dates_to_fetch:
        print(f"All {len(date_strs)} months are already cached.")
        return
        
    print(f"Fetching {len(dates_to_fetch)} new months concurrently...")
    
    def process_date(ds):
        try:
            constraints, _ = strategist.get_weekly_constraints(ds, n_samples=3)
            
            with cache_lock:
                cache[ds] = {
                    "regime": constraints.regime_classification,
                    "confidence": constraints.confidence_score,
                    "vol_target": constraints.max_portfolio_volatility_target,
                    "caps": constraints.sector_exposure_caps,
                    "safe_haven": constraints.recommended_safe_haven
                }
                # Save incrementally safely
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed for {ds}: {e}")
            return False

    try:
        # Run API calls concurrently (max 5 workers to avoid OpenAI rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            list(tqdm(executor.map(process_date, dates_to_fetch), total=len(dates_to_fetch), desc="Caching LLM Constraints"))
                
    except KeyboardInterrupt:
        print("\nCaching interrupted. Progress is saved.")
        sys.exit(0)
        
    print(f"\nCache generation complete. Saved to {cache_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    generate_cache()
