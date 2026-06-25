"""
Standalone LLM Strategist Evaluator

Evaluates the LLMStrategist independently from the RL pipeline.
Runs the LLM against hand-labeled historical ground truth regimes and calculates
accuracy, precision, and recall.
"""

import os
import sys
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.strategist.llm_agent import LLMStrategist

logger = logging.getLogger(__name__)

# Ground-truth hand-labeled dates spanning 2011-2024
EVAL_DATA = [
    # Crisis / Risk-Off
    {"date": "2020-03-16", "true_regime": "risk-off", "desc": "COVID Crash / 0% Rates"},
    {"date": "2022-09-22", "true_regime": "risk-off", "desc": "Aggressive 75bp Hike / Peak Inflation Fear"},
    {"date": "2015-12-17", "true_regime": "transitional", "desc": "First rate hike since 2006 (Lift-off)"},
    
    # Bull / Risk-On
    {"date": "2017-12-14", "true_regime": "risk-on", "desc": "Steady low-rate expansion"},
    {"date": "2021-11-04", "true_regime": "risk-on", "desc": "Tapering announced but rates still low, market euphoric"},
    {"date": "2023-07-27", "true_regime": "risk-on", "desc": "Possible final hike, soft landing narrative"},
    
    # Transitional / Choppy
    {"date": "2019-08-01", "true_regime": "transitional", "desc": "Insurance cuts mid-cycle"},
    {"date": "2022-03-17", "true_regime": "transitional", "desc": "First rate hike of modern tightening cycle"},
    {"date": "2018-12-04", "true_regime": "transitional", "desc": "Tariffs, strong ISM, but emerging market concerns"}
]

def run_evaluation():
    print("\n" + "="*60)
    print("RUNNING ISOLATED LLM STRATEGIST EVALUATION")
    print("="*60)
    
    strategist = LLMStrategist()
    
    results = []
    y_true = []
    y_pred = []
    
    # Run with n_samples=3 for majority vote self-consistency
    for item in EVAL_DATA:
        date_str = item["date"]
        true_regime = item["true_regime"]
        desc = item["desc"]
        
        print(f"\nEvaluating Date: {date_str} | True Regime: {true_regime}")
        print(f"Context: {desc}")
        
        try:
            constraints, _ = strategist.get_weekly_constraints(date_str, n_samples=3)
            pred_regime = constraints.regime_classification
            conf = constraints.confidence_score
            
            y_true.append(true_regime)
            y_pred.append(pred_regime)
            
            match = "[PASS]" if pred_regime == true_regime else "[FAIL]"
            print(f"  Prediction: {pred_regime} | Conf: {conf}/10 | {match}")
            
            results.append({
                "Date": date_str,
                "Description": desc,
                "True Regime": true_regime,
                "Predicted Regime": pred_regime,
                "Confidence": conf,
                "Correct": pred_regime == true_regime
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate {date_str}: {e}")
            
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc:.2%}\n")
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("\nConfusion Matrix:")
        labels = ["risk-on", "risk-off", "transitional"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
        print(df_cm)
    else:
        print("No successful evaluations.")
        
    # Save results
    results_df = pd.DataFrame(results)
    out_path = os.path.join(PROJECT_ROOT, "data", "llm_evaluation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_evaluation()
