"""
Orchestrator: Joint Offline RL + LLM Training Pipeline

Executes the full pipeline in the most efficient manner possible:
1. Sequentially builds the data and corpus.
2. Parallellizes LLM API calls and Multi-Seed RL training via multiprocessing.
3. Automatically triggers evaluations once training completes.
"""

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def run_cmd(cmd, desc):
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> {desc}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with exit code {result.returncode}")
        sys.exit(1)

def run_background_cmd(cmd, desc):
    print(f"[{time.strftime('%H:%M:%S')}] >>> Launching in background: {desc}")
    return subprocess.Popen(cmd, shell=True)

def run_pipeline():
    print("="*60)
    print("STARTING FULL TRAINING PIPELINE")
    print("="*60)

    # 1. Data Foundation
    # run_cmd(f'python "{os.path.join(SRC_DIR, "strategist", "build_corpus.py")}"', "Building Macro Corpus")
    # run_cmd(f'python "{os.path.join(SRC_DIR, "rl", "generate_expert_data.py")}"', "Generating Expert Features & Dataset")

    # 2. Parallel Training (LLM Cache & RL Seeds)
    print("\n" + "="*60)
    print("STARTING PARALLEL TRAINING TASKS")
    print("="*60)

    processes = []
    
    # Launch LLM Cache Generator (Internally parallelized with ThreadPoolExecutor)
    # llm_cmd = f'python "{os.path.join(SRC_DIR, "eval", "cache_llm_constraints.py")}"'
    # processes.append(("LLM Monthly Cache", run_background_cmd(llm_cmd, "LLM Monthly Cache")))
    
    # Launch 10 RL Trainings in parallel (5 seeds x 2 algos)
    seeds = [42, 43, 44, 45, 46]
    for algo in ["CQL", "TD3BC"]:
        for seed in seeds:
            desc = f"RL Trainer: {algo} Seed {seed}"
            cmd = f'python "{os.path.join(SRC_DIR, "rl", "train_offline.py")}" --algo {algo} --seed {seed}'
            processes.append((desc, run_background_cmd(cmd, desc)))
            
    print("\nWaiting for all 11 background processes to finish...")
    print("(This may take a while depending on hardware and API limits)")
    
    failed = False
    for desc, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[FAIL] {desc} failed with code {p.returncode}")
            failed = True
        else:
            print(f"[SUCCESS] {desc} finished successfully")
            
    if failed:
        print("\nERROR: One or more training tasks failed. Aborting pipeline.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE. STARTING EVALUATIONS.")
    print("="*60)

    # 3. Evaluations
    run_cmd(f'python "{os.path.join(SRC_DIR, "eval", "evaluate_strategist.py")}"', "LLM Vacuum Evaluation")
    run_cmd(f'python "{os.path.join(SRC_DIR, "eval", "policy_explainability.py")}"', "Policy Feature Attribution")
    run_cmd(f'python "{os.path.join(SRC_DIR, "eval", "synthetic_stress.py")}"', "Synthetic Stress Scenarios")
    run_cmd(f'python "{os.path.join(SRC_DIR, "eval", "walk_forward.py")}"', "Walk-Forward Ablation Table")

    print("\n" + "="*60)
    print("PIPELINE FINISHED SUCCESSFULLY! 🚀")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
