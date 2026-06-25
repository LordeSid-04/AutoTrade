import os
import shutil
import glob
import re

def recover_cql_models():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(project_root, "d3rlpy_logs")
    data_dir = os.path.join(project_root, "data")
    
    seeds = [42, 43, 44, 45, 46]
    timestamp = "20260622025330"
    
    print("Starting CQL model recovery...")
    for seed in seeds:
        folder_name = f"CQL_seed_{seed}_{timestamp}"
        folder_path = os.path.join(logs_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"ERROR: Could not find logs folder for seed {seed} at {folder_path}")
            continue
            
        d3_files = glob.glob(os.path.join(folder_path, "model_*.d3"))
        if not d3_files:
            print(f"ERROR: No .d3 checkpoints found in {folder_path}")
            continue
            
        max_step = -1
        best_model_path = None
        
        for f in d3_files:
            basename = os.path.basename(f)
            # Match model_XXXX.d3
            match = re.search(r"model_(\d+)\.d3", basename)
            if match:
                step = int(match.group(1))
                if step > max_step:
                    max_step = step
                    best_model_path = f
                    
        if best_model_path:
            target_path = os.path.join(data_dir, f"cql_full_seed_{seed}.d3")
            print(f"Seed {seed}: Promoting {os.path.basename(best_model_path)} (Step {max_step}) to {os.path.basename(target_path)}")
            shutil.copy2(best_model_path, target_path)
            
    print("Recovery complete!")

if __name__ == "__main__":
    recover_cql_models()
