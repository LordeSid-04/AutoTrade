"""
Offline RL Training with Conservative Q-Learning (CQL) and TD3+BC
Features multi-seed reporting and custom SharedAssetEncoder.
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import h5py
import torch
import torch.nn as nn
from dataclasses import dataclass

import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.models.encoders import EncoderFactory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from src.env.trading_env import MultiAssetTradingEnv

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- Custom Shared Asset Encoder ---
# Respects the permutation structure of the assets (15 assets x 10 features)
# rather than flattening everything into a single MLP

from src.rl.custom_encoders import register_custom_encoders, SharedAssetEncoderFactory
register_custom_encoders()

def load_h5_dataset(h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Expert dataset not found at {h5_path}. Run generate_expert_data.py first.")
    
    print(f"Loading expert dataset from {h5_path} using h5py...")
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    with h5py.File(h5_path, 'r') as f:
        obs_keys = [k for k in f.keys() if k.startswith('observations_')]
        num_episodes = len(obs_keys)
        
        for i in range(num_episodes):
            obs = f[f'observations_{i}'][()]
            acts = f[f'actions_{i}'][()]
            rews = f[f'rewards_{i}'][()]
            
            terms = np.zeros(len(obs), dtype=np.float32)
            terms[-1] = 1.0
            
            observations.append(obs)
            actions.append(acts)
            rewards.append(rews)
            terminals.append(terms)
            
    obs_array = np.concatenate(observations, axis=0)
    act_array = np.concatenate(actions, axis=0)
    rew_array = np.concatenate(rewards, axis=0)
    term_array = np.concatenate(terminals, axis=0)
    
    return MDPDataset(
        observations=obs_array,
        actions=act_array,
        rewards=rew_array,
        terminals=term_array
    )

def train_offline_rl(dataset, algo_class, algo_name, seed, n_steps=50000, n_steps_per_epoch=1000):
    print(f"Initializing {algo_name} Offline RL model for seed {seed}...")
    
    np.random.seed(seed)
    d3rlpy.seed(seed)
    torch.manual_seed(seed)
    
    encoder_factory = SharedAssetEncoderFactory(feature_size=256)
    
    if algo_name == "CQL":
        model = d3rlpy.algos.CQLConfig(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            alpha_learning_rate=1e-4,
            batch_size=256,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        ).create(device="cpu")
    else:
        # TD3+BC
        model = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        ).create(device="cpu")
    
    print(f"Starting {algo_name} training for seed {seed}...")
    
    model.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress=True,
        experiment_name=f"{algo_name}_seed_{seed}"
    )
    
    model_path = os.path.join(DATA_DIR, f"{algo_name.lower()}_full_seed_{seed}.d3")
    model.save(model_path)
    print(f"Trained model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline RL Trainer")
    parser.add_argument("--algo", type=str, choices=["CQL", "TD3BC", "ALL"], default="ALL", help="Algorithm to train")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to train")
    args = parser.parse_args()

    h5_path = os.path.join(DATA_DIR, "expert_dataset.h5")
    try:
        dataset = load_h5_dataset(h5_path)
        
        seeds = [42, 43, 44, 45, 46] if args.seed is None else [args.seed]
        
        if args.algo == "ALL":
            algorithms = [("CQL", d3rlpy.algos.CQL), ("TD3BC", d3rlpy.algos.TD3PlusBC)]
        elif args.algo == "CQL":
            algorithms = [("CQL", d3rlpy.algos.CQL)]
        else:
            algorithms = [("TD3BC", d3rlpy.algos.TD3PlusBC)]
            
        for algo_name, algo_class in algorithms:
            for seed in seeds:
                train_offline_rl(dataset, algo_class, algo_name, seed, n_steps=50000, n_steps_per_epoch=1000)
                
        # Only write metadata if running the full default sweep
        if args.algo == "ALL" and args.seed is None:
            metadata = {
                "algorithms": ["CQL", "TD3BC"],
                "seeds": [42, 43, 44, 45, 46],
                "n_steps": 50000,
                "n_steps_per_epoch": 1000,
                "batch_size": 256,
            }
            meta_path = os.path.join(DATA_DIR, "training_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Training metadata saved to {meta_path}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)
