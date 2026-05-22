"""Training script. Usage: python src/train.py --config configs/riyadh_4x4.yaml --cost-model S"""
import argparse, yaml, os, numpy as np
from environment import HVACEnv
from agent import create_agent

def train(args):
    with open(args.config) as f: config = yaml.safe_load(f)
    env = HVACEnv(config, cost_model_name=args.cost_model)
    agent = create_agent(env, seed=args.seed)
    agent.learn(total_timesteps=args.episodes * 720)
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{config.get('city','unknown')}_{config['n_zones']}z_{args.cost_model}"
    agent.save(path)
    print(f"Saved to {path}.zip")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cost-model", default="S", choices=["L","E","S"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=200000)
    train(p.parse_args())
