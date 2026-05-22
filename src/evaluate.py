"""Evaluation script. Usage: python src/evaluate.py --model checkpoints/riyadh_16z_S.zip --config configs/riyadh_4x4.yaml"""
import argparse, yaml, numpy as np
from environment import HVACEnv
from agent import load_agent

def thermostat_baseline(env, n_ep=30):
    costs = []
    for _ in range(n_ep):
        obs, _ = env.reset(); ep_cost = 0; done = False; prev = np.zeros(env.n_zones, dtype=int)
        while not done:
            actions = np.array([1 if env.temperatures[i] >= env.t_max-0.5 else (0 if env.temperatures[i] <= env.t_min+0.5 else prev[i]) for i in range(env.n_zones)])
            obs, _, done, _, info = env.step(actions); ep_cost += info["cost"]; prev = actions.copy()
        costs.append(ep_cost)
    return np.mean(costs), np.std(costs)

def evaluate_rbrl(env, model_path, n_ep=30):
    agent = load_agent(model_path, env); costs = []
    for _ in range(n_ep):
        obs, _ = env.reset(); ep_cost = 0; done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action); ep_cost += info["cost"]
        costs.append(ep_cost)
    return np.mean(costs), np.std(costs)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--config", required=True)
    p.add_argument("--cost-model", default="S"); p.add_argument("--n-weeks", type=int, default=30)
    args = p.parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    env = HVACEnv(config, cost_model_name=args.cost_model)
    tm, ts = thermostat_baseline(env, args.n_weeks)
    rm, rs = evaluate_rbrl(env, args.model, args.n_weeks)
    print(f"THERM: {tm:.2f}±{ts:.2f} | RBRL: {rm:.2f}±{rs:.2f} | Saving: {(tm-rm)/tm*100:.1f}%")
