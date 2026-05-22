"""PPO agent wrapper. Reference: Table 4 of the paper."""
from stable_baselines3 import PPO
import torch

DEFAULT_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, policy_kwargs=dict(net_arch=dict(pi=[256,128], vf=[256,128]), activation_fn=torch.nn.ReLU))

def create_agent(env, params=None, seed=42, verbose=1):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return PPO("MlpPolicy", env, seed=seed, verbose=verbose, device="auto", **p)

def load_agent(path, env):
    return PPO.load(path, env=env)
