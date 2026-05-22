"""
Main Training Script for PI-PPO
Usage: python train_pippo.py --config configs/villa_5zone.yaml --month july --seed 0
"""

import argparse
import yaml
import numpy as np
import torch
from agent import PPOAgent, PPOTrainer
from reward import compute_reward, OMEGA
from surrogate import FeasibilitySurrogate
from environment import build_5zone_villa, build_compound, compute_heat_balance


def train(config_path: str, month: str, seed: int, n_episodes: int = None):
    """Train PI-PPO agent."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    n_zones = config['n_zones']
    comfort_low = np.array(config['comfort_low'])
    comfort_high = np.array(config['comfort_high'])
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Build environment
    if n_zones <= 5:
        zones = build_5zone_villa()
    else:
        zones = build_compound(n_zones, seed=42)
    
    # State dimension: x(t) + m(t-1) + T_a + E_cum + k + dx(t) + t_hour
    state_dim = n_zones + n_zones + 1 + 1 + 1 + n_zones + 1
    
    # Initialize agent
    agent = PPOAgent(state_dim, n_zones)
    trainer = PPOTrainer(agent, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10)
    
    # Initialize surrogate
    surrogate = FeasibilitySurrogate(n_zones)
    
    # Training parameters
    if n_episodes is None:
        n_episodes = {5: 5000, 20: 10000, 50: 15000, 100: 15000}.get(n_zones, 10000)
    
    steps_per_day = 96  # 24h / 15min
    
    print(f"Training PI-PPO: {n_zones} zones, {month}, seed={seed}")
    print(f"Episodes: {n_episodes}, State dim: {state_dim}")
    print(f"Comfort: [{comfort_low[0]}, {comfort_high[0]}]°C")
    print(f"Reward weights: {OMEGA}")
    
    best_reward = -float('inf')
    
    for episode in range(n_episodes):
        # Reset episode
        temperatures = np.random.uniform(comfort_low, comfort_high)
        modes = np.zeros(n_zones, dtype=int)
        e_cum = 0.0
        episode_reward = 0.0
        
        # Episode loop (1 day = 96 steps)
        states, actions, log_probs_list, rewards, values, dones = [], [], [], [], [], []
        
        for step in range(steps_per_day):
            t_hour = step * 0.25
            # Approximate T_a based on month (simplified)
            T_a = get_ambient_temp(t_hour, month)
            
            # Compute temperature derivatives
            dx_obs = np.zeros(n_zones)
            for i, zone in enumerate(zones):
                q = zone.K_i * (T_a - temperatures[i]) + zone.Q_int + zone.Q_cool * modes[i]
                dx_obs[i] = q / zone.C_i * 900  # 15-min step
            
            # Surrogate estimate
            k = min(n_zones, max(1, int(np.ceil(sum(0.48 for _ in range(n_zones))))))
            
            # Build state
            state = np.concatenate([
                temperatures, modes.astype(float),
                [T_a], [e_cum / 6000], [k],
                dx_obs, [t_hour / 24]
            ])
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = agent.select_action(state_tensor, k)
            
            # Compute predicted derivatives
            dx_pred = np.zeros(n_zones)
            for i, zone in enumerate(zones):
                q = zone.K_i * (T_a - temperatures[i]) + zone.Q_int + zone.Q_cool * action[i]
                dx_pred[i] = q / zone.C_i * 900
            
            # Compute reward
            reward, _ = compute_reward(
                action, modes, temperatures, dx_obs, dx_pred,
                comfort_low, comfort_high, e_cum, sum(0.48 for _ in range(n_zones)), k
            )
            
            # Step environment
            new_temps = compute_heat_balance(temperatures, T_a, action, zones)
            e_cum += 1.8 * np.sum(action) * 0.25
            
            states.append(state)
            actions.append(action)
            log_probs_list.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(0)
            
            temperatures = new_temps
            modes = action
            episode_reward += reward
        
        # PPO update
        next_value = 0
        advantages, returns = trainer.compute_gae(
            rewards, values, next_value, dones
        )
        trainer.update(states, actions, log_probs_list, returns, advantages)
        
        if episode % 500 == 0:
            print(f"Episode {episode}/{n_episodes}: reward={episode_reward:.1f}, E_cum={e_cum:.0f} kWh")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.state_dict(), f'checkpoints/pippo_{n_zones}z_{month}_seed{seed}.pt')
    
    print(f"Training complete. Best reward: {best_reward:.1f}")


def get_ambient_temp(t_hour: float, month: str) -> float:
    """Approximate Jeddah ambient temperature (simplified TMY3 profile)."""
    profiles = {
        'january': (18, 29, 14),    # (min, max, peak_hour)
        'april':   (23, 35, 14),
        'july':    (29, 43, 14),
        'october': (25.5, 37, 14),
    }
    t_min, t_max, t_peak = profiles.get(month, (29, 43, 14))
    # Sinusoidal approximation
    phase = 2 * np.pi * (t_hour - t_peak + 6) / 24
    return (t_max + t_min) / 2 + (t_max - t_min) / 2 * np.cos(phase)


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--month', type=str, default='july')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=None)
    args = parser.parse_args()
    
    train(args.config, args.month, args.seed, args.episodes)
