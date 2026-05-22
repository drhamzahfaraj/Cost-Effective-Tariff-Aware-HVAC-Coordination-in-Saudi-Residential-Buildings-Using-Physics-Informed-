"""
PPO Agent with Top-k Projection for HVAC Scheduling
Policy: 2×256 neurons, ReLU, separate actor-critic heads
Action: Bernoulli per zone, projected to ||m||_1 ≤ k
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np


class PPOAgent(nn.Module):
    """Proximal Policy Optimization agent for HVAC scheduling."""
    
    def __init__(self, state_dim: int, n_zones: int, hidden_dim: int = 256):
        super().__init__()
        self.n_zones = n_zones
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head: probability of ON for each zone
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, n_zones),
            nn.Sigmoid(),
        )
        
        # Critic head: state value
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value
    
    def select_action(self, state: torch.Tensor, k: int) -> tuple:
        """Select action with top-k projection."""
        probs, value = self.forward(state)
        
        # Sample from Bernoulli
        dist = Bernoulli(probs)
        raw_action = dist.sample()
        
        # Top-k projection: keep at most k ON
        if raw_action.sum() > k:
            # Keep top-k by probability
            _, topk_idx = torch.topk(probs.squeeze(), k)
            projected = torch.zeros_like(raw_action)
            projected[0, topk_idx] = 1.0
        else:
            projected = raw_action
        
        log_prob = dist.log_prob(projected).sum(dim=-1)
        
        return projected.numpy().flatten().astype(int), log_prob, value


class PPOTrainer:
    """PPO training loop with clipped surrogate objective."""
    
    def __init__(self, agent, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
    
    def compute_gae(self, rewards, values, next_value, dones, lam=0.95):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO clipped surrogate update (Equation 11 in paper)."""
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            probs, values = self.agent(states)
            dist = Bernoulli(probs)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            entropy = dist.entropy().sum(dim=-1).mean()
            
            loss = actor_loss + critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
