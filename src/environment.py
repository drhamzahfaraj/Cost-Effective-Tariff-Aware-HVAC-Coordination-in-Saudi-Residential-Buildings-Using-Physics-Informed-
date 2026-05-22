"""RC thermal simulation environment for multi-zone HVAC scheduling.
First-order RC model (Eq. 2) as a Gymnasium environment.
Reference: Section 3.1 of the paper.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .cost_models import get_cost_model
from .rules import RuleLayer

class HVACEnv(gym.Env):
    def __init__(self, config, cost_model_name="S", epw_data=None, solar_data=None):
        super().__init__()
        self.n_zones = config["n_zones"]
        self.dt = config.get("dt", 1.0)
        self.c_th = config.get("c_th", 77.2)
        self.lambda_ext = config.get("lambda_ext", 20.5)
        self.lambda_win = config.get("lambda_win", 12.5)
        self.lambda_iz = config.get("lambda_iz", 12.5)
        self.q_hvac = config.get("q_hvac", 2.0)
        self.p_hvac = config.get("p_hvac", 0.67)
        self.shgc = config.get("shgc", 0.25)
        self.a_gl = config.get("a_gl", 2.0)
        self.t_min = config.get("t_min", 22.0)
        self.t_max = config.get("t_max", 26.0)
        self.c_sw = config.get("c_sw", 0.15)
        self.adjacency = np.array(config.get("adjacency", np.zeros((self.n_zones, self.n_zones))))
        h = np.arange(8760)
        self.t_out_year = epw_data if epw_data is not None else 28 + 7*np.sin(2*np.pi*(h-6)/24) + 12*np.sin(2*np.pi*h/8760)
        self.solar_year = solar_data if solar_data is not None else np.maximum(0, 800*np.sin(np.pi*(h%24-6)/12))
        self.cost_model = get_cost_model(cost_model_name)
        self.rule_layer = RuleLayer(self.t_min, self.t_max)
        self.observation_space = spaces.Box(-50, 100, shape=(self.n_zones + 5,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.n_zones)
        self._reset_state()

    def _reset_state(self):
        self.temperatures = np.full(self.n_zones, 24.0)
        self.prev_actions = np.zeros(self.n_zones, dtype=int)
        self.time_since_switch = np.ones(self.n_zones) * 10.0
        self.e_cum, self.hour, self.step_count = 0.0, 0, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self.hour = self.np_random.integers(0, 8760 - 720) if self.np_random else 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=int)
        h = self.hour % 8760
        t_out = self.t_out_year[h]
        q_sol = self.shgc * self.a_gl * self.solar_year[h] / 1000.0
        total_cost, overrides = 0.0, 0
        for i in range(self.n_zones):
            final_action, reason = self.rule_layer.apply(action[i], self.temperatures[i], self.prev_actions[i], self.time_since_switch[i])
            if reason != "agent": overrides += 1
            action[i] = final_action
            neighbours = np.where(self.adjacency[i] > 0)[0]
            t_iz = np.mean(self.temperatures[neighbours]) if len(neighbours) > 0 else self.temperatures[i]
            liz_total = self.lambda_iz * len(neighbours) if len(neighbours) > 0 else 0.0
            heat_W = -self.q_hvac*1000*action[i] - (self.lambda_ext+self.lambda_win)*(self.temperatures[i]-t_out) - liz_total*(self.temperatures[i]-t_iz) + q_sol*1000
            self.temperatures[i] += self.dt * 3600 / (self.c_th * 1000) * heat_W
            total_cost += self.cost_model.interval_cost(action[i], self.p_hvac, self.dt, self.e_cum, self.c_sw, self.prev_actions[i])
            self.e_cum += self.p_hvac * action[i] * self.dt
            self.time_since_switch[i] = 0.0 if action[i] != self.prev_actions[i] else self.time_since_switch[i] + self.dt
        self.prev_actions = action.copy()
        self.hour += 1; self.step_count += 1
        cv = sum(self.rule_layer.comfort_violation(t) for t in self.temperatures)
        reward = -total_cost - 10.0 * cv
        return self._get_obs(), reward, self.step_count >= 720, False, {"cost": total_cost, "e_cum": self.e_cum, "overrides": overrides, "comfort_violation": cv}

    def _get_obs(self):
        h = self.hour % 8760
        return np.concatenate([self.temperatures, [self.t_out_year[h], self.t_out_year[(h+1)%8760], self.t_out_year[(h+2)%8760], self.e_cum/6000, np.sin(2*np.pi*(self.hour%24)/24)]]).astype(np.float32)
