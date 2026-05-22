"""Rule-based safety layer for RBRL.
R1 (Force-on): T >= Tmax - eps_g -> u=1
R2 (Force-off): T <= Tmin + eps_g -> u=0
R3 (Cycle guard): prevent switching within minimum cycle time
Reference: Section 5.1 of the paper.
"""
class RuleLayer:
    def __init__(self, t_min=22.0, t_max=26.0, eps_g=0.5, min_cycle_h=1.0):
        self.t_min, self.t_max = t_min, t_max
        self.eps_g, self.min_cycle_h = eps_g, min_cycle_h

    def apply(self, agent_action, temperature, prev_action, time_since_switch):
        if temperature >= self.t_max - self.eps_g: return 1, "R1"
        if temperature <= self.t_min + self.eps_g: return 0, "R2"
        if agent_action != prev_action and time_since_switch < self.min_cycle_h:
            return prev_action, "R3"
        return agent_action, "agent"

    def comfort_violation(self, temperature):
        if temperature > self.t_max: return temperature - self.t_max
        if temperature < self.t_min: return self.t_min - temperature
        return 0.0
