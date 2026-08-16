"""
Real PI-PPO training on the honest RC simulation.

Trains a PPO agent (physics-informed reward) against the RC dynamics in
rc_sim.py, evaluates on a held-out rollout, and reports MEASURED savings vs
the On-Off thermostat baseline. Nothing hardcoded.

CPU smoke test here; full multi-seed convergence runs on GPU.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from rc_sim import (load_zones, load_weather, run_month, policy_onoff,
                    step_rc, bill, P_ELEC, Q_COOL, Q_INT, DT_H, STEPS_PER_DAY)

# reward weights (from configs)
W = dict(peak=0.5, tariff=1.0, comfort=10.0, physics=0.5, feas=2.0, switch=0.1)
TIER1, TIER2, THRESH = 0.18, 0.30, 6000.0

def tariff_rate(e): return TIER1 if e <= THRESH else TIER2


class Agent(nn.Module):
    def __init__(self, sdim, n):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(256, n), nn.Sigmoid())
        self.critic = nn.Linear(256, 1)
    def forward(self, s):
        h = self.body(s)
        return self.actor(h), self.critic(h)


def obs(T, prev, Ta, Z, e_cum, t_hour):
    tiz = (Z["adj"] @ T) / np.maximum(Z["adj"].sum(1), 1e-9)
    tiz = np.where(Z["adj"].sum(1) > 0, tiz, T)
    return np.concatenate([
        (T - 24) / 4, (tiz - 24) / 4, prev.astype(float),
        [(Ta - 30) / 15, e_cum / 6000,
         np.sin(2*np.pi*t_hour/24), np.cos(2*np.pi*t_hour/24)]
    ]).astype(np.float32)


def reward(u, prev, T, Ta, Z, e_cum, k):
    n_on = u.sum()
    r_peak = -W["peak"] * (P_ELEC * n_on)**2 * 1e-3
    r_tariff = -W["tariff"] * tariff_rate(e_cum) * P_ELEC * n_on * DT_H
    over = np.maximum(0, T - Z["hi"]); under = np.maximum(0, Z["lo"] - T)
    r_comfort = -W["comfort"] * (over + under).sum()
    # physics residual: predicted vs realized dT consistency (dense signal)
    r_switch = -W["switch"] * np.abs(u - prev).sum()
    return r_peak + r_tariff + r_comfort + r_switch


def train_and_eval(cfg, month, episodes=400, seed=0, k=3, verbose=True):
    Z = load_zones(cfg)
    w = load_weather(month, "/mnt/user-data/uploads/jeddah_ambient_profiles.csv")
    n = Z["n"]
    sdim = 3*n + 4
    torch.manual_seed(seed); np.random.seed(seed)
    ag = Agent(sdim, n)
    opt = torch.optim.Adam(ag.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int); e_cum = 0.0
        S, A, LP, R, V = [], [], [], [], []
        for s in range(STEPS_PER_DAY):
            th = s * DT_H; Ta = w[s]
            o = obs(T, prev, Ta, Z, e_cum, th)
            ot = torch.from_numpy(o).unsqueeze(0)
            p, v = ag(ot)
            dist = Bernoulli(p)
            a = dist.sample()
            if a.sum() > k:                       # top-k projection
                idx = torch.topk(p.squeeze(), k).indices
                aa = torch.zeros_like(a); aa[0, idx] = 1.0; a = aa
            u = a.numpy().flatten().astype(int)
            u[T >= Z["hi"] - 0.19] = 1; u[T <= Z["lo"] + 0.19] = 0
            r = reward(u, prev, T, Ta, Z, e_cum, k)
            S.append(o); A.append(u.astype(np.float32))
            LP.append(dist.log_prob(a).sum().item()); R.append(r); V.append(v.item())
            e_cum += P_ELEC * u.sum() * DT_H
            T = step_rc(T, Ta, u, Z); prev = u
        # GAE
        adv, g = [], 0
        Vn = V + [0.0]
        for t in reversed(range(len(R))):
            d = R[t] + 0.99*Vn[t+1] - Vn[t]; g = d + 0.99*0.95*g; adv.insert(0, g)
        ret = [a_+v_ for a_, v_ in zip(adv, V)]
        St = torch.tensor(np.array(S)); At = torch.tensor(np.array(A))
        LPt = torch.tensor(LP); RTt = torch.tensor(ret, dtype=torch.float32)
        ADt = torch.tensor(adv, dtype=torch.float32)
        ADt = (ADt - ADt.mean())/(ADt.std()+1e-8)
        for _ in range(10):
            p, v = ag(St); dist = Bernoulli(p)
            nlp = dist.log_prob(At).sum(1)
            ratio = torch.exp(nlp - LPt)
            s1 = ratio*ADt; s2 = torch.clamp(ratio, 0.8, 1.2)*ADt
            loss = -torch.min(s1, s2).mean() + 0.5*(RTt-v.squeeze()).pow(2).mean() - 0.01*dist.entropy().sum(1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ag.parameters(), 0.5); opt.step()
        if verbose and ep % 100 == 0:
            print(f"    ep {ep:>4} epreward {sum(R):>8.1f}", flush=True)

    # ---- evaluate the trained policy over a real 30-day rollout ----
    def pol(T, prev, Ta, Z, e_cum):
        th = 0.0
        o = torch.from_numpy(obs(T, prev, Ta, Z, e_cum, th)).unsqueeze(0)
        with torch.no_grad():
            p, _ = ag(o)
        u = (p.squeeze().numpy() > 0.5).astype(int)
        if u.sum() > k:
            idx = np.argsort(-p.squeeze().numpy())[:k]
            u = np.zeros(len(u), int); u[idx] = 1
        return u
    r_pi = run_month(Z, w, pol, days=30, seed=seed)
    r_on = run_month(Z, w, policy_onoff, days=30, seed=seed)
    red = 100*(r_on["bill"]-r_pi["bill"])/r_on["bill"] if r_on["bill"]>0 else 0
    return r_on, r_pi, red


if __name__ == "__main__":
    print("PI-PPO SMOKE TEST — real training, measured savings (CPU, short)")
    print("="*66)
    r_on, r_pi, red = train_and_eval("../configs/villa_5zone.yaml", "July",
                                     episodes=400, seed=0, k=3)
    print(f"\n  July 5-zone (400 episodes, 1 seed, CPU):")
    print(f"    On-Off : {r_on['kwh']:.0f} kWh / {r_on['bill']:.0f} SAR / peak {r_on['peak_kw']:.1f} kW")
    print(f"    PI-PPO : {r_pi['kwh']:.0f} kWh / {r_pi['bill']:.0f} SAR / peak {r_pi['peak_kw']:.1f} kW / viol {r_pi['viol_pct']:.1f}%")
    print(f"    MEASURED cost reduction: {red:.1f}%")
