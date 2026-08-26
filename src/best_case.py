"""
Behaviour-cloning warm-start for the RC study.

The sibling codebase established that PPO from scratch settles into a bad local
optimum (over-cycles, cost rises); the fix is to clone a dynamic-programming
expert that DOES pre-cool, then fine-tune. This reproduces that on the
real RC dynamics, so we measure PI-PPO's true best-case saving.

Per-zone DP ignores coupling (a 2nd-order effect for a demonstration); PPO
fine-tuning recovers it. Nothing hardcoded -- the expert is solved, the policy
is fitted, savings are measured from real rollouts.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from rc_sim import (load_zones, load_weather, run_month, policy_onoff,
                    step_rc, bill, P_ELEC, Q_COOL, Q_INT, DT_H, STEPS_PER_DAY)
from train_real import Agent, obs, reward, tariff_rate


def dp_expert_zone(K, C, weather_day, lo, hi, price_rate=0.18, ngrid=60):
    """Optimal ON/OFF policy for one uncoupled zone over one day by DP.

    Cost = energy price * ON. Returns POL[step, temp_bin, prev] in {0,1}.
    """
    grid = np.linspace(lo + 0.19, hi - 0.19, ngrid)
    H = STEPS_PER_DAY
    INF = 1e18
    V = np.zeros((ngrid, 2))
    POL = np.zeros((H, ngrid, 2), dtype=np.int8)

    def nxt(T, u, Ta):
        q = Q_COOL * u + K * (Ta - T) + Q_INT
        return T + q / C * DT_H * 3600.0

    for k in range(H - 1, -1, -1):
        Ta = weather_day[k]
        Vn = np.full((ngrid, 2), INF)
        for i, T in enumerate(grid):
            for pu in (0, 1):
                best, ba = INF, 0
                for u in (0, 1):
                    Tn = nxt(T, u, Ta)
                    if Tn < grid[0] - 1e-6 or Tn > grid[-1] + 1e-6:
                        # infeasible -> heavy penalty (comfort)
                        c = INF
                        v = c
                    else:
                        j = int(np.clip(round((Tn - grid[0]) / (grid[1]-grid[0])), 0, ngrid-1))
                        c = price_rate * P_ELEC * u * DT_H
                        if u == 1 and pu == 0:
                            c += 0.04
                        v = c + V[j, u]
                    if v < best:
                        best, ba = v, u
                Vn[i, pu] = best
                POL[k, i, pu] = ba
        V = Vn
    return POL, grid


def collect_demos(Z, weather_day, weeks=8, seed=0):
    """Roll DP experts (one per zone) through the real coupled env; record (obs,action)."""
    rng = np.random.default_rng(seed)
    n = Z["n"]
    # one DP per zone (respects each zone's own K,C)
    experts = [dp_expert_zone(Z["K"][z], Z["C"][z], weather_day, Z["lo"][z], Z["hi"][z])
               for z in range(n)]
    O, A = [], []
    for _ in range(weeks):
        T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int); e_cum = 0.0
        for s in range(STEPS_PER_DAY):
            Ta = weather_day[s]
            u = np.zeros(n, int)
            for z in range(n):
                POL, grid = experts[z]
                i = int(np.clip(round((T[z]-grid[0])/(grid[1]-grid[0])), 0, len(grid)-1))
                u[z] = POL[s, i, prev[z]]
            u[T >= Z["hi"] - 0.19] = 1; u[T <= Z["lo"] + 0.19] = 0
            O.append(obs(T, prev, Ta, Z, e_cum, s*DT_H)); A.append(u.astype(np.float32))
            e_cum += P_ELEC * u.sum() * DT_H
            T = step_rc(T, Ta, u, Z); prev = u
    return np.array(O, np.float32), np.array(A, np.float32)


def clone(ag, O, A, epochs=80, lr=1e-3):
    opt = torch.optim.Adam(ag.parameters(), lr=lr)
    X, Y = torch.tensor(O), torch.tensor(A)
    for ep in range(epochs):
        p, _ = ag(X)
        loss = F.binary_cross_entropy(p, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((ag(X)[0] > 0.5).float() == Y).float().mean().item()
    return acc


def finetune(ag, Z, weather_day, episodes=300, seed=0, k=3, lr=1e-5):
    opt = torch.optim.Adam(ag.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = Z["n"]
    for ep in range(episodes):
        T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int); e_cum = 0.0
        S, Ac, LP, R, V = [], [], [], [], []
        for s in range(STEPS_PER_DAY):
            Ta = weather_day[s]; o = obs(T, prev, Ta, Z, e_cum, s*DT_H)
            ot = torch.from_numpy(o).unsqueeze(0)
            p, v = ag(ot); dist = Bernoulli(p); a = dist.sample()
            if a.sum() > k:
                idx = torch.topk(p.squeeze(), k).indices
                aa = torch.zeros_like(a); aa[0, idx] = 1.0; a = aa
            u = a.numpy().flatten().astype(int)
            u[T >= Z["hi"]-0.19] = 1; u[T <= Z["lo"]+0.19] = 0
            r = reward(u, prev, T, Ta, Z, e_cum, k)
            S.append(o); Ac.append(u.astype(np.float32))
            LP.append(dist.log_prob(a).sum().item()); R.append(r); V.append(v.item())
            e_cum += P_ELEC*u.sum()*DT_H; T = step_rc(T, Ta, u, Z); prev = u
        adv, g = [], 0; Vn = V+[0.0]
        for t in reversed(range(len(R))):
            d = R[t]+0.99*Vn[t+1]-Vn[t]; g = d+0.99*0.95*g; adv.insert(0, g)
        ret = [a_+v_ for a_, v_ in zip(adv, V)]
        St = torch.tensor(np.array(S)); At = torch.tensor(np.array(Ac))
        LPt = torch.tensor(LP); RTt = torch.tensor(ret, dtype=torch.float32)
        ADt = torch.tensor(adv, dtype=torch.float32); ADt = (ADt-ADt.mean())/(ADt.std()+1e-8)
        for _ in range(10):
            p, v = ag(St); dist = Bernoulli(p); nlp = dist.log_prob(At).sum(1)
            ratio = torch.exp(nlp-LPt); s1 = ratio*ADt; s2 = torch.clamp(ratio,0.8,1.2)*ADt
            loss = -torch.min(s1,s2).mean()+0.5*(RTt-v.squeeze()).pow(2).mean()-0.001*dist.entropy().sum(1).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ag.parameters(),0.5); opt.step()
    return ag


def best_case(cfg, month, k, seed=0):
    Z = load_zones(cfg)
    w = load_weather(month, "/mnt/user-data/uploads/jeddah_ambient_profiles.csv")
    n = Z["n"]; ag = Agent(3*n+4, n)
    torch.manual_seed(seed); np.random.seed(seed)
    O, A = collect_demos(Z, w, weeks=8, seed=seed)
    acc = clone(ag, O, A, epochs=80)
    ag = finetune(ag, Z, w, episodes=300, seed=seed, k=k)
    def pol(T, prev, Ta, Z, e_cum):
        o = torch.from_numpy(obs(T, prev, Ta, Z, e_cum, 0.0)).unsqueeze(0)
        with torch.no_grad(): p, _ = ag(o)
        u = (p.squeeze().numpy() > 0.5).astype(int)
        if u.sum() > k:
            idx = np.argsort(-p.squeeze().numpy())[:k]; u = np.zeros(len(u), int); u[idx] = 1
        return u
    r_pi = run_month(Z, w, pol, days=30, seed=seed)
    r_on = run_month(Z, w, policy_onoff, days=30, seed=seed)
    red = 100*(r_on["bill"]-r_pi["bill"])/r_on["bill"] if r_on["bill"]>0 else 0
    return r_on, r_pi, red, acc


if __name__ == "__main__":
    print("BEST-CASE PI-PPO (BC warm-start + fine-tune) — 5-zone Jeddah villa, July")
    print("="*72)
    r_on, r_pi, red, acc = best_case("../configs/villa_5zone.yaml", "July", k=3, seed=0)
    print(f"  BC clone accuracy: {acc:.3f}")
    print(f"  On-Off : {r_on['kwh']:.0f} kWh / {r_on['bill']:.0f} SAR / peak {r_on['peak_kw']:.1f}")
    print(f"  PI-PPO : {r_pi['kwh']:.0f} kWh / {r_pi['bill']:.0f} SAR / peak {r_pi['peak_kw']:.1f} / viol {r_pi['viol_pct']:.1f}%")
    print(f"  MEASURED best-case cost reduction: {red:.1f}%")
    print(f"  (Jeddah 5-zone DP optimality bound from sibling codebase: ~10.5%)")
