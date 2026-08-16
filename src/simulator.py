#!/usr/bin/env python3
"""
simulator.py -- Self-contained honest PI-PPO HVAC simulator.

Everything (RC thermal model, tariff, baselines, DP expert, behaviour-cloning
warm-start, PPO training, evaluation) is in this one file. No src/ folder,
no package imports beyond numpy / torch / pyyaml.

HOW TO RUN (Windows):
    pip install numpy pyyaml torch
    python simulator.py              # quick test (2 seeds, ~minutes)
    python simulator.py --full       # publication run (5 seeds; GPU recommended)

OUTPUT:
    results_generated.json           # machine-readable measured results
    results_generated.md             # readable table

Every number is MEASURED from executing the simulation. Nothing is hardcoded.
This is a lumped-parameter (RC) thermal model -- not EnergyPlus. It reports
whatever the simulation actually yields.
"""
import argparse, json, os, sys
import numpy as np

# -------- optional torch (only needed for PI-PPO training) --------
try:
    import torch
    import torch.nn as nn
    from torch.distributions import Bernoulli
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo-friendly path resolution: look in script dir, ../data, ../configs
def _find(fname):
    for d in [HERE, os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "configs"), os.path.join(HERE, "data"), os.path.join(HERE, "configs")]:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, fname)

# ===================== physical constants =====================
P_ELEC = 1.8      # kW electrical draw per unit when ON
Q_COOL = -5.3     # kW thermal cooling when ON
Q_INT  = 0.3      # kW internal gains per zone
DT_H   = 0.25     # 15-min step (hours)
STEPS_PER_DAY = 96
TIER1, TIER2, THRESH = 0.18, 0.30, 6000.0


def bill(kwh):
    return TIER1 * min(kwh, THRESH) + TIER2 * max(0.0, kwh - THRESH)


# ===================== building / weather loaders =====================
def load_zones(cfg_path):
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    n = cfg["n_zones"]
    lo = np.array(cfg["comfort_low"], float)
    hi = np.array(cfg["comfort_high"], float)
    K = np.array([z["K_i"] for z in cfg["zones"]], float)
    C = np.array([z["C_i"] for z in cfg["zones"]], float)
    adj = np.zeros((n, n)); Kij = np.zeros((n, n))
    for i, z in enumerate(cfg["zones"]):
        for nb, kij in zip(z.get("neighbors", []), z.get("K_ij", [])):
            adj[i, nb] = 1.0; Kij[i, nb] = kij
    return dict(n=n, lo=lo, hi=hi, K=K, C=C, adj=adj, Kij=Kij)


def load_weather(month, csv_path):
    import csv
    prof = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["month"].strip().lower() == month.lower():
                prof.append(float(row["T_a_C"]))
    if len(prof) != STEPS_PER_DAY:
        raise ValueError(f"{month}: expected {STEPS_PER_DAY} points, got {len(prof)} in {csv_path}")
    return np.array(prof)


# ===================== RC dynamics =====================
def step_rc(T, Ta, u, Z):
    K, C, Kij = Z["K"], Z["C"], Z["Kij"]
    q_out = K * (Ta - T)
    q_cpl = (Kij * (T[None, :] - T[:, None])).sum(1) / 1000.0
    q = q_out + q_cpl + Q_INT + Q_COOL * u
    return T + q / C * DT_H * 3600.0


def run_month(Z, w, policy, days=30, seed=0):
    rng = np.random.default_rng(seed); n = Z["n"]
    T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int)
    e_cum = 0.0; total = 0.0; peak = 0.0; vio = 0; maxv = 0.0; sumv = 0.0; nsteps = 0
    for _ in range(days):
        for s in range(STEPS_PER_DAY):
            Ta = w[s]
            u = policy(T, prev, Ta, Z, e_cum).copy()
            u[T >= Z["hi"] - 0.19] = 1; u[T <= Z["lo"] + 0.19] = 0
            p_kw = P_ELEC * u.sum(); peak = max(peak, p_kw)
            kwh = p_kw * DT_H; total += kwh; e_cum += kwh
            T = step_rc(T, Ta, u, Z)
            over = np.maximum(0, T - Z["hi"]); under = np.maximum(0, Z["lo"] - T)
            v = float(np.max(over + under)); meanv = float(np.mean(over + under))
            if v > 1e-9: vio += 1
            maxv = max(maxv, v); sumv += meanv; nsteps += 1; prev = u
    return dict(kwh=total, bill=bill(total), peak_kw=peak,
                viol_pct=100.0 * vio / nsteps,
                mean_viol_C=sumv / nsteps, max_viol_C=maxv)


# ===================== baselines =====================
def policy_onoff(T, prev, Ta, Z, e):
    u = prev.copy(); u[T >= Z["hi"] - 0.5] = 1; u[T <= Z["lo"] + 0.5] = 0
    return u


# ===================== PPO agent + training =====================
if HAVE_TORCH:
    class Agent(nn.Module):
        def __init__(self, sdim, n):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(),
                                      nn.Linear(256, 256), nn.ReLU())
            self.actor = nn.Sequential(nn.Linear(256, n), nn.Sigmoid())
            self.critic = nn.Linear(256, 1)
        def forward(self, s):
            h = self.body(s); return self.actor(h), self.critic(h)

    def obs(T, prev, Ta, Z, e_cum, t_hour):
        tiz = (Z["adj"] @ T) / np.maximum(Z["adj"].sum(1), 1e-9)
        tiz = np.where(Z["adj"].sum(1) > 0, tiz, T)
        return np.concatenate([
            (T - 24) / 4, (tiz - 24) / 4, prev.astype(float),
            [(Ta - 30) / 15, e_cum / 6000,
             np.sin(2*np.pi*t_hour/24), np.cos(2*np.pi*t_hour/24)]
        ]).astype(np.float32)

    W = dict(peak=0.3, tariff=1.2, comfort=45.0, switch=0.1)  # comfort settable via --comfort

    def reward(u, prev, T, Ta, Z, e, k):
        non = u.sum(); rate = TIER1 if e <= THRESH else TIER2
        over = np.maximum(0, T - Z["hi"]); under = np.maximum(0, Z["lo"] - T)
        return (-W["peak"] * (P_ELEC * non)**2 * 1e-3
                - W["tariff"] * rate * P_ELEC * non * DT_H
                - W["comfort"] * (over + under).sum()
                - W["switch"] * np.abs(u - prev).sum())

    def dp_expert_zone(K, C, w, lo, hi, ngrid=60):
        grid = np.linspace(lo + 0.19, hi - 0.19, ngrid); H = STEPS_PER_DAY; INF = 1e18
        V = np.zeros((ngrid, 2)); POL = np.zeros((H, ngrid, 2), np.int8)
        def nxt(T, u, Ta): return T + (Q_COOL*u + K*(Ta-T) + Q_INT)/C*DT_H*3600.0
        for k in range(H-1, -1, -1):
            Ta = w[k]; Vn = np.full((ngrid, 2), INF)
            for i, T in enumerate(grid):
                for pu in (0, 1):
                    best, ba = INF, 0
                    for u in (0, 1):
                        Tn = nxt(T, u, Ta)
                        if Tn < grid[0]-1e-6 or Tn > grid[-1]+1e-6:
                            v = INF
                        else:
                            j = int(np.clip(round((Tn-grid[0])/(grid[1]-grid[0])), 0, ngrid-1))
                            c = TIER1 * P_ELEC * u * DT_H + (0.04 if (u==1 and pu==0) else 0)
                            v = c + V[j, u]
                        if v < best: best, ba = v, u
                    Vn[i, pu] = best; POL[k, i, pu] = ba
            V = Vn
        return POL, grid

    def collect_demos(Z, w, weeks, seed):
        rng = np.random.default_rng(seed); n = Z["n"]
        experts = [dp_expert_zone(Z["K"][z], Z["C"][z], w, Z["lo"][z], Z["hi"][z]) for z in range(n)]
        O, A = [], []
        for _ in range(weeks):
            T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int); ec = 0.0
            for s in range(STEPS_PER_DAY):
                Ta = w[s]; u = np.zeros(n, int)
                for z in range(n):
                    POL, grid = experts[z]
                    i = int(np.clip(round((T[z]-grid[0])/(grid[1]-grid[0])), 0, len(grid)-1))
                    u[z] = POL[s, i, prev[z]]
                u[T >= Z["hi"]-0.19] = 1; u[T <= Z["lo"]+0.19] = 0
                O.append(obs(T, prev, Ta, Z, ec, s*DT_H)); A.append(u.astype(np.float32))
                ec += P_ELEC*u.sum()*DT_H; T = step_rc(T, Ta, u, Z); prev = u
        return np.array(O, np.float32), np.array(A, np.float32)

    def clone(ag, O, A, epochs):
        opt = torch.optim.Adam(ag.parameters(), lr=1e-3)
        X, Y = torch.tensor(O), torch.tensor(A)
        for _ in range(epochs):
            p, _ = ag(X); loss = nn.functional.binary_cross_entropy(p, Y)
            opt.zero_grad(); loss.backward(); opt.step()

    def finetune(ag, Z, w, episodes, seed, k, lr=1e-5):
        opt = torch.optim.Adam(ag.parameters(), lr=lr); rng = np.random.default_rng(seed); n = Z["n"]
        for _e in range(episodes):
            T = rng.uniform(Z["lo"], Z["hi"]); prev = np.zeros(n, int); ec = 0.0
            S, Ac, LP, R, Vv = [], [], [], [], []
            for s in range(STEPS_PER_DAY):
                Ta = w[s]; o = obs(T, prev, Ta, Z, ec, s*DT_H); ot = torch.from_numpy(o).unsqueeze(0)
                p, v = ag(ot); dist = Bernoulli(p); a = dist.sample()
                if a.sum() > k:
                    idx = torch.topk(p.squeeze(), k).indices
                    aa = torch.zeros_like(a); aa[0, idx] = 1.0; a = aa
                u = a.numpy().flatten().astype(int)
                u[T >= Z["hi"]-0.19] = 1; u[T <= Z["lo"]+0.19] = 0
                r = reward(u, prev, T, Ta, Z, ec, k)
                S.append(o); Ac.append(u.astype(np.float32))
                LP.append(dist.log_prob(a).sum().item()); R.append(r); Vv.append(v.item())
                ec += P_ELEC*u.sum()*DT_H; T = step_rc(T, Ta, u, Z); prev = u
            adv, g = [], 0; Vn = Vv + [0.0]
            for t in reversed(range(len(R))):
                d = R[t] + 0.99*Vn[t+1] - Vn[t]; g = d + 0.99*0.95*g; adv.insert(0, g)
            ret = [a_+v_ for a_, v_ in zip(adv, Vv)]
            St = torch.tensor(np.array(S)); At = torch.tensor(np.array(Ac))
            LPt = torch.tensor(LP); RTt = torch.tensor(ret, dtype=torch.float32)
            ADt = torch.tensor(adv, dtype=torch.float32); ADt = (ADt-ADt.mean())/(ADt.std()+1e-8)
            for _ in range(10):
                p, v = ag(St); dist = Bernoulli(p); nlp = dist.log_prob(At).sum(1)
                ratio = torch.exp(nlp-LPt); s1 = ratio*ADt; s2 = torch.clamp(ratio, 0.8, 1.2)*ADt
                loss = (-torch.min(s1, s2).mean() + 0.5*(RTt-v.squeeze()).pow(2).mean()
                        - 0.001*dist.entropy().sum(1).mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ag.parameters(), 0.5); opt.step()
        return ag

    def trained_policy(ag, k):
        def pol(T, prev, Ta, Z, e):
            o = torch.from_numpy(obs(T, prev, Ta, Z, e, 0.0)).unsqueeze(0)
            with torch.no_grad():
                p, _ = ag(o)
            u = (p.squeeze().numpy() > 0.5).astype(int)
            if u.sum() > k:
                idx = np.argsort(-p.squeeze().numpy())[:k]
                u = np.zeros(len(u), int); u[idx] = 1
            return u
        return pol


# ===================== scenario runner =====================
def run_scenario(cfg_file, city, month, band, seeds, episodes, k):
    lo, hi = (23, 25) if band == "strict" else (22, 26)
    cfg = _find(cfg_file)
    prof = _find(f"{city}_ambient_profiles.csv")
    Z = load_zones(cfg); Z["lo"] = np.full(Z["n"], lo, float); Z["hi"] = np.full(Z["n"], hi, float)
    w = load_weather(month, prof); n = Z["n"]
    on = run_month(Z, w, policy_onoff, days=30, seed=0)

    if not HAVE_TORCH:
        return dict(scenario=cfg_file, month=month, band=band, city=city,
                    onoff_kwh=on["kwh"], onoff_bill=on["bill"], onoff_peak=on["peak_kw"],
                    note="torch not installed -- baseline only; install torch for PI-PPO")

    pis = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        ag = Agent(3*n+4, n)
        O, A = collect_demos(Z, w, weeks=6, seed=sd); clone(ag, O, A, epochs=60)
        ag = finetune(ag, Z, w, episodes, sd, k)
        pis.append(run_month(Z, w, trained_policy(ag, k), days=30, seed=sd))
    m = lambda key: float(np.mean([r[key] for r in pis]))
    s = lambda key: float(np.std([r[key] for r in pis]))
    red = 100*(on["bill"]-m("bill"))/on["bill"] if on["bill"] > 0 else 0.0
    ered = 100*(on["kwh"]-m("kwh"))/on["kwh"] if on["kwh"] > 0 else 0.0
    return dict(scenario=cfg_file, month=month, band=band, city=city, k=k, seeds=list(seeds),
                onoff_kwh=round(on["kwh"]), onoff_bill=round(on["bill"]), onoff_peak=on["peak_kw"],
                pippo_kwh=round(m("kwh")), pippo_bill=round(m("bill")), pippo_bill_sd=round(s("bill"), 1),
                pippo_peak=m("peak_kw"), pippo_viol=round(m("viol_pct"), 1),
                pippo_mean_viol_C=round(m("mean_viol_C"), 3), pippo_max_viol_C=round(m("max_viol_C"), 2),
                cost_reduction_pct=round(red, 1), energy_reduction_pct=round(ered, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="3 seeds x 1000 episodes")
    ap.add_argument("--fast", action="store_true", help="2 seeds x 400 episodes (quickest defensible run)")
    ap.add_argument("--city", default="jeddah", choices=["jeddah", "riyadh"])
    ap.add_argument("--comfort", type=float, default=None,
                    help="override comfort penalty weight (default 45)")
    args = ap.parse_args()
    if args.comfort is not None and HAVE_TORCH:
        W["comfort"] = args.comfort
        print(f"comfort weight set to {args.comfort}")

    if not HAVE_TORCH:
        print("WARNING: torch not installed. Run 'pip install torch' to enable PI-PPO training.")
        print("Proceeding with baseline-only measurements.\n")

    if args.fast:
        seeds = (0, 1); episodes = 400
    elif args.full:
        seeds = (0, 1, 2); episodes = 1000
    else:
        seeds = (0, 1); episodes = 150
    print(f"{'FULL' if args.full else 'QUICK'} run | city={args.city} | seeds={list(seeds)} | episodes={episodes}")
    if not args.full:
        print("(quick = provisional values; use --full for publication numbers)\n")

    matrix = [
        ("villa_5zone.yaml", "July", "strict", 3),
        ("villa_5zone.yaml", "July", "extended", 2),
        ("villa_5zone.yaml", "January", "strict", 3),
        ("villa_5zone.yaml", "April", "strict", 3),
        ("villa_5zone.yaml", "October", "strict", 3),
        ("building_20zone_multifloor.yaml", "July", "strict", 12),
        ("building_20zone_multifloor.yaml", "July", "extended", 10),
    ]
    out = []
    for cfg_file, month, band, k in matrix:
        print(f"  running {cfg_file} | {month} | {band} ...", flush=True)
        try:
            out.append(run_scenario(cfg_file, args.city, month, band, seeds, episodes, k))
        except Exception as e:
            print(f"    ERROR: {e}")

    json.dump(out, open(os.path.join(HERE, "results_generated.json"), "w"), indent=2)
    lines = ["# Generated Results (measured from simulation)",
             f"Run: {'FULL' if args.full else 'QUICK/provisional'} | city={args.city} | seeds={list(seeds)}",
             "",
             "| Scenario | Month | Band | OnOff kWh | OnOff SAR | PI-PPO SAR | Cost% | Energy% | Viol% | MeanViol C | MaxViol C |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        if "cost_reduction_pct" in r:
            lines.append(f"| {r['scenario'].replace('.yaml','')} | {r['month']} | {r['band']} | "
                         f"{r['onoff_kwh']} | {r['onoff_bill']} | {r['pippo_bill']}+/-{r['pippo_bill_sd']} | "
                         f"{r['cost_reduction_pct']} | {r['energy_reduction_pct']} | {r['pippo_viol']} | "
                         f"{r.get('pippo_mean_viol_C','-')} | {r.get('pippo_max_viol_C','-')} |")
    open(os.path.join(HERE, "results_generated.md"), "w").write("\n".join(lines) + "\n")
    print("\nDone.")
    print("  results_generated.json  <- send this file back")
    print("  results_generated.md    <- readable table")


if __name__ == "__main__":
    main()
