"""
Honest RC-simulation environment for the PI-PPO study.

This is a REAL, runnable multi-zone RC thermal simulation. It integrates the
heat-balance ODE (Eq. 1) with inter-zone coupling over a 30-day month at 15-min
resolution, driven by the measured Jeddah diurnal profiles, and MEASURES energy,
bill, peak and comfort violations from the resulting trajectory.

It is NOT EnergyPlus and does not claim to be. It is a lumped-parameter RC model,
exactly as the manuscript must be rewritten to state.

Every number this produces comes from executing the dynamics below. Nothing is
hardcoded.
"""
from __future__ import annotations
import numpy as np
import yaml
import os

# Saudi two-tier tariff (Eq. 3) -- the one genuinely-verified piece
TIER1, TIER2, THRESH = 0.18, 0.30, 6000.0

def bill(kwh: float) -> float:
    return TIER1 * min(kwh, THRESH) + TIER2 * max(0.0, kwh - THRESH)

# AC unit model (from configs / manuscript)
P_ELEC = 1.8      # kW electrical draw when ON
Q_COOL = -5.3     # kW thermal cooling when ON
Q_INT  = 0.3      # kW internal gains per zone
DT_H   = 0.25     # 15-min step
STEPS_PER_DAY = 96


def load_zones(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    n = cfg["n_zones"]
    lo = np.array(cfg["comfort_low"], float)
    hi = np.array(cfg["comfort_high"], float)
    if "zones" in cfg:                       # explicit villa
        K = np.array([z["K_i"] for z in cfg["zones"]], float)
        C = np.array([z["C_i"] for z in cfg["zones"]], float)
        adj = np.zeros((n, n))
        Kij = np.zeros((n, n))
        for i, z in enumerate(cfg["zones"]):
            for nb, kij in zip(z.get("neighbors", []), z.get("K_ij", [])):
                adj[i, nb] = 1.0
                Kij[i, nb] = kij
    else:                                    # procedural compound
        g = cfg.get("generation", {})
        rng = np.random.default_rng(g.get("seed", 42))
        base = yaml.safe_load(open(os.path.join(os.path.dirname(cfg_path),
                                                g.get("base", "villa_5zone.yaml"))))
        bK = np.array([z["K_i"] for z in base["zones"]])
        bC = np.array([z["C_i"] for z in base["zones"]])
        pert = g.get("perturbation_pct", 15) / 100.0
        K = np.array([bK[i % len(bK)] * (1 + rng.uniform(-pert, pert)) for i in range(n)])
        C = np.array([bC[i % len(bC)] * (1 + rng.uniform(-pert, pert)) for i in range(n)])
        klo, khi = g.get("K_ij_range", [0.03, 0.06])
        adj = np.zeros((n, n)); Kij = np.zeros((n, n))
        for i in range(n - 1):               # chain topology
            adj[i, i+1] = adj[i+1, i] = 1.0
            k = rng.uniform(klo, khi)
            Kij[i, i+1] = Kij[i+1, i] = k
    return dict(n=n, lo=lo, hi=hi, K=K, C=C, adj=adj, Kij=Kij)


def load_weather(month, profiles_csv):
    import csv
    prof = []
    with open(profiles_csv) as f:
        for row in csv.DictReader(f):
            if row["month"].strip().lower() == month.lower():
                prof.append(float(row["T_a_C"]))
    if len(prof) != STEPS_PER_DAY:
        raise ValueError(f"{month}: expected {STEPS_PER_DAY} pts, got {len(prof)}")
    return np.array(prof)                     # one representative day, 15-min


def step_rc(T, Ta, u, Z):
    """One 15-min RC update with inter-zone coupling. Returns new T."""
    K, C, Kij = Z["K"], Z["C"], Z["Kij"]
    q_out = K * (Ta - T)                              # kW, envelope
    q_cpl = (Kij * (T[None, :] - T[:, None])).sum(1) / 1000.0  # W->kW coupling
    q_cool = Q_COOL * u
    q = q_out + q_cpl + Q_INT + q_cool               # kW
    dT = q / C * DT_H * 3600.0                        # C over the step
    return T + dT


def run_month(Z, weather_day, policy, days=30, seed=0):
    """Real rollout over `days`. Measures everything from the trajectory."""
    rng = np.random.default_rng(seed)
    n = Z["n"]
    T = rng.uniform(Z["lo"], Z["hi"])
    prev = np.zeros(n, int)
    e_cum = 0.0
    total_kwh = 0.0
    peak_kw = 0.0
    viol_steps = 0
    max_viol = 0.0
    for d in range(days):
        for s in range(STEPS_PER_DAY):
            Ta = weather_day[s]
            u = policy(T, prev, Ta, Z, e_cum)
            # hard comfort safety (R1/R2): force ON near upper, OFF near lower
            u = u.copy()
            u[T >= Z["hi"] - 0.19] = 1
            u[T <= Z["lo"] + 0.19] = 0
            p_kw = P_ELEC * u.sum()
            peak_kw = max(peak_kw, p_kw)
            kwh = p_kw * DT_H
            total_kwh += kwh
            e_cum += kwh
            T = step_rc(T, Ta, u, Z)
            over = np.maximum(0, T - Z["hi"]); under = np.maximum(0, Z["lo"] - T)
            v = float(np.max(over + under))
            if v > 1e-9: viol_steps += 1
            max_viol = max(max_viol, v)
            prev = u
    return dict(kwh=total_kwh, bill=bill(total_kwh), peak_kw=peak_kw,
                max_viol=max_viol,
                viol_pct=100.0 * viol_steps / (days * STEPS_PER_DAY))


# ---- baseline policies (real, not stubs) ----
def policy_onoff(T, prev, Ta, Z, e_cum):
    """Independent thermostat with 0.5C hysteresis."""
    u = prev.copy()
    u[T >= Z["hi"] - 0.5] = 1
    u[T <= Z["lo"] + 0.5] = 0
    return u

def policy_gs(k):
    """Green-scheduling lazy controller with concurrency limit k."""
    def pol(T, prev, Ta, Z, e_cum):
        u = prev.copy()
        u[T <= Z["lo"] + 0.05] = 0
        crit = (prev == 0) & (T >= Z["hi"] - 0.05)
        u[crit] = 1
        if u.sum() > k:                       # shed coolest non-critical
            on = np.where(u == 1)[0]
            nonc = on[~crit[on]]
            order = nonc[np.argsort(T[nonc])]
            for i in order[:int(u.sum() - k)]:
                u[i] = 0
        return u
    return pol
