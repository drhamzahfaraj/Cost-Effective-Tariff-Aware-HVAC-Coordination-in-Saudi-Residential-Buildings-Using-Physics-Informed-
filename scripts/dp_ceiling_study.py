import sys, os, numpy as np
sys.path.insert(0,'src')
from rc_sim import load_weather, run_month, policy_onoff, step_rc

def get_weather(month):
    for p in ['configs/jeddah_ambient_profiles.csv','/tmp/jeddah_ambient_profiles.csv']:
        if os.path.exists(p): return load_weather(month, p)
    raise FileNotFoundError
def bill(k): return 0.18*min(k,6000)+0.30*max(0,k-6000)

Q_COOL=-5.3; Q_INT=0.3; DT_H=0.25; STEPS=96; TIER1=0.18; P_ELEC=1.8
baseK=np.array([0.28,0.25,0.24,0.22,0.22]); baseC=np.array([3600,3000,3000,2400,2400])

def dp_zone(K,C,w,lo,hi,ngrid=80):
    grid=np.linspace(lo+0.19,hi-0.19,ngrid); INF=1e18
    V=np.zeros((ngrid,2)); POL=np.zeros((STEPS,ngrid,2),np.int8)
    def nxt(T,u,Ta): return T+(Q_COOL*u+K*(Ta-T)+Q_INT)/C*DT_H*3600.0
    for k in range(STEPS-1,-1,-1):
        Ta=w[k]; Vn=np.full((ngrid,2),INF)
        for i,T in enumerate(grid):
            for pu in (0,1):
                best,ba=INF,0
                for u in (0,1):
                    Tn=nxt(T,u,Ta)
                    if Tn<grid[0]-1e-6 or Tn>grid[-1]+1e-6: v=INF
                    else:
                        j=int(np.clip(round((Tn-grid[0])/(grid[1]-grid[0])),0,ngrid-1))
                        c=TIER1*P_ELEC*u*DT_H+(0.04 if(u==1 and pu==0)else 0); v=c+V[j,u]
                    if v<best: best,ba=v,u
                Vn[i,pu]=best; POL[k,i,pu]=ba
        V=Vn
    return POL,grid

def run_dp(K,C,w,lo,hi,days=30):
    POL,grid=dp_zone(K,C,w,lo,hi)
    T=(lo+hi)/2; prev=0; tot=0.0
    for _ in range(days):
        for s in range(STEPS):
            i=int(np.clip(round((T-grid[0])/(grid[1]-grid[0])),0,len(grid)-1))
            u=POL[s,i,prev]
            if T>=hi-0.19:u=1
            if T<=lo+0.19:u=0
            tot+=P_ELEC*u*DT_H
            T=T+(Q_COOL*u+K*(w[s]-T)+Q_INT)/C*DT_H*3600.0; prev=u
    return tot

w=get_weather("July")
print("=== DP-OPTIMAL CEILING vs SATURATION (single zone, energy savings vs On-Off) ===")
print(f"{'Kscale':>7} {'duty%':>7} {'OnOff kWh':>10} {'DP kWh':>8} {'save%':>7}")
for ks in [0.5,0.7,0.9,1.0,1.1,1.3,1.5]:
    K=0.25*ks; C=3000
    lo,hi=23.,25.
    # On-off single zone
    T=(lo+hi)/2; prev=0; on=0.0
    for _ in range(30):
        for s in range(STEPS):
            if T>=hi-0.5:prev=1
            if T<=lo+0.5:prev=0
            on+=P_ELEC*prev*DT_H
            T=T+(Q_COOL*prev+K*(w[s]-T)+Q_INT)/C*DT_H*3600.0
    dp=run_dp(K,C,w,lo,hi)
    duty=on/(1.8*720)*100
    save=100*(on-dp)/on if on>0 else 0
    print(f"{ks:>7} {duty:>6.1f} {on:>10.0f} {dp:>8.0f} {save:>6.1f}")
