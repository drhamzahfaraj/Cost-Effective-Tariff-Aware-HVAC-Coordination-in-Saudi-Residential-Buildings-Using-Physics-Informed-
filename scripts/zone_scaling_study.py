import sys, os, numpy as np
sys.path.insert(0,'src')
from rc_sim import load_weather, run_month, policy_onoff, policy_gs
def gw(m):
    for p in ['configs/jeddah_ambient_profiles.csv','/tmp/jeddah_ambient_profiles.csv']:
        if os.path.exists(p): return load_weather(m,p)
def bill(k): return 0.18*min(k,6000)+0.30*max(0,k-6000)
baseK=np.array([0.28,0.25,0.24,0.22,0.22]); baseC=np.array([3600,3000,3000,2400,2400])
def make(n,seed=42):
    rng=np.random.default_rng(seed)
    K=np.array([baseK[i%5]*(1+rng.uniform(-0.15,0.15)) for i in range(n)])
    C=np.array([baseC[i%5]*(1+rng.uniform(-0.15,0.15)) for i in range(n)])
    adj=np.zeros((n,n));Kij=np.zeros((n,n))
    for i in range(n-1):adj[i,i+1]=adj[i+1,i]=1;Kij[i,i+1]=Kij[i+1,i]=0.05
    return dict(n=n,lo=np.full(n,23.),hi=np.full(n,25.),K=K,C=C,adj=adj,Kij=Kij)
w=gw("July")
print("ZONE-SCALING TABLE DATA (matched physics, July strict):")
for n in [5,10,20,40]:
    Z=make(n); on=run_month(Z,w,policy_onoff,days=30,seed=0)
    duty=on['kwh']/(1.8*n*720)*100; t2=max(0,on['kwh']-6000)/on['kwh']*100
    print(f"  n={n:>3}: OnOff {on['kwh']:>6.0f} kWh, duty {duty:.0f}%, Tier-2 frac {t2:.0f}%")
