import numpy as np, json, time
from scipy.special import ndtr
T, dt = 5.0, 1e-3; K = int(T/dt)
t = np.arange(1,K+1)*dt
w = np.array([3*np.sqrt(2), 3*np.pi, 12.0, 12.3]); a=np.full(4,0.5); chi=np.zeros(4); a0=1.0
G = lambda s: 2/(1+s)**2; Gj=G(1j*w)
x = G(0)*a0 + sum(a[m]*abs(Gj[m])*np.cos(w[m]*t+chi[m]+np.angle(Gj[m])) for m in range(4))
dx=np.diff(x); VT=np.abs(dx).sum()
n_ext=int(np.sum(np.sign(dx[1:])!=np.sign(dx[:-1])))

def sod(z,D):
    """returns event flags eps (K,), reference r (K,) with r[0]=nan, overshoots list"""
    eps=np.zeros(K,bool); r=np.empty(K); r[0]=np.nan; eps[0]=True
    last=z[0]; ov=[]
    for k in range(1,K):
        r[k]=last
        if abs(z[k]-last)>=D:
            eps[k]=True; ov.append(abs(z[k]-last)-D); last=z[k]
    return eps,r,np.array(ov)

def phi(u): return np.exp(-0.5*u*u)/np.sqrt(2*np.pi)
def cfun(al,be):
    P=np.where(al>0, ndtr(-al)-ndtr(-be), ndtr(be)-ndtr(al))
    pa,pb=phi(al),phi(be)
    cev=1-P-al*pa+be*pb
    cno=np.where(P>1e-300,(pa-pb)**2/np.maximum(P,1e-300),0.0)
    return cev,cno

# regressors
psi=np.outer(t,w)+chi
H=np.column_stack([np.full(K,a0)]+[c for m in range(4) for c in (a[m]*np.cos(psi[:,m]),-a[m]*np.sin(psi[:,m]))])

# --- noise-free: max feasible Delta on grid ---
N0=320
def count(D): return int(sod(x,D)[0].sum())
grid=np.arange(0.0022,0.0026,1e-7); Ns=np.array([count(D) for D in grid])
Dc=grid[Ns>=N0].max()
grid2=np.arange(Dc-1e-7,Dc+1e-7,1e-10); Ns2=np.array([count(D) for D in grid2])
Dstar=grid2[Ns2>=N0].max()
eps0,_,ov0=sod(x,Dstar); Nnf=int(eps0.sum())
res=dict(VT=VT,n_ext=n_ext,maxstep=float(np.abs(dx).max()),Dstar=float(Dstar),Nnf=Nnf,
         ov_nf=float(ov0.mean()), ov_pred=float(dt/2*np.sum(dx**2/dt)/np.sum(np.abs(dx))),
         nonmono=[(0.00231050,count(0.00231050)),(0.00231075,count(0.00231075))],
         ymin=float(x.min()),ymax=float(x.max()),Cmin=float((a*abs(Gj)).min()))
print(res)

# --- noisy MC ---
R=2000; rng=np.random.default_rng(20261005)
Z=rng.standard_normal((R,K))     # common draws, scaled by sigma_d
out={}
for d in [5,20,50]:
    s=Dstar/d; cnt=np.zeros(R); ovz=np.zeros(R); S=np.zeros(R); Sev=np.zeros(R)
    Wsum=np.zeros(K); t0=time.time()
    for i in range(R):
        z=x+s*Z[i]; eps,r,ov=sod(z,Dstar)
        cnt[i]=eps.sum(); ovz[i]=ov.mean()
        al=(r[1:]-Dstar-x[1:])/s; be=(r[1:]+Dstar-x[1:])/s
        cev,cno=cfun(al,be); c=np.concatenate([[1.0],cev+cno])
        S[i]=c.sum(); Sev[i]=1.0+cev.sum(); Wsum+=c
    wk=Wsum/R
    Jsod=(H*wk[:,None]).T@H/s**2
    # constant-output control
    ccnt=np.array([sod(2.0+s*Z[i],Dstar)[0].sum() for i in range(300)])
    N=int(np.floor(cnt.mean())); ti=np.arange(1,N+1)*T/N; psiN=np.outer(ti,w)+chi
    Hp=np.column_stack([np.full(N,a0)]+[c for m in range(4) for c in (a[m]*np.cos(psiN[:,m]),-a[m]*np.sin(psiN[:,m]))])
    Jper=Hp.T@Hp/s**2
    ev_s=np.linalg.eigvalsh(Jsod); ev_p=np.linalg.eigvalsh(Jper)
    out[d]=dict(sigma=s,B=cnt.mean(),se=cnt.std(ddof=1)/np.sqrt(R),sd=cnt.std(ddof=1),R=R,
                ovz=ovz.mean(),ovz_rel=ovz.mean()/Dstar,S=S.mean(),S_se=S.std(ddof=1)/np.sqrt(R),Sev=Sev.mean(),
                const_ctrl=ccnt.mean(),N=N,
                logdet_sod=float(np.sum(np.log(ev_s))),cond_sod=float(ev_s[-1]/ev_s[0]),lam_min_sod=float(ev_s[0]),
                logdet_per=float(np.sum(np.log(ev_p))),cond_per=float(ev_p[-1]/ev_p[0]),rank_per=int(np.sum(ev_p>1e-9*ev_p[-1])),
                w_min=float(wk.min()),w_mean=float(wk.mean()),time=time.time()-t0)
    print(d,out[d])
json.dump(dict(res=res,out=out),open('mc83.json','w'),indent=1)
