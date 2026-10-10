import numpy as np, json, time
from scipy.stats import norm
exec(open('mc83.py').read().split('# --- noise-free')[0])
D=0.002384694482635785
R=10000; seed=20261006; rng=np.random.default_rng(seed)
Z=rng.standard_normal((R,K))
out={}
for d in [5,20,50]:
    s=D/d; t0=time.time()
    cnt=np.zeros(R); ovz=np.zeros(R); S=np.zeros(R); Wsum=np.zeros(K)
    for i in range(R):
        z=x+s*Z[i]; eps,r,ov=sod(z,D)
        cnt[i]=eps.sum(); ovz[i]=ov.mean()
        al=(r[1:]-D-x[1:])/s; be=(r[1:]+D-x[1:])/s
        cev,cno=cfun(al,be); c=np.concatenate([[1.0],cev+cno]); S[i]=c.sum(); Wsum+=c
    wk=Wsum/R
    out[d]=dict(sigma=s,B=cnt.mean(),se=cnt.std(ddof=1)/np.sqrt(R),sd=cnt.std(ddof=1),
                B_lo=cnt.mean()-2*cnt.std(ddof=1)/np.sqrt(R),B_hi=cnt.mean()+2*cnt.std(ddof=1)/np.sqrt(R),
                floor=int(np.floor(cnt.mean())),ovz=ovz.mean(),ovz_rel=ovz.mean()/D,
                S=S.mean(),S_se=S.std(ddof=1)/np.sqrt(R),cnt_min=int(cnt.min()),cnt_max=int(cnt.max()),time=time.time()-t0)
    np.save(f'wk_d{d}.npy',wk); np.save(f'cnt_d{d}.npy',cnt)
    print(d,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[d].items()})
# constant-output control, R=2000, first-order prediction
ctrl={}
for d in [5,20,50]:
    s=D/d; c=np.array([sod(2.0+s*Z[i],D)[0].sum() for i in range(2000)])
    pred=1+(K-1)*2*norm.cdf(-d/np.sqrt(2))
    ctrl[d]=dict(mean=c.mean(),se=c.std(ddof=1)/np.sqrt(2000),pred_first_order=pred)
    print("ctrl",d,ctrl[d])
json.dump(dict(out=out,ctrl=ctrl,R=R,seed=seed,Delta=D),open('mc836.json','w'),indent=1)
