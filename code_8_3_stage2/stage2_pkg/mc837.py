"""8.3.7: weight sum S, J_sod(u_unif), J_per(N) and generalized eigenvalues at d=20, with batch standard errors."""
import numpy as np, json
from scipy.linalg import eigh
exec(open('mc83.py').read().split('# --- noise-free')[0])   # x, H, K, T, sod(), cfun()
D = 0.002384694482635785; d = 20; s = D/d; N = 316; R = 10000; B = 10
rng = np.random.default_rng(20261006); Z = rng.standard_normal((R,K))   # same draws as mc836.py
ti = np.arange(1,N+1)*T/N; ps = np.outer(ti,w)+chi
Hp = np.column_stack([np.full(N,a0)]+[c for m in range(4) for c in (a[m]*np.cos(ps[:,m]),-a[m]*np.sin(ps[:,m]))])
Jper = Hp.T@Hp/s**2; ev_p = np.linalg.eigvalsh(Jper)
batches = []; Wtot = np.zeros(K); Sev_all = []; Sno_all = []
for b in range(B):
    Wsum = np.zeros(K); S = []
    for i in range(b*R//B,(b+1)*R//B):
        z = x+s*Z[i]; eps, r, ov = sod(z, D)                 # r[k] = reference BEFORE the update at k
        al = (r[1:]-D-x[1:])/s; be = (r[1:]+D-x[1:])/s
        cev, cno = cfun(al, be); c = np.concatenate([[1.0], cev+cno])
        Wsum += c; S.append(c.sum()); Sev_all.append(1.0 + cev.sum()); Sno_all.append(cno.sum())
    wk = Wsum/(R//B); Wtot += Wsum
    Jsod = (H*wk[:,None]).T@H/s**2; ev = np.linalg.eigvalsh(Jsod)
    batches.append(dict(S=np.mean(S), logdet=np.sum(np.log(ev)), lam=eigh(Jsod, Jper, eigvals_only=True)))
wk = Wtot/R; Jsod = (H*wk[:,None]).T@H/s**2; ev = np.linalg.eigvalsh(Jsod); lam, V = eigh(Jsod, Jper)
se = lambda a: np.std(a, axis=0, ddof=1)/np.sqrt(B)
print("J_per: rank", int((ev_p>1e-9*ev_p[-1]).sum()), "cond", ev_p[-1]/ev_p[0], "logdet", np.sum(np.log(ev_p)))
print("J_sod: rank", int((ev>1e-9*ev[-1]).sum()), "cond", ev[-1]/ev[0], "logdet", np.sum(np.log(ev)), "batch se", se([b['logdet'] for b in batches]))
print("S", np.mean([b['S'] for b in batches]), "batch se", se([b['S'] for b in batches]))
print("S_ev", np.mean(Sev_all), "S_no", np.mean(Sno_all), "  (S_ev - B) + S_no = S - B)")
print("lambda_i", np.round(lam,4), "se", np.round(se(np.array([b['lam'] for b in batches])),4))
print("Delta m_D", np.sum(np.log(lam)), "m_E", lam[0])
a_worst = Jper @ V[:, 0]; a_worst /= np.linalg.norm(a_worst)   # coefficients of the worst CRLB-ratio combination (= 1/lambda_1)
print("a_worst", np.round(a_worst,3))
print("CRLB diag ratio sod/per", np.round(np.diag(np.linalg.inv(Jsod))/np.diag(np.linalg.inv(Jper)),3))
np.save('wk_d20_10000.npy', wk)
