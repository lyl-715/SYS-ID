"""§8.5 stage 1: separate exploration. (A) amplitude scan at zero phases on the simplex p_m=a_m^2;
(B) phase scan at uniform amplitudes on a 6^4 grid. Single scenario (Delta_*, sigma=Delta_*/20).
Common random numbers (R=200, seed 20261010) for every candidate and for the uniform reference.
Per candidate: paired send-count difference D (mean, se), B_u, log det J_u, Delta m_D vs (C) and (B),
m_E vs (C) and (B), nine generalized eigenvalues vs (C), mean event gap, window AC energy."""
import numpy as np, json, time, itertools, sys
from scipy.linalg import eigh
exec(open('mc83.py').read().split('# --- noise-free')[0])   # x (uniform), K, T, t, w, G, Gj, a0, sod, cfun
D = 0.002384694482635785; s = D/20; R = 200
rng = np.random.default_rng(20261010); Z = rng.standard_normal((R, K))
Nper = 316; ti = np.arange(1, Nper+1)*T/Nper

def regressors(a, chi, tt):
    ps = np.outer(tt, w)+chi
    return np.column_stack([np.full(len(tt), a0)]+[c for m in range(4) for c in (a[m]*np.cos(ps[:, m]), -a[m]*np.sin(ps[:, m]))])
def output(a, chi):
    return G(0)*a0 + sum(a[m]*abs(Gj[m])*np.cos(w[m]*t+chi[m]+np.angle(Gj[m])) for m in range(4))
def evaluate(a, chi):
    """returns dict of per-candidate quantities (CRN over the R draws)."""
    Hc = regressors(a, chi, t); xc = output(a, chi)
    cnt = np.zeros(R); gaps = []; Wsum = np.zeros(K)
    for i in range(R):
        z = xc + s*Z[i]; eps, r, _ = sod(z, D); cnt[i] = eps.sum()
        ev = np.where(eps)[0]; gaps.append(np.diff(ev).mean())
        al = (r[1:]-D-xc[1:])/s; be = (r[1:]+D-xc[1:])/s; cev, cno = cfun(al, be)
        Wsum += np.concatenate([[1.0], cev+cno])
    wk = Wsum/R; J = (Hc*wk[:, None]).T@Hc/s**2
    uac = xc-G(0)*a0; Eac = np.mean(uac**2)   # window AC energy of the *output* (1/T int x_ac^2); input AC energy = sum a^2/2 by normalisation
    uin = sum(a[m]*np.cos(w[m]*t+chi[m]) for m in range(4)); Ein = np.mean(uin**2)
    return dict(cnt=cnt, J=J, gap=float(np.mean(gaps)), Ein=float(Ein), Eout=float(Eac))

# reference (C): uniform input, same draws
a_u = np.full(4, 0.5); chi_u = np.zeros(4)
ref = evaluate(a_u, chi_u); Jc = ref['J']; cnt_u = ref['cnt']
Hp = regressors(a_u, chi_u, ti); Jb = Hp.T@Hp/s**2
ldC = np.linalg.slogdet(Jc)[1]; ldB = np.linalg.slogdet(Jb)[1]
print(f"reference (C): B_u={cnt_u.mean():.2f}, logdet={ldC:.3f}; (B): logdet={ldB:.3f}; offset C-B={ldC-ldB:.3f}")

def summarize(a, chi, res, tag):
    Dr = res['cnt']-cnt_u; ld = np.linalg.slogdet(res['J'])[1]
    lamC = eigh(res['J'], Jc, eigvals_only=True); lamB = eigh(res['J'], Jb, eigvals_only=True)
    return dict(tag=tag, a=a.tolist(), chi=chi.tolist(), B=float(res['cnt'].mean()), D=float(Dr.mean()), D_se=float(Dr.std(ddof=1)/np.sqrt(R)),
                dmD_C=float(ld-ldC), dmD_B=float(ld-ldB), mE_C=float(lamC[0]), mE_B=float(lamB[0]), lamC=lamC.tolist(),
                gap=res['gap'], Ein=res['Ein'], Eout=res['Eout'])

part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
results = []; t0 = time.time()
# (A) amplitude scan: compositions of 10 into 4 positive parts -> p_m = n_m/10, a_m = sqrt(p_m)
comps = [c for c in itertools.product(range(1, 10), repeat=4) if sum(c) == 10]
for c in (comps if part == 0 else []):
    a = np.sqrt(np.array(c)/10.0); results.append(summarize(a, chi_u, evaluate(a, chi_u), 'amp'))
print(f"amplitude scan: {len(comps)} candidates, {time.time()-t0:.0f}s")
# (B) phase scan: chi_m in {0, pi/3, ..., 5pi/3}, uniform amplitudes
t1 = time.time(); grid = np.arange(6)*np.pi/3
allphase = list(itertools.product(grid, repeat=4)); q = len(allphase)//4
sel = allphase[(part-1)*q:part*q] if part >= 1 else []
for c in sel:
    chi = np.array(c); results.append(summarize(a_u, chi, evaluate(a_u, chi), 'phase'))
print(f"phase scan part {part}: {len(sel)} candidates, {time.time()-t1:.0f}s")
json.dump(dict(R=R, seed=20261010, Delta=D, sigma=s, Nper=Nper, ldC=ldC, ldB=ldB, B_u=float(cnt_u.mean()), results=results), open(f'mc85_scan_part{part}.json', 'w'))
print(f'saved mc85_scan_part{part}.json')
