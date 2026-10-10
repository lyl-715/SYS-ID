"""§8.5 stage 1 post-processing (reproduces every number in section_8_5_CN.md beyond mc85_scan.py):
near-uniform amplitude grid, budget classification with the uniform-reference exception, summary statistics,
eta^2 decomposition, paired batch standard errors for selected candidates, interpretation checks, figures."""
import json, numpy as np, itertools, sys
from scipy.linalg import eigh
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('mc85_scan.py').read().split('part = int(sys.argv[1])')[0])   # setup, evaluate(), summarize(), reference (C), Z, cnt_u, Jc, Jb

def qual(r):
    """8.3.6(g): three-way decision; the uniform reference qualifies by definition."""
    if np.allclose(r['chi'], 0) and np.allclose(np.array(r['a'])**2, 0.25): return 'ok'
    return 'ok' if r['D']+2*r['D_se'] < 0 else ('over' if r['D']-2*r['D_se'] > 0 else 'undec')
def counts(rs): return {k: sum(qual(r) == k for r in rs) for k in ['ok', 'over', 'undec']}

# --- near-uniform amplitude grid: p_m = n_m/20, n_m in {3..7}, sum 20 (85 points, includes uniform) ---
try: amp20 = json.load(open('mc85_amp20.json'))
except FileNotFoundError:
    amp20 = [summarize(np.sqrt(np.array(c)/20.0), chi_u, evaluate(np.sqrt(np.array(c)/20.0), chi_u), 'amp20')
             for c in itertools.product(range(3, 8), repeat=4) if sum(c) == 20]
    json.dump(amp20, open('mc85_amp20.json', 'w'))
res = []
for k in range(5): res += json.load(open(f'mc85_scan_part{k}.json'))['results']
amp10 = [r for r in res if r['tag'] == 'amp']; ph = [r for r in res if r['tag'] == 'phase']

def stats(rs, name):
    Dm = np.array([r['dmD_C'] for r in rs]); B = np.array([r['B'] for r in rs])
    print(f"{name}: n={len(rs)} dmD range [{Dm.min():.3f},{Dm.max():.3f}] B range [{B.min():.1f},{B.max():.1f}] counts {counts(rs)} corr(B,dmD)={np.corrcoef(B,Dm)[0,1]:.3f}")
    okb = sorted([r for r in rs if qual(r) == 'ok'], key=lambda r: -r['dmD_C'])[:3]
    for r in okb: print(f"   top qualified: p={np.round(np.array(r['a'])**2,2)} chi/(pi/3)={np.round(np.array(r['chi'])/(np.pi/3)).astype(int)} dmD={r['dmD_C']:+.3f} mE={r['mE_C']:.3f} B={r['B']:.1f} D={r['D']:+.1f}+-{r['D_se']:.1f}")
stats(amp10, 'amplitude coarse'); stats(amp20, 'amplitude near-uniform'); stats(ph, 'phase')
over_low_p1 = [r for r in amp20 if qual(r) == 'over' and np.array(r['a'])[0]**2 <= 0.25]
print("near-uniform: over-budget with p1<=0.25:", [(np.round(np.array(r['a'])**2,2).tolist(), round(r['D'],2)) for r in over_low_p1][:3])

# --- phase scan: eta^2 decomposition (full factorial 6^4) ---
chis = np.array([r['chi'] for r in ph]); k = np.round(chis/(np.pi/3)).astype(int)
Dm = np.array([r['dmD_C'] for r in ph]); B = np.array([r['B'] for r in ph]); mE = np.array([r['mE_C'] for r in ph])
def eta2(y, g): g = np.asarray(g); return sum(y[g == v].size*(y[g == v].mean()-y.mean())**2 for v in np.unique(g))/y.size/y.var()
print("eta2 dmD main:", [round(eta2(Dm, k[:, m]), 3) for m in range(4)], "diff(3,4):", round(eta2(Dm, (k[:, 2]-k[:, 3]) % 6), 3), "diff(1,2):", round(eta2(Dm, (k[:, 0]-k[:, 1]) % 6), 3))
print("eta2 dmD joint pairs:", {f"({i+1},{j+1})": round(eta2(Dm, k[:, i]*6+k[:, j]), 3) for i, j in itertools.combinations(range(4), 2)})
print("eta2 B diff(3,4):", round(eta2(B, (k[:, 2]-k[:, 3]) % 6), 3))
d34 = (k[:, 2]-k[:, 3]) % 6
print("level means dmD by (chi3-chi4):", [round(Dm[d34 == v].mean(), 3) for v in range(6)], " B:", [round(B[d34 == v].mean(), 1) for v in range(6)])
print("slope dmD vs B (nats/send):", round(np.polyfit(B, Dm, 1)[0], 4))
print("qualified & dmD>0:", sum(qual(r) == 'ok' and r['dmD_C'] > 0 for r in ph), "| of which mE>=0.5:", sum(qual(r) == 'ok' and r['dmD_C'] > 0 and r['mE_C'] >= 0.5 for r in ph), "| qualified & mE>=0.8 excluding uniform:", sum(qual(r) == 'ok' and r['mE_C'] >= 0.8 and not np.allclose(r['chi'], 0) for r in ph))
def pick(c): return [r for r in ph if np.allclose(np.round(np.array(r['chi'])/(np.pi/3)).astype(int), c)][0]
for c in [(5, 0, 1, 4), (5, 1, 3, 3), (4, 4, 5, 2), (3, 2, 3, 2), (3, 0, 1, 4)]:
    r = pick(c); print(c, f"B={r['B']:.2f} dmD={r['dmD_C']:+.4f} mE={r['mE_C']:.4f} D={r['D']:+.2f}+-{r['D_se']:.2f} {qual(r)} gap={r['gap']:.2f} steps = {r['gap']*dt:.4f} s")

# --- paired batch se of Delta m_D (10 batches x 20 draws), lambda_min ratio, rotation-only control ---
def Qmat(chi):
    """fixed amplitudes: H_chi = H_0 Q_chi with Q block-diagonal rotations of each (cos, -sin) column pair."""
    Q = np.eye(9)
    for m in range(4):
        c, sn = np.cos(chi[m]), np.sin(chi[m]); i = 1+2*m; Q[i:i+2, i:i+2] = [[c, -sn], [sn, c]]
    return Q
def paired_se(chi):
    Hc = regressors(a_u, chi, t); xc = output(a_u, chi); Hu = regressors(a_u, chi_u, t); lds = []
    for b in range(10):
        Wc = np.zeros(K); Wu = np.zeros(K)
        for i in range(b*20, (b+1)*20):
            for xx, W in [(xc, Wc), (x, Wu)]:
                z = xx+s*Z[i]; eps, r, _ = sod(z, D); al = (r[1:]-D-xx[1:])/s; be = (r[1:]+D-xx[1:])/s; cev, cno = cfun(al, be); W += np.concatenate([[1.0], cev+cno])
        lds.append(np.linalg.slogdet((Hc*(Wc/20)[:, None]).T@Hc)[1]-np.linalg.slogdet((Hu*(Wu/20)[:, None]).T@Hu)[1])
    return np.mean(lds), np.std(lds, ddof=1)/np.sqrt(10)
for c in [(5, 0, 1, 4), (5, 1, 3, 3)]:
    chi = np.array(c)*np.pi/3; m, se = paired_se(chi); Ju = evaluate(a_u, chi)['J']
    Q = Qmat(chi); assert np.allclose(regressors(a_u, chi, t), regressors(a_u, chi_u, t)@Q)
    Jrot = Q.T@Jc@Q                                   # same weights as the reference, rotated regressors only
    print(c, f"paired dmD={m:.4f} se={se:.4f} | lam_min(Ju)/lam_min(Jc)={np.linalg.eigvalsh(Ju)[0]/np.linalg.eigvalsh(Jc)[0]:.4f} cond(Ju)={np.linalg.cond(Ju):.1f} cond(Jc)={np.linalg.cond(Jc):.1f}")
    print(f"   rotation-only (same weights): mE={eigh(Jrot, Jc, eigvals_only=True)[0]:.7f}, logdet diff={np.linalg.slogdet(Jrot)[1]-np.linalg.slogdet(Jc)[1]:.2e}; ordered eigenvalue ratios Ju/Jc: {np.round(np.linalg.eigvalsh(Ju)/np.linalg.eigvalsh(Jc),3)}")

# --- figures ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for data, axi, title in [(amp10+amp20, ax[0], 'amplitude scans, chi=0 (169 points)'), (ph, ax[1], 'phase scan, uniform amplitudes (1296 points)')]:
    Bq = np.array([r['B'] for r in data]); Dq = np.array([r['dmD_C'] for r in data]); mq = np.array([r['mE_C'] for r in data])
    sc = axi.scatter(Bq, Dq, c=mq, cmap='viridis', s=12, vmin=0, vmax=1); axi.axvline(cnt_u.mean(), color='k', lw=0.8, ls='--'); axi.axhline(0, color='k', lw=0.5)
    axi.set_xlabel('expected sends  B_u'); axi.set_ylabel('Delta m_D vs (C)'); axi.set_title(title, fontsize=10)
plt.colorbar(sc, ax=ax[1], label='m_E vs (C)'); plt.tight_layout(); plt.savefig('fig85_scatter.png', dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(range(6), [Dm[d34 == v].mean() for v in range(6)], marker='o'); ax[0].set_xlabel('(chi3 - chi4) / (pi/3)'); ax[0].set_ylabel('mean Delta m_D over the 216 grid points'); ax[0].axhline(0, color='k', lw=0.5)
ax[1].plot(range(6), [B[d34 == v].mean() for v in range(6)], marker='o'); ax[1].set_xlabel('(chi3 - chi4) / (pi/3)'); ax[1].set_ylabel('mean B_u over the 216 grid points'); ax[1].axhline(cnt_u.mean(), color='k', lw=0.8, ls='--')
plt.tight_layout(); plt.savefig('fig85_pair34.png', dpi=150); print("figures saved")
