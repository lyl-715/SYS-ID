"""§8.5 stage 2, task 4: final validation of 5-8 fixed candidates on two independent batches.
Final set (fixed from the R=2000 pool of stage2_local.py, in this order, at most 8): the best qualified candidate for
each target (unconstrained; m_E^(C) >= 0.3, 0.5, 0.7), the uniform reference, per front region (A: m_E<0.1,
B: 0.1<=m_E<0.3, C: m_E>=0.3) the front point with the largest Delta m_D^(C), the stage-1 phase-only best P5014, and
the best qualified candidate with non-uniform amplitudes.
Batch 1 (seed 20261013, R=4000, the same draws for all candidates and the reference): J, paired send-count differences vs
uniform, Delta m_D^(C),(B) with paired batch se (10x400), nine generalized eigenvalues vs (C), worst combination
a = J_C v_min with v_min^T J_C v_min = 1 (parameter order G0,Re1,Im1,Re2,Im2,Re3,Im3,Re4,Im4; sign: largest |a_i| > 0),
event rate per clock point q_k = (1/R) sum_r eps_k^(r) saved with x_k and w_k, gap quantiles (clock steps),
window mean AC power of input and output. q_k summaries: mean over all k; p99 and max over k>=1 (the first clock point
is always an event).
Batch 2 (seed 20261014, R=4000): ML covariance check as in mc84.py (functions nll_grad_hess, mle, report exec'd verbatim
from mc84.py with H = the candidate's regressors; CRLB = inverse of the batch-1 J of the same candidate): sd ratios with
batch se, bias t-values, Hotelling T^2, whitened eigenvalues with the Wishart benchmark (5000 draws, seed 1).
Run from code_8_3_stage2/stage2_pkg/:  python3 stage2_final.py
Outputs (stage2/): stage2_final.log, final_batch1.json, final_qxw.npz, final_J.npz, final_batch2.json, final_theta_mle.npz"""
exec(open('stage2_common.py').read())
from scipy.special import log_ndtr
from scipy.stats import f as fdist
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage2_final.log')
R4, SEED_B1, SEED_B2 = 4000, 20261013, 20261014
print(f"=== stage2_final.py: batch 1 seed {SEED_B1} R={R4}; batch 2 seed {SEED_B2} R={R4} ===")
check_trigger()

# --- fix the final set ----------------------------------------------------------------------------------------------
P = json.load(open(f'{OUT}/pool_R2000.json')); poolr = {r['id']: r for r in P['records']}; best = P['best']
fr = json.load(open(f'{OUT}/front_R2000.json'))['front']
roles = {}
def addf(i, role): roles.setdefault(i, []).append(role)
addf(best['none'], 'best unconstrained')
for lab in ('0.3', '0.5', '0.7'): addf(best[lab], f'best m_E>={lab}')
addf('UNIF', 'uniform reference')
for g, fn in [('A', lambda m: m < 0.1), ('B', lambda m: 0.1 <= m < 0.3), ('C', lambda m: m >= 0.3)]:
    c = [r for r in fr if fn(r['mE_C']) and r['id'] != 'UNIF']
    if c: addf(max(c, key=lambda r: r['dmD_C'])['id'], f'front region {g}')
addf('P5014', 'stage-1 phase-only best')
nu = [r for r in P['records'] if r['dec'] == 'ok' and not np.allclose(r['p'], 0.25)]
addf(max(nu, key=lambda r: r['dmD_C'])['id'], 'best non-uniform amplitudes')
final_ids = list(roles)[:8]
print(f"final set ({len(final_ids)}), R=2000 values (seed 20261012) for reference:")
for i in final_ids:
    r = poolr[i]; print(f"  {i:16s} [{'; '.join(roles[i])}] p={fmt_p(r)} chi={fmt_chi(r)} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} D={r['D']:+.2f}+-{r['D_se']:.2f}")
finals = [dict(id=i, a=poolr[i]['a'], chi=poolr[i]['chi'], roles=roles[i]) for i in final_ids]

# --- batch 1 ----------------------------------------------------------------------------------------------------------
t1 = time.time()
Z1 = np.random.default_rng(SEED_B1).standard_normal((R4, K))
ref = evaluate_crn(a_u, chi_u, Z1, extras=True); Jc = ref['J']; ldC = np.linalg.slogdet(Jc)[1]
raw = eval_many(finals, Z1, extras=True); del Z1
print(f"\nbatch 1 evaluated in {time.time()-t1:.0f} s; reference (C): B_u={ref['cnt'].mean():.3f}, log det J_C={ldC:.4f}, (B) offset C-B={ldC-ldB:+.4f}")
b1 = []; qxw = dict(t=t); Js = dict(J_C=Jc)
for c, res in zip(finals, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R4, SEED_B1); r['roles'] = c['roles']
    lam, V = eigh(res['J'], Jc); v = V[:, 0]; a_w = Jc@v; a_w *= np.sign(a_w[np.argmax(np.abs(a_w))])
    r.update(vJv=float(v@Jc@v), a_worst=a_w.tolist(), crlb_ratio_worst=float(a_w@np.linalg.solve(res['J'], a_w)/(a_w@np.linalg.solve(Jc, a_w))),
             q_mean=float(res['q'].mean()), q_max=float(res['q'][1:].max()), q_p99=float(np.quantile(res['q'][1:], 0.99)), J=res['J'].tolist())   # k=0 is always an event
    b1.append(r); qxw[f"{c['id']}_q"] = res['q']; qxw[f"{c['id']}_x"] = res['x']; qxw[f"{c['id']}_w"] = res['wk']; Js[c['id']] = res['J']
qxw['UNIFref_q'] = ref['q']; qxw['UNIFref_x'] = ref['x']; qxw['UNIFref_w'] = ref['wk']
np.savez_compressed(f'{OUT}/final_qxw.npz', **qxw); np.savez(f'{OUT}/final_J.npz', **Js)

print(f"\nTable 13. Final candidates, batch 1 (R={R4}, seed {SEED_B1}); D = paired send-count difference vs uniform on the same draws; "
      f"dmD_C se = paired batch se 10x{R4//10}")
print(f"{'id':16s} {'B':>8s} {'D +- se':>15s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'dmD_B':>7s} {'mE_C':>7s} {'(se)':>8s} {'mE_B':>6s}")
for r in b1: print(f"{r['id']:16s} {r['B']:8.3f} {r['D']:+7.3f} +- {r['D_se']:5.3f} {r['dec']:5s} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['dmD_B']:+7.3f} {r['mE_C']:7.4f} ({r['mE_se']:6.4f}) {r['mE_B']:6.3f}")
for r in b1:
    if r['id'] != 'UNIF': continue
    print(f"  (UNIF in the pool vs reference: D={r['D']:+.3f}, dmD_C={r['dmD_C']:+.1e}; reference B_u={ref['cnt'].mean():.3f} +- {ref['cnt'].std(ddof=1)/np.sqrt(R4):.3f} (se))")
print(f"\nTable 14. Nine generalized eigenvalues vs (C), batch 1 (R={R4}, seed {SEED_B1})")
for r in b1: print(f"{r['id']:16s} {lam_table_row(r)}")
print(f"\nTable 15. Worst combination a = J_C v_min, v_min^T J_C v_min = 1 (batch 1); CRLB variance of a^T theta is 1/lambda_min times the reference's")
print(f"{'id':16s} " + ' '.join(f"{n:>7s}" for n in PNAMES) + f" {'1/lam_min':>9s} {'check':>7s} {'vJv':>6s}")
for r in b1: print(f"{r['id']:16s} " + ' '.join(f"{v:+7.3f}" for v in r['a_worst']) + f" {1/r['mE_C']:9.2f} {r['crlb_ratio_worst']:7.2f} {r['vJv']:6.3f}")
print("Table 15b. Same directions normalised to unit Euclidean norm (as in mc837.py), for reading which parameters dominate")
for r in b1: print(f"{r['id']:16s} " + ' '.join(f"{v:+7.3f}" for v in np.array(r['a_worst'])/np.linalg.norm(r['a_worst'])))
print(f"\nTable 16. Gaps (clock steps, pooled over R={R4}), event rate q_k, window mean AC power (batch 1)")
print(f"{'id':16s} {'gap':>6s} {'min':>4s} {'q01':>4s} {'q05':>4s} {'q25':>4s} {'q50':>4s} {'q75':>4s} {'q95':>4s} {'q99':>4s} {'max':>5s} | {'q_mean':>7s} {'q_p99':>6s} {'q_max':>6s} | {'P_in':>7s} {'P_out':>9s}")
for r in b1:
    g = r['gapq']
    print(f"{r['id']:16s} {r['gap']:6.2f} {g['min']:4d} {g['q01']:4d} {g['q05']:4d} {g['q25']:4d} {g['q50']:4d} {g['q75']:4d} {g['q95']:4d} {g['q99']:4d} {g['max']:5d} | "
          f"{r['q_mean']:7.4f} {r['q_p99']:6.3f} {r['q_max']:6.3f} | {r['Ein']:7.4f} {r['Eout']:9.3e}")
print("Table 16b. Input descriptors of the final candidates (prediction 3): p, ||p - 1/4||_1, (chi3 - chi4) mod 2pi")
for r in b1: print(f"{r['id']:16s} p={fmt_p(r)} ||p-1/4||_1={np.abs(np.array(r['p'])-0.25).sum():.3f} chi3-chi4={np.mod(r['chi'][2]-r['chi'][3], TWO_PI):.3f}")
srt = sorted([r for r in b1 if r['id'] != 'UNIF'], key=lambda r: -r['dmD_C'])
print("paired differences of Delta m_D^(C) between consecutive final candidates (batch 1):")
for r1, r2 in zip(srt[:-1], srt[1:]): print(f"  {r1['id']} - {r2['id']}: {r1['dmD_C']-r2['dmD_C']:+.4f} (paired batch se {diff_se(r1, r2):.4f})")
r5 = [r for r in b1 if r['id'] == 'P5014'][0]
print("paired differences of Delta m_D^(C) vs the stage-1 phase-only best P5014 (batch 1):")
for r in b1:
    if r['id'] != 'P5014': print(f"  {r['id']} - P5014: {r['dmD_C']-r5['dmD_C']:+.4f} (paired batch se {diff_se(r, r5):.4f})")
json.dump(dict(R=R4, seed=SEED_B1, ldC=ldC, B_u=float(ref['cnt'].mean()), B_u_se=float(ref['cnt'].std(ddof=1)/np.sqrt(R4)), records=b1), open(f'{OUT}/final_batch1.json', 'w'), indent=1)
print(f"batch 1 done, {time.time()-t1:.0f} s")

# --- batch 2: ML covariance check as in mc84.py --------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")   # as in mc84.py: the unused tail branch of np.where may evaluate log1p(-1)
src84 = open('mc84.py').read()
defs84 = src84.split('rng = np.random.default_rng(seed); Z = rng.standard_normal((R, K))')[1].split('t0 = time.time(); TH')[0]
report84 = 'def report(TH, label):'+src84.split('def report(TH, label):')[1].split('out = dict(R=R')[0]
Z2b = np.random.default_rng(SEED_B2).standard_normal((R4, K))
CH = 250
def ml_chunk(task):
    j, i0 = task; c = finals[j]; a = np.asarray(c['a']); chi = np.mod(np.asarray(c['chi']), TWO_PI)
    ns = dict(np=np, H=regressors(a, chi, t), K=K, D=D, s=s, log_ndtr=log_ndtr, ndtr=ndtr); exec(defs84, ns)
    xc = output(a, chi); zz = xc+s*Z2b[i0:i0+CH]; eps, rr = sod_batch(zz, D)
    TH = np.zeros((CH, 9)); TL = np.zeros((CH, 9)); conv = 0; nit = 0
    for i in range(CH):
        TH[i], TL[i], suc, it = ns['mle'](zz[i], eps[i], rr[i]); conv += suc; nit += it
    return j, i0, TH, TL, conv, nit
t2 = time.time()
with pool(4) as pl: outs = pl.map(ml_chunk, [(j, i0) for j in range(len(finals)) for i0 in range(0, R4, CH)], chunksize=1)
print(f"\nbatch 2: {len(finals)}x{R4} ML fits in {time.time()-t2:.0f} s")
# Wishart benchmark of mc84.py (seed 1, 5000 draws): the draws do not depend on the candidate, so the null distribution
# of max_i |lambda_i(W) - 1| is generated once and reused (identical numbers to rerunning mc84's loop per candidate)
rngW = np.random.default_rng(1); M = 5000; p_ = 9
nullmax = np.array([np.max(np.abs(np.linalg.eigvalsh((X := rngW.standard_normal((R4-1, p_))).T@X/(R4-1))-1)) for _ in range(M)])
b2 = []; thetas = {}
for j, c in enumerate(finals):
    parts = sorted([o for o in outs if o[0] == j], key=lambda o: o[1])
    TH = np.vstack([o[2] for o in parts]); TL = np.vstack([o[3] for o in parts]); conv = sum(o[4] for o in parts); nit = sum(o[5] for o in parts)
    Jf = np.array(b1[j]['J']); C_ = np.linalg.inv(Jf)
    a = np.asarray(c['a']); chi = np.mod(np.asarray(c['chi']), TWO_PI); assert np.allclose(regressors(a, chi, t)@theta0, output(a, chi))
    ns = dict(np=np, R=R4, theta0=theta0, C=C_, names=PNAMES); exec(report84, ns)
    print(f"\n######## {c['id']} [{'; '.join(c['roles'])}]: converged {conv}/{R4}, mean Newton iterations {nit/R4:.1f}; CRLB from batch 1 J (seed {SEED_B1})")
    rm = ns['report'](TH, 'MLE, censored likelihood'); rl = ns['report'](TL, 'LS on sent samples only (control)')
    Sg = np.cov(TH.T, ddof=1); L_ = np.linalg.cholesky(Jf); Wm = L_.T@Sg@L_; obs = float(np.max(np.abs(np.linalg.eigvalsh(Wm)-1)))
    tail = float(np.mean(nullmax >= obs))
    m_ = TH.mean(0)-theta0; T2 = float(R4*m_@np.linalg.solve(Sg, m_)); pT2 = float(1-fdist.cdf((R4-p_)/(p_*(R4-1))*T2, p_, R4-p_))
    print(f" whitened eigenvalues (MLE): max|lambda-1| = {obs:.4f}; Wishart benchmark tail probability = {tail:.3f} ({M} draws, seed 1)")
    print(f" Hotelling T2 = {T2:.3f}, p = {pT2:.3f}")
    b2.append(dict(id=c['id'], roles=c['roles'], R=R4, seed=SEED_B2, converged=int(conv), mean_iter=nit/R4, mle=rm, ls_sent=rl,
                   max_abs_lam_minus_1=obs, wishart_tail=tail, T2=T2, p_T2=pT2))
    thetas[f"{c['id']}_mle"] = TH; thetas[f"{c['id']}_ls"] = TL
print(f"\nTable 17. ML covariance check, batch 2 (R={R4}, seed {SEED_B2}); CRLB = J^-1 from batch 1 (R={R4}, seed {SEED_B1}); Wishart benchmark 5000 draws seed 1")
print(f"{'id':16s} {'conv':>9s} {'sd ratio min':>12s} {'max':>6s} {'max|t|':>6s} {'logdet diff':>11s} {'max|lam-1|':>10s} {'W-tail':>6s} {'T2':>7s} {'p(T2)':>6s} | LS-sent logdet diff")
for r in b2:
    m = r['mle']
    print(f"{r['id']:16s} {r['converged']:4d}/{R4} {min(m['ratio']):12.3f} {max(m['ratio']):6.3f} {max(abs(v) for v in m['t']):6.2f} {m['logdet_diff']:+11.3f} "
          f"{r['max_abs_lam_minus_1']:10.4f} {r['wishart_tail']:6.3f} {r['T2']:7.2f} {r['p_T2']:6.3f} | {r['ls_sent']['logdet_diff']:+.3f}")
json.dump(dict(R=R4, seed=SEED_B2, crlb_from=dict(R=R4, seed=SEED_B1), records=b2), open(f'{OUT}/final_batch2.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/final_theta_mle.npz', **thetas)
log_runtime('stage2_final', T_START, f'{len(finals)} final candidates, batch 1 + batch 2 at R={R4}')
