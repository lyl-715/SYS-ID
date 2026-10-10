"""§8.5 stage 2 follow-up (report §8). Run from code_8_3_stage2/stage2_pkg/ after stage2_final.py:  python3 stage2_followup.py
(1) Budget margin of the final candidates, margin = -(D + 2 se) in sends (batch 1 values, seed 20261013), and a re-run of the
    paired 8.3.6(g) decision for NM-0.3-P4312, AB-0.5, NM-0.7-UNIF and UNIF on an independent batch 3 (seed 20261015,
    R=4000; the same draws for the three candidates and the uniform reference).
(2) Independent check of the Im1 bias: the ML covariance check of stage2_final.py (its batch-2 code, lines from
    '# --- batch 2' up to the json.dump of final_batch2.json, exec'd verbatim; only the printed batch/table labels are
    renamed) for NM-none-P5514 and AB-0.5 on batch 4 (seed 20261016, R=4000), CRLB = inverse of the batch-1 J as before.
    Batch 2 and batch 4 are reported side by side and never combined.
(3) Window input power vs long-run power: P_in = window mean AC input power, approximated by the clock-grid average
    (1/K) sum_k u_ac(t_k)^2 (as in evaluate_crn / Table G), with the continuous integral (1/T) int_0^T u_ac^2 dt (analytic,
    checked by quadrature) as a check column; long-run sum a_m^2/2, ratio, and the window output AC power P_out (grid).
(4) Review checks (no new draws): q_k at the turning points of x_k (batch 1); bias sizes per parameter max|b_i|/s_i and
    joint sqrt(T^2/R), T^2/R = b^T S^-1 b (batches 2 and 4); empirical worst-combination variance ratios from the ML samples
    (batch 2) next to the FIM value 1/lambda_min (batch 1); largest m_E in the region-A Nelder-Mead runs under lambda0 >= 0.3.
Outputs (stage2/): stage2_followup.log, followup.json, followup_theta_mle.npz"""
exec(open('stage2_common.py').read())
from scipy.special import log_ndtr
from scipy.stats import f as fdist
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage2_followup.log')
R4, SEED_B1, SEED_B3, SEED_B4 = 4000, 20261013, 20261015, 20261016
print(f"=== stage2_followup.py: budget margin + batch 3 (seed {SEED_B3}, R={R4}); independent bias check batch 4 (seed {SEED_B4}, R={R4}); window power ===")
check_trigger()
B1 = json.load(open(f'{OUT}/final_batch1.json')); b1all = {r['id']: r for r in B1['records']}
B2 = json.load(open(f'{OUT}/final_batch2.json')); b2all = {r['id']: r for r in B2['records']}
out = {}

# --- (1a) budget margin from batch 1 --------------------------------------------------------------------------------------
print(f"\nTable F1a. Budget margin, batch 1 (R={R4}, seed {SEED_B1}): margin = -(D + 2 se) in sends (> 0 means qualified with that margin); "
      f"relative to the reference budget B_u = {B1['B_u']:.3f}")
print(f"{'id':16s} {'D':>8s} {'se':>6s} {'2se':>6s} {'margin':>8s} {'margin/B_u':>10s} {'dec':5s}")
rows = []
for r in B1['records']:
    mg = 0.0-(r['D']+2*r['D_se']); rows.append(dict(id=r['id'], D=r['D'], D_se=r['D_se'], margin=mg, margin_pct=100*mg/B1['B_u'], dec=r['dec']))
    note = '  (qualifies by definition; D = 0, se = 0)' if r['id'] == 'UNIF' else ''
    print(f"{r['id']:16s} {r['D']:+8.3f} {r['D_se']:6.3f} {2*r['D_se']:6.3f} {mg:+8.3f} {100*mg/B1['B_u']:+9.3f}% {r['dec']:5s}{note}")
out['margin_batch1'] = rows

# --- (1b) paired decision on an independent batch 3 -------------------------------------------------------------------------
ids3 = ['NM-0.3-P4312', 'AB-0.5', 'NM-0.7-UNIF', 'UNIF']
c3 = [dict(id=i, a=b1all[i]['a'], chi=b1all[i]['chi']) for i in ids3]
t1 = time.time(); Z3 = np.random.default_rng(SEED_B3).standard_normal((R4, K))
ref3 = evaluate_crn(a_u, chi_u, Z3); raw3 = eval_many(c3, Z3); del Z3
print(f"\nbatch 3 evaluated in {time.time()-t1:.0f} s; reference (C) uniform: B_u = {ref3['cnt'].mean():.3f} +- {ref3['cnt'].std(ddof=1)/np.sqrt(R4):.3f} (se)")
print(f"Table F1b. Paired 8.3.6(g) decision on batch 3 (R={R4}, seed {SEED_B3}, same draws for the candidates and the reference); "
      "batch-1 decision for comparison (not combined)")
print(f"{'id':16s} {'B':>8s} {'D':>8s} {'se':>6s} {'margin':>8s} {'margin/B_u':>10s} {'dec':5s} | {'dmD_C (se)':>15s} {'mE_C':>7s} | batch 1: {'D +- se':>15s} {'dec':5s}")
rows = []
for c, res in zip(c3, raw3):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref3, R4, SEED_B3); q = b1all[c['id']]
    mg = 0.0-(r['D']+2*r['D_se']); Bu3 = float(ref3['cnt'].mean())
    rows.append(dict(id=r['id'], B=r['B'], D=r['D'], D_se=r['D_se'], margin=mg, margin_pct=100*mg/Bu3, dec=r['dec'], dmD_C=r['dmD_C'], dmD_se=r['dmD_se'], mE_C=r['mE_C']))
    print(f"{r['id']:16s} {r['B']:8.3f} {r['D']:+8.3f} {r['D_se']:6.3f} {mg:+8.3f} {100*mg/Bu3:+9.3f}% {r['dec']:5s} | {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:7.4f} | "
          f"{q['D']:+7.3f} +- {q['D_se']:5.3f} {q['dec']:5s}")
flips = [r['id'] for r in rows if r['dec'] != 'ok']
print(f"decisions on batch 3: {dict((r['id'], r['dec']) for r in rows)}; not qualified on batch 3: {flips if flips else 'none'}")
out['batch3'] = dict(R=R4, seed=SEED_B3, B_u=float(ref3['cnt'].mean()), records=rows)

# --- (2) independent bias check: the batch-2 code of stage2_final.py, verbatim, on batch 4 --------------------------------------
src_final = open('stage2_final.py').read()
sec = src_final[src_final.index('# --- batch 2: ML covariance check as in mc84.py'):src_final.index("json.dump(dict(R=R4, seed=SEED_B2")]
sec = sec.replace('print(f"\\nbatch 2: ', 'print(f"\\nbatch 4 (independent bias check): ').replace(
      'Table 17. ML covariance check, batch 2', 'Table F2a. ML covariance check, batch 4 (independent bias check)')   # labels only
assert 'batch 4 (independent bias check): ' in sec and 'Table F2a.' in sec
ids4 = ['NM-none-P5514', 'AB-0.5']
finals = [dict(id=i, a=b1all[i]['a'], chi=b1all[i]['chi'], roles=b1all[i]['roles']) for i in ids4]
b1 = [b1all[i] for i in ids4]            # batch-1 records aligned with `finals` (the code uses b1[j]['J'] as the CRLB)
SEED_B2 = SEED_B4                         # the exec'd code draws its noise from SEED_B2
print(f"\n=== independent bias check: batch 4, seed {SEED_B4}, R={R4}; candidates {ids4}; CRLB from batch 1 (seed {SEED_B1}) ===")
t2_ = time.time(); exec(sec); del Z2b
print(f"batch 4 done in {time.time()-t2_:.0f} s")
print(f"\nTable F2b. Im1 bias and joint bias, batch 4 (R={R4}, seed {SEED_B4}) next to batch 2 (R={R4}, seed 20261014); "
      "each batch on its own, not combined; CRLB from batch 1 (seed 20261013) in both")
print(f"{'id':16s} {'batch':>6s} {'seed':>9s} {'T2':>7s} {'p(T2)':>6s} {'reject 5%':>9s} {'Im1 bias/sd':>11s} {'Im1 t':>6s} {'sd ratio min':>12s} {'max':>6s}")
rows = []
for r4 in b2:
    for lab, seed_, rr in [('2', 20261014, b2all[r4['id']]), ('4', SEED_B4, r4)]:
        m = rr['mle']; bs = m['bias'][2]/m['sd'][2]
        rows.append(dict(id=rr['id'], batch=lab, seed=seed_, T2=rr['T2'], p_T2=rr['p_T2'], im1_bias_sd=bs, im1_t=m['t'][2], ratio_min=min(m['ratio']), ratio_max=max(m['ratio'])))
        print(f"{rr['id']:16s} {lab:>6s} {seed_:9d} {rr['T2']:7.2f} {rr['p_T2']:6.3f} {('yes' if rr['p_T2'] < 0.05 else 'no'):>9s} {bs:+11.3f} {m['t'][2]:+6.2f} "
              f"{min(m['ratio']):12.3f} {max(m['ratio']):6.3f}")
rep = [r['id'] for r in rows if r['batch'] == '4' and r['p_T2'] < 0.05]
print(f"5%-level rejection of zero joint bias on batch 4: {rep if rep else 'none'} (batch 2 rejected for both NM-none-P5514 and AB-0.5)")
out['batch4'] = dict(R=R4, seed=SEED_B4, crlb_from=dict(R=R4, seed=SEED_B1), summary=rows, records=b2)
np.savez_compressed(f'{OUT}/followup_theta_mle.npz', **thetas)

# --- (3) window input power vs long-run power -------------------------------------------------------------------------------
def pin_continuous(a, chi):
    """(1/T) int_0^T (sum_m a_m cos(w_m t + chi_m))^2 dt, exact (product-to-sum, termwise integration)."""
    tot = 0.0
    for m in range(4):
        for n in range(4):
            for al, be in ((w[m]-w[n], chi[m]-chi[n]), (w[m]+w[n], chi[m]+chi[n])):
                tot += 0.5*a[m]*a[n]*(T*np.cos(be) if abs(al) < 1e-14 else (np.sin(al*T+be)-np.sin(be))/al)
    return tot/T
from scipy.integrate import quad
print(f"\nTable F3. Window mean AC power over T = {T} s (clock-grid average (1/K) sum_k u_ac(t_k)^2, as in Table G) vs long-run power, "
      "final candidates (deterministic); check column: continuous (1/T) int u_ac^2 dt")
print(f"{'id':16s} {'P_in window':>11s} {'long-run sum a^2/2':>18s} {'ratio':>6s} {'P_in/P_in(UNIF)':>15s} {'P_out window':>12s} {'Table G P_in':>12s} {'chi3-chi4':>9s} {'P_in continuous':>15s}")
rows = []; PinU = [r['Ein'] for r in B1['records'] if r['id'] == 'UNIF'][0]
for r in B1['records']:
    a = np.asarray(r['a']); chi = np.mod(np.asarray(r['chi']), TWO_PI)
    uin = sum(a[m]*np.cos(w[m]*t+chi[m]) for m in range(4)); Pin = float(np.mean(uin**2))
    Pout = float(np.mean((output(a, chi)-G(0)*a0)**2)); Plr = float(np.sum(a**2)/2)
    assert abs(Pin-r['Ein']) < 1e-12 and abs(Pout-r['Eout']) < 1e-15
    Pc = pin_continuous(a, chi); Pq = quad(lambda tt: sum(a[m]*np.cos(w[m]*tt+chi[m]) for m in range(4))**2, 0, T, limit=400)[0]/T
    assert abs(Pc-Pq) < 1e-9, (Pc, Pq)
    rows.append(dict(id=r['id'], P_in=Pin, P_longrun=Plr, ratio=Pin/Plr, ratio_unif=Pin/PinU, P_out=Pout, P_in_continuous=Pc))
    print(f"{r['id']:16s} {Pin:11.4f} {Plr:18.4f} {Pin/Plr:6.3f} {Pin/PinU:15.3f} {Pout:12.4e} {r['Ein']:12.4f} {np.mod(chi[2]-chi[3], TWO_PI):9.3f} {Pc:15.6f}")
print(f"  grid vs continuous, UNIF: {[q['P_in'] for q in rows if q['id'] == 'UNIF'][0]:.6f} vs {[q['P_in_continuous'] for q in rows if q['id'] == 'UNIF'][0]:.6f}; "
      f"max |grid - continuous| over the final candidates = {max(abs(q['P_in']-q['P_in_continuous']) for q in rows):.6f}")
out['window_power'] = rows

# --- (4a) event rate at the turning points of x_k (batch 1 q_k; t > 1 s to leave out the start-up transient) --------------------
Q = np.load(f'{OUT}/final_qxw.npz'); rows = []
print(f"\nTable F4. Event rate q_k at the turning points of x_k, t > 1 s, batch 1 (R={R4}, seed {SEED_B1}); "
      "'at tp' = q_k at the clock point where x_k changes direction, 'min +-5' = smallest q_k within +-5 clock steps")
print(f"{'id':16s} {'n tp':>4s} {'q at tp: min':>12s} {'max':>7s} {'min +-5: min':>12s} {'max':>7s} | {'mean q_k':>8s}")
for r in B1['records']:
    xq = Q[f"{r['id']}_x"]; qq = Q[f"{r['id']}_q"]; dx_ = np.diff(xq); tp = np.where(np.sign(dx_[1:]) != np.sign(dx_[:-1]))[0]+1; tp = tp[t[tp] > 1.0]
    at = np.array([qq[k] for k in tp]); mn = np.array([qq[max(1, k-5):k+6].min() for k in tp])
    rows.append(dict(id=r['id'], n_tp=int(tp.size), q_at_tp=at.tolist(), q_min_pm5=mn.tolist()))
    print(f"{r['id']:16s} {tp.size:4d} {at.min():12.4f} {at.max():7.4f} {mn.min():12.4f} {mn.max():7.4f} | {qq.mean():8.4f}")
allat = np.concatenate([q['q_at_tp'] for q in rows]); allmn = np.concatenate([q['q_min_pm5'] for q in rows])
print(f"  all {allat.size} turning points: q at tp in [{allat.min():.4f}, {allat.max():.4f}]; min within +-5 steps in [{allmn.min():.4f}, {allmn.max():.4f}]; "
      f"q resolution 1/R = {1/R4:.5f}")
out['turning_points'] = rows

# --- (4b) bias size: per parameter vs joint (batch 2 all seven, batch 4 two; each batch on its own) -----------------------------
print(f"\nTable F5. Bias size, ML estimates: per-parameter max_i |b_i|/s_i (and its MSE share max_i (b_i/s_i)^2), joint standardised "
      f"bias sqrt(T^2/R) and T^2/R = b^T S^-1 b = largest (c^T b)^2 / c^T S c over linear combinations c (R={R4})")
print(f"{'id':16s} {'batch':>5s} {'max|b_i|/s_i':>12s} {'param':>5s} {'max (b_i/s_i)^2':>15s} {'sqrt(T2/R)':>10s} {'T2/R':>8s} {'p(T2)':>6s}")
rows = []
for lab, recs_ in [('2', B2['records']), ('4', b2)]:
    for rr in recs_:
        m = rr['mle']; z = np.array(m['bias'])/np.array(m['sd']); i = int(np.argmax(np.abs(z)))
        rows.append(dict(id=rr['id'], batch=lab, max_b_s=float(abs(z[i])), param=PNAMES[i], T2_over_R=rr['T2']/R4, joint=float(np.sqrt(rr['T2']/R4)), p_T2=rr['p_T2']))
        print(f"{rr['id']:16s} {lab:>5s} {abs(z[i]):12.3f} {PNAMES[i]:>5s} {100*z[i]**2:14.2f}% {np.sqrt(rr['T2']/R4):10.4f} {100*rr['T2']/R4:7.2f}% {rr['p_T2']:6.3f}")
out['bias_size'] = rows

# --- (4c) worst combination: FIM ratio vs empirical ML-sample ratios (batch 2 ML samples, a from batch 1) -------------------------
TH2 = np.load(f'{OUT}/final_theta_mle.npz'); Jc1 = np.load(f'{OUT}/final_J.npz')['J_C']; SU = np.cov(TH2['UNIF_mle'].T, ddof=1)
print(f"\nTable F6. Worst combination a = J_C v_min (batch 1): FIM ratio 1/lambda_min vs empirical variance ratios of a^T theta_hat from the "
      f"batch-2 ML samples (R={R4}, seed 20261014; common draws for all candidates)")
print(f"{'id':16s} {'FIM 1/lam_min':>13s} {'aS_u a / aS_UNIF a':>18s} {'aS_u a / aJ_C^-1 a':>18s}")
rows = []
for r in B1['records']:
    a_ = np.array(r['a_worst']); Su = np.cov(TH2[f"{r['id']}_mle"].T, ddof=1)
    e1 = float(a_@Su@a_/(a_@SU@a_)); e2 = float(a_@Su@a_/(a_@np.linalg.solve(Jc1, a_)))
    rows.append(dict(id=r['id'], fim=1/r['mE_C'], emp_vs_unif=e1, emp_vs_JC=e2))
    print(f"{r['id']:16s} {1/r['mE_C']:13.2f} {e1:18.2f} {e2:18.2f}")
print(f"  relative se of one sample variance at R={R4} (Gaussian): sqrt(2/(R-1)) = {np.sqrt(2/(R4-1)):.4f}")
out['worst_combination_empirical'] = rows

# --- (4d) Nelder-Mead runs under lambda0 >= 0.3 from region A: was the m_E floor violated at every evaluation? -----------------
LR = json.load(open(f'{OUT}/local_runs.json')); rows = []
print("\nTable F7. Region-A Nelder-Mead runs (start P5014) under lambda0 >= 0.3: largest m_E^(C) over all evaluations of the run (R=500 search values)")
for r in LR['runs']:
    if r['lambda0'] is not None and r['start'] == 'P5014':
        mx = max(v['mE'] for v in r['archive']); rows.append(dict(target=r['target'], start=r['start'], n=len(r['archive']), max_mE=mx))
        print(f"  lambda0={r['target']:4s} start {r['start']}: {len(r['archive'])} evaluations, max m_E = {mx:.4f} -> floor violated throughout: {mx < r['lambda0']}")
out['regionA_runs'] = rows
json.dump(out, open(f'{OUT}/followup.json', 'w'), indent=1)
log_runtime('stage2_followup', T_START, f'batch 3 (seed {SEED_B3}) decisions for {len(ids3)} inputs, batch 4 (seed {SEED_B4}) ML check for {len(ids4)}, window power')
