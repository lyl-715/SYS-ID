"""§8.5 stage 2, task 3: local refinement by Nelder-Mead (scipy) with penalties, from 3 starting points per target
taken from different regions of the R=2000 empirical front (stage2_refine.py).
Targets: unconstrained max Delta m_D^(C), and max Delta m_D^(C) subject to m_E^(C) >= lambda0, lambda0 in {0.3,0.5,0.7}.
Parametrisation (protocol): x = (z1,z2,z3, chi1..chi4), p = softmax(z1,z2,z3,0), a_m = sqrt(p_m); chi continuous inside
the search, reduced mod 2*pi only when forming the input and when saving.
Search objective (R=500 inside, maxfev=150 per start):
    f(x) = -Delta m_D^(C) + MU_B*max(0, D + 2 se) + MU_E*max(0, lambda0 - m_E^(C)),   MU_B = 1 nat/send, MU_E = 20 nat
(D, se: paired send-count difference vs the uniform reference on the same R=500 draws).  The penalty only steers the
search; budget decisions and rankings use the paired R=2000 decision of 8.3.6(g).
Modelling choices not fixed by the protocol (stated, not settled): search draws = seed 20261010 extended to R=500 (the
first 200 rows are the stage-1/LHS draws; independent of the ranking draws 20261012); penalty weights MU_B, MU_E;
initial simplex x0 + h e_j with h = 0.25 in z and 0.25 rad in chi; front regions A: m_E<0.1, B: 0.1<=m_E<0.3,
C: m_E>=0.3 (uniform included); in each region the start is the front point with the smallest R=2000 value of the
target's penalised objective.
Every new candidate (each run's result and, per target, the best penalised point visited by any run) is re-evaluated
at R=2000 on the refinement draws (seed 20261012) before any ranking; rankings use R=2000 estimates only.
Control for prediction 1 ("most of any gain comes from phases"): for each target, the best qualified feasible candidate
produced by its search is also evaluated with the same phases and exactly uniform amplitudes (id PO-...), on the same
R=2000 draws; these controls enter the R=2000 pool like any other evaluated candidate.
Run from code_8_3_stage2/stage2_pkg/:  python3 stage2_local.py
Outputs (stage2/): stage2_local.log, local_runs.json, local_R2000.json, pool_R2000.json, front_R2000.json"""
exec(open('stage2_common.py').read())
from scipy.optimize import minimize
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage2_local.log')
RS, SEEDS = 500, 20261010
R2, SEED2 = 2000, 20261012
MU_B, MU_E, H_Z, H_CHI, MAXFEV = 1.0, 20.0, 0.25, 0.25, 150
TARGETS = [('none', None), ('0.3', 0.3), ('0.5', 0.5), ('0.7', 0.7)]
REGIONS = [('A', lambda m: m < 0.1), ('B', lambda m: 0.1 <= m < 0.3), ('C', lambda m: m >= 0.3)]
print(f"=== stage2_local.py: Nelder-Mead, R={RS} inside (seed {SEEDS}), maxfev={MAXFEV}, MU_B={MU_B}, MU_E={MU_E}, h=({H_Z},{H_CHI}); "
      f"re-evaluation R={R2} seed {SEED2} ===")
check_trigger()
ref2 = json.load(open(f'{OUT}/refine_R2000.json')); recs2 = ref2['records']
ok2 = [r for r in recs2 if r['dec'] == 'ok']; fr2 = [ok2[i] for i in pareto_front(ok2)]
ZS = np.random.default_rng(SEEDS).standard_normal((RS, K))
REFS = evaluate_crn(a_u, chi_u, ZS); ldCs = np.linalg.slogdet(REFS['J'])[1]
print(f"search reference (uniform, R={RS}, seed {SEEDS}): B_u={REFS['cnt'].mean():.3f}, log det J_C={ldCs:.4f}")

def x_to_input(x): return np.sqrt(p_of_z(np.asarray(x[:3]))), np.mod(np.asarray(x[3:]), TWO_PI)
def input_to_x(p, chi): p = np.asarray(p); return np.concatenate([np.log(p[:3]/p[3]), np.asarray(chi, float)])
def eval_s(x):
    a, chi = x_to_input(x); res = evaluate_crn(a, chi, ZS); Dr = res['cnt']-REFS['cnt']
    return dict(dmD=float(np.linalg.slogdet(res['J'])[1]-ldCs), mE=float(eigh(res['J'], REFS['J'], eigvals_only=True)[0]),
                D=float(Dr.mean()), Dse=float(Dr.std(ddof=1)/np.sqrt(RS)))
def pen(dmD, mE, D_, Dse, l0):
    return -dmD + MU_B*max(0.0, D_+2*Dse) + (MU_E*max(0.0, l0-mE) if l0 is not None else 0.0)
def pen_rec(r, l0): return pen(r['dmD_C'], r['mE_C'], r['D'], r['D_se'], l0)

def nm_run(task):
    lab, l0, sid, x0 = task; x0 = np.asarray(x0, float); arch = []
    def f(x):
        v = eval_s(x); v['x'] = np.asarray(x).tolist(); v['f'] = pen(v['dmD'], v['mE'], v['D'], v['Dse'], l0); arch.append(v); return v['f']
    sim = np.vstack([x0]+[x0+h*e for h, e in zip([H_Z]*3+[H_CHI]*4, np.eye(7))])
    t0 = time.time()
    res = minimize(f, x0, method='Nelder-Mead', options=dict(maxfev=MAXFEV, initial_simplex=sim, xatol=1e-4, fatol=1e-5))
    ib = int(np.argmin([v['f'] for v in arch]))
    return dict(target=lab, lambda0=l0, start=sid, x0=x0.tolist(), x=np.asarray(res.x).tolist(), f=float(res.fun), nfev=int(res.nfev),
                nit=int(res.nit), msg=str(res.message), start_eval=arch[0], best_eval=arch[ib], archive=arch, time=time.time()-t0)

# --- starting points: per target, per front region, the front point with the smallest R=2000 penalised objective --------
print(f"\nR=2000 front (stage2_refine.py) by region: " + '; '.join(f"{g}: " + ','.join(r['id'] for r in fr2 if fn(r['mE_C'])) for g, fn in REGIONS))
tasks = []
for lab, l0 in TARGETS:
    for g, fn in REGIONS:
        cand = [r for r in fr2 if fn(r['mE_C'])]; st = min(cand, key=lambda r: pen_rec(r, l0))
        tasks.append((lab, l0, st['id'], input_to_x(st['p'], st['chi']).tolist()))
        print(f"  target lambda0={lab:4s} region {g}: start {st['id']:6s} (R=2000: dmD_C={st['dmD_C']:+.3f}, mE_C={st['mE_C']:.3f}, penalised f={pen_rec(st, l0):+.3f})")
t1 = time.time()
with pool(4) as pl: runs = pl.map(nm_run, tasks, chunksize=1)
print(f"\n{len(runs)} Nelder-Mead runs, {sum(r['nfev'] for r in runs)} evaluations at R={RS}, wall {time.time()-t1:.0f} s")

print(f"\nTable 8. Nelder-Mead runs (values at R={RS}, seed {SEEDS}; f = penalised objective of the run's target)")
print(f"{'target':6s} {'start':6s} {'nfev':>4s} {'f(start)':>9s} {'f(end)':>9s} | end: {'dmD_C':>7s} {'mE_C':>6s} {'D +- se':>13s}  p, chi (mod 2pi)")
for r in runs:
    b = r['best_eval']; a, chi = x_to_input(r['x'])
    print(f"{r['target']:6s} {r['start']:6s} {r['nfev']:4d} {r['start_eval']['f']:+9.3f} {r['f']:+9.3f} | {b['dmD']:+7.3f} {b['mE']:6.3f} {b['D']:+6.2f}+-{b['Dse']:.2f}  "
          f"p=({','.join(f'{v:.3f}' for v in a**2)}) chi=({','.join(f'{v:.3f}' for v in chi)})")

# --- new candidates: each run's result + per target the best penalised point over all archives -----------------------------
new = {}
def add(cid_, x, why):
    a, chi = x_to_input(x); key = tuple(np.round(a**2, 7))+tuple(np.round(chi, 7))
    if key in new: new[key]['why'].append(why); return
    new[key] = dict(id=cid_, a=a.tolist(), chi=chi.tolist(), x=list(x), why=[why])
startkeys = {}
for r in recs2: startkeys[tuple(np.round(np.array(r['p']), 7))+tuple(np.round(np.mod(r['chi'], TWO_PI), 7))] = r['id']
for r in runs: add(f"NM-{r['target']}-{r['start']}", r['x'], f"run {r['target']}/{r['start']}")
allv = [v for r in runs for v in r['archive']]
for lab, l0 in TARGETS:
    b = min(allv, key=lambda v: pen(v['dmD'], v['mE'], v['D'], v['Dse'], l0)); add(f"AB-{lab}", b['x'], f"archive-best {lab}")
newl = []
for key, c in new.items():
    if key in startkeys: print(f"  {c['id']} ({'; '.join(c['why'])}) coincides with refined candidate {startkeys[key]} -> no new evaluation"); continue
    newl.append(c)
print(f"\n{len(newl)} new candidates -> re-evaluation at R={R2} (seed {SEED2}), same draws and reference as stage2_refine.py")
Z2 = np.random.default_rng(SEED2).standard_normal((R2, K)); ref = evaluate_crn(a_u, chi_u, Z2)
assert np.linalg.slogdet(ref['J'])[1] == ref2['ldC'], "reference differs from stage2_refine.py"
raw = eval_many(newl, Z2); newrecs = []
s500 = {tuple(np.round(np.array(v['x']), 12)): v for v in allv}
for c, res in zip(newl, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src='nm'); r['why'] = c['why']; r['x'] = c['x']
    r['R500'] = s500.get(tuple(np.round(np.array(c['x']), 12)), {}); newrecs.append(r)
print(f"\nTable 9. New candidates at R={R2} (seed {SEED2}); R={RS} search values for comparison")
print(f"{'id':16s} {'B':>7s} {'D +- se':>14s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'(se)':>8s} | {'R500 dmD':>8s} {'mE':>6s}  p, chi")
for r in sorted(newrecs, key=lambda r: -r['dmD_C']):
    q = r['R500']
    print(f"{r['id']:16s} {r['B']:7.2f} {r['D']:+7.2f} +- {r['D_se']:4.2f} {r['dec']:5s} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:6.3f} ({r['mE_se']:6.4f}) | "
          f"{q.get('dmD', np.nan):+8.3f} {q.get('mE', np.nan):6.3f}  p={fmt_p(r)} chi={fmt_chi(r)}")
d5 = np.array([r['dmD_C']-r['R500']['dmD'] for r in newrecs if r['R500']])
print(f"R=2000 minus R=500 (search optimism) for dmD_C: mean {d5.mean():+.4f}, range [{d5.min():+.4f},{d5.max():+.4f}]")
print(f"\nTable 10. Nine generalized eigenvalues vs (C), new candidates, R={R2}")
for r in sorted(newrecs, key=lambda r: -r['dmD_C']): print(f"{r['id']:16s} {lam_table_row(r)}")

# --- control: same phases, exactly uniform amplitudes, for each target's search-best (prediction 1) ---------------------
def runs_of(lab, r): return any(w.startswith(f"run {lab}/") or w == f"archive-best {lab}" for w in r['why'])
sbest = {}
for lab, l0 in TARGETS:
    mine = [r for r in newrecs if r['dec'] == 'ok' and (l0 is None or r['mE_C'] >= l0-1e-9) and runs_of(lab, r)]
    if mine: sbest[lab] = max(mine, key=lambda r: r['dmD_C'])
ctrl = {}
for lab, r in sbest.items():
    if not np.allclose(r['p'], 0.25): ctrl.setdefault(r['id'], dict(id=f"PO-{r['id']}", a=a_u.tolist(), chi=r['chi'], of=r['id']))
ctrl = list(ctrl.values()); rawc = eval_many(ctrl, Z2); ctrlrecs = []
for c, res in zip(ctrl, rawc):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src='ctrl'); r['why'] = [f"phase-only projection of {c['of']}"]; ctrlrecs.append(r)
startof = {f"NM-{t[0]}-{t[2]}": t[2] for t in tasks}
byid = {r['id']: r for r in recs2+newrecs+ctrlrecs}
print(f"\nTable 9b. Prediction-1 control at R={R2}: gain of each search-best over its start, split into the phase-only projection "
      "(same chi, p = 1/4) and the remainder (amplitudes + interaction); paired batch se in brackets")
for c, rc in zip(ctrl, ctrlrecs):
    w_ = byid[c['of']]; st = byid.get(startof.get(c['of'], ''), None)
    line = f"  {c['of']:16s} dmD_C={w_['dmD_C']:+.3f} | {rc['id']:19s} dmD_C={rc['dmD_C']:+.3f} mE_C={rc['mE_C']:.3f} D={rc['D']:+.2f}+-{rc['D_se']:.2f} {rc['dec']:5s}"
    line += f" | amplitude part (winner - projection) {w_['dmD_C']-rc['dmD_C']:+.4f} ({diff_se(w_, rc):.4f})"
    if st is not None:
        line += f"; start {st['id']} {st['dmD_C']:+.3f}: phase part (projection - start) {rc['dmD_C']-st['dmD_C']:+.4f} ({diff_se(rc, st):.4f})"
    print(line)

# --- ranking among all R=2000 estimates (refined 40 + new + controls) ----------------------------------------------------
newrecs = newrecs+ctrlrecs
poolr = recs2+newrecs; okp = [r for r in poolr if r['dec'] == 'ok']; fr = [okp[i] for i in pareto_front(okp)]
print(f"\nTable 11. Empirical Pareto front of the evaluated qualified candidates, R={R2} (seed {SEED2}), pool of {len(poolr)} "
      f"(refined {len(recs2)} + new {len(newrecs)-len(ctrlrecs)} + controls {len(ctrlrecs)}): {len(fr)} points")
for r in fr:
    print(f"  {r['id']:16s} p={fmt_p(r)} chi={fmt_chi(r)} B={r['B']:7.2f} D={r['D']:+6.2f}+-{r['D_se']:.2f} dmD_C={r['dmD_C']:+.3f}({r['dmD_se']:.3f}) mE_C={r['mE_C']:.3f}({r['mE_se']:.4f})")
print(f"\nTable 12. Per target, R={R2}: best qualified in the pool with m_E^(C) >= lambda0, and best produced by that target's search")
best = {}; pool_best = []; search_best = []
for lab, l0 in TARGETS:
    feas = [r for r in okp if l0 is None or r['mE_C'] >= l0-1e-9]; srt = sorted(feas, key=lambda r: -r['dmD_C']); b = srt[0]; best[lab] = b['id']
    mine = [r for r in newrecs if r['src'] == 'nm' and r['dec'] == 'ok' and (l0 is None or r['mE_C'] >= l0-1e-9) and runs_of(lab, r)]
    starts = [r for r in recs2 if r['id'] in {t[2] for t in tasks if t[0] == lab} and r['dec'] == 'ok' and (l0 is None or r['mE_C'] >= l0-1e-9)]
    sb = max(mine, key=lambda r: r['dmD_C']) if mine else None; st = max(starts, key=lambda r: r['dmD_C']) if starts else None
    pool_best.append(b['dmD_C']); search_best.append(sb['dmD_C'] if sb else np.nan)
    print(f"  lambda0={lab:4s}: pool best {b['id']:16s} dmD_C={b['dmD_C']:+.3f} (se {b['dmD_se']:.3f}) mE_C={b['mE_C']:.3f} (se {b['mE_se']:.4f}) D={b['D']:+.2f}+-{b['D_se']:.2f} "
          f"runner-up {srt[1]['id']} {srt[1]['dmD_C']:+.3f} (diff {b['dmD_C']-srt[1]['dmD_C']:+.4f}, paired se {diff_se(b, srt[1]):.4f})")
    print(f"               search best (qualified, feasible, from this target's runs) "
          + (f"{sb['id']} dmD_C={sb['dmD_C']:+.3f} mE_C={sb['mE_C']:.3f}" if sb else "none")
          + (f"; best feasible start {st['id']} dmD_C={st['dmD_C']:+.3f}; gain {sb['dmD_C']-st['dmD_C']:+.4f} (paired se {diff_se(sb, st):.4f})" if sb and st else ""))
print(f"P2 (R={R2}): pool max dmD_C for lambda0 = none/0.3/0.5/0.7: {np.round(pool_best, 3).tolist()} -> non-increasing: {all(np.diff(pool_best) <= 0)}")
sbv = np.array(search_best); fin = np.isfinite(sbv)
print(f"    per-target search outcomes: {np.round(sbv, 3).tolist()} -> non-increasing over the targets with an outcome: {all(np.diff(sbv[fin]) <= 0)}")
lam_grid = np.round(np.arange(0, 1.0001, 0.05), 2)
Dst = [max(r['dmD_C'] for r in okp if r['mE_C'] >= l-1e-9) for l in lam_grid]
print("    pool max dmD_C over qualified with mE_C >= lambda0 (grid): " + ', '.join(f"{l:.2f}:{d_:+.3f}" for l, d_ in zip(lam_grid, Dst)) + f" -> non-increasing: {all(np.diff(Dst) <= 0)}")
print("    front shape, segment slopes d mE / d dmD along the R=2000 front:")
for r1, r2 in zip(fr[:-1], fr[1:]):
    print(f"     {r1['id']}({r1['dmD_C']:+.3f},{r1['mE_C']:.3f}) -> {r2['id']}({r2['dmD_C']:+.3f},{r2['mE_C']:.3f}): {(r2['mE_C']-r1['mE_C'])/(r2['dmD_C']-r1['dmD_C']):+.2f} per nat")
lo_ = [r for r in fr if r['mE_C'] < 0.1]; hi_ = [r for r in fr if r['mE_C'] >= 0.6]
print(f"    front points with mE_C<0.1: {len(lo_)}, dmD_C span {max(r['dmD_C'] for r in lo_)-min(r['dmD_C'] for r in lo_):.3f} nat, mE_C span {max(r['mE_C'] for r in lo_)-min(r['mE_C'] for r in lo_):.3f}; "
      f"with mE_C>=0.6: {len(hi_)}, dmD_C span {max(r['dmD_C'] for r in hi_)-min(r['dmD_C'] for r in hi_):.3f} nat, mE_C span {max(r['mE_C'] for r in hi_)-min(r['mE_C'] for r in hi_):.3f}")
ua = [r for r in okp if not np.allclose(r['p'], 0.25)]; bu = max(ua, key=lambda r: r['dmD_C'])
bq = max(okp, key=lambda r: r['dmD_C']); p5 = [r for r in poolr if r['id'] == 'P5014'][0]
print(f"P1 (R={R2}): best qualified in pool {bq['id']} dmD_C={bq['dmD_C']:+.3f} p={fmt_p(bq)}; stage-1 phase-only best P5014 at R={R2}: {p5['dmD_C']:+.3f}; "
      f"difference {bq['dmD_C']-p5['dmD_C']:+.4f} (paired se {diff_se(bq, p5):.4f}); best qualified with non-uniform amplitudes: {bu['id']} dmD_C={bu['dmD_C']:+.3f} p={fmt_p(bu)}")
print("P3 (read off), top-8 qualified in the pool: (chi3-chi4) mod 2pi, ||p-1/4||_1")
for r in sorted(okp, key=lambda r: -r['dmD_C'])[:8]:
    print(f"   {r['id']:16s} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} chi3-chi4={np.mod(r['chi'][2]-r['chi'][3], TWO_PI):.3f} ||p-1/4||_1={np.abs(np.array(r['p'])-0.25).sum():.3f} p={fmt_p(r)}")
json.dump(dict(RS=RS, seedS=SEEDS, MU_B=MU_B, MU_E=MU_E, h=[H_Z, H_CHI], maxfev=MAXFEV, tasks=tasks, runs=runs), open(f'{OUT}/local_runs.json', 'w'))
json.dump(dict(R=R2, seed=SEED2, records=newrecs, controls=[r['id'] for r in ctrlrecs]), open(f'{OUT}/local_R2000.json', 'w'), indent=1)
json.dump(dict(R=R2, seed=SEED2, ldC=ref2['ldC'], records=poolr, best=best), open(f'{OUT}/pool_R2000.json', 'w'), indent=1)
json.dump(dict(R=R2, seed=SEED2, front=fr), open(f'{OUT}/front_R2000.json', 'w'), indent=1)
log_runtime('stage2_local', T_START, f'{len(runs)} NM runs, {len(newrecs)-len(ctrlrecs)} new candidates + {len(ctrlrecs)} controls at R={R2}')
