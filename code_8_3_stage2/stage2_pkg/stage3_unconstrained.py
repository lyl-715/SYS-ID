"""§8.5 stage 3, experiment 2: the power-only problem (sum a_m^2 = 1 by the parametrisation; no send budget, no m_E floor).
Nelder-Mead on -Delta m_D^(C) (the stage-2 nm_run with MU_B = 0 and lambda0 = None), R=500 inside (seed 20261010 extended, as in
stage 2), maxfev=200, same parametrisation and initial simplex as stage 2, from the 10 highest-Delta m_D candidates of
stage2/merged_R200.json regardless of budget. Every endpoint is re-evaluated at R=2000 (seed 20261012, same draws and reference as the
stage-2 pool). For the best endpoint and the next five: the SoD-vs-matched-periodic advantage
    log det J_u - log det J_per(u, floor(B_u)),  J_per = H_u(t_i)^T H_u(t_i) / sigma^2,  t_i = i T/N, i = 1..N, N = floor(B_u)
(the candidate's own regressors at N equispaced times, as the (B) reference of mc85_scan.py but for the candidate).
The result is "the best evaluated point of the unbudgeted problem", not an optimum.
Run from code_8_3_stage2/stage2_pkg/:  python3 stage3_unconstrained.py      Outputs (stage3/): stage3_unconstrained.log, unconstrained.json"""
exec(open('stage3_common.py').read())
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage3_unconstrained.log')
MU_B = 0.0                                   # no budget penalty; with lambda0=None pen() = -Delta m_D
print(f"=== stage3_unconstrained.py: power-only problem, Nelder-Mead on -dmD_C, R={RS} inside (seed {SEEDS}), maxfev={MAXFEV}, h=({H_Z},{H_CHI}); "
      f"re-evaluation R={R2} seed {SEED2} ===")
check_trigger()
M = json.load(open(f'{S2}/merged_R200.json'))['records']
starts = sorted(M, key=lambda r: -r['dmD_C'])[:10]
print("starts: the 10 highest dmD_C in stage2/merged_R200.json (R=200, seed 20261010), any decision")
for r in starts:
    print(f"  {r['id']:6s} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} B={r['B']:.1f} D={r['D']:+.1f} {r['dec']:5s} p={fmt_p(r)} chi={fmt_chi(r)}")
setup_search()
tasks = [('unbudgeted', None, r['id'], input_to_x(r['p'], r['chi']).tolist()) for r in starts]
t1 = time.time()
with pool(4) as pl: runs = pl.map(nm_run, tasks, chunksize=1)
print(f"\n{len(runs)} Nelder-Mead runs, {sum(r['nfev'] for r in runs)} evaluations at R={RS}, wall {time.time()-t1:.0f} s")

# --- endpoints, with provenance, re-evaluated at R=2000 -------------------------------------------------------------------------
P, rec2 = stage2_pool()
Z2 = np.random.default_rng(SEED2).standard_normal((R2, K)); ref = evaluate_crn(a_u, chi_u, Z2)
assert np.linalg.slogdet(ref['J'])[1] == P['ldC'], "R=2000 reference differs from the stage-2 pool reference"
ends = {}
for r in runs:
    a, chi = x_to_input(r['x']); k = key_of(a**2, chi)
    o = [dict(target=q['target'], start=q['start'], eval=i+1) for q in runs for i, v in enumerate(q['archive']) if v['x'] == r['x']]
    if k in ends: ends[k]['selected_by'].append(dict(target='unbudgeted', how='run result')); continue
    ends[k] = dict(id=f"U-{r['start']}", a=a.tolist(), chi=chi.tolist(), x=r['x'], origin_run=o, selected_by=[dict(target='unbudgeted', how='run result')],
                   R500=r['best_eval'], start=r['start'], start_R500=r['start_eval'])
endl = list(ends.values()); raw = eval_many(endl, Z2); recs = []
for c, res in zip(endl, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src='unbudgeted')
    r.update({k: c[k] for k in ('x', 'origin_run', 'selected_by', 'R500', 'start', 'start_R500')}); r['J'] = res['J'].tolist(); recs.append(r)
recs.sort(key=lambda r: -r['dmD_C'])
st = {r['id']: r for r in starts}
print(f"\nTable U1. Endpoints of the unbudgeted search at R={R2} (seed {SEED2}; same draws and reference as the stage-2 pool), sorted by dmD_C; "
      f"start value at R=200 (seed 20261010) and end value at R={RS} (seed {SEEDS}) for comparison")
print(f"{'id':8s} {'start R200':>10s} {'end R500':>8s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'B':>8s} {'D +- se':>15s} {'dec':5s}  p, chi, origin")
for r in recs:
    print(f"{r['id']:8s} {st[r['start']]['dmD_C']:+10.3f} {r['R500']['dmD']:+8.3f} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:6.3f} {r['B']:8.2f} "
          f"{r['D']:+7.2f} +- {r['D_se']:4.2f} {r['dec']:5s}  p={fmt_p(r)} chi={fmt_chi(r)} origin {', '.join(o['start']+'#'+str(o['eval']) for o in r['origin_run'])}")
b = recs[0]
print(f"\nBest evaluated point of the unbudgeted problem (not an optimum): {b['id']} (origin {b['origin_run'][0]['start']}#{b['origin_run'][0]['eval']})")
print(f"  p = {fmt_p(b)}, chi (mod 2pi) = {fmt_chi(b)}, chi3-chi4 = {np.mod(b['chi'][2]-b['chi'][3], TWO_PI):.3f}")
print(f"  R={R2}, seed {SEED2}: B = {b['B']:.2f}, D = {b['D']:+.2f} +- {b['D_se']:.2f} ({b['dec']}), dmD_C = {b['dmD_C']:+.3f} (paired batch se {b['dmD_se']:.3f}), "
      f"mE_C = {b['mE_C']:.4f} (se {b['mE_se']:.4f}), dmD_B = {b['dmD_B']:+.3f}")
print(f"  nine generalized eigenvalues vs (C): {lam_table_row(b)}")
print(f"  p1 > 0.5: {b['p'][0] > 0.5} (p1 = {b['p'][0]:.3f}); budget-qualified best of the stage-2 pool for comparison: NM-none-P5514 dmD_C = {rec2['NM-none-P5514']['dmD_C']:+.3f}; "
      f"difference {b['dmD_C']-rec2['NM-none-P5514']['dmD_C']:+.3f} (paired se {diff_se(b, rec2['NM-none-P5514']):.3f}) for {b['B']-rec2['NM-none-P5514']['B']:+.1f} sends")

# --- SoD vs matched periodic sampling --------------------------------------------------------------------------------------------
def adv(r_, J):
    N = int(np.floor(r_['B'])); tt = np.arange(1, N+1)*T/N
    Hp_ = regressors(np.asarray(r_['a']), np.mod(np.asarray(r_['chi']), TWO_PI), tt); Jper = Hp_.T@Hp_/s**2
    return N, float(np.linalg.slogdet(J)[1]-np.linalg.slogdet(Jper)[1])
NU, advU = adv(dict(B=float(ref['cnt'].mean()), a=a_u.tolist(), chi=chi_u.tolist()), ref['J'])
print(f"\nTable U2. SoD vs matched periodic sampling, R={R2} (seed {SEED2}): log det J_u - log det J_per(u, floor(B_u)), periodic reference built from the "
      f"candidate's own regressors at N = floor(B_u) equispaced times")
print(f"{'id':8s} {'B_u':>8s} {'N':>4s} {'log det J_u':>11s} {'advantage':>9s} {'dmD_C':>7s}")
rows = []
for r in recs[:6]:
    N, av = adv(r, np.array(r['J'])); rows.append(dict(id=r['id'], N=N, advantage=av)); r['sod_vs_periodic'] = dict(N=N, advantage=av)
    print(f"{r['id']:8s} {r['B']:8.2f} {N:4d} {np.linalg.slogdet(np.array(r['J']))[1]:11.3f} {av:+9.3f} {r['dmD_C']:+7.3f}")
print(f"{'UNIF':8s} {ref['cnt'].mean():8.2f} {NU:4d} {np.linalg.slogdet(ref['J'])[1]:11.3f} {advU:+9.3f} {0:+7.3f}   (uniform reference, same definition)")
json.dump(dict(R=R2, seed=SEED2, RS=RS, seedS=SEEDS, maxfev=MAXFEV, MU_B=MU_B, starts=[r['id'] for r in starts], tasks=tasks,
               runs=runs, records=recs,
               sod_vs_periodic=rows, sod_vs_periodic_uniform=dict(N=NU, advantage=advU)), open(f'{OUT}/unconstrained.json', 'w'))
log_runtime('stage3_unconstrained', T_START, f'{len(runs)} NM runs (maxfev {MAXFEV}), {len(recs)} endpoints at R={R2}')
