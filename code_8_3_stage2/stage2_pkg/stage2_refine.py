"""§8.5 stage 2, task 2: re-evaluate the ~40 candidates selected by stage2_post.py at R=2000 on fresh common draws
(seed 20261012), the same draws for every candidate and for the uniform reference (C).  Recomputes the paired
send-count difference D +- se, the 8.3.6(g) decision, Delta m_D^(C) with its paired batch se (10 batches x 200 draws),
the nine generalized eigenvalues vs (C), and re-ranks among the R=2000 estimates only.
Run from code_8_3_stage2/stage2_pkg/:  python3 stage2_refine.py
Outputs (stage2/): stage2_refine.log, refine_R2000.json, front_R2000_refine.json"""
exec(open('stage2_common.py').read())
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage2_refine.log')
R2, SEED2 = 2000, 20261012
print(f"=== stage2_refine.py: R={R2}, seed {SEED2}, paired batch se 10x{R2//10} ===")
check_trigger()
sel = json.load(open(f'{OUT}/refine_selection.json'))
Z = np.random.default_rng(SEED2).standard_normal((R2, K))
ref = evaluate_crn(a_u, chi_u, Z)
ldC = np.linalg.slogdet(ref['J'])[1]
print(f"reference (C) uniform: B_u={ref['cnt'].mean():.3f} (sd {ref['cnt'].std(ddof=1):.2f}), log det J_C={ldC:.4f}, (B) log det={ldB:.4f}, offset C-B={ldC-ldB:+.4f}")
t1 = time.time(); raw = eval_many(sel, Z); print(f"evaluated {len(sel)} candidates at R={R2} in {time.time()-t1:.0f} s (4 workers)")
recs = []
for c, res in zip(sel, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src=c['src'])
    r.update(why=c['why'], R200=dict(dmD_C=c['dmD_C'], mE_C=c['mE_C'], D=c['D'], D_se=c['D_se'], dec=c['dec'], B=c['B']))
    recs.append(r)
u = [r for r in recs if r['id'] == 'UNIF'][0]
print(f"self-check: UNIF re-evaluated in the pool: D={u['D']:+.3f}, dmD_C={u['dmD_C']:+.2e}, mE_C={u['mE_C']:.6f} (expected 0, 0, 1)")

print(f"\nTable 4. R={R2} (seed {SEED2}) re-evaluation of the {len(recs)} selected candidates, sorted by Delta m_D^(C); "
      "dmD_C = full-R estimate, (se) = paired batch se 10x200; R=200 values for comparison")
print(f"{'id':6s} {'sel':6s} {'B':>7s} {'D +- se':>14s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'(se)':>8s} {'dmD_B':>7s} {'mE_B':>6s} | {'R200: dmD_C':>11s} {'mE_C':>6s} {'dec':5s}")
for r in sorted(recs, key=lambda r: -r['dmD_C']):
    q = r['R200']
    print(f"{r['id']:6s} {'+'.join(r['why']):6s} {r['B']:7.2f} {r['D']:+7.2f} +- {r['D_se']:4.2f} {r['dec']:5s} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:6.3f} ({r['mE_se']:6.4f}) {r['dmD_B']:+7.3f} {r['mE_B']:6.3f} | "
          f"{q['dmD_C']:+11.3f} {q['mE_C']:6.3f} {q['dec']:5s}")
dd = np.array([r['dmD_C']-r['R200']['dmD_C'] for r in recs if r['id'] != 'UNIF'])
dm = np.array([r['mE_C']-r['R200']['mE_C'] for r in recs if r['id'] != 'UNIF'])
print(f"R=2000 minus R=200 over the {dd.size} non-reference candidates: dmD_C mean {dd.mean():+.3f} (sd {dd.std(ddof=1):.3f}, range [{dd.min():+.3f},{dd.max():+.3f}]); "
      f"mE_C mean {dm.mean():+.3f} (range [{dm.min():+.3f},{dm.max():+.3f}])")
chg = [(r['id'], r['R200']['dec'], r['dec']) for r in recs if r['R200']['dec'] != r['dec']]
print(f"decision changes R=200 -> R=2000: {len(chg)}: {chg}")
print(f"decisions at R=2000: {dict((k, sum(r['dec'] == k for r in recs)) for k in ('ok', 'over', 'undec'))}")

print(f"\nTable 5. Nine generalized eigenvalues vs (C) at R={R2} (ascending)")
for r in sorted(recs, key=lambda r: -r['dmD_C']): print(f"{r['id']:6s} {lam_table_row(r)}")

ok = [r for r in recs if r['dec'] == 'ok']; fr = [ok[i] for i in pareto_front(ok)]
print(f"\nTable 6. Empirical Pareto front of the evaluated qualified candidates at R={R2} (seed {SEED2}): {len(fr)} points")
for r in fr:
    print(f"  {r['id']:6s} p={fmt_p(r)} chi={fmt_chi(r)} B={r['B']:7.2f} D={r['D']:+6.2f}+-{r['D_se']:.2f} dmD_C={r['dmD_C']:+.3f}({r['dmD_se']:.3f}) mE_C={r['mE_C']:.3f}({r['mE_se']:.4f})")
print(f"\nTable 7. Re-ranking at R={R2} only: best qualified candidate subject to m_E^(C) >= lambda0")
best = {}
for lab, l0 in [('none', None), ('0.3', 0.3), ('0.5', 0.5), ('0.7', 0.7)]:
    feas = [r for r in ok if l0 is None or r['mE_C'] >= l0-1e-9]; b = max(feas, key=lambda r: r['dmD_C']); best[lab] = b['id']
    sec = sorted(feas, key=lambda r: -r['dmD_C'])[1] if len(feas) > 1 else None
    print(f"  lambda0={lab:4s}: {b['id']:6s} dmD_C={b['dmD_C']:+.3f} (se {b['dmD_se']:.3f}) mE_C={b['mE_C']:.3f} (se {b['mE_se']:.4f}) D={b['D']:+.2f}+-{b['D_se']:.2f}"
          + (f"   runner-up {sec['id']} {sec['dmD_C']:+.3f} (se {sec['dmD_se']:.3f}), mE_C={sec['mE_C']:.3f}; difference {b['dmD_C']-sec['dmD_C']:+.4f} (paired batch se {diff_se(b, sec):.4f})" if sec else ''))
vals = [max(r['dmD_C'] for r in ok if l0 is None or r['mE_C'] >= l0-1e-9) for l0 in (None, 0.3, 0.5, 0.7)]
print(f"P2 at R={R2} (refined set): max dmD_C for lambda0 = none/0.3/0.5/0.7: {np.round(vals, 3).tolist()} -> non-increasing: {all(np.diff(vals) <= 0)}")
bl = max([r for r in ok if r['src'] == 'lhs'], key=lambda r: r['dmD_C'])
print(f"P1 at R={R2} (refined set): best qualified {best['none']} dmD_C={max(vals):+.3f}; best qualified LHS (amplitude+phase) {bl['id']} dmD_C={bl['dmD_C']:+.3f} (se {bl['dmD_se']:.3f}); "
      f"stage-1 reference value +0.597 was an R=200 estimate, P5014 at R=2000: {[r['dmD_C'] for r in recs if r['id']=='P5014'][0]:+.3f}")
json.dump(dict(R=R2, seed=SEED2, ldC=ldC, B_u=float(ref['cnt'].mean()), records=recs, best=best), open(f'{OUT}/refine_R2000.json', 'w'), indent=1)
json.dump(dict(R=R2, seed=SEED2, front=fr), open(f'{OUT}/front_R2000_refine.json', 'w'), indent=1)
log_runtime('stage2_refine', T_START, f'{len(recs)} candidates at R={R2}')
