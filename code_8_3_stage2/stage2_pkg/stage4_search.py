"""§8.5 stage 4, Script 1: added search budget in the full 7-dim space (budget on), and the unbudgeted problem.
Rules: stage4/design.md §1 (frozen before this script ran). Summary:
- per lambda0 in {none, 0.3, 0.5, 0.7}: 12 Nelder-Mead starts never used as starts in stages 2-3: 6 from the current
  empirical front (stage3/front_union_R2000.json) stratified by m_E (L [0,0.1), M [0.1,0.5), H [0.5,inf), 2 each, cyclic
  dealing of one random permutation per stratum, seed 20261018), 6 by simple random sampling without replacement from the
  qualified R=200 pool (stage2/merged_R200.json, dec 'ok');
- stage-2 nm_run unchanged (penalty MU_B = 1 nat/send on max(0, D+2se), MU_E = 20, simplex steps 0.25), MAXFEV = 300, inner
  R=500 on the stage-2 search draws (seed 20261010);
- unbudgeted problem: the 10 highest-Delta m_D candidates of stage2/merged_R200.json not used as starts in stages 2-3,
  nm_run with MU_B = 0 and no floor (as stage3_unconstrained.py), MAXFEV = 300;
- per run two kept points: the endpoint and the archived point with the largest Delta m_D among those satisfying the run's
  own constraints on the search draws (D + 2 Dse < 0 and m_E >= lambda0); none -> endpoint only, logged; deduplicated by key;
- every start and every kept point evaluated at R=2000 (seed 20261012, the stage-2/3 ranking draws), merged with the
  stage-2/3 pool, empirical front recomputed; per floor previous vs new best with paired batch se and the exploratory
  interval +- t_{9,0.975} se; start -> endpoint changes; unbudgeted previous vs new best; representatives for Script 3.
Intervals here are exploratory diagnostics (design.md §1.8, statement (i)).
Run from code_8_3_stage2/stage2_pkg/:  python3 stage4_search.py
Outputs (stage4/): stage4_search.log, search_runs.json, search_R2000.json, pool_R2000.json, front_R2000.json, fig_front_stage4.png"""
exec(open('stage4_common.py').read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage4_search.log')
MAXFEV = 300
STRATA = [('L', 0.0, 0.1), ('M', 0.1, 0.5), ('H', 0.5, np.inf)]
N_FRONT, N_POOL, PER_STRATUM, N_UNB = 6, 6, 2, 10
print(f"=== stage4_search.py: Nelder-Mead maxfev={MAXFEV}, R={RS} inside (seed {SEEDS}), MU_B={MU_B}, MU_E={MU_E}, h=({H_Z},{H_CHI}); "
      f"start selection seed {SEED_SEL}; re-evaluation R={R2} seed {SEED2} ===")
check_trigger()

# --- stage-2/3 pool, used starts, current front ------------------------------------------------------------------------------
S23, ldC23 = s23_pool(); k23 = {kkey(r): r for r in S23}
used = used_starts()
ok23 = [r for r in S23 if r['dec'] == 'ok']; fr23 = [ok23[i] for i in pareto_front(ok23)]
fr3 = json.load(open(f'{S3}/front_union_R2000.json'))['front']
assert [r['id'] for r in fr23] == [r['id'] for r in fr3], "recomputed front differs from stage3/front_union_R2000.json"
print(f"stage-2/3 pool: {len(S23)} records at R={R2} (seed {SEED2}); qualified {len(ok23)}; current empirical front {len(fr23)} points "
      f"(= stage3/front_union_R2000.json); stage-2/3 Nelder-Mead starts: {sum(len(v) for v in used.values())} tasks, {len(used)} distinct keys")

# --- front starts: strata, cyclic dealing --------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED_SEL)
elig = [r for r in fr23 if kkey(r) not in used]
print(f"\nTable S1. Current front ({len(fr23)}) and eligibility as a start (design.md §1.3); strata on R={R2} m_E^(C): "
      + ', '.join(f"{g} [{lo},{hi})" for g, lo, hi in STRATA))
for r in fr23:
    k = kkey(r); g = [g for g, lo, hi in STRATA if lo <= r['mE_C'] < hi][0]
    print(f"  {r['id']:30s} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} stratum {g} "
          + (f"USED ({'; '.join(f'{os.path.basename(f)}:{t}/{s_}' for f, t, s_ in used[k])})" if k in used else 'eligible'))
perm = {}
for g, lo, hi in STRATA:
    E = sorted([r for r in elig if lo <= r['mE_C'] < hi], key=lambda r: r['id']); perm[g] = [E[i] for i in rng.permutation(len(E))]
    print(f"  stratum {g}: {len(E)} eligible; permutation (seed {SEED_SEL}): {[r['id'] for r in perm[g]]}")
def allocation():
    """2 per stratum; a stratum with fewer eligible points gives its shortfall to the nearest other stratum (ties: higher m_E)"""
    al = {g: min(PER_STRATUM, len(perm[g])) for g, _, _ in STRATA}; gs = [g for g, _, _ in STRATA]
    short = N_FRONT-sum(al.values())
    while short > 0:
        cand = [h for h in gs if len(perm[h]) > al[h]]
        if not cand: break
        deficit = [g for g in gs if len(perm[g]) < PER_STRATUM][0]
        h = min(cand, key=lambda h: (abs(gs.index(h)-gs.index(deficit)), -gs.index(h))); al[h] += 1; short -= 1
    return al
alloc = allocation()
if sum(alloc.values()) < N_FRONT: print(f"  fewer than {N_FRONT} eligible front points: {sum(alloc.values())} front starts per floor (logged; not filled)")
front_starts = {lab: [] for lab, _ in FLOORS}; ptr = {g: 0 for g, _, _ in STRATA}
for lab, _ in FLOORS:
    for g, _, _ in STRATA:
        n_ = len(perm[g])
        for _ in range(alloc[g]):
            while perm[g][ptr[g] % n_]['id'] in [r['id'] for r in front_starts[lab]]: ptr[g] += 1
            front_starts[lab].append(perm[g][ptr[g] % n_]); ptr[g] += 1
fkeys = {kkey(r) for v in front_starts.values() for r in v}

# --- R=200 pool starts and unbudgeted starts -------------------------------------------------------------------------------------
M200 = json.load(open(f'{S2}/merged_R200.json'))['records']
cand200 = sorted([r for r in M200 if r['dec'] == 'ok' and kkey(r) not in used and kkey(r) not in fkeys and not is_uniform(r)], key=lambda r: r['id'])
draw = rng.choice(len(cand200), len(FLOORS)*N_POOL, replace=False)
pool_starts = {lab: [cand200[i] for i in draw[j*N_POOL:(j+1)*N_POOL]] for j, (lab, _) in enumerate(FLOORS)}
print(f"\nqualified R=200 pool (stage2/merged_R200.json, dec 'ok'): {sum(r['dec'] == 'ok' for r in M200)}; eligible after removing used starts, "
      f"front starts and UNIF: {len(cand200)}; {len(draw)} drawn without replacement (seed {SEED_SEL}, after the permutations)")
ub_starts = [r for r in sorted(M200, key=lambda r: -r['dmD_C']) if kkey(r) not in used][:N_UNB]
assert [r['id'] for r in ub_starts] == ['L0299', 'L1573', 'L2029', 'L0295', 'L2207', 'L2375', 'L1870', 'L0190', 'L2916', 'L2066'], "unbudgeted starts differ from design.md §1.5"
print(f"\nTable S2. Starts per floor (R={R2} values for front starts; R=200 values, seed 20261010, for pool starts)")
for lab, l0 in FLOORS:
    for r in front_starts[lab]:
        g = [g for g, lo, hi in STRATA if lo <= r['mE_C'] < hi][0]
        print(f"  {lab:4s} front {g}  {r['id']:30s} R2000: dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} D={r['D']:+.2f}+-{r['D_se']:.2f} {r['dec']}")
    for r in pool_starts[lab]:
        print(f"  {lab:4s} R200     {r['id']:30s} R200:  dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} D={r['D']:+.2f}+-{r['D_se']:.2f} {r['dec']}  p={fmt_p(r)} chi={fmt_chi(r)}")
print("unbudgeted starts (the 10 highest R=200 dmD_C of stage2/merged_R200.json not used as stage-2/3 starts):")
for r in ub_starts: print(f"  {r['id']:6s} R200: dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} B={r['B']:.1f} D={r['D']:+.1f} {r['dec']} p={fmt_p(r)} chi={fmt_chi(r)}")
ndup = sum(len(v) for v in front_starts.values())-len(fkeys)
print(f"front starts: {sum(len(v) for v in front_starts.values())} slots, {len(fkeys)} distinct points ({ndup} re-uses across floors, unavoidable: "
      f"{len(elig)} eligible for {len(FLOORS)*N_FRONT} slots)")

# --- Nelder-Mead runs ----------------------------------------------------------------------------------------------------------
start_src = {}
tasks = []
for lab, l0 in FLOORS:
    for r in front_starts[lab]+pool_starts[lab]:
        tasks.append((lab, l0, r['id'], input_to_x(r['p'], r['chi']).tolist())); start_src[(lab, r['id'])] = 'front' if r in front_starts[lab] else 'R200'
utasks = [('unbudgeted', None, r['id'], input_to_x(r['p'], r['chi']).tolist()) for r in ub_starts]
setup_search()
t0 = time.time(); _ = eval_s(np.array(tasks[0][3])); print(f"one search evaluation at R={RS}: {time.time()-t0:.2f} s (single process)")
MU_B = 1.0; t1 = time.time()
with pool(4) as pl: runs = pl.map(nm_run, tasks, chunksize=1)
print(f"\n{len(runs)} budgeted Nelder-Mead runs, {sum(r['nfev'] for r in runs)} evaluations at R={RS}, wall {time.time()-t1:.0f} s")
MU_B = 0.0; t1 = time.time()
with pool(4) as pl: uruns = pl.map(nm_run, utasks, chunksize=1)
MU_B = 1.0
print(f"{len(uruns)} unbudgeted Nelder-Mead runs (MU_B=0, no floor), {sum(r['nfev'] for r in uruns)} evaluations at R={RS}, wall {time.time()-t1:.0f} s")
allruns = runs+uruns
json.dump(dict(RS=RS, seedS=SEEDS, MU_B=1.0, MU_E=MU_E, h=[H_Z, H_CHI], maxfev=MAXFEV, seed_selection=SEED_SEL, tasks=tasks, utasks=utasks,
               start_source={f"{k[0]}/{k[1]}": v for k, v in start_src.items()}, runs=runs, uruns=uruns), open(f'{OUT}/search_runs.json', 'w'))

# --- archive rule: endpoint + constrained best per run ---------------------------------------------------------------------------
xorig = {}
for r in allruns:
    for i, v in enumerate(r['archive']): xorig.setdefault(tuple(v['x']), []).append(dict(target=r['target'], start=r['start'], eval=i+1))
def satisfies(v, r):
    if r['target'] == 'unbudgeted': return True
    return v['D']+2*v['Dse'] < 0 and (r['lambda0'] is None or v['mE'] >= r['lambda0'])
kept = {}; no_sat = []; inpool = []
def keep(x, why, rid):
    a, chi = x_to_input(x); k = key_of(a**2, chi)
    if k not in kept:
        kept[k] = dict(id=rid, a=a.tolist(), chi=chi.tolist(), x=list(x), kept_as=[], origin_run=xorig[tuple(x)], R500=None)
    kept[k]['kept_as'].append(why)
    return k
run_kept = []
for r in allruns:
    arch = r['archive']; ie = [i for i, v in enumerate(arch) if v['x'] == r['x']]; assert ie, f"endpoint of {r['target']}/{r['start']} not in its archive"
    tag = 'S4U' if r['target'] == 'unbudgeted' else f"S4-{r['target']}"
    ke = keep(r['x'], dict(run=f"{r['target']}/{r['start']}", kind='endpoint', eval=ie[0]+1), f"{tag}-{r['start']}")
    kept[ke]['R500'] = kept[ke]['R500'] or arch[ie[0]]
    sat = [i for i, v in enumerate(arch) if satisfies(v, r)]
    if sat:
        ib = max(sat, key=lambda i: (arch[i]['dmD'], -i))
        kc = keep(arch[ib]['x'], dict(run=f"{r['target']}/{r['start']}", kind='constrained best', eval=ib+1), f"{tag}C-{r['start']}")
        kept[kc]['R500'] = kept[kc]['R500'] or arch[ib]
    else:
        kc = None; no_sat.append(f"{r['target']}/{r['start']}")
    run_kept.append(dict(run=f"{r['target']}/{r['start']}", end_key=ke, cb_key=kc, end_eval=ie[0]+1, cb_eval=(ib+1 if sat else None), n_sat=len(sat)))
print(f"\nArchive rule: {2*len(allruns)-len(no_sat)} kept points from {len(allruns)} runs -> {len(kept)} distinct after deduplication by key")
print(f"runs with no archived point satisfying their own constraints on the search draws (endpoint only kept; constraint not relaxed): "
      f"{len(no_sat)}: {no_sat if no_sat else 'none'}")

# --- R=2000 evaluation: starts, then kept points not already evaluated --------------------------------------------------------------
Z2 = np.random.default_rng(SEED2).standard_normal((R2, K)); ref = evaluate_crn(a_u, chi_u, Z2)
assert np.linalg.slogdet(ref['J'])[1] == ldC23, "R=2000 reference differs from the stage-2/3 pool reference"
starts_all = {}
for lab, l0 in FLOORS:
    for r in front_starts[lab]+pool_starts[lab]: starts_all.setdefault(kkey(r), r)
for r in ub_starts: starts_all.setdefault(kkey(r), r)
sl = list(starts_all.items())
t1 = time.time(); raw = eval_many([dict(a=r['a'], chi=r['chi']) for _, r in sl], Z2)
CMP = ('B', 'D', 'D_se', 'dmD_C', 'dmD_se', 'mE_C', 'mE_se', 'dmD_B', 'mE_B', 'lamC', 'ldb')
newrecs = []; nrep = 0; rec_of = dict(k23)
for (k, r), res in zip(sl, raw):
    q = summarize_s2(r['id'], r['a'], r['chi'], res, ref, R2, SEED2, src='stage4-start')
    if k in k23:
        assert all(q[c] == k23[k][c] for c in CMP), f"start {r['id']}: R=2000 re-evaluation differs from its stage-2/3 record"
        nrep += 1
    else:
        q['R200'] = {c: r[c] for c in ('B', 'D', 'D_se', 'dec', 'dmD_C', 'mE_C')}; newrecs.append(q); rec_of[k] = q
print(f"\n{len(sl)} distinct starts evaluated at R={R2} in {time.time()-t1:.0f} s; {nrep} already in the stage-2/3 pool reproduce their stored records "
      f"exactly ({', '.join(CMP)}); {len(sl)-nrep} new start records")
todo = [(k, c) for k, c in kept.items() if k not in rec_of]
for k, c in kept.items():
    if k in rec_of: print(f"  kept point {c['id']} ({'; '.join(w['run']+' '+w['kind'] for w in c['kept_as'])}) coincides with {rec_of[k]['id']} -> not re-evaluated")
t1 = time.time(); raw = eval_many([dict(a=c['a'], chi=c['chi']) for _, c in todo], Z2)
for (k, c), res in zip(todo, raw):
    q = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src='stage4')
    q.update(x=c['x'], kept_as=c['kept_as'], origin_run=c['origin_run'], R500=c['R500']); newrecs.append(q); rec_of[k] = q
for k, c in kept.items(): rec_of[k].setdefault('stage4_kept_as', c['kept_as'])
print(f"{len(todo)} new kept points evaluated at R={R2} in {time.time()-t1:.0f} s")
merged = S23+newrecs; assert len({kkey(r) for r in merged}) == len(merged)
okm = [r for r in merged if r['dec'] == 'ok']; frm = [okm[i] for i in pareto_front(okm)]
new_ids = {r['id'] for r in newrecs}

print(f"\nTable S3. New stage-4 records at R={R2} (seed {SEED2}), sorted by dmD_C (top 40; all in search_R2000.json); "
      "kept_as = run / kind#evaluation; R500 = search value at that evaluation")
print(f"{'id':34s} {'B':>7s} {'D +- se':>14s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'(se)':>7s} | {'R500 dmD':>8s} {'mE':>6s}  kept_as")
for r in sorted(newrecs, key=lambda r: -r['dmD_C'])[:40]:
    q = r.get('R500') or {}
    print(f"{r['id']:34s} {r['B']:7.2f} {r['D']:+7.2f} +- {r['D_se']:4.2f} {r['dec']:5s} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:6.3f} ({r['mE_se']:.4f}) | "
          f"{q.get('dmD', np.nan):+8.3f} {q.get('mE', np.nan):6.3f}  {'; '.join(w['run']+' '+w['kind']+'#'+str(w['eval']) for w in r.get('kept_as', [])) or 'start'}")
kr = [r for r in newrecs if r.get('R500')]
d5 = np.array([r['dmD_C']-r['R500']['dmD'] for r in kr])
print(f"R=2000 minus R=500 (search optimism) for dmD_C over the {len(kr)} new kept points: mean {d5.mean():+.4f}, median {np.median(d5):+.4f}, range [{d5.min():+.4f},{d5.max():+.4f}]")
dec_ = {k_: sum(r['dec'] == k_ for r in kr) for k_ in ('ok', 'over', 'undec')}
print(f"decisions of the new kept points at R={R2}: {dec_}")

print(f"\nTable S4. Empirical Pareto front of the evaluated candidates, R={R2} (seed {SEED2}): merged pool {len(merged)} (stage 2-3: {len(S23)}, "
      f"stage 4: {len(newrecs)}), qualified {len(okm)}; front {len(frm)} points ('*' = new in stage 4)")
for r in frm:
    print(f"  {'*' if r['id'] in new_ids else ' '} {r['id']:34s} p={fmt_p(r)} chi={fmt_chi(r)} B={r['B']:7.2f} D={r['D']:+6.2f}+-{r['D_se']:.2f} "
          f"dmD_C={r['dmD_C']:+.3f}({r['dmD_se']:.3f}) mE_C={r['mE_C']:.3f}({r['mE_se']:.4f})")
print(f"  stage-3 front points no longer on the front: {sorted({r['id'] for r in fr23}-{r['id'] for r in frm})}")

# --- per floor: previous vs new best ----------------------------------------------------------------------------------------------
print(f"\nTable S5. Per floor, R={R2} (seed {SEED2}): previous best (stage-2/3 pool) vs new best (merged pool); paired difference with batch se "
      f"(10x200, same draws); exploratory interval +- t_9,0.975 se = +- {T9:.3f} se; threshold 0.02 nat")
print(f"{'lambda0':7s} {'previous best':30s} {'dmD_C':>7s} {'mE_C':>6s} | {'new best':34s} {'dmD_C':>7s} {'mE_C':>6s} {'margin':>7s} | "
      f"{'diff':>8s} {'(se)':>8s} {'interval':>19s} | {'pt>0.02':>7s} {'lo>0.02':>7s}")
per_floor = []
for lab, l0 in FLOORS:
    b0 = max([r for r in S23 if feasible(r, l0)], key=lambda r: r['dmD_C']); b1 = max([r for r in merged if feasible(r, l0)], key=lambda r: r['dmD_C'])
    d = b1['dmD_C']-b0['dmD_C']; se = diff_se(b1, b0) if b1['id'] != b0['id'] else 0.0; lo, hi = d-T9*se, d+T9*se
    row = dict(lambda0=lab, prev=b0['id'], prev_dmD=b0['dmD_C'], prev_mE=b0['mE_C'], new=b1['id'], new_dmD=b1['dmD_C'], new_mE=b1['mE_C'],
               new_margin=-(b1['D']+2*b1['D_se']), diff=d, diff_se=se, lo=lo, hi=hi, pt_above=bool(d > 0.02), lo_above=bool(lo > 0.02))
    per_floor.append(row)
    print(f"{lab:7s} {b0['id']:30s} {b0['dmD_C']:+7.3f} {b0['mE_C']:6.3f} | {b1['id']:34s} {b1['dmD_C']:+7.3f} {b1['mE_C']:6.3f} {row['new_margin']:+7.3f} | "
          f"{d:+8.4f} ({se:6.4f}) [{lo:+8.4f},{hi:+8.4f}] | {('yes' if row['pt_above'] else 'no'):>7s} {('yes' if row['lo_above'] else 'no'):>7s}")
    feas = sorted([r for r in merged if feasible(r, l0)], key=lambda r: -r['dmD_C'])[:4]
    print(f"        top-4 feasible in the merged pool: " + '; '.join(f"{r['id']} {r['dmD_C']:+.3f} (mE {r['mE_C']:.3f})" for r in feas))
nb = [r['new_dmD'] for r in per_floor]
print(f"new best per floor none/0.3/0.5/0.7: {np.round(nb, 3).tolist()} -> non-increasing: {all(np.diff(nb) <= 0)}")
lam_grid = np.round(np.arange(0, 1.0001, 0.05), 2)
D0 = [max(r['dmD_C'] for r in S23 if r['dec'] == 'ok' and r['mE_C'] >= l-1e-9) for l in lam_grid]
D1 = [max(r['dmD_C'] for r in okm if r['mE_C'] >= l-1e-9) for l in lam_grid]
print("max dmD_C over qualified with mE_C >= lambda0, stage-2/3 pool -> merged: " + ', '.join(f"{l:.2f}:{a_:+.3f}->{b_:+.3f}" for l, a_, b_ in zip(lam_grid, D0, D1)))

# --- own-search outcomes and search budget ------------------------------------------------------------------------------------------
print(f"\nTable S6. Search budget added vs improvement found (Nelder-Mead evaluations at R={RS}; stage-3 grids, 625 points each at R=200, not counted)")
nf = {}
for f, st in ((f'{S2}/local_runs.json', 'stage2'), (f'{S3}/dense_runs.json', 'stage3'), (f'{S3}/unconstrained.json', 'stage3')):
    for r in json.load(open(f))['runs']: nf.setdefault((st, r['target']), [0, 0]); nf[(st, r['target'])][0] += 1; nf[(st, r['target'])][1] += r['nfev']
for r in allruns: nf.setdefault(('stage4', r['target']), [0, 0]); nf[('stage4', r['target'])][0] += 1; nf[('stage4', r['target'])][1] += r['nfev']
print(f"{'target':10s} {'stage 2 runs/evals':>18s} {'stage 3 runs/evals':>18s} {'stage 4 runs/evals':>18s} | {'own-search best (stage 4 origin)':45s} | {'improvement of the floor best (se)':>34s}")
own = {}
for (lab, l0), row in zip(FLOORS+[('unbudgeted', None)], per_floor+[None]):
    mine = [r for r in newrecs if r['src'] == 'stage4' and any(o['target'] == lab for o in r['origin_run']) and (lab == 'unbudgeted' or feasible(r, l0))]
    sb = max(mine, key=lambda r: r['dmD_C']) if mine else None; own[lab] = sb['id'] if sb else None
    imp = f"{row['diff']:+.4f} ({row['diff_se']:.4f})" if row else '(see Table S8)'
    print(f"{lab:10s} " + ' '.join(f"{nf.get((st, lab), [0, 0])[0]:>8d}/{nf.get((st, lab), [0, 0])[1]:<9d}" for st in ('stage2', 'stage3', 'stage4'))
          + f" | {(f'{sb['id']} {sb['dmD_C']:+.3f} mE {sb['mE_C']:.3f}' if sb else 'none'):45s} | {imp:>34s}")

# --- start -> endpoint ---------------------------------------------------------------------------------------------------------------
print(f"\nTable S7. Start -> endpoint per budgeted run, R={R2} (seed {SEED2}). feasible = qual 'ok' and m_E^(C) >= lambda0 (budget only for none). "
      "(i) feasible starts: dmD change end - start with paired batch se; (ii) infeasible starts: feasibility transition with the failing constraint(s). "
      "cb = constrained best on the search draws")
print(f"{'run':40s} {'src':5s} {'start dmD':>9s} {'mE':>6s} {'start fails':16s} | {'end dmD':>8s} {'mE':>6s} {'end fails':16s} | {'change (se)':>17s} | {'cb dmD':>7s} {'mE':>6s} {'cb fails':12s}")
s7 = []
for r, rk in zip(allruns, run_kept):
    if r['target'] == 'unbudgeted': continue
    l0 = r['lambda0']; st = rec_of[[k for k, v in starts_all.items() if v['id'] == r['start']][0]]; en = rec_of[rk['end_key']]; cb = rec_of[rk['cb_key']] if rk['cb_key'] else None
    fs, fe = fails(st, l0), fails(en, l0); ch = en['dmD_C']-st['dmD_C']; se = diff_se(en, st) if en['id'] != st['id'] else 0.0
    s7.append(dict(run=rk['run'], src=start_src[(r['target'], r['start'])], start=st['id'], end=en['id'], cb=cb['id'] if cb else None, start_feasible=fs == '', end_feasible=fe == '',
                   cb_feasible=(fails(cb, l0) == '') if cb else None, start_dmD=st['dmD_C'], end_dmD=en['dmD_C'], change=ch, change_se=se, start_fails=fs, end_fails=fe,
                   start_mE=st['mE_C'], end_mE=en['mE_C'], cb_dmD=cb['dmD_C'] if cb else None, cb_mE=cb['mE_C'] if cb else None))
    print(f"{rk['run']:40s} {start_src[(r['target'], r['start'])]:5s} {st['dmD_C']:+9.3f} {st['mE_C']:6.3f} {fs or 'feasible':16s} | {en['dmD_C']:+8.3f} {en['mE_C']:6.3f} {fe or 'feasible':16s} | "
          + (f"{ch:+8.4f} ({se:.4f})" if fs == '' else f"{'(ii) '+('->feasible' if fe == '' else '->infeasible'):>17s}") + " | "
          + (f"{cb['dmD_C']:+7.3f} {cb['mE_C']:6.3f} {fails(cb, l0) or 'feasible':12s}" if cb else f"{'none (endpoint only)':>27s}"))
print("Summary (i), starts feasible at R=2000:")
for lab, _ in FLOORS+[('all', None)]:
    g = [q for q in s7 if q['start_feasible'] and (lab == 'all' or q['run'].startswith(lab+'/'))]
    if not g: print(f"  {lab:4s}: no feasible start"); continue
    ch = np.array([q['change'] for q in g]); se = np.array([q['change_se'] for q in g])
    print(f"  {lab:4s}: n={len(g):2d}, end feasible {sum(q['end_feasible'] for q in g)}/{len(g)}; change mean {ch.mean():+.4f}, median {np.median(ch):+.4f}, "
          f"range [{ch.min():+.4f},{ch.max():+.4f}]; change > +2se: {int(np.sum(ch > 2*se))}, < -2se: {int(np.sum(ch < -2*se))}, |change| <= 2se: {int(np.sum(np.abs(ch) <= 2*se))}")
print("Summary (ii), starts infeasible at R=2000 (transition to the endpoint; failing constraints at the start):")
for lab, _ in FLOORS+[('all', None)]:
    g = [q for q in s7 if not q['start_feasible'] and (lab == 'all' or q['run'].startswith(lab+'/'))]
    if not g: print(f"  {lab:4s}: no infeasible start"); continue
    fc = {}
    for q in g: fc[q['start_fails']] = fc.get(q['start_fails'], 0)+1
    print(f"  {lab:4s}: n={len(g):2d}, infeasible->feasible {sum(q['end_feasible'] for q in g)}, infeasible->infeasible {sum(not q['end_feasible'] for q in g)}; "
          f"constrained-best point feasible at R=2000: {sum(bool(q['cb_feasible']) for q in g)}; start fails: {fc}")
by_src = {}
for q in s7: by_src.setdefault(q['src'], []).append(q)
for sname, g in by_src.items():
    print(f"  by start source {sname}: {len(g)} runs; start feasible {sum(q['start_feasible'] for q in g)}, end feasible {sum(q['end_feasible'] for q in g)}; "
          f"best end dmD_C among feasible ends: " + (f"{max(q['end_dmD'] for q in g if q['end_feasible']):+.3f}" if any(q['end_feasible'] for q in g) else 'none'))
same = {}
for r, rk in zip(allruns, run_kept):
    if r['target'] != 'unbudgeted': same.setdefault((r['start'], rk['end_key']), []).append(r['target'])
dupl = {k: v for k, v in same.items() if len(v) > 1}
print(f"  runs from the same start ending at the same point under different floors: " + ('; '.join(f"{k[0]}: {', '.join(v)}" for k, v in dupl.items()) if dupl else 'none'))

# --- unbudgeted ---------------------------------------------------------------------------------------------------------------------
print(f"\nTable S8. Unbudgeted problem (power only), R={R2} (seed {SEED2}); start R=200 and R=2000 values, endpoint R=500 and R=2000; any decision")
print(f"{'start':6s} {'R200':>7s} {'R2000':>7s} | {'endpoint':12s} {'R500':>7s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'B':>8s} {'D +- se':>15s} {'dec':5s} | change (se)  p, chi")
urows = []
for r, rk in zip(allruns, run_kept):
    if r['target'] != 'unbudgeted': continue
    st = rec_of[kkey([q for q in ub_starts if q['id'] == r['start']][0])]; en = rec_of[rk['end_key']]; s200 = [q for q in ub_starts if q['id'] == r['start']][0]
    ch = en['dmD_C']-st['dmD_C']; se = diff_se(en, st)
    urows.append(dict(start=r['start'], end=en['id'], start_dmD=st['dmD_C'], end_dmD=en['dmD_C'], change=ch, change_se=se, p=en['p'], chi=en['chi'], B=en['B'], D=en['D']))
    print(f"{r['start']:6s} {s200['dmD_C']:+7.3f} {st['dmD_C']:+7.3f} | {en['id']:12s} {r['best_eval']['dmD']:+7.3f} {en['dmD_C']:+7.3f} ({en['dmD_se']:5.3f}) {en['mE_C']:6.3f} {en['B']:8.2f} "
          f"{en['D']:+7.2f} +- {en['D_se']:4.2f} {en['dec']:5s} | {ch:+.3f} ({se:.3f})  p={fmt_p(en)} chi={fmt_chi(en)}")
u0 = max(S23, key=lambda r: r['dmD_C']); u1 = max(merged, key=lambda r: r['dmD_C'])
du = u1['dmD_C']-u0['dmD_C']; seu = diff_se(u1, u0) if u1['id'] != u0['id'] else 0.0
unb = dict(prev=u0['id'], prev_dmD=u0['dmD_C'], new=u1['id'], new_dmD=u1['dmD_C'], diff=du, diff_se=seu, lo=du-T9*seu, hi=du+T9*seu,
           pt_above=bool(du > 0.02), lo_above=bool(du-T9*seu > 0.02), new_p=u1['p'], new_chi=u1['chi'], new_B=u1['B'], new_D=u1['D'], new_mE=u1['mE_C'])
print(f"unbudgeted best evaluated: previous {u0['id']} {u0['dmD_C']:+.3f} -> merged {u1['id']} {u1['dmD_C']:+.3f} (p={fmt_p(u1)}, B={u1['B']:.2f}, mE={u1['mE_C']:.3f}); "
      f"difference {du:+.4f} (paired se {seu:.4f}), interval [{unb['lo']:+.4f},{unb['hi']:+.4f}]; point > 0.02: {'yes' if unb['pt_above'] else 'no'}; "
      f"lower end > 0.02: {'yes' if unb['lo_above'] else 'no'}")
top = sorted(merged, key=lambda r: -r['dmD_C'])[:6]
print("  six highest dmD_C in the merged pool (any decision): " + '; '.join(f"{r['id']} {r['dmD_C']:+.3f} p1={r['p'][0]:.3f}" for r in top))

# --- representatives for Script 3 ------------------------------------------------------------------------------------------------------
reps = representatives(merged)
print(f"\nRepresentatives for stage4_scenarios.py (design.md §3: strictest floor first, feasible non-uniform, largest dmD_C not already chosen):")
byid = {r['id']: r for r in merged}
for lab, v in reps.items():
    if v is None: print(f"  {lab:4s}: no non-uniform feasible candidate (no representative)"); continue
    r = byid[v['id']]
    print(f"  {lab:4s}: {v['id']:34s} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} (floor best {v['floor_best']}; replaced: {v['replaced']})")

json.dump(dict(R=R2, seed=SEED2, records=newrecs), open(f'{OUT}/search_R2000.json', 'w'), indent=1)
json.dump(dict(R=R2, seed=SEED2, ldC=ldC23, records=merged), open(f'{OUT}/pool_R2000.json', 'w'), indent=1)
json.dump(dict(R=R2, seed=SEED2, front=frm, per_floor=per_floor, unbudgeted=unb, unbudgeted_runs=urows, start_end=s7, representatives=reps,
               own_search=own, no_satisfying_point=no_sat, front_starts={k: [r['id'] for r in v] for k, v in front_starts.items()},
               pool_starts={k: [r['id'] for r in v] for k, v in pool_starts.items()}, unbudgeted_starts=[r['id'] for r in ub_starts]),
          open(f'{OUT}/front_R2000.json', 'w'), indent=1)

# --- figure: front before and after -------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID_, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID_, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 9, 'legend.frameon': False})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for grp, col, lab_, mk in [([r for r in okm if r['id'] not in new_ids], '#2a78d6', 'stage 2-3 (qualified)', 'o'), ([r for r in okm if r['id'] in new_ids], '#eb6834', 'stage 4 (qualified)', 's')]:
    ax.plot([r['dmD_C'] for r in grp], [r['mE_C'] for r in grp], mk, ms=4.5, color=col, mec=SURF, mew=0.6, label=lab_, zorder=3)
ax.step([r['dmD_C'] for r in fr23], [r['mE_C'] for r in fr23], where='post', color=MUTED, lw=1.0, ls='--', label='stage-3 empirical front')
ax.step([r['dmD_C'] for r in frm], [r['mE_C'] for r in frm], where='post', color=INK2, lw=1.2, label='stage-4 empirical front')
for l0 in (0.3, 0.5, 0.7): ax.axhline(l0, color=GRID_, lw=0.9, ls=':', zorder=1)
ax.set_xlabel(r'$\Delta m_D^{(C)}$ (nat)'); ax.set_ylabel(r'$m_E^{(C)}$'); ax.set_xlim(left=-0.3)
ax.set_title(f'Empirical Pareto front of the evaluated candidates, R={R2}, seed {SEED2}', loc='left', fontsize=10, color=INK); ax.legend(loc='lower left', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_front_stage4.png', dpi=150); plt.close(fig)
log_runtime('stage4_search', T_START, f'{len(runs)} budgeted + {len(uruns)} unbudgeted NM runs (maxfev {MAXFEV}), {len(newrecs)} new records at R={R2}')
