"""§8.5 stage 4, Script 2: confirmation on a frozen pool with one independent batch (rules: stage4/design.md §2 and the prompt).
Two invocations, run from code_8_3_stage2/stage2_pkg/:
  python3 stage4_confirm.py freeze   -> freezes and logs, BEFORE any noise is drawn: the pool (all points of the Script-1 front,
                                        stage4/front_R2000.json, plus the 7 stage-2 finalists, deduplicated by key, no truncation),
                                        the incumbent per floor (Script-1 best at R=2000), delta = 0.02 nat, confidence 0.95, the method
                                        (batch-mean t-intervals, 20 batches of 400, 19 df), the stopping rule (this single batch, no
                                        augmentation) and the comparison family (budget intervals vs UNIF and m_E intervals for the n-1
                                        non-uniform candidates, Delta m_D differences for all C(n,2) unordered pairs; M = 2(n-1) + C(n,2),
                                        t_{19, 1-0.05/(2M)} on every interval). Writes stage4/confirm_frozen.json, stage4_confirm_freeze.log.
  python3 stage4_confirm.py draw     -> refuses to run unless confirm_frozen.json is tracked by git and unmodified; draws seed 20261020,
                                        R=8000 common random numbers (the same draws for every candidate and the uniform reference);
                                        point estimates from the full R=8000 matrices, standard errors from the 20 paired batch statistics
                                        (differences formed within each batch first); nonlinearity diagnostic (mean of batch statistics -
                                        full-sample statistic, not a bias bound); classification and candidate near-optimal sets per floor.
                                        Writes stage4/stage4_confirm.log, confirm_R8000.json.
Evaluation: evaluate_crn (stage2_common.py, unchanged) with nbatch=20, i.e. batches of 400 consecutive draws.
Intervals are batch-mean approximations for a logdet-of-mean estimator; nothing is claimed about unevaluated inputs."""
exec(open('stage4_common.py').read())
import subprocess, hashlib, itertools
MODE = sys.argv[1] if len(sys.argv) > 1 else ''
assert MODE in ('freeze', 'draw'), "usage: python3 stage4_confirm.py freeze|draw"
R8, NB = 8000, 20; BS = R8//NB
DELTA, CONF = 0.02, 0.95
FROZEN = f'{OUT}/confirm_frozen.json'
T_START = time.time()

if MODE == 'freeze':
    sys.stdout = Tee(f'{OUT}/stage4_confirm_freeze.log')
    print(f"=== stage4_confirm.py freeze: no noise is drawn in this invocation ===")
    PM = json.load(open(f'{OUT}/pool_R2000.json')); byid = {r['id']: r for r in PM['records']}
    FR = json.load(open(f'{OUT}/front_R2000.json'))
    fin_ids = [r['id'] for r in json.load(open(f'{S2}/final_batch1.json'))['records']]; assert len(fin_ids) == 7
    poolc = {}
    for r, role in [(r, 'front') for r in FR['front']]+[(byid[i], 'stage-2 finalist') for i in fin_ids]:
        k = kkey(r)
        if k in poolc: poolc[k]['roles'].append(role); continue
        poolc[k] = dict(id=r['id'], a=r['a'], chi=r['chi'], p=r['p'], roles=[role],
                        R2000=dict(B=r['B'], D=r['D'], D_se=r['D_se'], dec=r['dec'], dmD_C=r['dmD_C'], dmD_se=r['dmD_se'], mE_C=r['mE_C'], mE_se=r['mE_se']))
    plist = list(poolc.values()); n = len(plist)
    assert sum(is_uniform(c) for c in plist) == 1, "UNIF must be in the pool exactly once"
    inc = {row['lambda0']: row['new'] for row in FR['per_floor']}
    for lab, i in inc.items(): assert i in [c['id'] for c in plist], f"incumbent {i} ({lab}) not in the pool"
    M = 2*(n-1)+n*(n-1)//2; alpha = 1-CONF; q = 1-alpha/(2*M); tc = float(tdist.ppf(q, NB-1))
    frozen = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), pool=plist, n=n, incumbents=inc, delta=DELTA, confidence=CONF,
                  method=f"batch-mean t-intervals: {NB} batches of {BS} consecutive draws, {NB-1} df; point estimates from the full R={R8} matrices, "
                         f"se = sd(batch statistics, ddof=1)/sqrt({NB}), differences formed within each batch first",
                  stopping_rule=f"this single batch (seed {SEED_CONF}, R={R8}); no augmentation",
                  family=dict(budget=f"{n-1} intervals for D = B_c - B_UNIF (non-uniform candidates)", mE=f"{n-1} intervals for m_E^(C) (non-uniform candidates; one per candidate, serving all floors)",
                              pairs=f"{n*(n-1)//2} intervals for Delta m_D^(C)(i) - Delta m_D^(C)(j), all unordered pairs (contains every comparison vs UNIF and vs the incumbents)",
                              M=M, rule="each unique interval counted once; t_{19, 1-0.05/(2M)} applied to every interval"),
                  t_quantile_level=q, t_crit=tc, seed=SEED_CONF, R=R8, nbatch=NB, batch_size=BS, floors=[lab for lab, _ in FLOORS],
                  classifier=dict(budget="confirmed if upper end <= 0, rejected if lower end > 0, else undecided",
                                  mE="m_E >= lambda0 confirmed if lower end >= lambda0, rejected if upper end < lambda0, else undecided",
                                  UNIF="definitional values: budget confirmed, m_E = 1, Delta m_D = 0",
                                  feasibility="rejected if budget rejected or (lambda0 != none and m_E rejected); confirmed if budget confirmed and (lambda0 = none or m_E confirmed); else undecided",
                                  sets="challengers = confirmed feasible; c excluded if some challenger b has lower end of (dmD(b) - dmD(c)) > delta; "
                                       "candidate near-optimal set = feasibility not rejected and not excluded; groups: confirmed-feasible, feasibility-undecided"))
    print(f"pool: {n} candidates (Script-1 front {len(FR['front'])} points + 7 stage-2 finalists, deduplicated by key; no truncation)")
    print(f"{'#':>3s} {'id':34s} {'roles':28s} | R=2000 (seed 20261012): {'dmD_C':>7s} {'mE_C':>6s} {'D +- se':>14s} {'dec':5s}")
    for j, c in enumerate(plist):
        q_ = c['R2000']
        print(f"{j+1:3d} {c['id']:34s} {', '.join(c['roles']):28s} | {q_['dmD_C']:+7.3f} {q_['mE_C']:6.3f} {q_['D']:+6.2f} +- {q_['D_se']:4.2f} {q_['dec']:5s}")
    print("incumbent per floor (Script-1 best at R=2000): " + '; '.join(f"{lab}: {i}" for lab, i in inc.items()))
    print(f"delta = {DELTA} nat; confidence {CONF}; method: {frozen['method']}")
    print(f"stopping rule: {frozen['stopping_rule']}")
    print(f"comparison family: {frozen['family']['budget']}; {frozen['family']['mE']}; {frozen['family']['pairs']}")
    print(f"M = 2(n-1) + C(n,2) = 2*{n-1} + {n*(n-1)//2} = {M}; critical value t_{{19, 1-0.05/(2M)}} = t_{{19, {q:.8f}}} = {tc:.4f}")
    print(f"draw: seed {SEED_CONF}, R={R8} (not drawn here)")
    json.dump(frozen, open(FROZEN, 'w'), indent=1)
    print(f"written {FROZEN}; sha256 {hashlib.sha256(open(FROZEN, 'rb').read()).hexdigest()}")
    log_runtime('stage4_confirm freeze', T_START, f'pool n={n}, M={M}, t={tc:.4f}')
    sys.exit(0)

# --- draw: only from a committed, unmodified frozen file --------------------------------------------------------------------------------
def git(*a): return subprocess.run(['git', *a], capture_output=True, text=True)
assert git('ls-files', '--error-unmatch', FROZEN).returncode == 0, f"{FROZEN} is not tracked by git: commit it before drawing"
assert git('diff', '--quiet', 'HEAD', '--', FROZEN).returncode == 0, f"{FROZEN} differs from HEAD: commit it before drawing"
sys.stdout = Tee(f'{OUT}/stage4_confirm.log')
F = json.load(open(FROZEN)); plist = F['pool']; n = F['n']; tc = F['t_crit']; M = F['family']['M']; inc = F['incumbents']
assert F['seed'] == SEED_CONF and F['R'] == R8 and F['nbatch'] == NB and F['delta'] == DELTA
print(f"=== stage4_confirm.py draw: frozen pool from {FROZEN} (commit {git('log', '-1', '--format=%h', '--', FROZEN).stdout.strip()}, "
      f"sha256 {hashlib.sha256(open(FROZEN, 'rb').read()).hexdigest()[:16]}), n={n}, M={M}, t={tc:.4f}; seed {SEED_CONF}, R={R8}, {NB} batches of {BS} ===")
check_trigger()

def _eval_worker_nb(args):
    a, chi = args
    return evaluate_crn(a, chi, _ZW, nbatch=NB)
def eval_many_nb(cands, Zb):
    global _ZW
    _ZW = Zb
    with pool(4) as pl: return pl.map(_eval_worker_nb, [(c['a'], c['chi']) for c in cands], chunksize=1)

t1 = time.time()
Z8 = np.random.default_rng(SEED_CONF).standard_normal((R8, K))
ref = evaluate_crn(a_u, chi_u, Z8, nbatch=NB); raw = eval_many_nb(plist, Z8); del Z8
ldC = np.linalg.slogdet(ref['J'])[1]; cu = ref['cnt']
print(f"evaluated reference + {n} candidates at R={R8} in {time.time()-t1:.0f} s; reference (C) uniform: B_u = {cu.mean():.3f} +- {cu.std(ddof=1)/np.sqrt(R8):.3f} (se), "
      f"log det J_C = {ldC:.4f}")

def bse(v): return float(np.std(v, ddof=1)/np.sqrt(NB))
C = []
for c, res in zip(plist, raw):
    dc = res['cnt']-cu; Db = dc.reshape(NB, BS).mean(1)
    ld = np.linalg.slogdet(res['J'])[1]; dmD = float(ld-ldC); dmDb = res['ldb']-ref['ldb']
    mE = float(eigh(res['J'], ref['J'], eigvals_only=True)[0]); mEb = np.array([eigh(res['Jbat'][b], ref['Jbat'][b], eigvals_only=True)[0] for b in range(NB)])
    u = is_uniform(c)
    if u: assert dc.max() == 0 and dc.min() == 0 and dmD == 0.0 and np.all(dmDb == 0), "UNIF self-check failed"
    C.append(dict(id=c['id'], roles=c['roles'], uniform=u, B=float(res['cnt'].mean()), D=float(dc.mean()), D_se=bse(Db), D_batch_mean=float(Db.mean()),
                  dmD=dmD, dmD_se=bse(dmDb), dmD_nonlin=float(dmDb.mean()-dmD), mE=mE, mE_se=bse(mEb), mE_nonlin=float(mEb.mean()-mE),
                  ldb=res['ldb'].tolist(), R2000=c['R2000']))
print("UNIF self-check (evaluated as a pool candidate on the same draws): D = 0 and Delta m_D = 0 exactly, batch statistics 0: passed")
for c in C:
    if c['uniform']:
        c.update(D_lo=0.0, D_hi=0.0, budget='confirmed (definition)', mE=1.0, mE_lo=1.0, mE_hi=1.0)
    else:
        c.update(D_lo=c['D']-tc*c['D_se'], D_hi=c['D']+tc*c['D_se'], mE_lo=c['mE']-tc*c['mE_se'], mE_hi=c['mE']+tc*c['mE_se'])
        c['budget'] = 'confirmed' if c['D_hi'] <= 0 else ('rejected' if c['D_lo'] > 0 else 'undecided')
    c['dmD_lo'] = c['dmD']-tc*c['dmD_se']; c['dmD_hi'] = c['dmD']+tc*c['dmD_se']
    c['mEstat'] = {}; c['feas'] = {}
    for lab, l0 in FLOORS:
        if l0 is None: ms = 'n/a'
        elif c['uniform']: ms = 'confirmed (definition)'
        else: ms = 'confirmed' if c['mE_lo'] >= l0 else ('rejected' if c['mE_hi'] < l0 else 'undecided')
        c['mEstat'][lab] = ms; bs_ = c['budget'].split(' ')[0]; ms_ = ms.split(' ')[0]
        if bs_ == 'rejected' or ms_ == 'rejected': f_ = 'rejected'
        elif bs_ == 'confirmed' and ms_ in ('confirmed', 'n/a'): f_ = 'confirmed'
        else: f_ = 'undecided'
        c['feas'][lab] = f_
byid = {c['id']: c for c in C}
pairs = {}
for i, j in itertools.combinations(range(n), 2):
    a_, b_ = C[i], C[j]; d = a_['dmD']-b_['dmD']; se = bse(np.array(a_['ldb'])-np.array(b_['ldb']))
    pairs[(a_['id'], b_['id'])] = dict(i=a_['id'], j=b_['id'], diff=d, se=se, lo=d-tc*se, hi=d+tc*se)
assert len(pairs) == n*(n-1)//2
def pdiff(b, c):
    """interval for dmD(b) - dmD(c) from the unordered pair"""
    if (b, c) in pairs: q = pairs[(b, c)]; return q['diff'], q['se'], q['lo'], q['hi']
    q = pairs[(c, b)]; return -q['diff'], q['se'], -q['hi'], -q['lo']
nint = sum(not c['uniform'] for c in C)*2+len(pairs); assert nint == M, (nint, M)
print(f"intervals formed: {sum(not c['uniform'] for c in C)} budget + {sum(not c['uniform'] for c in C)} m_E + {len(pairs)} pairwise = {nint} = M; each uses t = {tc:.4f}")

print(f"\nTable C1. Confirmation batch (seed {SEED_CONF}, R={R8}, {NB}x{BS}); point estimates from the full matrices; se from the {NB} paired batch statistics; "
      f"intervals = point +- {tc:.4f} se (approximate overall confidence {CONF} over the M={M} intervals); R=2000 values (seed 20261012) for comparison")
print(f"{'id':34s} {'D':>7s} {'(se)':>6s} {'interval':>17s} {'budget':10s} | {'dmD_C':>7s} {'(se)':>6s} {'interval':>17s} | {'mE_C':>7s} {'(se)':>7s} {'interval':>17s} | "
      f"{'mE>=0.3':9s} {'mE>=0.5':9s} {'mE>=0.7':9s} | R2000: {'dmD_C':>7s} {'mE_C':>6s} {'D':>6s}")
for c in C:
    q_ = c['R2000']
    print(f"{c['id']:34s} {c['D']:+7.3f} ({c['D_se']:.3f}) [{c['D_lo']:+7.3f},{c['D_hi']:+7.3f}] {c['budget'].split(' ')[0]:10s} | {c['dmD']:+7.3f} ({c['dmD_se']:.4f}) "
          f"[{c['dmD_lo']:+7.3f},{c['dmD_hi']:+7.3f}] | {c['mE']:7.4f} ({c['mE_se']:.5f}) [{c['mE_lo']:.4f},{c['mE_hi']:.4f}] | "
          + ' '.join(f"{c['mEstat'][lab].split(' ')[0]:9s}" for lab, _ in FLOORS[1:]) + f" | {q_['dmD_C']:+7.3f} {q_['mE_C']:6.3f} {q_['D']:+6.2f}")
print("(UNIF: definitional values; its D and m_E intervals are not part of the family)")
print(f"\nTable C1b. Nonlinearity diagnostic (not a bias bound): mean of the {NB} batch statistics minus the full-sample statistic")
print(f"{'id':34s} {'dmD: batch mean - full':>23s} {'(in dmD se)':>11s} {'mE: batch mean - full':>22s} {'(in mE se)':>10s}")
for c in C:
    if c['uniform']: print(f"{c['id']:34s} {'0 (identical to the reference)':>23s}"); continue
    print(f"{c['id']:34s} {c['dmD_nonlin']:+23.5f} {c['dmD_nonlin']/c['dmD_se']:+11.2f} {c['mE_nonlin']:+22.6f} {c['mE_nonlin']/c['mE_se']:+10.2f}")
nl = np.array([c['dmD_nonlin'] for c in C if not c['uniform']]); nm = np.array([c['mE_nonlin'] for c in C if not c['uniform']])
print(f"  range over the non-uniform candidates: dmD [{nl.min():+.5f},{nl.max():+.5f}], mE [{nm.min():+.6f},{nm.max():+.6f}]")
cnt = {k: sum(c['budget'].split(' ')[0] == k for c in C if not c['uniform']) for k in ('confirmed', 'undecided', 'rejected')}
print(f"budget status of the {n-1} non-uniform candidates: {cnt}")

print(f"\nTable C2. Pairwise Delta m_D^(C) differences, all C(n,2) = {len(pairs)} unordered pairs (i - j; interval = diff +- {tc:.4f} se)")
for q in pairs.values(): print(f"  {q['i']:34s} - {q['j']:34s} {q['diff']:+8.4f} ({q['se']:.4f}) [{q['lo']:+8.4f},{q['hi']:+8.4f}]")

sets = {}
print(f"\nTable C3. Candidate near-optimal sets per floor (delta = {DELTA} nat; challengers = candidates confirmed feasible for the floor)")
for lab, l0 in FLOORS:
    ch = [c for c in C if c['feas'][lab] == 'confirmed']
    excl = {}
    for c in C:
        best = None
        for b in ch:
            if b['id'] == c['id']: continue
            d, se, lo, hi = pdiff(b['id'], c['id'])
            if lo > DELTA and (best is None or lo > best[1]): best = (b['id'], lo, d, se)
        if best: excl[c['id']] = best
    conf_ = [c for c in C if c['feas'][lab] == 'confirmed' and c['id'] not in excl]
    und_ = [c for c in C if c['feas'][lab] == 'undecided' and c['id'] not in excl]
    rej_ = [c for c in C if c['feas'][lab] == 'rejected']
    ic = byid[inc[lab]]
    sets[lab] = dict(challengers=[c['id'] for c in ch], near_opt_confirmed=[c['id'] for c in conf_], near_opt_undecided=[c['id'] for c in und_],
                     excluded={k: dict(by=v[0], lo=v[1], diff=v[2], se=v[3]) for k, v in excl.items()}, feasibility_rejected=[c['id'] for c in rej_],
                     undecided_feasibility=[c['id'] for c in C if c['feas'][lab] == 'undecided'],
                     incumbent=dict(id=ic['id'], feasibility=ic['feas'][lab], excluded_by=excl.get(ic['id'], [None])[0], in_set=ic['id'] in [c['id'] for c in conf_+und_]))
    print(f"\n  floor lambda0 = {lab}: {len(ch)} challengers (confirmed feasible): {', '.join(c['id'] for c in ch)}")
    print(f"    feasibility undecided: {len(sets[lab]['undecided_feasibility'])}: {', '.join(sets[lab]['undecided_feasibility']) or 'none'}; "
          f"feasibility rejected: {len(rej_)}: {', '.join(c['id'] for c in rej_) or 'none'}")
    print(f"    candidate near-optimal set, confirmed-feasible group ({len(conf_)}):")
    for c in sorted(conf_, key=lambda c: -c['dmD']):
        print(f"      {c['id']:34s} dmD_C={c['dmD']:+.4f} [{c['dmD_lo']:+.4f},{c['dmD_hi']:+.4f}] mE={c['mE']:.4f} D={c['D']:+.3f} (budget {c['budget'].split(' ')[0]}, m_E {c['mEstat'][lab].split(' ')[0]})")
    print(f"    candidate near-optimal set, feasibility-undecided group ({len(und_)}):")
    for c in sorted(und_, key=lambda c: -c['dmD']):
        print(f"      {c['id']:34s} dmD_C={c['dmD']:+.4f} [{c['dmD_lo']:+.4f},{c['dmD_hi']:+.4f}] mE={c['mE']:.4f} D={c['D']:+.3f} (budget {c['budget'].split(' ')[0]}, m_E {c['mEstat'][lab].split(' ')[0]})")
    print(f"    excluded (by the challenger with the largest lower end of dmD(b) - dmD(c)):")
    for k, v in sorted(excl.items(), key=lambda kv: kv[1][1]):
        print(f"      {k:34s} by {v[0]:34s} diff {v[2]:+.4f} (se {v[3]:.4f}), lower end {v[1]:+.4f} > {DELTA}")
    ist = sets[lab]['incumbent']
    print(f"    incumbent {ic['id']}: feasibility {ic['feas'][lab]}" + ("" if ic['feas'][lab] == 'confirmed' else " -- NOT confirmed feasible, so it cannot act as a challenger")
          + (f"; excluded by {ist['excluded_by']}" if ist['excluded_by'] else '') + f"; in the candidate near-optimal set: {ist['in_set']}")
    grp = lambda L: ', '.join(c['id'] for c in sorted(L, key=lambda c: -c['dmD'])) or 'none'
    print(f"    Within this frozen pool, at approximate overall confidence {CONF}, the candidate near-optimal set at floor lambda0 = {lab} is "
          f"{{confirmed-feasible: {grp(conf_)}}} and {{feasibility-undecided: {grp(und_)}}}.")
print("\nMembership in the candidate near-optimal set means that the candidate has not been excluded; it does not certify that every member is feasible "
      "or within delta of the true best feasible candidate in the frozen pool.")
print("Intervals are batch-mean approximations for a logdet-of-mean estimator; nothing is claimed about unevaluated inputs.")
print("\nUndecided feasibilities per floor: " + '; '.join(f"{lab}: {len(sets[lab]['undecided_feasibility'])}" for lab, _ in FLOORS)
      + f"; total {sum(len(sets[lab]['undecided_feasibility']) for lab, _ in FLOORS)}")
print("m_E undecided (candidate, floor): " + (', '.join(f"{c['id']}@{lab}" for c in C for lab, _ in FLOORS[1:] if c['mEstat'][lab] == 'undecided') or 'none'))
json.dump(dict(frozen=F, R=R8, seed=SEED_CONF, nbatch=NB, B_u=float(cu.mean()), ldC=float(ldC), candidates=C, pairs=list(pairs.values()), sets=sets),
          open(f'{OUT}/confirm_R8000.json', 'w'), indent=1)
log_runtime('stage4_confirm draw', T_START, f'n={n} candidates at R={R8}, M={M}')
