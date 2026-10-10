"""§8.5 stage 2, task 1: merge the 3000 LHS candidates with the 1465 stage-1 candidates (all R=200, seed 20261010
common draws, reference (C) = uniform input on the same draws), deduplicate, apply the budget decision of
8.3.6(g) (mc85_post.py::qual; the uniform reference qualifies by definition), compute the empirical Pareto front of
the qualified candidates in (Delta m_D^(C), m_E^(C)), check predictions 1-4 of mc85_stage2_protocol.md, and select
~40 candidates for the R=2000 refinement.  Run from code_8_3_stage2/stage2_pkg/:  python3 stage2_post.py
Outputs (stage2/): stage2_post.log, merged_R200.json, front_R200.json, refine_selection.json"""
exec(open('stage2_common.py').read())
from scipy.stats import spearmanr
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage2_post.log')
print("=== stage2_post.py: merge, decisions, empirical Pareto front at R=200 (seed 20261010) ===")
check_trigger()

# --- load -------------------------------------------------------------------------------------------------
s1 = []
for k in range(5): s1 += json.load(open(f'mc85_scan_part{k}.json'))['results']
s1 += json.load(open('mc85_amp20.json'))
lhs_idx = {0: range(0, 500)}; lhs_idx.update({k: range(500+(k-2)*250, 500+(k-1)*250) for k in range(2, 12)})
lhs = []
for k, rg in lhs_idx.items():
    part = json.load(open(f'mc85_lhs_part{k}.json')); assert len(part) == len(rg)
    for r, i in zip(part, rg): r['lhs_index'] = i; lhs.append(r)
# regenerate the LHS design exactly as mc85_lhs.py and check the stored inputs and their index blocks
rs = np.random.default_rng(20261011); P = rs.dirichlet(np.ones(4), size=3000); Ul = np.zeros((3000, 4))
for m in range(4): Ul[:, m] = (rs.permutation(3000)+rs.random(3000))/3000*2*np.pi
mis = max(max(np.abs(np.array(r['a'])-np.sqrt(P[r['lhs_index']])).max(), np.abs(np.array(r['chi'])-Ul[r['lhs_index']]).max()) for r in lhs)
print(f"loaded stage-1: {len(s1)} (amp {sum(r['tag']=='amp' for r in s1)}, phase {sum(r['tag']=='phase' for r in s1)}, amp20 {sum(r['tag']=='amp20' for r in s1)}); "
      f"LHS: {len(lhs)}; max |stored - regenerated LHS design| = {mis:.1e}")

def cid(r):
    if r['tag'] == 'lhs': return f"L{r['lhs_index']:04d}"
    if r['tag'] == 'phase': return 'P' + ''.join(str(int(v)) for v in np.round(np.array(r['chi'])/(np.pi/3)))
    if r['tag'] == 'amp': return 'A' + ''.join(str(int(v)) for v in np.round(np.array(r['a'])**2*10))
    return 'Q' + ''.join(str(int(v)) for v in np.round(np.array(r['a'])**2*20))
groups = {}
for r in s1+lhs:
    r['id'] = cid(r); r['src'] = r['tag']
    key = tuple(np.round(np.array(r['a'])**2, 9))+tuple(np.round(np.mod(r['chi'], TWO_PI), 9))
    groups.setdefault(key, []).append(r)
merged = []
for g in groups.values():
    r = dict(g[0]); r['dup_ids'] = [q['id'] for q in g[1:]]
    if len(g) > 1:
        same = all(all(q[k] == g[0][k] for k in ('B', 'D', 'dmD_C', 'mE_C')) for q in g[1:])
        print(f"  duplicate group {[q['id'] for q in g]}: identical stored values = {same}")
    if is_uniform(r): r['id'] = 'UNIF'
    r['p'] = (np.array(r['a'])**2).tolist(); r['chi'] = np.mod(r['chi'], TWO_PI).tolist(); r['dec'] = qual(r); r['R'] = 200; r['seed'] = 20261010
    merged.append(r)
print(f"merged: {len(s1)+len(lhs)} -> {len(merged)} unique candidates after deduplication")

# --- end-to-end reproduction of stored values with the accelerated pipeline (R=200, seed 20261010) ---------
Z200 = np.random.default_rng(20261010).standard_normal((200, K))
ref200 = evaluate_crn(a_u, chi_u, Z200)
byid = {r['id']: r for r in merged}
dev = 0.0
for i in ['P5014', 'A4222', 'Q5735', 'L0000', 'L2999']:
    r0 = byid[i]; rr = summarize_s2(i, r0['a'], r0['chi'], evaluate_crn(r0['a'], r0['chi'], Z200), ref200, 200, 20261010)
    dev = max(dev, max(abs(rr[k]-r0[k]) for k in ('B', 'D', 'D_se', 'dmD_C', 'dmD_B', 'mE_C', 'mE_B', 'gap', 'Ein', 'Eout')), np.abs(np.array(rr['lamC'])-r0['lamC']).max())
print(f"reproduction of 5 stored candidates (P5014, A4222, Q5735, L0000, L2999) with evaluate_crn at R=200: max |diff| = {dev:.1e}; "
      f"reference ldC = {np.linalg.slogdet(ref200['J'])[1]:.6f}, B_u = {ref200['cnt'].mean():.2f}")

# --- counts, ranges, correlations ---------------------------------------------------------------------------
def cnts(rs): return {k: sum(r['dec'] == k for r in rs) for k in ('ok', 'over', 'undec')}
srcs = [('stage-1 amp (84)', 'amp'), ('stage-1 amp20', 'amp20'), ('stage-1 phase', 'phase'), ('LHS', 'lhs')]
print("\nTable 1. Counts and ranges (R=200, seed 20261010); decision 8.3.6(g), uniform reference qualifies by definition")
print(f"{'set':18s} {'n':>5s} {'ok':>5s} {'over':>5s} {'undec':>5s}   {'dmD_C range':>18s}   {'mE_C range':>14s}   {'B range':>14s}  {'corr(B,dmD)':>11s} {'Spearman':>8s}")
for name, tg in srcs+[('all merged', None)]:
    rs_ = [r for r in merged if tg is None or r['src'] == tg]
    Dm = np.array([r['dmD_C'] for r in rs_]); mE = np.array([r['mE_C'] for r in rs_]); B = np.array([r['B'] for r in rs_]); c = cnts(rs_)
    print(f"{name:18s} {len(rs_):5d} {c['ok']:5d} {c['over']:5d} {c['undec']:5d}   [{Dm.min():+7.3f},{Dm.max():+7.3f}]   [{mE.min():.3f},{mE.max():.3f}]   [{B.min():6.1f},{B.max():6.1f}]  {np.corrcoef(B, Dm)[0, 1]:11.3f} {spearmanr(B, Dm)[0]:8.3f}")
ok = [r for r in merged if r['dec'] == 'ok']
Dq = np.array([r['dmD_C'] for r in ok]); mq = np.array([r['mE_C'] for r in ok]); Bq = np.array([r['B'] for r in ok])
print(f"qualified only: n={len(ok)}, dmD_C [{Dq.min():+.3f},{Dq.max():+.3f}], mE_C [{mq.min():.3f},{mq.max():.3f}], corr(B,dmD)={np.corrcoef(Bq, Dq)[0, 1]:.3f}")

def row(r, extra=''):
    return (f"{r['id']:6s} {r['src']:5s} p={fmt_p(r)} chi={fmt_chi(r)} B={r['B']:6.2f} D={r['D']:+6.2f}+-{r['D_se']:.2f} {r['dec']:5s} "
            f"dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} dmD_B={r['dmD_B']:+.3f}{extra}")

# --- empirical Pareto front of the qualified candidates -----------------------------------------------------
fi = pareto_front(ok); front = [ok[i] for i in fi]
print(f"\nTable 2. Empirical Pareto front of the evaluated qualified candidates (R=200, seed 20261010): {len(front)} points")
for r in front: print('  '+row(r))
json.dump(dict(R=200, seed=20261010, front=front), open(f'{OUT}/front_R200.json', 'w'), indent=1)

# --- predictions ---------------------------------------------------------------------------------------------
print("\n--- prediction checks at R=200 (screening level; re-checked at R=2000 in stage2_refine/local) ---")
best = max(ok, key=lambda r: r['dmD_C']); okl = [r for r in ok if r['src'] == 'lhs']; bestl = max(okl, key=lambda r: r['dmD_C'])
oknew = [r for r in ok if r['src'] in ('lhs',)]
print(f"P1: stage-1 phase-only best (qualified) = P5014 dmD_C=+{byid['P5014']['dmD_C']:.3f}; prediction: joint best qualified in (+0.597, +1.097)")
print(f"    best qualified overall: {best['id']} dmD_C={best['dmD_C']:+.3f}; best qualified LHS (amplitude+phase): {bestl['id']} dmD_C={bestl['dmD_C']:+.3f} mE_C={bestl['mE_C']:.3f} B={bestl['B']:.1f}")
print(f"    LHS qualified with dmD_C > +0.597: {sum(r['dmD_C'] > 0.597 for r in okl)};  LHS (any decision) with dmD_C > +0.597: {sum(r['dmD_C'] > 0.597 for r in merged if r['src']=='lhs')} "
      f"(of which over budget: {sum(r['dmD_C'] > 0.597 and r['dec']=='over' for r in merged if r['src']=='lhs')})")
print("    top-5 qualified LHS:"); [print('     '+row(r)) for r in sorted(okl, key=lambda r: -r['dmD_C'])[:5]]
lam_grid = np.round(np.arange(0, 1.0001, 0.05), 2)
Dstar = [max(r['dmD_C'] for r in ok if r['mE_C'] >= l-1e-9) for l in lam_grid]   # 1e-9: m_E of the reference is 1 up to rounding
arg = [max((r for r in ok if r['mE_C'] >= l-1e-9), key=lambda r: r['dmD_C'])['id'] for l in lam_grid]
print("P2: max dmD_C over qualified with mE_C >= lambda0:")
for l, d_, i_ in zip(lam_grid, Dstar, arg): print(f"    lambda0={l:.2f}  max dmD_C={d_:+.3f}  ({i_})")
print(f"    non-increasing in lambda0: {all(np.diff(Dstar) <= 0)}")
seg = [(front[j], front[j+1]) for j in range(len(front)-1)]
print("    front segments (sorted by dmD_C desc): slope d mE / d dmD")
for r1, r2 in seg:
    print(f"     {r1['id']}({r1['dmD_C']:+.3f},{r1['mE_C']:.3f}) -> {r2['id']}({r2['dmD_C']:+.3f},{r2['mE_C']:.3f}): slope {(r2['mE_C']-r1['mE_C'])/(r2['dmD_C']-r1['dmD_C']):+.2f} per nat")
lo = [r for r in front if r['mE_C'] < 0.1]; hi = [r for r in front if r['mE_C'] >= 0.6]
print(f"    front points with mE_C<0.1: {len(lo)}, dmD_C span {max(r['dmD_C'] for r in lo)-min(r['dmD_C'] for r in lo):.3f} nat; "
      f"with mE_C>=0.6: {len(hi)}, dmD_C span {max(r['dmD_C'] for r in hi)-min(r['dmD_C'] for r in hi):.3f} nat, mE_C span {max(r['mE_C'] for r in hi)-min(r['mE_C'] for r in hi):.3f}")
# P3 (read off): relative phase chi3-chi4 and power distribution of the best candidates
L = [r for r in merged if r['src'] == 'lhs']; chiL = np.array([r['chi'] for r in L]); DmL = np.array([r['dmD_C'] for r in L]); BL = np.array([r['B'] for r in L])
def eta2(y, g): return sum(y[g == v].size*(y[g == v].mean()-y.mean())**2 for v in np.unique(g))/y.size/y.var()
def bins(v, nb=6): return np.floor(np.mod(v, TWO_PI)/(TWO_PI/nb)).astype(int)
print("P3 (not predicted; read off). LHS cloud, 6 bins: eta^2 of dmD_C / of B on")
for lab, v in [('chi1', chiL[:, 0]), ('chi2', chiL[:, 1]), ('chi3', chiL[:, 2]), ('chi4', chiL[:, 3]), ('chi3-chi4', chiL[:, 2]-chiL[:, 3]),
               ('chi1-chi2', chiL[:, 0]-chiL[:, 1]), ('chi1-chi4', chiL[:, 0]-chiL[:, 3]), ('chi2-chi3', chiL[:, 1]-chiL[:, 2])]:
    print(f"    {lab:10s} eta2(dmD)={eta2(DmL, bins(v)):.3f}  eta2(B)={eta2(BL, bins(v)):.3f}")
pL = np.array([r['p'] for r in L])
print(f"    LHS: eta2(dmD) on p1 sextile-bins = {eta2(DmL, np.minimum((pL[:, 0]*6).astype(int), 5)):.3f}; corr(p1, dmD)={np.corrcoef(pL[:, 0], DmL)[0, 1]:.3f}, corr(p1, B)={np.corrcoef(pL[:, 0], BL)[0, 1]:.3f}")
top = sorted(ok, key=lambda r: -r['dmD_C'])[:10]
print("    top-10 qualified: (chi3-chi4) mod 2pi and ||p - 1/4||_1")
for r in top: print(f"     {r['id']:6s} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.3f} chi3-chi4={np.mod(r['chi'][2]-r['chi'][3], TWO_PI):.3f} ||p-1/4||_1={np.abs(np.array(r['p'])-0.25).sum():.3f}")
topl = sorted(okl, key=lambda r: -r['dmD_C'])[:20]
print(f"    top-20 qualified LHS: mean p = {np.round(np.mean([r['p'] for r in topl], 0), 3)}, mean ||p-1/4||_1 = {np.mean([np.abs(np.array(r['p'])-0.25).sum() for r in topl]):.3f} "
      f"(all LHS: {np.mean(np.abs(pL-0.25).sum(1)):.3f}); resultant length of (chi3-chi4) = {np.abs(np.mean(np.exp(1j*np.array([r['chi'][2]-r['chi'][3] for r in topl])))):.3f}, "
      f"circular mean {np.mod(np.angle(np.mean(np.exp(1j*np.array([r['chi'][2]-r['chi'][3] for r in topl])))), TWO_PI):.3f} rad")
# P4: correlation and counter-examples across the LHS cloud
dB = BL[:, None]-BL[None, :]; dD = DmL[:, None]-DmL[None, :]
big = dB >= 10; ce = big & (dD <= -0.3)
print(f"P4: LHS cloud corr(B, dmD_C) Pearson={np.corrcoef(BL, DmL)[0, 1]:.3f} Spearman={spearmanr(BL, DmL)[0]:.3f} (stage 1: 0.76 amplitude, 0.91 phase)")
print(f"    counter-examples: ordered LHS pairs with B_i-B_j >= 10 sends: {big.sum()}; of these dmD_C,i <= dmD_C,j - 0.3 nat: {ce.sum()} ({ce.sum()/big.sum():.1%})")
ii, jj = np.unravel_index(np.argmax(np.where(ce, np.minimum(dB/10, -dD/0.3), -np.inf)), dB.shape)
print(f"    most pronounced: {L[ii]['id']} B={BL[ii]:.1f} dmD_C={DmL[ii]:+.3f} vs {L[jj]['id']} B={BL[jj]:.1f} dmD_C={DmL[jj]:+.3f}")
nd = (DmL[:, None] >= -1) & (DmL[None, :] >= -1); bn = big & nd; cn = ce & nd
ii, jj = np.unravel_index(np.argmax(np.where(cn, np.minimum(dB/10, -dD/0.3), -np.inf)), dB.shape)
print(f"    restricted to non-degenerate pairs (both dmD_C >= -1, {int((DmL >= -1).sum())} LHS points): {bn.sum()} pairs with B_i-B_j >= 10, of these {cn.sum()} ({cn.sum()/bn.sum():.1%}) with dmD_C,i <= dmD_C,j - 0.3;"
      f" most pronounced: {L[ii]['id']} B={BL[ii]:.1f} dmD_C={DmL[ii]:+.3f} vs {L[jj]['id']} B={BL[jj]:.1f} dmD_C={DmL[jj]:+.3f}")
qL = np.array([r['dec'] == 'ok' for r in L])
print(f"    qualified LHS only: Pearson={np.corrcoef(BL[qL], DmL[qL])[0, 1]:.3f}, Spearman={spearmanr(BL[qL], DmL[qL])[0]:.3f}")

# --- selection for the R=2000 refinement ----------------------------------------------------------------------
# F1: front of the qualified (ends + middle, includes the uniform reference); F2: second Pareto layer of the
# qualified; FL: front of the qualified LHS (amplitude+phase) candidates; BB: budget-boundary = undecided
# candidates (|D| <= 2 se) in the first two Pareto layers of (qualified + undecided).
fid = {r['id'] for r in front}; rest = [r for r in ok if r['id'] not in fid]; F2 = [rest[i] for i in pareto_front(rest)]
FL = [okl[i] for i in pareto_front(okl)]
ou = [r for r in merged if r['dec'] in ('ok', 'undec')]; L1 = [ou[i] for i in pareto_front(ou)]
l1id = {r['id'] for r in L1}; ou2 = [r for r in ou if r['id'] not in l1id]; L2 = [ou2[i] for i in pareto_front(ou2)]
BB = [r for r in L1+L2 if r['dec'] == 'undec']
sel = {}
for lab, grp in [('F1', front), ('F2', F2), ('FL', FL), ('BB', BB)]:
    for r in grp: sel.setdefault(r['id'], dict(r, why=[]))['why'].append(lab)
print(f"\nTable 3. Selected for refinement: {len(sel)} candidates (F1 front {len(front)}, F2 second layer {len(F2)}, "
      f"FL LHS front {len(FL)}, BB budget-boundary {len(BB)}; overlaps counted once; uniform reference included via F1)")
for r in sorted(sel.values(), key=lambda r: -r['dmD_C']): print('  '+row(r, '  ['+'+'.join(r['why'])+']'))
assert 'UNIF' in sel
json.dump(list(sel.values()), open(f'{OUT}/refine_selection.json', 'w'), indent=1)
json.dump(dict(R=200, seed=20261010, n=len(merged), records=merged), open(f'{OUT}/merged_R200.json', 'w'))
log_runtime('stage2_post', T_START, f'{len(merged)} candidates merged, {len(sel)} selected')
