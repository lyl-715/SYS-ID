"""§8.5 stage 4, Script 3: scenario robustness at d = 5 and d = 50 (sigma_d = Delta*/d), rules in stage4/design.md §3.
Representatives: the four non-uniform floor bests after Script 1 (replacement rule of design.md §3, re-derived here and asserted equal
to stage4/front_R2000.json) plus UNIF. Seed 20261021, R=4000; the same draws Z for d = 5 and d = 50 (CRN shared across the two d, so
the two scenarios are NOT independent batches). For each d, before any evaluation and before the worker pool forks:
  - the module global s (used by evaluate_crn for the noise z = x + s Z, for alpha, beta in c_k = cfun(.) and for J = H'WH / s^2)
    is rebound to Delta*/d; check_trigger() is re-run at that sigma;
  - Jb and ldB (used by summarize_s2 for Delta m_D^(B) and m_E^(B)) are deleted, then rebuilt as J_per(u_unif, N_d) =
    H_p'H_p / sigma_d^2 at t_i = i T / N_d, N_d = floor(B_hat_d) of the scenario's own uniform reference on the same draws;
    ti, Hp, Nper rebound likewise; rank (eigenvalues > 1e-9 max, mc83.py convention) and condition number logged;
  - the scenario's own uniform reference (C)_d on the same draws: B_hat_d with se, N_d.
Per representative: paired send-count difference vs (C)_d and the 8.3.6(g) decision (qual), Delta m_D^(C) (paired batch se, 10x400),
nine generalized eigenvalues vs (C)_d, m_E and 1/m_E, Delta m_D vs the matched periodic reference (B)_d, window input power (must equal
the d=20 record bitwise: consistency check). Addition (design.md §3): a d = 20 control on the same draws, so the direction of change
d=20 -> 5 and d=20 -> 50 is a paired comparison (paired se from per-draw send differences and per-batch log det differences).
Run from code_8_3_stage2/stage2_pkg/:  python3 stage4_scenarios.py      Outputs (stage4/): stage4_scenarios.log, scenarios.json"""
exec(open('stage4_common.py').read())
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage4_scenarios.log')
R4 = 4000; D_STAR = D
print(f"=== stage4_scenarios.py: d in (5, 50) + d=20 control, sigma_d = Delta*/d, Delta* = {D_STAR!r}; seed {SEED_SCEN}, R={R4} (CRN shared across d) ===")
check_trigger()

PM = json.load(open(f'{OUT}/pool_R2000.json')); recs = PM['records']; byid = {r['id']: r for r in recs}
FR = json.load(open(f'{OUT}/front_R2000.json'))
reps = representatives(recs); assert reps == FR['representatives'], "representatives differ from stage4/front_R2000.json"
RL = [(lab, byid[v['id']]) for lab, v in reps.items() if v is not None]+[('ref', byid['UNIF'])]
print("representatives (design.md §3; R=2000, seed 20261012 values at d=20):")
for lab, r in RL:
    print(f"  {lab:4s} {r['id']:34s} p={fmt_p(r)} chi={fmt_chi(r)} dmD_C={r['dmD_C']:+.3f} mE_C={r['mE_C']:.4f} 1/mE={1/r['mE_C']:.2f} D={r['D']:+.2f}+-{r['D_se']:.2f} {r['dec']} Ein={r['Ein']:.6f}")
try:
    CF = json.load(open(f'{OUT}/confirm_R8000.json')); cst = {c['id']: c for c in CF['candidates']}
    print("Script-2 status (seed 20261020, R=8000; does not change the representatives):")
    for lab, r in RL:
        c = cst.get(r['id'])
        if c: print(f"  {r['id']:34s} budget {c['budget']}; feasibility " + ', '.join(f"{l}: {c['feas'][l]}" for l, _ in FLOORS)
                    + f"; dmD_C={c['dmD']:+.4f} [{c['dmD_lo']:+.4f},{c['dmD_hi']:+.4f}], mE={c['mE']:.4f}")
        else: print(f"  {r['id']}: not in the frozen pool")
except FileNotFoundError: print("Script-2 output not found (status not reported)")

Z4 = np.random.default_rng(SEED_SCEN).standard_normal((R4, K))
SC = {}
for d_ in (5, 50, 20):
    t1 = time.time()
    s = D_STAR/d_                                                       # rebinds the module global used by evaluate_crn / cfun / J
    for _g in ('Jb', 'ldB', 'Hp', 'ti', 'Nper'): globals().pop(_g, None)  # no leftover d=20 (B) reference
    print(f"\n=== d = {d_}{' (control, same draws)' if d_ == 20 else ''}: sigma = Delta*/{d_} = {s:.6e} ===")
    check_trigger()
    ref = evaluate_crn(a_u, chi_u, Z4); cu = ref['cnt']
    Bd = float(cu.mean()); Bse = float(cu.std(ddof=1)/np.sqrt(R4)); Nd = int(np.floor(Bd))
    Nper = Nd; ti = np.arange(1, Nd+1)*T/Nd; Hp = regressors(a_u, chi_u, ti); Jb = Hp.T@Hp/s**2; ldB = np.linalg.slogdet(Jb)[1]
    evp = np.linalg.eigvalsh(Jb); rank = int(np.sum(evp > 1e-9*evp[-1])); cond = float(evp[-1]/evp[0])
    ldC = np.linalg.slogdet(ref['J'])[1]
    print(f"uniform reference (C)_{d_}: B_hat = {Bd:.3f} +- {Bse:.3f} (se), sd {cu.std(ddof=1):.2f}; N_{d_} = floor(B_hat) = {Nd}; log det J_C = {ldC:.4f}")
    print(f"J_per(u_unif, N={Nd}) at sigma_{d_}: rank {rank}/9, condition number {cond:.4e}, log det {ldB:.4f}; (C)-(B) offset log det J_C - log det J_per = {ldC-ldB:+.4f}")
    raw = eval_many([dict(a=r['a'], chi=r['chi']) for _, r in RL], Z4)
    rows = []
    for (lab, r0), res in zip(RL, raw):
        q = summarize_s2(r0['id'], r0['a'], r0['chi'], res, ref, R4, SEED_SCEN, src=f'd={d_}')
        assert q['dmD_B'] == np.linalg.slogdet(res['J'])[1]-ldB, "Delta m_D^(B) does not use the rebuilt J_per"
        assert q['Ein'] == r0['Ein'], f"window input power of {r0['id']} differs from its d=20 record"
        q.update(floor=lab, cnt=res['cnt'].tolist(), inv_mE=1/q['mE_C']); rows.append(q)
    SC[d_] = dict(d=d_, sigma=s, B_u=Bd, B_u_se=Bse, N=Nd, ldC=float(ldC), ldB=float(ldB), offset_CB=float(ldC-ldB), Jper_rank=rank, Jper_cond=cond,
                  cnt_u=cu.tolist(), ldb_u=ref['ldb'].tolist(), rows=rows, time=time.time()-t1)
    print(f"Table Sc-{d_}. d = {d_}, R={R4}, seed {SEED_SCEN}; D = paired send-count difference vs (C)_{d_} (decision by qual, 8.3.6(g)); "
          f"dmD_C se = paired batch se 10x{R4//10}; dmD_B vs J_per(u_unif, {Nd}) at sigma_{d_}; P_in = window input power (= d=20 record: checked)")
    print(f"{'floor':5s} {'id':34s} {'B':>8s} {'D +- se':>15s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>7s} {'(se)':>7s} {'1/mE':>7s} {'dmD_B':>8s} {'P_in':>8s}")
    for q in rows:
        print(f"{q['floor']:5s} {q['id']:34s} {q['B']:8.3f} {q['D']:+7.3f} +- {q['D_se']:5.3f} {q['dec']:5s} {q['dmD_C']:+7.3f} ({q['dmD_se']:5.3f}) {q['mE_C']:7.4f} "
              f"({q['mE_se']:.4f}) {q['inv_mE']:7.2f} {q['dmD_B']:+8.3f} {q['Ein']:8.5f}")
    print(f"Nine generalized eigenvalues vs (C)_{d_}:")
    for q in rows: print(f"  {q['id']:34s} {lam_table_row(q)}")
    print(f"d = {d_} done in {SC[d_]['time']:.0f} s")
del Z4

print(f"\nTable Sc-cmp. Direction of change relative to the d = 20 control on the same draws (paired: send differences per draw, log det per batch)")
print(f"{'floor':5s} {'id':34s} {'d':>3s} | {'D_d':>8s} {'D_20':>8s} {'D_d - D_20 (se)':>18s} | {'dmD_d':>7s} {'dmD_20':>7s} {'diff (se)':>17s} {'ratio':>6s} | "
      f"{'1/mE_d':>7s} {'1/mE_20':>7s} {'ratio':>6s} | {'dec_d':5s} {'dec_20':6s}")
cmp_rows = []
for i, (lab, r0) in enumerate(RL):
    if lab == 'ref': continue
    q20 = SC[20]['rows'][i]
    for d_ in (5, 50):
        q = SC[d_]['rows'][i]
        dd = (np.array(q['cnt'])-np.array(SC[d_]['cnt_u']))-(np.array(q20['cnt'])-np.array(SC[20]['cnt_u'])); dd_se = float(dd.std(ddof=1)/np.sqrt(R4))
        lb = (np.array(q['ldb'])-np.array(SC[d_]['ldb_u']))-(np.array(q20['ldb'])-np.array(SC[20]['ldb_u'])); lb_se = float(lb.std(ddof=1)/np.sqrt(lb.size))
        row = dict(floor=lab, id=r0['id'], d=d_, D=q['D'], D_se=q['D_se'], D20=q20['D'], dD=q['D']-q20['D'], dD_se=dd_se, dmD=q['dmD_C'], dmD20=q20['dmD_C'],
                   ddmD=q['dmD_C']-q20['dmD_C'], ddmD_se=lb_se, inv_mE=q['inv_mE'], inv_mE20=q20['inv_mE'], dec=q['dec'], dec20=q20['dec'], mE=q['mE_C'], dmD_B=q['dmD_B'],
                   dmD_se=q['dmD_se'], mag=int(np.floor(np.log10(q['inv_mE']))) if q['inv_mE'] > 0 else None)
        cmp_rows.append(row)
        print(f"{lab:5s} {r0['id']:34s} {d_:3d} | {q['D']:+8.3f} {q20['D']:+8.3f} {row['dD']:+9.3f} ({dd_se:.3f}) | {q['dmD_C']:+7.3f} {q20['dmD_C']:+7.3f} {row['ddmD']:+8.3f} ({lb_se:.3f}) "
              f"{q['dmD_C']/q20['dmD_C']:6.2f} | {q['inv_mE']:7.2f} {q20['inv_mE']:7.2f} {q['inv_mE']/q20['inv_mE']:6.2f} | {q['dec']:5s} {q20['dec']:6s}")
print("\nPer representative at d = 5 and d = 50 (budget feasibility by qual; sign of the D gain with its paired batch se; m_E below/above 1 and the order of magnitude of 1/m_E):")
for lab, r0 in RL:
    if lab == 'ref': continue
    parts = []
    for d_ in (5, 50):
        q = [c for c in cmp_rows if c['id'] == r0['id'] and c['d'] == d_][0]
        parts.append(f"d={d_}: budget {q['dec']} (D={q['D']:+.3f}+-{q['D_se']:.3f}), dmD_C {q['dmD']:+.3f} (se {q['dmD_se']:.3f}, "
                     f"{'positive' if q['dmD'] > 0 else 'negative'}{', > 2 se' if abs(q['dmD']) > 2*q['dmD_se'] else ', within 2 se'}), m_E {q['mE']:.4f} "
                     f"({'below' if q['mE'] < 1 else 'above'} 1), 1/m_E {q['inv_mE']:.2f} (order 10^{q['mag']})")
    print(f"  {lab:4s} {r0['id']}: " + ' | '.join(parts))
print("\nUniform reference across d (same draws): " + '; '.join(f"d={d_}: B_hat={SC[d_]['B_u']:.3f}+-{SC[d_]['B_u_se']:.3f}, N={SC[d_]['N']}, (C)-(B) offset {SC[d_]['offset_CB']:+.4f}, "
                                                              f"J_per rank {SC[d_]['Jper_rank']}, cond {SC[d_]['Jper_cond']:.3e}" for d_ in (5, 20, 50)))
print("CRN: the d = 5 and d = 50 scenarios (and the d = 20 control) use the same standard-normal draws scaled by sigma_d; they are not independent batches.")
for d_ in SC:
    for q in SC[d_]['rows']: q.pop('cnt')
    SC[d_].pop('cnt_u')
json.dump(dict(R=R4, seed=SEED_SCEN, Delta=D_STAR, representatives=reps, scenarios={str(k): v for k, v in SC.items()}, comparison=cmp_rows),
          open(f'{OUT}/scenarios.json', 'w'), indent=1)
log_runtime('stage4_scenarios', T_START, f'{len(RL)} inputs at d in (5, 50) + d=20 control, R={R4}')
