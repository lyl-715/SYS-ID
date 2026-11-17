"""Round 19 driver (prompt reports_CN/prompt_round19_two_stage_rule_EN.md). Input U.

Usage (from this directory):  R13_INPUT=U python3 run_round19.py <stage>
  impl     implementation checks (not prompt checks): the simulator against acc18.simulate_B (SOD, RULE, RULE_FB) and
           acc18.batch_A (eval11 random-number order), lik_data against estimator_r8.StageData, receiver windows
                                                                                                  -> results/impl.json
  checks   prompt §5 (a) constant SoD throughout, (b) rule throughout, true model, no feedback, e = 1.577; 10 000 trajectories
           (10 batches of 1000, noise seeds 193000000 + i). Stops (exit 1) if either fails.     -> results/checks.json
  main     prompt §3-§4: primary K1 = 2500 (1000 replications), secondary K1 = 1250 / 3750 (300 each), control (K1 = 2500,
           true model in stage 2, 300), baseline (constant SoD throughout, 1000), noise seeds 190000000 + i
                                                                       -> results/rows.csv, results/main.npz, results/main.json
  summary  prompt §6-§7: information, bootstrap, budget, model, estimator, failures, predictions -> results/round19.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, csv
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
from multiprocessing import Pool
import two_stage19 as t
a = t.a
s, K, B, P = t.s, t.K, t.B, t.P
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
WORKERS = 4
IMPL_SEED0 = 195000000
CHK_R, CHK_BATCHES = 10000, 10
CHK_A_LOGDET, CHK_A_EN = 197.201, 316.89
CHK_B_LOGDET = 197.201+10.30
CONFIGS = {                                       # name: (K1, mode, replications)
    'primary': (2500, 'est', 1000),
    'K1_1250': (1250, 'est', 300),
    'K1_3750': (3750, 'est', 300),
    'control': (2500, 'true', 300),
    'baseline': (2500, 'sod', 1000),
}


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

class Stop(RuntimeError): pass

def jl(v):
    if isinstance(v, dict): return {str(k): jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return jl(v.tolist())
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def jdump(obj, path):
    tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(jl(obj), f, indent=1)
    os.replace(tmp, path)

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__)


# --- impl ------------------------------------------------------------------------------------------------------------------
def eval11_noise(R, seed):
    """(K, R) noise in the random-number order of eval11._mc_batch / acc18.batch_A (one standard_normal(R) per step)."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal(R) for _ in range(K)])

def stage_impl(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 19 stage impl start; versions {versions()}")
    res = dict(versions=versions(), checks={})
    ok_all = True
    # I1-I3: Sim against acc18.simulate_B (record, windows) on 40 replications
    NZ = t.noise_matrix(range(40), IMPL_SEED0)
    for trig in ('SOD', 'RULE', 'RULE_FB'):
        ev_r, z_r, lo_r, hi_r = a.simulate_B(trig, NZ.T)
        sm = t.Sim(NZ)
        if trig == 'SOD': sm.sod(K)
        else: sm.rule(K, t.XK, a.E_RULE, trig == 'RULE_FB')
        ok = bool(np.array_equal(sm.ev.T, ev_r) and np.array_equal(sm.z.T, z_r) and np.array_equal(sm.lo.T[:, 1:], lo_r[:, 1:])
                  and np.array_equal(sm.hi.T[:, 1:], hi_r[:, 1:]))
        out(f"IMPL I1 Sim vs acc18.simulate_B('{trig}') (40 replications, e = {a.E_RULE}): record and windows bitwise equal {ok} "
            f"-> {'PASS' if ok else 'FAIL'}")
        res['checks'][f'simulate_B_{trig}'] = ok; ok_all &= ok
    # I4: Sim against acc18.batch_A (information weights, counts) in the eval11 random-number order
    for trig in ('SOD', 'RULE', 'RULE_FB'):
        seed = IMPL_SEED0+1000; R = 200
        o = a.batch_A((trig, R, seed))
        sm = t.Sim(eval11_noise(R, seed))
        if trig == 'SOD': sm.sod(K)
        else: sm.rule(K, t.XK, a.E_RULE, trig == 'RULE_FB')
        ld = t.logdet_rows(sm.J_cond())
        ok = bool(np.array_equal(sm.Wk, o['Wk']) and np.array_equal(sm.N, o['N']) and np.array_equal(sm.S, o['S'])
                  and np.array_equal(ld, o['ld_cond']))
        out(f"IMPL I2 Sim vs acc18.batch_A('{trig}', {R}, {seed}): Wk, N, S, per-run log det J_cond bitwise equal {ok} "
            f"-> {'PASS' if ok else 'FAIL'}")
        res['checks'][f'batch_A_{trig}'] = ok; ok_all &= ok
    # I5: lik_data against estimator_r8.StageData for a whole SoD record; receiver windows (SoD throughout)
    sm = t.Sim(NZ); sm.sod(K)
    ok5 = True
    for j in range(5):
        ev = sm.ev[:, j]; zt = np.where(ev, sm.z[:, j], np.nan); kk = np.flatnonzero(~ev)
        lo, hi = t.receiver_windows(ev, zt, K, None, None, False, False)
        d = t.lik_data(ev, zt, lo[kk], hi[kk]); d0 = a.e8.StageData(ev, zt, ('sod',))
        ok5 &= bool(np.array_equal(d.lo, d0.lo) and np.array_equal(d.up, d0.up) and np.array_equal(d.He, d0.He)
                    and np.array_equal(d.ze, d0.ze) and np.array_equal(d.Hn, d0.Hn) and d.N == d0.N
                    and np.array_equal(lo[kk], sm.lo[kk, j]) and np.array_equal(hi[kk], sm.hi[kk, j]))
    out(f"IMPL I3 lik_data / receiver_windows vs estimator_r8.StageData(ev, z, ('sod',)) and the sensor (5 SoD records): equal {ok5} "
        f"-> {'PASS' if ok5 else 'FAIL'}")
    res['checks']['lik_data_sod'] = ok5; ok_all &= ok5
    # I6: receiver windows of a two-stage record (perturbed model, feedback) equal the sensor's, at every step p >= 1
    rng = np.random.default_rng(IMPL_SEED0+2000)
    th = [t.THETA_STAR*(1+0.01*rng.standard_normal(P)) for _ in range(40)]
    sm = t.Sim(NZ); sm.sod(2500)
    xb = np.column_stack([t.predict(x) for x in th]); sm.rule(K, xb, t.E0, True)
    ok6 = True
    for j in range(40):
        ev = sm.ev[:, j]; zt = np.where(ev, sm.z[:, j], np.nan)
        lo, hi = t.receiver_windows(ev, zt, 2500, t.predict(th[j]), t.E0, True, True)
        ok6 &= bool(np.array_equal(lo[1:], sm.lo[1:, j]) and np.array_equal(hi[1:], sm.hi[1:, j]))
    out(f"IMPL I4 receiver windows of 40 two-stage records (K1 = 2500, theta* perturbed by 1 %, feedback) equal the sensor's at "
        f"every step: {ok6} -> {'PASS' if ok6 else 'FAIL'}")
    res['checks']['receiver_two_stage'] = ok6; ok_all &= ok6
    # I7: K1 = 1 with N_1 = 1: the generalised plan is the existing one (covered by I1/I2 RULE_FB); state it
    res['passed'] = bool(ok_all); res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'impl.json'))
    if not ok_all: raise Stop("implementation check failed")
    out(f"=== ROUND 19 stage impl done in {res['wall_s']:.1f} s")


# --- checks (a), (b) --------------------------------------------------------------------------------------------------------
def _chk_batch(args):
    kind, b, R = args
    NZ = t.noise_matrix(range(b*R, (b+1)*R), t.CHK_SEED0)
    sm = t.Sim(NZ)
    if kind == 'a': sm.sod(K)
    else: sm.rule(K, t.XK, t.E0, False)
    return dict(kind=kind, b=b, R=R, Wk=sm.Wk.copy(), N=sm.N.copy(), ld_run=t.logdet_rows(sm.J_cond()))

def stage_checks(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 19 stage checks start; versions {versions()}")
    Rb = CHK_R//CHK_BATCHES
    jobs = [(kind, b, Rb) for kind in ('a', 'b') for b in range(CHK_BATCHES)]
    with Pool(WORKERS) as pool: outs = pool.map(_chk_batch, jobs)
    res = dict(versions=versions(), R=CHK_R, batches=CHK_BATCHES, noise_seeds=[t.CHK_SEED0, t.CHK_SEED0+CHK_R-1], checks={})
    ok_all = True
    for kind in ('a', 'b'):
        oo = sorted([o for o in outs if o['kind'] == kind], key=lambda o: o['b'])
        W = sum(o['Wk'] for o in oo)/CHK_R; I = float(t.logdet(t.J_of_weights(W)))
        Ib = np.array([t.logdet(t.J_of_weights(o['Wk']/o['R'])) for o in oo]); se = float(Ib.std(ddof=1)/np.sqrt(len(oo)))
        N = np.concatenate([o['N'] for o in oo]); EN = float(N.mean()); EN_se = float(N.std(ddof=1)/np.sqrt(N.size))
        ldr = np.concatenate([o['ld_run'] for o in oo])
        if kind == 'a':
            zI = (I-CHK_A_LOGDET)/se; zN = (EN-CHK_A_EN)/EN_se
            ok = bool(abs(zI) <= 3 and abs(zN) <= 3)
            out(f"CHECK (a) constant SoD throughout: log det J {I!r} (batch SE {se:.5f}) vs 197.201: diff {I-CHK_A_LOGDET:+.5f}, z {zI:+.2f}; "
                f"E[N] {EN:.4f} (SE {EN_se:.4f}) vs 316.89: diff {EN-CHK_A_EN:+.4f}, z {zN:+.2f} -> {'PASS' if ok else 'FAIL'}")
            out(f"CHECK (a) context: vs the archive 197.20097668224076 diff {I-t.I_SOD:+.5f}; E[N] vs B {EN-B:+.4f}; sd N {N.std(ddof=1):.3f}; "
                f"frac N > B {(N > B).mean():.4f}; per-run log det mean {ldr.mean():.4f} sd {ldr.std(ddof=1):.4f}")
            res['checks']['a'] = dict(logdet=I, logdet_batch_se=se, z_logdet=zI, EN=EN, EN_se=EN_se, z_EN=zN, sd_N=float(N.std(ddof=1)),
                                      frac_over_B=float((N > B).mean()), ld_run_mean=float(ldr.mean()), ld_run_sd=float(ldr.std(ddof=1)),
                                      batch_logdets=Ib.tolist(), passed=ok)
        else:
            zI = (I-CHK_B_LOGDET)/se
            ok = bool(abs(zI) <= 3)
            out(f"CHECK (b) rule throughout, true model, no feedback, e = {t.E0}: log det J {I!r} (batch SE {se:.5f}) vs 197.201 + 10.30 = "
                f"{CHK_B_LOGDET:.3f}: diff {I-CHK_B_LOGDET:+.5f}, z {zI:+.2f} -> {'PASS' if ok else 'FAIL'}")
            out(f"CHECK (b) context: gain over the archive SoD {I-t.I_SOD:+.4f} nat; E[N] {EN:.4f} (SE {EN_se:.4f}), sd N {N.std(ddof=1):.3f}, "
                f"frac N > B {(N > B).mean():.4f}; per-run log det mean {ldr.mean():.4f} sd {ldr.std(ddof=1):.4f}")
            res['checks']['b'] = dict(logdet=I, logdet_batch_se=se, z_logdet=zI, gain_vs_archive=I-t.I_SOD, EN=EN, EN_se=EN_se,
                                      sd_N=float(N.std(ddof=1)), frac_over_B=float((N > B).mean()), ld_run_mean=float(ldr.mean()),
                                      ld_run_sd=float(ldr.std(ddof=1)), batch_logdets=Ib.tolist(), passed=ok)
        ok_all &= ok
    res['passed'] = bool(ok_all); res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'checks.json'))
    if not ok_all: raise Stop("prompt §5 check failed: stop and report, do not continue")
    out(f"=== ROUND 19 stage checks done in {res['wall_s']:.1f} s")


# --- main -------------------------------------------------------------------------------------------------------------------
def subset(sim, cols):
    o = t.Sim.__new__(t.Sim)
    for k, v in sim.__dict__.items():
        if isinstance(v, np.ndarray):
            if k == 'Wk': o.Wk = None
            elif v.ndim == 2: setattr(o, k, v[:, cols].copy())
            else: setattr(o, k, v[cols].copy())
        else: setattr(o, k, v)
    o.R = len(cols)
    o.Wk = np.zeros(K); o.Wk[:o.p] = o.C[:o.p].sum(1)
    return o

def stage_main(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 19 stage main start; versions {versions()}")
    chk = json.load(open(os.path.join(RES, 'checks.json')))
    if not chk.get('passed'): raise Stop("checks did not pass")
    R1 = CONFIGS['primary'][2]; R3 = CONFIGS['K1_1250'][2]
    NZ = t.noise_matrix(range(R1))
    out(f"SETUP noise seeds {t.NOISE_SEED0} + i, i = 0..{R1-1}; e0 = {t.E0}, T = {t.FB_T}, m = {t.FB_M}; clip [{t.E_LO}, {t.E_HI}]; "
        f"restart seeds stage 1 {t.S1_SEED0} + i, final {t.FJ_SEED0} + i")
    # stage 1 (constant SoD) for the three K1
    st1 = {}
    for K1 in (2500, 1250, 3750):
        R = R1 if K1 == 2500 else R3
        sm = t.Sim(NZ[:, :R]); sm.sod(K1); st1[K1] = sm
        out(f"STAGE1 K1 = {K1}: {R} replications simulated; N_1 mean {sm.N.mean():.3f} sd {sm.N.std(ddof=1):.3f} min {sm.N.min():.0f} "
            f"max {sm.N.max():.0f}")
    jobs = [(i, K1, st1[K1].ev[:K1, i], st1[K1].z[:K1, i], st1[K1].lo[:K1, i], st1[K1].hi[:K1, i])
            for K1 in (2500, 1250, 3750) for i in range(st1[K1].R)]
    tt = time.perf_counter()
    with Pool(WORKERS) as pool: o1 = pool.map(t.stage1_job, jobs, chunksize=8)
    S1 = {K1: sorted([o for o in o1 if o['K1'] == K1], key=lambda o: o['i']) for K1 in (2500, 1250, 3750)}
    okw1 = all(o['windows_ok'] for o in o1)
    out(f"STAGE1 CML: {len(o1)} estimates in {time.perf_counter()-tt:.1f} s; receiver windows equal the sensor's in every run: {okw1} "
        f"-> {'PASS' if okw1 else 'FAIL'}")
    if not okw1: raise Stop("stage-1 receiver windows differ")
    for K1 in (2500, 1250, 3750):
        na = [o['i'] for o in S1[K1] if not o['converged']]
        assert all(o['N1'] == int(st1[K1].N[o['i']]) for o in S1[K1])
        out(f"STAGE1 K1 = {K1}: CML not accepted in {len(na)} of {len(S1[K1])} replications {na} (fallback theta_LS used as theta_tilde); "
            f"first call not accepted {sum(not o['att1']['accept'] for o in S1[K1])}; LS rank 9 in every run "
            f"{all(o['LS_rank'] == 9 and o['LS_usable'] for o in S1[K1])}")
    TT = {K1: np.array([o['theta'] for o in S1[K1]]) for K1 in S1}
    # stage 2
    sims = {}
    for name, (K1, mode, R) in CONFIGS.items():
        base = st1[K1] if R == st1[K1].R else subset(st1[K1], np.arange(R))
        sm = base.copy() if base is st1[K1] else base
        if mode == 'sod':
            sm.sod(K)
        else:
            xbar = t.XK if mode == 'true' else np.column_stack([t.predict(TT[K1][j]) for j in range(R)])
            sm.rule(K, xbar, t.E0, True)
        sims[name] = sm
        N2 = sm.N-sm.ev[:K1].sum(0)
        out(f"STAGE2 {name}: K1 {K1}, mode {mode}, R {R}: N mean {sm.N.mean():.3f} sd {sm.N.std(ddof=1):.3f} max {sm.N.max():.0f}; "
            f"N_2 mean {N2.mean():.3f}; frac N > B {(sm.N > B).mean():.4f}; log det J {t.logdet(t.J_of_weights(sm.Wk/R)):.4f}")
    # final estimates
    jobs = []
    for name, (K1, mode, R) in CONFIGS.items():
        sm = sims[name]
        jobs += [(name, i, K1, mode, sm.ev[:, i], sm.z[:, i], TT[K1][i], sm.lo[:, i], sm.hi[:, i]) for i in range(R)]
    tt = time.perf_counter()
    with Pool(WORKERS) as pool: of = pool.map(t.final_job, jobs, chunksize=8)
    F = {name: sorted([o for o in of if o['tag'] == name], key=lambda o: o['i']) for name in CONFIGS}
    okwf = all(o['windows_ok'] for o in of)
    out(f"FINAL CML: {len(of)} estimates in {time.perf_counter()-tt:.1f} s; receiver windows equal the sensor's in every run: {okwf} "
        f"-> {'PASS' if okwf else 'FAIL'}")
    if not okwf: raise Stop("final receiver windows differ")
    # rows, arrays
    rows = []; arrays = {}
    for name, (K1, mode, R) in CONFIGS.items():
        sm = sims[name]; s1 = S1[K1]; fo = F[name]
        N1 = sm.ev[:K1].sum(0); N2 = sm.ev[K1:].sum(0); N = sm.ev.sum(0)
        assert np.array_equal(N, sm.N.astype(int))
        ldr = t.logdet_rows(sm.J_cond())
        arrays[f'{name}_Jc'] = sm.J_cond(); arrays[f'{name}_W'] = sm.Wk/R
        arrays[f'{name}_theta_tilde'] = TT[K1][:R]; arrays[f'{name}_theta_final'] = np.array([o['theta'] for o in fo])
        for i in range(R):
            pe = t.pred_errors(TT[K1][i], K1)
            plan2 = B-t.FB_M-N1[i]
            if mode == 'sod':
                e2 = dict(e_mean2='', e_min2='', e_max2='', u_hit_lo='', u_hit_hi='', abnormal='')
            else:
                ee = sm.e[K1:, i]; uu = sm.u[K1:, i]
                e2 = dict(e_mean2=float(ee.mean()), e_min2=float(ee.min()), e_max2=float(ee.max()), u_hit_lo=int((uu <= t.E_LO).any()),
                          u_hit_hi=int((uu >= t.E_HI).any()), abnormal=int(N2[i] < 10 or N2[i] > 2*plan2))
            row = dict(config=name, i=i, K1=K1, mode=mode, noise_seed=t.NOISE_SEED0+i, N_1=int(N1[i]), N_2=int(N2[i]), N=int(N[i]),
                       over_B=int(N[i] > B), plan2=float(plan2), **e2, pe_rms=pe[0], pe_max=pe[1], pe_rms_stage2=pe[2],
                       s1_converged=int(s1[i]['converged']), s1_used=s1[i]['used'], s1_fallback=s1[i]['fallback'] or '',
                       s1_nit1=s1[i]['att1']['nit'], s1_decr1=s1[i]['att1']['decr'], s1_msg1=s1[i]['att1']['msg'],
                       fin_converged=int(fo[i]['converged']), fin_used=fo[i]['used'], fin_fallback=fo[i]['fallback'] or '',
                       fin_nit1=fo[i]['att1']['nit'], fin_decr1=fo[i]['att1']['decr'], fin_msg1=fo[i]['att1']['msg'],
                       ld_Jcond=float(ldr[i]))
            for j in range(P): row[f'theta_tilde_{j+1}'] = float(TT[K1][i, j])
            for j in range(P): row[f'theta_final_{j+1}'] = float(fo[i]['theta'][j])
            rows.append(row)
    with open(os.path.join(RES, 'rows.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    np.savez_compressed(os.path.join(RES, 'main.npz'), **arrays)
    res = dict(versions=versions(), configs={k: dict(K1=v[0], mode=v[1], R=v[2]) for k, v in CONFIGS.items()},
               seeds=dict(noise=[t.NOISE_SEED0, t.NOISE_SEED0+R1-1], stage1_restart=t.S1_SEED0, final_restart=t.FJ_SEED0),
               windows_ok=dict(stage1=okw1, final=okwf), wall_s=time.perf_counter()-t0,
               cml_wall_s=dict(stage1=float(sum(o['wall_s'] for o in o1)), final=float(sum(o['wall_s'] for o in of))))
    jdump(res, os.path.join(RES, 'main.json'))
    out(f"=== ROUND 19 stage main done in {res['wall_s']:.1f} s")


if __name__ == '__main__':
    stg = sys.argv[1]
    out = Tee(os.path.join(LOGS, f'{stg}.log'))
    try:
        if stg == 'impl': stage_impl(out)
        elif stg == 'checks': stage_checks(out)
        elif stg == 'main': stage_main(out)
        elif stg == 'summary':
            import summary19; summary19.run(out)
        else: raise SystemExit(f"unknown stage {stg}")
    except Stop as ex:
        out(f"=== ROUND 19 stage {stg} STOPPED: {ex}")
        sys.exit(1)
