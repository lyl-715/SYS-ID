"""Round 18 driver (prompt reports_CN/prompt_round18_accumulated_information_EN.md). Input U, true model; rounds 4-17 read-only; no DP,
no grid evaluator.

Usage (from this directory):  R13_INPUT=U python3 run_round18.py <stage>
  A        §2: R = 1e4 trajectories per trigger (10 batches of 1000, seeds 20261115 + b); J_cond, J_sent per trajectory; checks against
           Round 14 (RULE), the SoD archive (SOD) and round 10 (PRED); bitwise checks of the simulation against rule14._mc_cont and
           mismatch15.run_job                                                               -> results/partA.json, results/J_A_*.npy
  B        §3: 500 experiments per trigger (seeds 20261116 + i, the same noise for SOD, RULE, PRED); LS and CML  -> results/partB.json
  C        §4: edge table E(rho); C0, C1, C2 tuned to the budget (R = 2000, seed 20261117); ten repetitions (seeds 20261117 + b)
                                                                                                            -> results/partC.json
  summary  §5-§7: Q1-Q5, predictions, checks -> results/round18.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
from multiprocessing import Pool
import acc18 as a
s, K, B, P = a.s, a.K, a.B, a.P
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
WORKERS = 4
A_R, A_BATCHES, A_SEED = 10000, 10, 20261115
B_NREP = 500
C_R, C_SEED, C_REPS, C_TOL = 2000, 20261117, 10, 0.3
R14_LOGDET, R14_LOGDET_SE = 207.4639397749701, 0.0005463919506154085      # round14/results/part1.json d.mc_up (e = 1.58)
R10_F0 = 208.95657763962944                                                   # round10/results/round10.json budgets.B316.history[0].f
REF_OUTSIDE = dict(RULE=dict(I=207.503, mean=207.501, sd=0.063, I_minus_mean=0.002, ld_sent=194.179),
                   SOD=dict(I=197.194, mean=197.160, sd=0.263, I_minus_mean=0.034, ld_sent=193.448))   # prompt §2 (not a gate)


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

class Stop(RuntimeError): pass

def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return jl(v.tolist())
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def jdump(obj, path):
    tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(jl(obj), f, indent=1)
    os.replace(tmp, path)

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__)

FROZEN = ('round2a', 'round2b', 'round2c', 'round2d', 'round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10', 'round11',
          'round13', 'round14', 'round15')
FROZEN_FILES = ('config.json', 'density_recursion_v1.py', 'evaluator_fast.py')

def cleanup_frozen(out):
    removed = []
    for rd in FROZEN:
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_): os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + [os.path.join(EV, r) for r in FROZEN] + [os.path.join(EV, f) for f in FROZEN_FILES],
                        capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on rounds 4-15 (all present earlier rounds) and the frozen files: "
        f"{'clean' if not st else st}")


# --- Part A ----------------------------------------------------------------------------------------------------------------------
def _chk_rule14(args):
    sys.path.insert(0, os.path.join(EV, 'round14'))
    import rule14
    return rule14._mc_cont(args)

def _chk_m15(args):
    return a.m15.run_job(args)

def stage_A(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 18 stage A start; versions {versions()}")
    out(f"SETUP e = {a.E_RULE!r} (round14 part1.json d.e_up); VF(T, m) = ({a.FB_T}, {a.FB_M}); PRED c = Phi^-1(1 - q/2) = {a.C_PRED!r}, "
        f"q = {a.Q_UNI!r}; Delta* = {a.D_STAR!r} = {a.D_STAR/s:.12f} sigma; R = {A_R} ({A_BATCHES} batches, seeds {A_SEED} + b)")
    jobs = [(trig, A_R//A_BATCHES, A_SEED+b) for trig in a.TRIGGERS for b in range(A_BATCHES)]
    chk = [(a.E_RULE, A_R//A_BATCHES, A_SEED)]
    chk_fb = [('chk', np.zeros(9), dict(kind='fb', e0=a.E_RULE, T=a.FB_T, m=a.FB_M), A_R//A_BATCHES, A_SEED)]
    with Pool(WORKERS) as pool:
        r14 = pool.apply_async(_chk_rule14, chk); r15 = pool.apply_async(_chk_m15, chk_fb)
        outs = pool.map(a.batch_A, jobs)
        r14 = r14.get(); r15 = r15.get()
    by = {trig: [o for o in outs if o['trig'] == trig] for trig in a.TRIGGERS}
    # bitwise checks of the simulation (batch 0) against Round 14's and Round 15's routines
    o0 = by['RULE'][0]
    ok14 = bool(np.array_equal(o0['Wk'], r14[0]) and np.array_equal(o0['N'], r14[1]) and np.array_equal(o0['S'], r14[2]))
    out(f"CHECK sim RULE batch 0 vs rule14._mc_cont(e = 1.58, 1000, {A_SEED}): Wk, N, S bitwise equal {ok14} -> {'PASS' if ok14 else 'FAIL'}")
    o0 = by['RULE_FB'][0]
    ok15 = bool(np.array_equal(o0['Wk'], r15['Wk']) and np.array_equal(o0['N'], r15['N']) and np.array_equal(o0['S'], r15['S'])
                and o0['esum'] == r15['esum'] and o0['nlo'] == r15['nlo'] and o0['nhi'] == r15['nhi'])
    out(f"CHECK sim RULE_FB batch 0 vs mismatch15.run_job(delta = 0, VF(100, 3), 1000, {A_SEED}): Wk, N, S, e sums bitwise equal {ok15} "
        f"-> {'PASS' if ok15 else 'FAIL'}")
    if not (ok14 and ok15): raise Stop("simulation differs from Round 14 / Round 15 routines")
    res = dict(versions=versions(), setup=dict(e=a.E_RULE, FB=[a.FB_T, a.FB_M], C_PRED=a.C_PRED, q=a.Q_UNI, R=A_R, batches=A_BATCHES, seed=A_SEED),
               checks=dict(sim_rule14_bitwise=ok14, sim_m15_bitwise=ok15), triggers={})
    for trig in a.TRIGGERS:
        rec, J, W, Jsm = a.summarise_A(by[trig])
        res['triggers'][trig] = rec
        np.save(os.path.join(RES, f'J_A_{trig}.npy'), J); np.save(os.path.join(RES, f'W_A_{trig}.npy'), W)
        np.save(os.path.join(RES, f'Jsent_A_{trig}.npy'), Jsm)
        corr = 'undefined (sd 0)' if rec['corr_ld_N'] is None else f"{rec['corr_ld_N']:+.4f}"
        out(f"A {trig}: I {rec['I']!r} (batch SE {rec['I_batch_se']:.5f}) dI {rec['dI']:+.4f}; per-run log det mean {rec['ld_run_mean']:.4f} "
            f"sd {rec['ld_run_sd']:.4f} P5 {rec['ld_run_p05']:.4f} P95 {rec['ld_run_p95']:.4f} [min {rec['ld_run_min']:.4f}, max {rec['ld_run_max']:.4f}]; "
            f"I - mean {rec['I_minus_mean']:.5f}; corr(log det, N) {corr}")
        out(f"A {trig}: log det mean(J_sent) {rec['ld_sent_of_mean']:.4f} (batch SE {rec['ld_sent_of_mean_batch_se']:.4f}); "
            f"value of the non-transmissions I - log det mean(J_sent) {rec['value_nontx']:.4f} nat (batch SE {rec['value_nontx_batch_se']:.4f}); "
            f"S {rec['S']:.3f} (SE {rec['S_se']:.3f}) vs E[N] {rec['EN']:.3f} (SE {rec['EN_se']:.3f}), sd N {rec['sd_N']:.3f}, frac N > B "
            f"{rec['frac_N_over_B']:.4f}; per-run log det J_sent mean {rec['ld_sent_run_mean']:.4f} sd {rec['ld_sent_run_sd']:.4f}; "
            f"cond J {rec['cond_J']:.3f}; |I_direct - I| {abs(rec['I_direct_minus_I']):.1e}"
            + (f"; e_k mean {rec['e_mean']:.4f} sd {rec['e_sd']:.4f}, at clip lo/hi {rec['frac_clip_lo']:.1e}/{rec['frac_clip_hi']:.1e}"
               if 'e_mean' in rec else ''))
    T = res['triggers']
    # checks (prompt §2)
    z_sod = (T['SOD']['I']-a.I_SOD)/T['SOD']['I_batch_se']
    se_rule = float(np.hypot(T['RULE']['I_batch_se'], R14_LOGDET_SE)); z_rule = (T['RULE']['I']-R14_LOGDET)/se_rule
    d_pred = T['PRED']['I']-R10_F0
    c_sod, c_rule, c_pred = abs(z_sod) <= 3, abs(z_rule) <= 3, abs(d_pred) <= 5e-5
    out(f"CHECK A SOD: I {T['SOD']['I']!r} vs archive {a.I_SOD!r}: diff {T['SOD']['I']-a.I_SOD:+.5f}, batch SE {T['SOD']['I_batch_se']:.5f}, "
        f"z {z_sod:+.2f} -> {'PASS' if c_sod else 'FAIL'}")
    out(f"CHECK A RULE: I {T['RULE']['I']!r} vs Round 14 continuous rule e = 1.580 {R14_LOGDET!r}: diff {T['RULE']['I']-R14_LOGDET:+.5f}, "
        f"combined SE {se_rule:.5f}, z {z_rule:+.2f} -> {'PASS' if c_rule else 'FAIL'}")
    out(f"CHECK A PRED: I {T['PRED']['I']!r} vs round 10 history[0].f {R10_F0!r}: diff {d_pred:+.2e} (J_cond is the same for every run: "
        f"per-run sd {T['PRED']['ld_run_sd']:.1e}; agrees at 4 dp: {c_pred}) -> {'PASS' if c_pred else 'FAIL'}")
    res['checks'].update(A_SOD=dict(z=z_sod, passed=bool(c_sod)), A_RULE=dict(z=z_rule, se=se_rule, passed=bool(c_rule)),
                         A_PRED=dict(diff=d_pred, passed=bool(c_pred)))
    for trig, ref in REF_OUTSIDE.items():
        t = T[trig]
        out(f"REF outside (not a gate) {trig}: I {t['I']:.3f} / {ref['I']}; mean {t['ld_run_mean']:.3f} / {ref['mean']}; sd {t['ld_run_sd']:.3f} / "
            f"{ref['sd']}; I - mean {t['I_minus_mean']:.3f} / {ref['I_minus_mean']}; log det mean(J_sent) {t['ld_sent_of_mean']:.3f} / {ref['ld_sent']}")
    res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'partA.json'))
    if not (c_sod and c_rule and c_pred): raise Stop("Part A check failed")
    out(f"=== ROUND 18 stage A done in {res['wall_s']:.1f} s")


# --- Part B ----------------------------------------------------------------------------------------------------------------------
def stage_B(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 18 stage B start; versions {versions()}")
    A = json.load(open(os.path.join(RES, 'partA.json')))
    trigs = ('SOD', 'RULE', 'PRED')
    NZ = np.vstack([a.noise_B(i) for i in range(B_NREP)])
    rng = np.random.default_rng(20261116)                      # random-number order check: eval11._mc_batch order with R = 1
    seq = np.array([rng.standard_normal(1)[0] for _ in range(K)])
    okn = bool(np.array_equal(seq, NZ[0]))
    out(f"CHECK B noise: experiment 0 drawn one value at a time (eval11._mc_batch order, R = 1) equals standard_normal(K): {okn} "
        f"-> {'PASS' if okn else 'FAIL'}")
    if not okn: raise Stop("noise order")
    res = dict(versions=versions(), N_rep=B_NREP, seeds=[20261116, 20261116+B_NREP-1], restart_seeds=[80261116, 80261116+B_NREP-1],
               triggers={}, checks=dict(noise_order=okn))
    jobs = []
    for trig in trigs:
        ev, z, lo, hi = a.simulate_B(trig, NZ)
        # the receiver's windows equal the sensor's at every non-transmitted step
        okw = True
        for i in range(B_NREP):
            zt = np.where(ev[i], z[i], np.nan); kk = np.flatnonzero(~ev[i])
            rl, rh = a.receiver_windows(trig, ev[i], zt)
            okw &= bool(np.array_equal(rl, lo[i, kk]) and np.array_equal(rh, hi[i, kk]))
        Ns = ev.sum(1)
        out(f"B {trig}: simulated {B_NREP} experiments: N mean {Ns.mean():.2f} sd {Ns.std(ddof=1):.2f} min {Ns.min()} max {Ns.max()}; "
            f"receiver windows equal the sensor's at every non-transmitted step: {okw} -> {'PASS' if okw else 'FAIL'}")
        if not okw: raise Stop("receiver windows differ")
        res['triggers'][trig] = dict(N_mean=float(Ns.mean()), N_sd=float(Ns.std(ddof=1)), N_min=int(Ns.min()), N_max=int(Ns.max()),
                                     windows_reconstructed=okw)
        jobs += [(trig, i, ev[i], z[i]) for i in range(B_NREP)]
    with Pool(WORKERS) as pool:
        outs = pool.map(a.estimate_B, jobs, chunksize=10)
    np.savez_compressed(os.path.join(RES, 'partB_estimates.npz'),
                        **{f'{t}_{k}': np.array([o[k] for o in outs if o['trig'] == t]) for t in trigs
                           for k in ('theta_LS', 'theta_CML', 'N', 'converged', 'acc1', 'acc2', 'nit1', 'decr1')})
    for trig in trigs:
        J = np.load(os.path.join(RES, f'J_A_{trig}.npy')); I = A['triggers'][trig]['I']
        oo = sorted([o for o in outs if o['trig'] == trig], key=lambda o: o['i'])
        thL = np.array([o['theta_LS'] for o in oo]); thC = np.array([o['theta_CML'] for o in oo])
        conv = np.array([o['converged'] for o in oo]); acc1 = np.array([o['acc1'] for o in oo])
        rank_ok = all(o['LS_rank'] == 9 and o['LS_usable'] for o in oo)
        stL = a.acc_stats(thL, J); stC = a.acc_stats(thC, J)
        R = res['triggers'][trig]
        R.update(I=I, LS=stL, CML=stC, LS_rank9_all=rank_ok, CML_not_accepted=int((~conv).sum()), CML_first_call_not_accepted=int((~acc1).sum()),
                 CML_restart_used=int(sum(o['used'] == 'restart' for o in oo)), CML_fallback=int(sum(o['fallback'] is not None for o in oo)),
                 CML_nit1_median=float(np.median([o['nit1'] for o in oo])), CML_decr1_max=float(max(o['decr1'] for o in oo)),
                 CML_messages=sorted(set(o['msg1'] for o in oo)), wall_s_sum=float(sum(o['wall_s'] for o in oo)),
                 not_accepted_runs=[o['i'] for o in oo if not o['converged']])
        if (~conv).any():
            R['CML_accepted_only'] = a.acc_stats(thC[conv], J)
        for lab, st in (('LS', stL), ('CML', stC)):
            out(f"B {trig} {lab}: bias/CRsd [{', '.join(f'{v:+.3f}' for v in st['bias_cr'])}] (MC SE {np.mean(st['bias_cr_se']):.3f} avg); "
                f"largest |bias| {st['bias_max_abs']:.3f} (param {st['bias_argmax']+1}); Mahalanobis {st['bias_mahalanobis']:.3f}")
            out(f"B {trig} {lab}: RMSE/CRsd [{', '.join(f'{v:.3f}' for v in st['rmse_cr'])}] min {st['rmse_cr_min']:.3f} max {st['rmse_cr_max']:.3f}; "
                f"D-hat {st['D_hat']:.3f} (I {I:.3f}, D-hat - I {st['D_hat']-I:+.3f}); -log det cov {st['neg_logdet_cov']:.3f} "
                f"(minus I {st['neg_logdet_cov']-I:+.3f})")
        out(f"B {trig} CML: not accepted {R['CML_not_accepted']}/{B_NREP} (first call not accepted {R['CML_first_call_not_accepted']}, "
            f"restart used {R['CML_restart_used']}, fallback {R['CML_fallback']}); first-call nit median {R['CML_nit1_median']:.0f}, "
            f"max Newton decrement {R['CML_decr1_max']:.1e}; LS rank 9 in every run: {rank_ok}")
    d = res['triggers']['SOD']['CML']['D_hat']-res['triggers']['SOD']['I']
    c_sod = abs(d) <= 0.3
    out(f"CHECK B SOD CML: D-hat {res['triggers']['SOD']['CML']['D_hat']:.3f} vs I {res['triggers']['SOD']['I']:.3f}: |D-hat - I| {abs(d):.3f} "
        f"<= 0.3 -> {'PASS' if c_sod else 'FAIL'} (round 8b stage 1, R = 300: -log det M1 196.881 vs log det J1 197.201)")
    res['checks']['B_SOD_CML'] = dict(D_hat_minus_I=d, passed=bool(c_sod))
    res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'partB.json'))
    out(f"=== ROUND 18 stage B done in {res['wall_s']:.1f} s")


# --- Part B, supplementary (user instruction after stage B; NOT the pre-registered estimator; supp18.py) ----------------------------
def stage_Bn(out):
    import supp18
    t0 = time.perf_counter()
    out(f"=== ROUND 18 stage Bn start (SUPPLEMENTARY: Newton steps for the CML runs not accepted in stage B; the pre-registered "
        f"estimator and its numbers are unchanged); versions {versions()}")
    Bj = json.load(open(os.path.join(RES, 'partB.json'))); A = json.load(open(os.path.join(RES, 'partA.json')))
    est = np.load(os.path.join(RES, 'partB_estimates.npz'))
    trigs = ('SOD', 'RULE', 'PRED')
    NZ = np.vstack([a.noise_B(i) for i in range(B_NREP)])
    jobs = []
    for trig in trigs:
        runs = Bj['triggers'][trig]['not_accepted_runs']
        conv = est[f'{trig}_converged']
        assert sorted(np.flatnonzero(~conv).tolist()) == sorted(runs), trig
        ev, z, _, _ = a.simulate_B(trig, NZ)
        jobs += [(trig, i, ev[i], z[i], est[f'{trig}_theta_LS'][i]) for i in runs]
    out(f"SETUP non-accepted runs: " + '; '.join(f"{t} {Bj['triggers'][t]['not_accepted_runs']}" for t in trigs)
        + f"; start = end point of the first L-BFGS-B call; full Newton steps; stop at eta <= {supp18.TOL:g} or {supp18.MAX_STEPS} steps")
    with Pool(WORKERS) as pool:
        outs = pool.map(supp18.supp_one, jobs)
    rep = all(o['reproduced'] for o in outs)
    out(f"CHECK Bn: estimator_r8b.estimate_stage1 re-run on the {len(outs)} non-accepted runs reproduces stage B (not accepted, "
        f"fallback theta_LS bitwise equal to the archive): {rep} -> {'PASS' if rep else 'FAIL'}")
    if not rep: raise Stop("stage B decision not reproduced")
    res = dict(versions=versions(), max_steps=supp18.MAX_STEPS, tol=supp18.TOL, start='first L-BFGS-B call end point', runs=[],
               triggers={}, checks=dict(reproduced=rep))
    for trig in trigs:
        J = np.load(os.path.join(RES, f'J_A_{trig}.npy')); I = A['triggers'][trig]['I']
        crsd = np.sqrt(np.diag(np.linalg.inv(J)))
        thC = est[f'{trig}_theta_CML'].copy(); thL = est[f'{trig}_theta_LS']
        oo = sorted([o for o in outs if o['trig'] == trig], key=lambda o: o['i'])
        nconv = 0
        for o in oo:
            nt = o['newton']; h = nt['hist']
            d_a1 = float(np.abs((o['theta_a1']-nt['theta'])/crsd).max()); d_a2 = float(np.abs((o['theta_a2']-nt['theta'])/crsd).max())
            d_ls = float(np.abs((o['theta_LS']-nt['theta'])/crsd).max())
            err_new = float(np.abs((nt['theta']-a.THETA_STAR)/crsd).max()); err_ls = float(np.abs((o['theta_LS']-a.THETA_STAR)/crsd).max())
            out(f"Bn {trig} run {o['i']} (N {o['N']}): call 1 success {o['a1']['success']} '{o['a1']['message']}' nit {o['a1']['nit']} "
                f"eta {o['a1']['decrement']:.2e} |g|inf {o['a1']['grad_inf']:.1e}; call 2 success {o['a2']['success']} '{o['a2']['message']}' "
                f"nit {o['a2']['nit']} eta {o['a2']['decrement']:.2e}; Newton: converged {nt['converged']} after {nt['steps']} steps "
                f"({nt['reason']}), eta path [{', '.join(f'{x['eta']:.1e}' for x in h)}], NLL change {h[-1]['f']-h[0]['f']:+.2e}; "
                f"max |theta - theta_Newton|/CRsd: call-1 end {d_a1:.2e}, call-2 end {d_a2:.2e}, theta_LS {d_ls:.2f}; "
                f"max |error|/CRsd: Newton {err_new:.2f}, theta_LS (pre-registered fallback) {err_ls:.2f}")
            res['runs'].append(dict(trig=trig, i=o['i'], N=o['N'], a1=o['a1'], a2=o['a2'], newton_converged=nt['converged'],
                                    newton_steps=nt['steps'], newton_reason=nt['reason'], newton_hist=nt['hist'],
                                    theta_newton=nt['theta'], theta_a1=o['theta_a1'], theta_a2=o['theta_a2'], theta_LS=o['theta_LS'],
                                    dist_a1_cr=d_a1, dist_a2_cr=d_a2, dist_LS_cr=d_ls, err_newton_cr=err_new, err_LS_cr=err_ls))
            if nt['converged']:
                thC[o['i']] = nt['theta']; nconv += 1
        st = a.acc_stats(thC, J); pre = Bj['triggers'][trig]['CML']
        res['triggers'][trig] = dict(I=I, n_not_accepted=len(oo), n_newton_converged=nconv, CML_newton_substituted=st)
        out(f"Bn {trig}: {nconv} of {len(oo)} non-accepted runs converge under the Newton steps; CML with those estimates substituted "
            f"(500 runs): bias/CRsd [{', '.join(f'{v:+.3f}' for v in st['bias_cr'])}]; largest |bias| {st['bias_max_abs']:.3f} "
            f"(pre-registered {pre['bias_max_abs']:.3f}); Mahalanobis {st['bias_mahalanobis']:.3f}")
        out(f"Bn {trig}: RMSE/CRsd [{', '.join(f'{v:.3f}' for v in st['rmse_cr'])}] min {st['rmse_cr_min']:.3f} max {st['rmse_cr_max']:.3f}; "
            f"D-hat {st['D_hat']:.3f} (I {I:.3f}, D-hat - I {st['D_hat']-I:+.3f}; pre-registered {pre['D_hat']-I:+.3f}); -log det cov "
            f"{st['neg_logdet_cov']:.3f} (minus I {st['neg_logdet_cov']-I:+.3f})")
    res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'partB_newton.json'))
    out(f"=== ROUND 18 stage Bn done in {res['wall_s']:.1f} s")


# --- Part C ----------------------------------------------------------------------------------------------------------------------
def tune(args):
    var, Jref, Sref = args
    hist = []; t0 = time.perf_counter()
    def f(par):
        o = a.run_C((var, par, C_R, C_SEED, Jref, Sref)); hist.append(o); return o['EN']
    if var == 'C0':
        lo_p, hi_p = 1.50, 1.70                                  # e: E[N] decreases with e
        mid = lambda x, y: 0.5*(x+y)
    else:
        k = int(np.flatnonzero(a.EV_HULL == 1.58)[0]); rho_mid = 0.5*(a.BP[k-1]+a.BP[k])
        lam0 = (P/Sref)/rho_mid; lo_p, hi_p = lam0/1.25, lam0*1.25   # lambda: E[N] decreases with lambda
        mid = lambda x, y: float(np.sqrt(x*y))
    f_lo, f_hi = f(lo_p), f(hi_p)
    while f_lo <= B:
        hi_p, f_hi = lo_p, f_lo; lo_p = lo_p-0.05 if var == 'C0' else lo_p/1.25; f_lo = f(lo_p)
    while f_hi >= B:
        lo_p, f_lo = hi_p, f_hi; hi_p = hi_p+0.05 if var == 'C0' else hi_p*1.25; f_hi = f(hi_p)
    best = None
    for it in range(60):
        m = mid(lo_p, hi_p); fm = f(m)
        if abs(fm-B) <= C_TOL: best = hist[-1]; break
        if fm > B: lo_p = m
        else: hi_p = m
    return dict(var=var, best=best, hist=[{k: v for k, v in h.items()} for h in hist], wall_s=time.perf_counter()-t0)

def stage_C(out):
    t0 = time.perf_counter()
    out(f"=== ROUND 18 stage C start; versions {versions()}")
    A = json.load(open(os.path.join(RES, 'partA.json')))
    Jref = np.load(os.path.join(RES, 'J_A_RULE.npy')); Wref = np.load(os.path.join(RES, 'W_A_RULE.npy')); Sref = float(Wref.sum())
    out(f"SETUP J_ref = mean J_cond of RULE (Part A, log det {a.logdet(Jref):.4f}); S_ref = sum W_ref = {Sref!r}; p = {P}; eps = {a.EPS_C2}; "
        f"R = {C_R}, tuning seed {C_SEED}, |E[N] - B| <= {C_TOL}")
    # edge table
    mono = bool(np.all(np.diff(a.EV_HULL) <= 0))
    rho = np.concatenate([np.logspace(-3, 1, 20001), a.BP*(1+1e-9), a.BP*(1-1e-9)])
    Eb = a.E_brute(rho); Et = a.E_of(rho); agree = bool(np.array_equal(Eb, Et))
    rs = np.sort(rho); mono_b = bool(np.all(np.diff(a.E_brute(rs)) <= 0))
    k158 = int(np.flatnonzero(a.EV_HULL == 1.58)[0])
    out(f"EDGE TABLE: {a.HULL.size} of {a.E_GRID.size} grid edges are on the upper envelope; breakpoints rho in [{a.BP[0]:.6f}, {a.BP[-1]:.6f}] "
        f"(E = 4.0 below, 0.5 above); E(rho) = 1.58 for rho in [{a.BP[k158-1]:.6f}, {a.BP[k158]:.6f}); E non-increasing along the table: {mono}; "
        f"table vs direct argmax on {rho.size} rho values (log grid 1e-3..10 and +-1e-9 around every breakpoint): equal {agree}; direct argmax "
        f"non-increasing: {mono_b} -> {'PASS' if (mono and agree and mono_b) else 'FAIL'}")
    if not (mono and agree and mono_b): raise Stop("edge table")
    gam = a.gamma_terminal(Jref)
    out(f"C1 weights: gamma_k = s^-2 h_k^T J_ref^-1 h_k: min {gam[1:].min():.5f} median {np.median(gam[1:]):.5f} max {gam[1:].max():.5f}; "
        f"sum_k W_ref,k gamma_k = {float(Wref@gam):.6f} (= p); p / S_ref = {P/Sref:.6f}")
    res = dict(versions=versions(), Sref=Sref, logdet_Jref=float(a.logdet(Jref)), eps=a.EPS_C2,
               edge_table=dict(n_hull=int(a.HULL.size), bp=a.BP, e=a.EV_HULL, monotone=mono, brute_equal=agree),
               gamma=dict(min=float(gam[1:].min()), median=float(np.median(gam[1:])), max=float(gam[1:].max()), W_gamma=float(Wref@gam)))
    with Pool(3) as pool:
        tuned = pool.map(tune, [(v, Jref, Sref) for v in ('C0', 'C1', 'C2')])
    res['tuning'] = {}
    for tr in tuned:
        v = tr['var']
        for h in tr['hist']:
            out(f"TUNE {v} {'e' if v == 'C0' else 'lambda'} = {h['par']!r}: E[N] {h['EN']:.3f} (I {h['I']:.4f}; {h['wall_s']:.1f} s)")
        if tr['best'] is None: raise Stop(f"tuning of {v} did not reach |E[N] - B| <= {C_TOL}")
        b = tr['best']
        out(f"TUNED {v}: {'e' if v == 'C0' else 'lambda'} = {b['par']!r} after {len(tr['hist'])} runs ({tr['wall_s']:.0f} s): E[N] {b['EN']:.3f} "
            f"(SE {b['EN_se']:.3f}, sd N {b['sd_N']:.2f}); I {b['I']!r} dI {b['dI']:+.4f}; per-run log det mean {b['ld_run_mean']:.4f} sd "
            f"{b['ld_run_sd']:.4f}; S {b['S']:.2f}; e_k P5/P50/P95 {b['e_p05']:.3f}/{b['e_p50']:.3f}/{b['e_p95']:.3f} (mean {b['e_mean']:.4f})"
            + (f"; rel weight mean {b['rel_mean']:.4f}, rho in [{b['rho_min']:.4f}, {b['rho_max']:.4f}], |J_running - J_direct|/|J| "
               f"{b['Jc_running_vs_direct']:.1e}" if v == 'C2' else '')
            + (f"; e_k at 0.5 / 4.0: {b['e_frac_lo']:.1e} / {b['e_frac_hi']:.1e}" if v != 'C0' else ''))
        res['tuning'][v] = dict(best=b, n_runs=len(tr['hist']), wall_s=tr['wall_s'],
                                hist=[dict(par=h['par'], EN=h['EN'], I=h['I']) for h in tr['hist']])
    pars = {v: res['tuning'][v]['best']['par'] for v in ('C0', 'C1', 'C2')}
    jobs = [(v, pars[v], C_R, C_SEED+b, Jref, Sref) for b in range(C_REPS) for v in ('C0', 'C1', 'C2')]
    with Pool(WORKERS) as pool:
        reps = pool.map(a.run_C, jobs)
    I = {v: np.array([o['I'] for o in reps if o['var'] == v]) for v in ('C0', 'C1', 'C2')}
    EN = {v: np.array([o['EN'] for o in reps if o['var'] == v]) for v in ('C0', 'C1', 'C2')}
    det_ok = all(o['I'] == res['tuning'][o['var']]['best']['I'] and o['EN'] == res['tuning'][o['var']]['best']['EN'] for o in reps if o['seed'] == C_SEED)
    out(f"CHECK C determinism: repetition b = 0 (seed {C_SEED}) equals the final tuning run for C0, C1, C2: {det_ok} -> {'PASS' if det_ok else 'FAIL'}")
    for v in ('C0', 'C1', 'C2'):
        out(f"REPS {v}: I per seed [{', '.join(f'{x:.4f}' for x in I[v])}]; mean {I[v].mean():.4f} sd {I[v].std(ddof=1):.4f}; "
            f"E[N] mean {EN[v].mean():.3f} (min {EN[v].min():.3f}, max {EN[v].max():.3f})")
    diffs = {}
    for lab, (u, w) in (('C1-C0', ('C1', 'C0')), ('C2-C1', ('C2', 'C1')), ('C2-C0', ('C2', 'C0'))):
        d = I[u]-I[w]; diffs[lab] = dict(per_seed=d, mean=float(d.mean()), se=float(d.std(ddof=1)/np.sqrt(C_REPS)),
                                          tuned_seed=float(res['tuning'][u]['best']['I']-res['tuning'][w]['best']['I']))
        dEN = EN[u]-EN[w]
        out(f"DIFF I({u}) - I({w}): mean over {C_REPS} seeds {d.mean():+.4f} nat, paired SE {d.std(ddof=1)/np.sqrt(C_REPS):.4f} "
            f"(per seed [{', '.join(f'{x:+.4f}' for x in d)}]; at the tuning seed {diffs[lab]['tuned_seed']:+.4f}); E[N] difference mean "
            f"{dEN.mean():+.3f}")
    res['reps'] = dict(I={v: I[v] for v in I}, EN={v: EN[v] for v in EN}, sd_I={v: float(I[v].std(ddof=1)) for v in I},
                       records=[{k: o[k] for k in ('var', 'seed', 'EN', 'I', 'ld_run_mean', 'ld_run_sd', 'S', 'e_p05', 'e_p50', 'e_p95')} for o in reps])
    res['diffs'] = diffs; res['checks'] = dict(determinism=det_ok)
    # certificate check
    viol = []
    for v in ('C0', 'C1', 'C2'):
        sd1 = float(I[v].std(ddof=1)); It = res['tuning'][v]['best']['I']; Im = float(I[v].mean())
        z1 = (It-a.UB_CERT)/sd1; zm = (Im-a.UB_CERT)/(sd1/np.sqrt(C_REPS))
        ok = z1 <= 3 and zm <= 3
        out(f"CHECK C certificate {v}: I(tuned) {It:.4f}, mean over seeds {Im:.4f} vs certificate {a.UB_CERT:.4f}: (I - UB)/SE = {z1:+.1f} "
            f"(single run, SE {sd1:.4f}), {zm:+.1f} (mean, SE {sd1/np.sqrt(C_REPS):.4f}) -> {'PASS' if ok else 'STOP'}")
        res['checks'][f'cert_{v}'] = dict(z_single=z1, z_mean=zm, passed=bool(ok))
        if not ok: viol.append(v)
    res['wall_s'] = time.perf_counter()-t0
    jdump(res, os.path.join(RES, 'partC.json'))
    if viol: raise Stop(f"certificate exceeded by more than 3 SE: {viol}")
    if not det_ok: raise Stop("determinism")
    out(f"=== ROUND 18 stage C done in {res['wall_s']:.1f} s")


def main():
    stage = sys.argv[1]
    out = Tee(os.path.join(LOGS, f'{stage}.log'))
    try:
        {'A': stage_A, 'B': stage_B, 'Bn': stage_Bn, 'C': stage_C}[stage](out) if stage != 'summary' else __import__('summary18').stage_summary(out)
    except Stop as e:
        out(f"STOP {e}"); cleanup_frozen(out); sys.exit(2)
    cleanup_frozen(out)

if __name__ == '__main__':
    main()
