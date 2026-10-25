"""Round 8, Part B: the full two-stage sequential pipeline, R = 300 replications (spec reports_CN/plan_round8_CN.md v3.1,
frozen; §1.1-§1.7). 本场景、固定设计规则下的序贯仿真结果.

Per replication r (one worker, one numba thread):
  noise    e^(1) = default_rng(20261028 + r), e^(2) = default_rng(20271028 + r), full length K, pre-generated (truth_r8.noise);
           both schemes use the same e^(2) array.
  stage 1  u_unif, constant Delta*, mc83.sod on z = x0 + sigma e^(1)  ->  Y_1 (N_1 transmissions, forced first one included).
  theta_1  estimator_r8.estimate_stage1 (LS initial value / theta_bk, L-BFGS-B, restart seed 30261028 + r, fallback).
  control  stage 2 = constant Delta* SoD on e^(2) (shared by both analyses: it does not depend on theta_hat_1).
  For each analysis A in {P: the frozen acceptance rule (primary), S: the supplementary rule (DECISION E-1)}:
    design   design_r8.design(theta_hat_1) under the runtime sentinel (isolation_r8.guard; a non-empty access log stops the run);
             a pre-listed design failure -> the sequential stage 2 is the constant-Delta* SoD, flag "design failed" (DECISION B-2).
    stage 2  truth_r8.sim_table(e^(2), T) (lookup from the actual reference, transmit iff outside the window) -> Y_2, N_2.
    post-hoc E[N_2 | T, theta_0], J_table(T; theta_0): posthoc_r8.eval_table (sigma/120 general path, table-anchored grid).
    joint    l_1 + l_2 with T fixed, from theta_hat_1, restart seed 40261028 + r, fallback theta_hat_1 -- for the sequential
             scheme and for the control.
    H_r      -grad^2 l_2(theta_0; Y_2, T) (the stage-2 observed information at theta_0), also for the control's record.
  Workers are spawned (fresh interpreters; no numba threading layer inherited through fork).
  The packed table goes to results/tables_packed/T_r<r>_<A>.npz.bz2 (written by the worker, round trip checked); the main
  process appends one JSON line per finished replication to results/archive_round8.jsonl (fsync). A re-run resumes.
Correctness failures (sentinel access, evaluator assertions, anything not in the spec §1.7 list) propagate and stop the run.

Usage (from this directory):
  python3 seq_pipeline.py run [--R 300] [--workers 4]          # the run (resumes)
  python3 seq_pipeline.py probe --R 4                           # smoke test on probe seeds 93000000 + r / 94000000 + r,
                                                                # r = 10001.., outputs under results/runs/probe/ (not committed)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np

RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
ANALYSES = (('P', 'frozen'), ('S', 'supp'))
PROBE_SEEDS = (93000000, 94000000); PROBE_R0 = 10000
G = {}                                                    # worker globals


def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def init_worker(b1, J1, outdir, probe):
    import numba; numba.set_num_threads(1)
    import declared, estimator_r8, tables_r8, isolation_r8, design_r8, truth_r8, posthoc_r8
    G.update(dk=declared, est=estimator_r8, tb=tables_r8, iso=isolation_r8, dz=design_r8, tr=truth_r8, ph=posthoc_r8,
             b1=b1, J1=np.array(J1), outdir=outdir, probe=probe)
    design_r8.warmup(); posthoc_r8.warmup()

def noise(r):
    tr = G['tr']
    if not G['probe']: return tr.noise(r), (tr.SEED1+r, tr.SEED2+r)
    s1, s2 = PROBE_SEEDS[0]+r, PROBE_SEEDS[1]+r
    return (np.random.default_rng(s1).standard_normal(tr.K), np.random.default_rng(s2).standard_normal(tr.K)), (s1, s2)

def one_replication(r):
    dk, est, tb, iso, dz, tr, ph = (G[k] for k in ('dk', 'est', 'tb', 'iso', 'dz', 'tr', 'ph'))
    TH0 = tr.THETA0; Hm = dk.H(); s = dk.s; nan = np.nan
    T = {}; t0 = time.perf_counter()
    (e1, e2), (s1, s2) = noise(r)
    # --- stage 1 and its estimate
    z1, ev1, _ = tr.sim_sod(e1); st1 = est.StageData(ev1, np.where(ev1, z1, nan), ('sod',))
    t = time.perf_counter(); o1 = est.estimate_stage1(st1, r); T['stage1_est'] = time.perf_counter()-t
    T['stage1_total'] = time.perf_counter()-t0
    # --- control's stage 2 (shared)
    t = time.perf_counter()
    z2c, ev2c, _ = tr.sim_sod(e2); st2c = est.StageData(ev2c, np.where(ev2c, z2c, nan), ('sod',))
    Hr_c = est.nll_grad_hess_theta(st2c, TH0)[2]
    T['control_stage2'] = time.perf_counter()-t
    out = dict(r=r, seeds=dict(e1=s1, e2=s2, restart_stage1=est.SEED_R1+r, restart_joint=est.SEED_RJ+r), N1=int(ev1.sum()),
               stage1=dict(theta_LS=o1['theta_LS'], LS_usable=o1['LS_usable'], LS_rank=o1['LS_rank'], init_kind=o1['init_kind'],
                           theta_init=o1['theta_init'], attempts=o1['attempts']),
               control=dict(N2=int(ev2c.sum()), H_r=Hr_c), analyses={}, probe=G['probe'])
    cache = {}
    for A, rule in ANALYSES:
        q = o1['rules'][rule]; th1 = np.array(q['theta'], float)
        a = dict(rule=rule, theta1=th1, stage1_converged=q['converged'], stage1_used=q['used'], stage1_fallback=q['fallback'],
                 err1=th1-TH0, eps1=float(np.max(np.abs(Hm@(th1-TH0)))/s))
        key = th1.tobytes()
        if key in cache:
            c = cache[key]; a['shared_with'] = c['A']
        else:
            c = dict(A=A); tt = {}
            iso.ACCESS_LOG.clear(); t = time.perf_counter()
            with iso.guard():
                d = dz.design(th1, f'r{r}{A}')
            tt['design'] = time.perf_counter()-t
            if iso.ACCESS_LOG:
                raise iso.SentinelAccess(f"replication {r} analysis {A}: sentinel accessed {iso.ACCESS_LOG}")
            c['design'] = {k: d.get(k) for k in ('status', 'reason', 'lam', 'gbar', 'gamma1', 'EN_hat', 'EN_sod_hat', 'S_hat', 'logdet_hat',
                                                 'F_hat', 'dual_part', 'z0', 'h', 'G', 'WMAX', 'xhat_min', 'xhat_max',
                                                 'mass_dev_max_hat', 'min_m_hat', 'mass_dev_max_sod_hat', 'min_m_sod_hat',
                                                 't_cache_s', 't_dp_s', 't_fwd_s', 't_total_s')}
            if d['status'] == 'ok':
                c['design']['J_sod_hat'] = d['J_sod_hat']; c['design']['gamma_mean_check'] = float(np.mean(d['gamma'][1:]))
                tab = tb.SeqTable.from_design(d, f'r{r}{A}')
                path = os.path.join(G['outdir'], 'tables_packed', f'T_r{r:03d}_{A}.npz.bz2')
                nbytes, sha = tb.pack(d, path, extra=dict(r=r, analysis=A))
                c['table'] = dict(file=os.path.relpath(path, HERE), bytes=nbytes, codes_sha256=sha)
                t = time.perf_counter(); z2, ev2, _ = tr.sim_table(e2, tab); tt['sim2'] = time.perf_counter()-t
                st2 = est.StageData(ev2, np.where(ev2, z2, nan), ('table', tab))
                t = time.perf_counter(); rec, J, rr = ph.eval_table(tab); tt['posthoc'] = time.perf_counter()-t
                c['posthoc'] = dict(EN2=float(rec['EN']), EN2_P=float(rec['EN_P']), S2=float(rec['S']), logdet2=float(rec['logdet']), J_table=J,
                                    mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']), assert_fail=int(rec['assert_fail']),
                                    boundary_steps=int(rec['boundary_steps']), grid_n0=int(rec['grid_n0']), boundary_gap=float(rec['boundary_gap']),
                                    inv_S=bool(rec['inv_S']), inv_J=bool(rec['inv_J']), inv_w=bool(rec['inv_w']))
                c['frac_asym'] = float(np.mean(st2.Mm != st2.Mp)) if len(st2.Mm) else 0.0
            else:                                                    # spec §1.7 fallback: stage 2 = constant Delta* SoD (DECISION B-2)
                z2, ev2, st2 = z2c, ev2c, st2c; tt['sim2'] = 0.0; tt['posthoc'] = 0.0
                c['table'] = None
                c['posthoc'] = dict(EN2=G['b1'], J_table=G['J1'], fallback_sod=True)
            del d
            c['N2'] = int(ev2.sum()); c['st2'] = st2
            c['H_r'] = est.nll_grad_hess_theta(st2, TH0)[2] if c['table'] is not None else Hr_c
            c['times'] = tt; cache[key] = c
        a.update(design=c['design'], table=c['table'], posthoc=c['posthoc'], N2=c['N2'], H_r=c['H_r'], times=dict(c['times']),
                 design_failed=c['design']['status'] != 'ok', frac_asym=c.get('frac_asym'))
        # joint estimates (T fixed): sequential and control
        t = time.perf_counter()
        j1, j2 = est.estimate_joint([st1, c['st2']], th1, r)
        thf, conv, used, fb = est.decide(j1, j2, rule, th1, 'theta_hat_1')
        k1, k2 = est.estimate_joint([st1, st2c], th1, r)
        thc, convc, usedc, fbc = est.decide(k1, k2, rule, th1, 'theta_hat_1')
        a['times']['joint'] = time.perf_counter()-t
        a['seq'] = dict(theta_final=thf, converged=conv, used=used, fallback=fb, attempts=[est._attempt_summary(j1), est._attempt_summary(j2)])
        a['sod'] = dict(theta_final=thc, converged=convc, used=usedc, fallback=fbc, attempts=[est._attempt_summary(k1), est._attempt_summary(k2)])
        a['success_all'] = bool(a['stage1_converged'] and not a['design_failed'] and conv and convc)
        out['analyses'][A] = a
    tp = out['analyses']['P']['times']
    out['times'] = dict(T, primary_work_s=T['stage1_total']+T['control_stage2']+tp.get('design', 0)+tp.get('sim2', 0)+tp.get('posthoc', 0)+tp['joint'],
                        total_s=time.perf_counter()-t0, pid=os.getpid())
    return jl(out)


def summary_line(o):
    p, q = o['analyses']['P'], o['analyses']['S']
    f = lambda a: (f"eps1 {a['eps1']:.3f} conv1 {int(a['stage1_converged'])} design {a['design']['status'][:6]} "
                   f"Ehat {a['design'].get('EN_hat') or float('nan'):.2f} EN2 {a['posthoc']['EN2']:.2f} N2 {a['N2']} "
                   f"joint {int(a['seq']['converged'])}/{int(a['sod']['converged'])}")
    return (f"REP r={o['r']:03d} N1 {o['N1']} N2c {o['control']['N2']} | P: {f(p)} | S: {f(q)}{' (shared)' if 'shared_with' in q else ''} | "
            f"t {o['times']['total_s']:.0f} s (primary {o['times']['primary_work_s']:.0f} s)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'probe'])
    ap.add_argument('--R', type=int, default=300); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    probe = a.mode == 'probe'
    outdir = os.path.join(RES, 'runs', 'probe') if probe else RES
    os.makedirs(os.path.join(outdir, 'tables_packed'), exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    arch = os.path.join(outdir, 'archive_round8.jsonl')
    logp = os.path.join(outdir if probe else LOGS, 'partB_pipeline.log')
    logf = open(logp, 'a')
    def P(m):
        print(m, flush=True); logf.write(m+'\n'); logf.flush()
    reps = [PROBE_R0+i for i in range(1, a.R+1)] if probe else list(range(1, a.R+1))
    done = set()
    if os.path.exists(arch):
        for line in open(arch):
            line = line.strip()
            if line: done.add(json.loads(line)['r'])
    todo = [r for r in reps if r not in done]
    import posthoc_r8 as ph
    ph.warmup()
    t0 = time.perf_counter(); recS, J1, _ = ph.eval_sod_ref()
    b1 = float(recS['EN'])
    P(f"=== seq_pipeline.py {a.mode} started {time.strftime('%Y-%m-%d %H:%M:%S')}; R = {a.R}, workers {a.workers}; done {len(done)}, to do {len(todo)}")
    P(f"B0 b_1 = E_theta0[N_SoD] (sigma/120, reuse path) = {b1!r}; log det J_1 = {float(np.linalg.slogdet(J1)[1])!r}; "
      f"mass dev {recS['mass_dev_max']!r}, min m {recS['min_m']!r}; {time.perf_counter()-t0:.1f} s")
    json.dump(dict(b1=b1, J1=np.asarray(J1).tolist(), logdet_J1=float(np.linalg.slogdet(J1)[1]), EN_P=float(recS['EN_P']), S=float(recS['S']),
                   mass_dev_max=float(recS['mass_dev_max']), min_m=float(recS['min_m']), path=recS['path']),
              open(os.path.join(outdir, 'sod_reference.json'), 'w'), indent=1)
    tw = time.perf_counter()
    with open(arch, 'a') as fa, mp.get_context('spawn').Pool(a.workers, initializer=init_worker, initargs=(b1, np.asarray(J1), outdir, probe)) as pool:
        n = 0
        for o in pool.imap_unordered(one_replication, todo, chunksize=1):
            fa.write(json.dumps(o)+'\n'); fa.flush(); os.fsync(fa.fileno()); n += 1
            P(summary_line(o)+f" [{n}/{len(todo)}, wall {time.perf_counter()-tw:.0f} s]")
    P(f"=== seq_pipeline.py {a.mode} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; {n} replications in {time.perf_counter()-tw:.1f} s wall")


if __name__ == '__main__':
    main()
