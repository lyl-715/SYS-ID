"""Round 8b, Part B: the two-stage sequential pipeline of round 8 v3.1 (§1.1-§1.7) with the round-8b convergence acceptance and
fresh seeds (spec reports_CN/plan_round8b_CN.md v2, frozen). Primary analysis only. 本场景、固定设计规则下的序贯仿真结果.

Every step is a round-8 module, imported unchanged (DECISION A-2): truth_r8 (simulator: mc83.sod for SoD stages, sim_table for
the table), design_r8 under isolation_r8.guard (a non-empty sentinel access log stops the run), tables_r8 (lookup, packed
copy), posthoc_r8 (E[N_2 | T, theta_0], J_table at sigma/120; b_1, J_1), estimator_r8 through estimator_r8b (the only change:
acceptance = result.success, finite, Cholesky of H_vartheta, Newton decrement <= 1e-5; restart seeds explicit). The per-
replication steps are those of round 8's seq_pipeline.one_replication for one analysis (that function loops over round 8's
two acceptance rules and reads its seeds from module constants, so it is recomposed here, step for step):

  noise    e^(1) = default_rng(S1 + r).standard_normal(K), e^(2) = default_rng(S2 + r).standard_normal(K), pre-generated;
           both schemes use the same e^(2) array.
  stage 1  constant Delta* SoD (truth_r8.sim_sod) -> Y_1; theta_hat_1 = estimator_r8b.estimate_stage1(Y_1, RS1 + r).
  control  stage 2 = constant Delta* SoD on e^(2); its H_r at theta_0.
  design   design_r8.design(theta_hat_1) under the sentinel; a pre-listed design failure -> stage 2 = constant Delta* SoD
           (round 8 DECISION B-2).
  stage 2  truth_r8.sim_table(e^(2), T) -> Y_2; post-hoc posthoc_r8.eval_table(T); H_r = -grad^2 l_2(theta_0).
  joint    estimator_r8b.estimate_joint for the sequential scheme and the control, from theta_hat_1, restart RJ + r,
           fallback theta_hat_1.
Seed families: 'official' (S1, S2, RS1, RJ) = (50261028, 50271028, 60261028, 70261028), r = 1..300; 'round8' = (20261028,
20271028, 30261028, 40261028), used only by the engineering reproduction of round 8's supplementary analysis S.

The archive keeps round 8's line layout with the single analysis under the key 'P' (= the primary analysis of this round,
rule 'newton_decrement'), so that round 8's stats_r8 functions read it unchanged.

Usage (from this directory):
  python3 seq_pipeline_r8b.py repro --reps 1,2,98,134   # E2: round-8 seeds; compares with round8/results/archive_round8.jsonl
                                                        # analysis S bit for bit -> logs/engineering_repro_S.log,
                                                        # results/engineering_repro_S.json (run outputs in results/runs/repro/)
  python3 seq_pipeline_r8b.py run [--R 300] [--workers 4]   # the official run (resumes); refuses to start unless A3 and E2
                                                            # passed -> results/archive_round8b.jsonl, results/tables_packed/
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R8 = os.path.join(EV, 'round8')
for _p in (R8, HERE):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)
import numpy as np

RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
SEEDS = dict(official=(50261028, 50271028, 60261028, 70261028), round8=(20261028, 20271028, 30261028, 40261028))
AN = 'P'
G = {}                                                    # worker globals


def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def init_worker(b1, J1, outdir, family):
    import numba; numba.set_num_threads(1)
    import declared, estimator_r8b, tables_r8, isolation_r8, design_r8, truth_r8, posthoc_r8
    G.update(dk=declared, eb=estimator_r8b, tb=tables_r8, iso=isolation_r8, dz=design_r8, tr=truth_r8, ph=posthoc_r8,
             b1=b1, J1=np.array(J1), outdir=outdir, family=family, seeds=SEEDS[family])
    design_r8.warmup(); posthoc_r8.warmup()

def one_replication(r):
    dk, eb, tb, iso, dz, tr, ph = (G[k] for k in ('dk', 'eb', 'tb', 'iso', 'dz', 'tr', 'ph'))
    S1, S2, RS1, RJ = G['seeds']
    TH0 = tr.THETA0; Hm = dk.H(); s = dk.s; nan = np.nan
    T = {}; t0 = time.perf_counter()
    e1 = np.random.default_rng(S1+r).standard_normal(tr.K); e2 = np.random.default_rng(S2+r).standard_normal(tr.K)
    # --- stage 1 and its estimate
    z1, ev1, _ = tr.sim_sod(e1); st1 = eb.StageData(ev1, np.where(ev1, z1, nan), ('sod',))
    t = time.perf_counter(); o1 = eb.estimate_stage1(st1, RS1+r); T['stage1_est'] = time.perf_counter()-t
    T['stage1_total'] = time.perf_counter()-t0
    # --- control's stage 2
    t = time.perf_counter()
    z2c, ev2c, _ = tr.sim_sod(e2); st2c = eb.StageData(ev2c, np.where(ev2c, z2c, nan), ('sod',))
    Hr_c = eb.nll_grad_hess_theta(st2c, TH0)[2]
    T['control_stage2'] = time.perf_counter()-t
    out = dict(r=r, family=G['family'], seeds=dict(e1=S1+r, e2=S2+r, restart_stage1=RS1+r, restart_joint=RJ+r), N1=int(ev1.sum()),
               stage1=dict(theta_LS=o1['theta_LS'], LS_usable=o1['LS_usable'], LS_rank=o1['LS_rank'], init_kind=o1['init_kind'],
                           theta_init=o1['theta_init'], attempts=o1['attempts']),
               control=dict(N2=int(ev2c.sum()), H_r=Hr_c), analyses={})
    th1 = np.array(o1['theta'], float)
    a = dict(rule=eb.RULE_NAME, theta1=th1, stage1_converged=o1['converged'], stage1_used=o1['used'], stage1_fallback=o1['fallback'],
             err1=th1-TH0, eps1=float(np.max(np.abs(Hm@(th1-TH0)))/s))
    # --- design under the sentinel
    tt = {}
    iso.ACCESS_LOG.clear(); t = time.perf_counter()
    with iso.guard():
        d = dz.design(th1, f'r{r}{AN}')
    tt['design'] = time.perf_counter()-t
    if iso.ACCESS_LOG:
        raise iso.SentinelAccess(f"replication {r}: sentinel accessed {iso.ACCESS_LOG}")
    design = {k: d.get(k) for k in ('status', 'reason', 'lam', 'gbar', 'gamma1', 'EN_hat', 'EN_sod_hat', 'S_hat', 'logdet_hat',
                                    'F_hat', 'dual_part', 'z0', 'h', 'G', 'WMAX', 'xhat_min', 'xhat_max',
                                    'mass_dev_max_hat', 'min_m_hat', 'mass_dev_max_sod_hat', 'min_m_sod_hat',
                                    't_cache_s', 't_dp_s', 't_fwd_s', 't_total_s')}
    frac_asym = None
    if d['status'] == 'ok':
        design['J_sod_hat'] = d['J_sod_hat']; design['gamma_mean_check'] = float(np.mean(d['gamma'][1:]))
        tab = tb.SeqTable.from_design(d, f'r{r}{AN}')
        path = os.path.join(G['outdir'], 'tables_packed', f'T_r{r:03d}.npz.bz2')
        nbytes, sha = tb.pack(d, path, extra=dict(r=r, analysis=AN, family=G['family']))
        table = dict(file=os.path.relpath(path, HERE), bytes=nbytes, codes_sha256=sha)
        t = time.perf_counter(); z2, ev2, _ = tr.sim_table(e2, tab); tt['sim2'] = time.perf_counter()-t
        st2 = eb.StageData(ev2, np.where(ev2, z2, nan), ('table', tab))
        t = time.perf_counter(); rec, J, rr = ph.eval_table(tab); tt['posthoc'] = time.perf_counter()-t
        posthoc = dict(EN2=float(rec['EN']), EN2_P=float(rec['EN_P']), S2=float(rec['S']), logdet2=float(rec['logdet']), J_table=J,
                       mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']), assert_fail=int(rec['assert_fail']),
                       boundary_steps=int(rec['boundary_steps']), grid_n0=int(rec['grid_n0']), boundary_gap=float(rec['boundary_gap']),
                       inv_S=bool(rec['inv_S']), inv_J=bool(rec['inv_J']), inv_w=bool(rec['inv_w']))
        frac_asym = float(np.mean(st2.Mm != st2.Mp)) if len(st2.Mm) else 0.0
        H_r = eb.nll_grad_hess_theta(st2, TH0)[2]
    else:                                                        # spec §1.7 fallback: stage 2 = constant Delta* SoD (round 8 DECISION B-2)
        z2, ev2, st2 = z2c, ev2c, st2c; tt['sim2'] = 0.0; tt['posthoc'] = 0.0
        table = None; posthoc = dict(EN2=G['b1'], J_table=G['J1'], fallback_sod=True); H_r = Hr_c
    del d
    a.update(design=design, table=table, posthoc=posthoc, N2=int(ev2.sum()), H_r=H_r, times=tt,
             design_failed=design['status'] != 'ok', frac_asym=frac_asym)
    # --- joint estimates (T fixed): sequential and control
    t = time.perf_counter()
    a['seq'] = eb.estimate_joint([st1, st2], th1, RJ+r)
    a['sod'] = eb.estimate_joint([st1, st2c], th1, RJ+r)
    a['times']['joint'] = time.perf_counter()-t
    a['success_all'] = bool(a['stage1_converged'] and not a['design_failed'] and a['seq']['converged'] and a['sod']['converged'])
    out['analyses'][AN] = a
    out['times'] = dict(T, primary_work_s=T['stage1_total']+T['control_stage2']+tt['design']+tt['sim2']+tt['posthoc']+a['times']['joint'],
                        total_s=time.perf_counter()-t0, pid=os.getpid())
    return jl(out)


def summary_line(o):
    a = o['analyses'][AN]
    return (f"REP r={o['r']:03d} N1 {o['N1']} N2c {o['control']['N2']} | eps1 {a['eps1']:.3f} conv1 {int(a['stage1_converged'])} "
            f"design {a['design']['status'][:6]} Ehat {a['design'].get('EN_hat') or float('nan'):.2f} EN2 {a['posthoc']['EN2']:.2f} "
            f"N2 {a['N2']} joint {int(a['seq']['converged'])}/{int(a['sod']['converged'])} | t {o['times']['total_s']:.0f} s")


# --- E2: comparison with round 8's supplementary analysis S ----------------------------------------------------------------
SKIP = {'times', 'shared_with', 'rule', 'table', 'restart_seed', 't_cache_s', 't_dp_s', 't_fwd_s', 't_total_s'}       # timings, labels

def _diff(new, old, path=''):
    """paths where two JSON values differ (floats compared exactly, as serialised)."""
    if isinstance(new, dict) and isinstance(old, dict):
        out = []
        for k in sorted(set(new) | set(old)):
            if k in SKIP: continue
            if k not in new or k not in old: out.append(f"{path}/{k} (missing in {'new' if k not in new else 'round 8'})"); continue
            out += _diff(new[k], old[k], f"{path}/{k}")
        return out
    if isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old): return [f"{path} (length {len(new)} vs {len(old)})"]
        return [x for i, (u, v) in enumerate(zip(new, old)) for x in _diff(u, v, f"{path}[{i}]")]
    return [] if new == old else [path]

def compare_with_round8(o, old):
    """o: a repro line of this pipeline; old: round 8's archive line of the same r. Compared: the noise-driven records, the
    stage-1 calls, and everything of round 8's analysis S except timings (times, t_*_s), the table file name / packed size (the
    packed copy's metadata names the analysis; the codes' sha256 is compared), the rule label, the restart-seed echo of the
    joint calls (compared through /seeds) and the archive-only key 'shared_with'."""
    a, s = o['analyses'][AN], old['analyses']['S']
    d = []
    d += _diff(o['seeds'], old['seeds'], '/seeds')
    d += [x for x in _diff(o['N1'], old['N1'], '/N1')]
    d += _diff(o['stage1'], old['stage1'], '/stage1')
    d += _diff(o['control'], old['control'], '/control')
    d += _diff(a, s, '/analysis')
    tab_ok = (a['table'] is None and s['table'] is None) or (a['table']['codes_sha256'] == s['table']['codes_sha256'])
    if not tab_ok: d.append('/analysis/table/codes_sha256')
    keys = ('theta1', 'eps1', 'stage1_converged', 'design', 'posthoc', 'N2', 'H_r', 'seq', 'sod', 'success_all')
    return dict(r=o['r'], n_diff=len(d), diffs=d[:50], table_codes_sha_equal=tab_ok, compared_top_keys=list(keys),
                seq_converged=a['seq']['converged'], sod_converged=a['sod']['converged'], seq_used=a['seq']['used'], sod_used=a['sod']['used'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'repro'])
    ap.add_argument('--R', type=int, default=300); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reps', type=str, default='1,2,98,134')
    a = ap.parse_args()
    repro = a.mode == 'repro'
    family = 'round8' if repro else 'official'
    outdir = os.path.join(RES, 'runs', 'repro') if repro else RES
    os.makedirs(os.path.join(outdir, 'tables_packed'), exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    arch = os.path.join(outdir, 'archive_round8b.jsonl')
    logp = os.path.join(LOGS, 'engineering_repro_S.log' if repro else 'partB_pipeline.log')
    logf = open(logp, 'a')
    def P(m):
        print(m, flush=True); logf.write(m+'\n'); logf.flush()
    if not repro:                                                # gate: A3 (spec) and E2 (engineering reproduction) passed
        chk = json.load(open(os.path.join(RES, 'partA_checks.json')))
        e2 = json.load(open(os.path.join(RES, 'engineering_repro_S.json')))
        if not (chk['all_required_pass'] and e2['all_equal']):
            P(f"=== refusing to start the official run: A3 all required pass {chk['all_required_pass']}, E2 all equal {e2['all_equal']}")
            sys.exit(2)
    reps = [int(x) for x in a.reps.split(',')] if repro else list(range(1, a.R+1))
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
    P(f"=== seq_pipeline_r8b.py {a.mode} started {time.strftime('%Y-%m-%d %H:%M:%S')}; seed family {family} {SEEDS[family]}; "
      f"replications {reps[0]}..{reps[-1]} ({len(reps)}), workers {a.workers}; done {len(done)}, to do {len(todo)}")
    P(f"B0 b_1 = E_theta0[N_SoD] (sigma/120, reuse path) = {b1!r}; log det J_1 = {float(np.linalg.slogdet(J1)[1])!r}; "
      f"mass dev {recS['mass_dev_max']!r}, min m {recS['min_m']!r}; {time.perf_counter()-t0:.1f} s")
    json.dump(dict(b1=b1, J1=np.asarray(J1).tolist(), logdet_J1=float(np.linalg.slogdet(J1)[1]), EN_P=float(recS['EN_P']), S=float(recS['S']),
                   mass_dev_max=float(recS['mass_dev_max']), min_m=float(recS['min_m']), path=recS['path']),
              open(os.path.join(outdir, 'sod_reference.json'), 'w'), indent=1)
    tw = time.perf_counter()
    with open(arch, 'a') as fa, mp.get_context('spawn').Pool(a.workers, initializer=init_worker, initargs=(b1, np.asarray(J1), outdir, family)) as pool:
        n = 0
        for o in pool.imap_unordered(one_replication, todo, chunksize=1):
            fa.write(json.dumps(o)+'\n'); fa.flush(); os.fsync(fa.fileno()); n += 1
            P(summary_line(o)+f" [{n}/{len(todo)}, wall {time.perf_counter()-tw:.0f} s]")
    P(f"=== seq_pipeline_r8b.py {a.mode} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; {n} replications in {time.perf_counter()-tw:.1f} s wall")
    if repro:
        new = {o['r']: o for o in (json.loads(l) for l in open(arch) if l.strip())}
        old = {o['r']: o for o in (json.loads(l) for l in open(os.path.join(R8, 'results', 'archive_round8.jsonl')) if l.strip())}
        ref8 = json.load(open(os.path.join(R8, 'results', 'sod_reference.json')))
        rows = [compare_with_round8(new[r], old[r]) for r in reps]
        for c in rows:
            P(f"E2 r={c['r']:03d} differences from round 8 analysis S: {c['n_diff']} {c['diffs']}; table codes sha256 equal {c['table_codes_sha_equal']}; "
              f"joint seq converged {c['seq_converged']} ({c['seq_used']}), control converged {c['sod_converged']} ({c['sod_used']})")
        b1_eq = b1 == ref8['b1'] and np.array_equal(np.asarray(J1), np.array(ref8['J1']))
        allq = all(c['n_diff'] == 0 and c['table_codes_sha_equal'] for c in rows) and b1_eq
        P(f"E2 b_1, J_1 equal to round 8's sod_reference.json: {b1_eq}")
        P(f"E2 reproduction of round 8's supplementary analysis S by the round-8b pipeline (r = {reps}): {'ALL EQUAL' if allq else 'DIFFERENCES'}")
        json.dump(dict(reps=reps, rows=rows, b1_J1_equal=bool(b1_eq), all_equal=bool(allq), seeds=SEEDS['round8']),
                  open(os.path.join(RES, 'engineering_repro_S.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
