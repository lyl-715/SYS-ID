"""Round 17 driver (prompt reports_CN/prompt_round17_absolute_diagnostics_EN.md). Parts A and B; Part C is on hold and not implemented.
Usage (from this directory):  R13_INPUT=U python3 run_round17.py partA partB summary
  partA    §1: eight designs, each J reproduced (archived log det to 1e-9) before its diagnostics are used -> results/partA.json, results/partA/J_*.npy
  partB    §2: M_min in {6, 12, 18, 24, 36, 48}: fit_rule17, sigma/24 and sigma/120, diagnostics of the sigma/24 J, continuous Monte Carlo scan
           -> results/partB.json (checkpointed per M_min)
  summary  §4-§6: tables, Q1-Q4, predictions, checks -> results/round17.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4, R10, R11, R14 = (os.path.join(EV, r) for r in ('round4', 'round10', 'round11', 'round14'))
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
import diag17 as d17
e13, r14 = d17.e13, d17.r14
ew, dr = e13.ew, e13.dr
s, K, G, B = d17.s, d17.K, d17.G, d17.B
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS, os.path.join(RES, 'partA'), os.path.join(RES, 'partB')): os.makedirs(_d, exist_ok=True)
REL = 1e-9
M_LIST = (6, 12, 18, 24, 36, 48)
E_STEP = 0.04
REF = dict(I_full=219.00850592919505, I_SoD=197.20097668224076, I_round4=205.03141309913676, I_T1=207.35521730257756, I_rule=207.5002944958199,
           I_classW=209.0685398686887, cond_full=98.67923064906208, sod_mu_min=0.048517708610531125, sod_mu_max=0.16545799228159686,
           sod_cond=94.92214093489969)
OUTSIDE_B = {6: (1.607, 207.584), 12: (1.598, 207.553), 18: (1.588, 207.522), 24: (1.578, 207.490), 36: (1.554, 207.416), 48: (1.528, 207.298)}


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

class Stop(RuntimeError): pass

def jdump(obj, path):
    tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def relerr(x, ref): return abs(float(x)-float(ref))/abs(float(ref))

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, TOL24=e13.TOL24, TOL120=e13.TOL120)

FROZEN = ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10', 'round11', 'round13', 'round14')
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
    out(f"CLEANUP removed empty folders {removed}; git status on rounds 4-14 and the frozen files: {'clean' if not st else st}")

def logdiag(out, name, dg):
    out(f"DIAG {name}: I {dg['I']!r} (I - I_full {dg['I_minus_full']:+.4f}); mu min {dg['mu_min']:.6f} max {dg['mu_max']:.6f} ratio {dg['mu_ratio']:.3f}; "
        f"geo mu {dg['geo_mu']:.6f} (identity rel {dg['geo_check_rel']:.1e}); cond {dg['cond']:.6f}; lambda min {dg['lam_min']:.4e} max {dg['lam_max']:.4e}; "
        f"sd ratio min {dg['sd_min']:.4f} geo {dg['sd_geo']:.4f} max {dg['sd_max']:.4f}; tr ratio {dg['tr_ratio']:.4f}; lambda_min ratio {dg['lam_min_ratio']:.4f}")
    out(f"DIAG {name}: mu = {[round(v, 6) for v in dg['mu']]}; sd ratios = {[round(v, 4) for v in dg['sd_ratio']]}")


# --- Part A ---------------------------------------------------------------------------------------------------------------------------
def stage_partA(out):
    res = dict(versions=versions(), designs={}, order=[])
    def add(key, label, J, I_ref, ref_src, EN, S, extra=None):
        dg = d17.diagnostics(J); rel = relerr(dg['I'], I_ref)
        np.save(os.path.join(RES, 'partA', f'J_{key}.npy'), J)
        res['designs'][key] = dict(label=label, I_ref=I_ref, I_ref_source=ref_src, I_rel=rel, reproduces=bool(rel <= REL), EN=EN, S=S,
                                   dI=dg['I']-REF['I_SoD'], diag=dg, **(extra or {}))
        res['order'].append(key)
        out(f"REPRO {key} ({label}): log det J {dg['I']!r} vs {I_ref!r} [{ref_src}] rel {rel:.1e} -> {'PASS' if rel <= REL else 'FAIL'}; E[N] {EN!r} S {S!r}")
        logdiag(out, key, dg)
        jdump(res, os.path.join(RES, 'partA.json'))
        if rel > REL: raise Stop(f"{key}: archived log det J does not reproduce ({rel:.2e})")
    # 1. full data
    add('full', 'full data', d17.J_FULL, REF['I_full'], 'prompt §1', float(K), float(K))
    # 2. constant SoD at Delta*: archived J; E[N], S from the accepted operator (whose J is compared with the archive)
    J_sod = np.load(os.path.join(R4, 'results', 'J_U_SoD.npy'))
    rec, J_acc, rr = ew.evaluate_table('U', ew.ConstTable(ew.SOD_M, ew.SOD_M))
    cmpj = dict(bitwise=bool(J_sod.tobytes() == np.asarray(J_acc, float).tobytes()), max_rel=float(np.max(np.abs(J_sod-J_acc))/np.max(np.abs(J_sod))))
    out(f"SOD archived J (round4/results/J_U_SoD.npy) vs accepted operator {rec.get('path')}: {cmpj}")
    add('sod', 'constant SoD at Delta*', J_sod, REF['I_SoD'], 'round4/results/J_U_SoD.npy; prompt §1', rec['EN'], rec['S'], dict(J_vs_accepted_operator=cmpj))
    # 3. round-4 table (as round 11's inputs stage: from_round4 with the old action set)
    with np.load(os.path.join(R4, 'results', 'table_U_W.npz')) as f: cw, st_, wd = f['codes'], f['start'], f['width']
    T4 = e13.from_round4(cw, st_, wd, e13.old_set()); rec, J, fw = e13.evaluate(T4)
    add('round4', 'round-4 table (half-widths, old action set)', J, REF['I_round4'], 'round4/results/mc.json["U/W"]["logdet_eval"]', rec['EN'], rec['S'])
    # 4. T1
    with np.load(os.path.join(R11, 'results', 'table_edge_iter.npz')) as f: c1 = np.array(f['codes'])
    rec, J, fw = e13.evaluate(e13.Table11(c1, e13.edge_set(), 'T1'))
    add('T1', 'T1 (edges, 100 actions)', J, REF['I_T1'], 'round11.json tables.edge_j1.I', rec['EN'], rec['S'])
    # 5. T' (round 11b's union action set and table class, read-only)
    sys.path.insert(0, os.path.join(R11, 'addendum')); import sprime as sp
    acts = sp.set_union()
    with np.load(os.path.join(R11, 'addendum', 'results', 'table_edge_Sprime.npz')) as f:
        cP = np.array(f['codes']); LNa, UNa = np.array(f['LN']), np.array(f['UN'])
    lncheck = bool(np.array_equal(acts.LN, LNa) and np.array_equal(acts.UN, UNa))
    I_tp = json.load(open(os.path.join(R11, 'addendum', 'results', 'round11b.json')))['Tprime']['I24']
    rec, J, fw = e13.evaluate(sp.TableG(cP, acts, 'Tprime'))
    out(f"TPRIME action set {acts.name} ({acts.n} actions); edge nodes LN/UN equal to the archived npz: {lncheck}")
    add('Tprime', "T' (edges, 884 actions)", J, I_tp, 'round11/addendum/results/round11b.json Tprime.I24', rec['EN'], rec['S'], dict(LN_UN_equal_archive=lncheck))
    # 6. one-parameter rule R(37, 13)
    rec, J = r14.evaluate(r14.Rule(37, 13), 1)
    add('rule', 'one-parameter rule R(37, 13)', J, REF['I_rule'], 'round14 part1 1b (sigma/24)', rec['EN'], rec['S'])
    # 7-8. round 10: fixed threshold on the prediction residual; class-W optimum
    sys.path.insert(0, R10); import ub
    r10 = json.load(open(os.path.join(R10, 'results', 'round10.json')))['budgets']['B316']
    prob = ub.Problem(e13.HK, s, ub.c_max, ub.dc_max)
    qu = (B-1.0)/(K-1.0); w = ub.c_max(np.full(K-1, qu)); J = prob.J(np.full(K-1, qu))
    add('fixed', 'fixed threshold on the prediction residual (no SoD), W_k = c_max(q)', J, r10['history'][0]['f'],
        'round10.json budgets.B316.history[0].f', 1.0+(K-1)*qu, 1.0+float(w.sum()), dict(q=qu, a=float(ub.a_sym(qu))))
    sol = ub.solve(ub.Problem(e13.HK, s, ub.c_max, ub.dc_max), B-1.0, tol=0.01)
    J = ub.Problem(e13.HK, s, ub.c_max, ub.dc_max).J(sol['q'])
    chk4 = dict(f=sol['f'], f_ref=r10['f'], f_equal=bool(sol['f'] == r10['f']), u_cert=sol['u_cert'], u_cert_ref=r10['u_cert'], u_cert_equal=bool(sol['u_cert'] == r10['u_cert']))
    chk4['pass_'] = bool(chk4['f_equal'] and chk4['u_cert_equal'])
    out(f"CHECK4 round 10 at the nominal budget: f {sol['f']!r} (archive {r10['f']!r}) equal {chk4['f_equal']}; u_cert {sol['u_cert']!r} (archive {r10['u_cert']!r}) "
        f"equal {chk4['u_cert_equal']} -> {'PASS' if chk4['pass_'] else 'FAIL'}")
    add('classW', 'class-W optimum (receiver-known windows), W_k = c_max(q_hat_k)', J, r10['f'], 'round10.json budgets.B316.f', 1.0+float(sol['q'].sum()),
        1.0+float(ub.c_max(sol['q']).sum()), dict(q_min=float(sol['q'].min()), q_max=float(sol['q'].max()), u_cert=sol['u_cert']))
    res['check4'] = chk4
    # reference values (outside computation)
    dsod = res['designs']['sod']['diag']; dfull = res['designs']['full']['diag']
    refc = dict(cond_full=(dfull['cond'], REF['cond_full']), sod_mu_min=(dsod['mu_min'], REF['sod_mu_min']), sod_mu_max=(dsod['mu_max'], REF['sod_mu_max']),
                sod_cond=(dsod['cond'], REF['sod_cond']))
    res['reference_values'] = {k: dict(value=v, ref=r, rel=relerr(v, r), pass_=bool(relerr(v, r) <= REL)) for k, (v, r) in refc.items()}
    for k, v in res['reference_values'].items(): out(f"REF {k}: {v['value']!r} vs {v['ref']!r} rel {v['rel']:.1e} -> {'PASS' if v['pass_'] else 'FAIL'}")
    res['geo_identity_pass'] = all(dd['diag']['geo_check_rel'] <= 1e-10 for dd in res['designs'].values())
    out(f"CHECK2 geometric-mean identity exp(mean log mu) = exp((I - I_full)/9) to 1e-10 for every design: {res['geo_identity_pass']} "
        f"(max rel {max(dd['diag']['geo_check_rel'] for dd in res['designs'].values()):.1e})")
    jdump(res, os.path.join(RES, 'partA.json'))
    if not (all(v['pass_'] for v in res['reference_values'].values()) and res['geo_identity_pass'] and chk4['pass_']): raise Stop("a Part A check failed")


# --- Part B ---------------------------------------------------------------------------------------------------------------------------
def mc_scan(e0, m, out):
    """continuous Monte Carlo: e from the lattice e_eff in steps of 0.04 towards E[N] = B until B is bracketed; linear interpolation of E[N] and
    log det J to E[N] = B (DECISION 17-5)."""
    runs = []
    def run(e):
        mc, J = d17.mc_cont(e, m); runs.append(mc)
        out(f"MC m={m:.2f} e={e:.6f}: EN {mc['EN_mc']!r} ({mc['EN_se']:.4f}) S {mc['S_mc']!r} logdet {mc['logdet_mc']!r} ({mc['logdet_batch_se']:.5f}); {mc['wall_s']:.1f} s")
        return mc
    a = run(e0); step = E_STEP if a['EN_mc'] > B else -E_STEP; e = e0
    while True:
        e = e+step; b = run(e)
        if (a['EN_mc']-B)*(b['EN_mc']-B) <= 0: break
        a = b
        if len(runs) > 15: raise Stop(f"MC scan: no bracket for m = {m}")
    t = (a['EN_mc']-B)/(a['EN_mc']-b['EN_mc'])
    e_fit = a['e']+t*(b['e']-a['e']); ld = a['logdet_mc']+t*(b['logdet_mc']-a['logdet_mc'])
    return dict(runs=runs, bracket=[a['e'], b['e']], e_fit=e_fit, logdet_at_B=ld, t=t)

def stage_partB(out):
    path = os.path.join(RES, 'partB.json'); res = json.load(open(path)) if os.path.exists(path) else dict(rows={})
    res['versions'] = versions()
    p14 = json.load(open(os.path.join(R14, 'results', 'part1.json')))['b']
    for M in M_LIST:
        key = str(M)
        if key in res['rows'] and res['rows'][key].get('done'): out(f"PARTB M_min={M}: done, skipped"); continue
        out(f"PARTB M_min = {M} ({M/24:.2f} sigma)")
        fit, rule, r24, J24, r120, J120 = d17.fit_rule17(B, M, out)
        np.save(os.path.join(RES, 'partB', f'J_M{M}_s24.npy'), J24); np.save(os.path.join(RES, 'partB', f'J_M{M}_s120.npy'), J120)
        dg = d17.diagnostics(J24); logdiag(out, f'M_min={M}', dg)
        row = dict(M_min=M, m=M/24, fit=fit, rec24={k: r24[k] for k in ('EN', 'S', 'logdet', 'cond', 'mass_dev_max', 'min_m')},
                   rec120={k: r120[k] for k in ('EN', 'S', 'logdet', 'cond', 'mass_dev_max', 'min_m')}, diag24=dg)
        if M == 24:
            c3 = dict(n_j=bool((fit['n'], fit['j']) == (p14['fit']['n'], p14['fit']['j'])),
                      rec24_bitwise=bool(all(r24[k] == p14['rec24'][k] for k in ('EN', 'S', 'logdet'))),
                      rec120_bitwise=bool(all(r120[k] == p14['rec120'][k] for k in ('EN', 'S', 'logdet'))))
            c3['pass_'] = all(c3.values()); row['check_round14'] = c3
            out(f"CHECK3 M_min = 24 vs Round 14 R(37, 13): {c3} -> {'PASS' if c3['pass_'] else 'FAIL'}")
            if not c3['pass_']: jdump(res, path); raise Stop("M_min = 24 does not reproduce Round 14's rule")
        res['rows'][key] = row; jdump(res, path)
        sc = mc_scan(fit['e_eff'], M/24, out)
        row['mc'] = sc; row['outside'] = dict(e_fit=OUTSIDE_B[M][0], logdet=OUTSIDE_B[M][1]); row['done'] = True
        res['rows'][key] = row; jdump(res, path)
        out(f"PARTB M_min={M} DONE: R({fit['n']}, {fit['j']}) e_eff {fit['e_eff']:.5f}; sigma/24 EN {r24['EN']:.3f} I {r24['logdet']:.4f}; sigma/120 EN {r120['EN']:.3f} "
            f"I {r120['logdet']:.4f}; mu {dg['mu_min']:.4f}..{dg['mu_max']:.4f} cond {dg['cond']:.3f}; MC e_fit {sc['e_fit']:.4f} logdet at B {sc['logdet_at_B']:.4f} "
            f"(outside {OUTSIDE_B[M][0]} / {OUTSIDE_B[M][1]})")


STAGES = ('partA', 'partB', 'summary')

def main():
    numba.set_num_threads(4)
    for stg in sys.argv[1:]:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        out = Tee(os.path.join(LOGS, f'{stg}.log'))
        out(f"=== ROUND 17 stage {stg} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
            f"numba {numba.__version__}; B = {B!r}; input {e13.NAME}; TOL24 {e13.TOL24:g}")
        t0 = time.perf_counter()
        try:
            if stg == 'partA': stage_partA(out)
            elif stg == 'partB': stage_partB(out)
            elif stg == 'summary':
                import summary17; summary17.stage(out)
        except (Stop, e13.AssertionStop, AssertionError, RuntimeError) as e:
            out(f"STOP {type(e).__name__}: {e}"); out(f"=== ROUND 17 stage {stg} STOPPED after {time.perf_counter()-t0:.1f} s"); cleanup_frozen(out); raise
        out(f"=== ROUND 17 stage {stg} done in {time.perf_counter()-t0:.1f} s")
        cleanup_frozen(out)


if __name__ == '__main__':
    main()
