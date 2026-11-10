"""Round 15 driver (prompt reports_CN/prompt_round15_rule_with_estimated_model_EN.md). Input U only; rounds 4-14 are read-only; no DP.

Usage (from this directory):  R13_INPUT=U python3 run_round15.py <stage> ...
  check1   §5 item 1: the 12 reference models, mismatched lattice rule R(37, 13) on sigma/24 vs the outside computation (gate: stop if not 1e-9)
  check2   §5 item 2: three reference models, sigma/120 evaluation and Monte Carlo of the mismatched lattice rule (R = 1e4), z-scores
  mc       §3: all continuous-rule jobs, checkpointed in results/jobs.jsonl (true model x 18 variants and check 4 at R = 20000; main set x 18
           variants; reference set x 4 variants; selection §4b; scaled and gross-error models x V0, VM*, VF*)
  summary  §4-§7: tables, selection, head-to-head, scaled/gross, figure, Q1-Q7, predictions, checks -> results/round15.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R14 = os.path.join(EV, 'round14')
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
import mismatch15 as m15
e13 = m15.e13
s, K, B = m15.s, m15.K, m15.B
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
JOBS = os.path.join(RES, 'jobs.jsonl')
WORKERS = 4; R_JOB = 2000; SEED = 20261112; R_TRUE_BATCHES = 10
E0_CHECK4 = 1.5768
REL = 1e-9
# §5 item 1: reference values (outside computation): (c, draw) -> (E[N], log det J, rms pred. err, max pred. err)
REF_C1 = {(1, 0): (293.130066163623, 206.94719294319475, 0.113856, 0.243846), (1, 1): (340.7463732234027, 207.79277045511677, 0.164476, 0.374621),
          (1, 2): (318.4508130092631, 207.4469157068209, 0.131573, 0.288575), (1, 3): (315.63872992664085, 207.39108535919198, 0.115342, 0.249696),
          (2, 0): (323.5285886160711, 207.5402691799476, 0.106571, 0.228429), (2, 1): (353.8547438394742, 207.70585912543362, 0.278381, 0.641947),
          (2, 2): (335.7276635272488, 207.65125283199484, 0.177686, 0.398959), (2, 3): (318.6371113922262, 207.3058071370323, 0.194471, 0.473770),
          (4, 0): (295.0571166894415, 206.94920759763048, 0.135837, 0.299670), (4, 1): (271.76169186770636, 206.1620347166347, 0.265468, 0.589348),
          (4, 2): (293.2396357394827, 206.6005178205391, 0.261610, 0.542670), (4, 3): (386.8182332082853, 208.14034340245658, 0.338782, 0.706108)}
CHECK2_MODELS = ('ref_c1_d1', 'ref_c2_d1', 'ref_c4_d3')


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

def e0_round14():
    d = json.load(open(os.path.join(R14, 'results', 'part1.json')))['d']
    return float(d['e_up']), d


# --- check 1 -----------------------------------------------------------------------------------------------------------------------
def stage_check1(out):
    M = m15.models()
    th = m15.theta_star(); xk_th = e13.HK@th
    res = dict(versions=versions(), theta_star=[float(v) for v in th], HK_theta_minus_XK_max_abs=float(np.abs(xk_th-e13.XK).max()),
               HK_theta_minus_XK_max_rel=float(np.abs(xk_th-e13.XK).max()/np.abs(e13.XK).max()), rows={})
    out(f"THETA* {res['theta_star']}; max |HK theta* - XK| = {res['HK_theta_minus_XK_max_abs']:.3e} ({res['HK_theta_minus_XK_max_rel']:.1e} relative)")
    ok = True
    for (c, d), ref in REF_C1.items():
        name = f'ref_c{c:g}_d{d}'; mod = M[name]
        rule = m15.MRule(mod['delta']); rec, J = m15.evaluate_mrule(rule, 1)
        r = dict(c=c, d=d, EN=rec['EN'], logdet=rec['logdet'], S=rec['S'], rms=mod['rms'], max=mod['max'], ref=dict(EN=ref[0], logdet=ref[1], rms=ref[2], max=ref[3]),
                 EN_rel=relerr(rec['EN'], ref[0]), logdet_rel=relerr(rec['logdet'], ref[1]), rms_abs=abs(mod['rms']-ref[2]), max_abs=abs(mod['max']-ref[3]),
                 mass_dev_max=rec['mass_dev_max'], fl_shift_min=int(rule.fl_shift.min()), fl_shift_max=int(rule.fl_shift.max()), wall_s=rec['wall_s'])
        r['pass_'] = bool(r['EN_rel'] <= REL and r['logdet_rel'] <= REL and r['rms_abs'] <= 5e-7 and r['max_abs'] <= 5e-7)
        ok &= r['pass_']; res['rows'][name] = r
        out(f"CHECK1 c={c} d={d}: EN {rec['EN']!r} (rel {r['EN_rel']:.1e}) logdet {rec['logdet']!r} (rel {r['logdet_rel']:.1e}); rms {mod['rms']:.6f} max {mod['max']:.6f} "
            f"(ref {ref[2]:.6f} / {ref[3]:.6f}); FLd - FL in [{r['fl_shift_min']}, {r['fl_shift_max']}]; mass dev {rec['mass_dev_max']:.2e} ({rec['wall_s']:.1f} s) "
            f"-> {'PASS' if r['pass_'] else 'FAIL'}")
    res['pass_'] = bool(ok)
    jdump(res, os.path.join(RES, 'check1.json'))
    out(f"CHECK1 all 12 reference models: {'PASS' if ok else 'FAIL'}")
    if not ok: raise Stop("check 1 does not reproduce: stop (prompt §5 item 1)")


# --- check 2 -----------------------------------------------------------------------------------------------------------------------
def stage_check2(out):
    M = m15.models(); c1 = json.load(open(os.path.join(RES, 'check1.json')))
    res = dict(versions=versions(), rows={}); ok = True
    for name in CHECK2_MODELS:
        rule = m15.MRule(M[name]['delta'])
        r120, J = m15.evaluate_mrule(rule, 5)
        mc = m15.mc_lattice_m(rule)
        z = dict(EN=(r120['EN']-mc['EN_mc'])/mc['EN_se'], S=(r120['S']-mc['S_mc'])/mc['S_se'], logdet=(r120['logdet']-mc['logdet_mc'])/mc['logdet_batch_se'])
        p = bool(max(abs(v) for v in z.values()) <= 3 and mc['n_clipped'] == 0); ok &= p
        res['rows'][name] = dict(sigma24=dict(EN=c1['rows'][name]['EN'], logdet=c1['rows'][name]['logdet'], S=c1['rows'][name]['S']),
                                 sigma120=dict(EN=r120['EN'], S=r120['S'], logdet=r120['logdet'], mass_dev_max=r120['mass_dev_max'], wall_s=r120['wall_s']),
                                 mc=mc, z=z, pass_=p)
        out(f"CHECK2 {name}: sigma/120 EN {r120['EN']!r} S {r120['S']!r} logdet {r120['logdet']!r} ({r120['wall_s']:.1f} s); MC (R = 1e4) EN {mc['EN_mc']!r} ({mc['EN_se']:.4f}) "
            f"S {mc['S_mc']!r} ({mc['S_se']:.4f}) logdet {mc['logdet_mc']!r} ({mc['logdet_batch_se']:.5f}); z EN {z['EN']:+.2f} S {z['S']:+.2f} logdet {z['logdet']:+.2f}; "
            f"clipped {mc['n_clipped']} -> {'PASS' if p else 'FAIL'}")
    res['pass_'] = bool(ok); jdump(res, os.path.join(RES, 'check2.json'))
    out(f"CHECK2 three models: {'PASS' if ok else 'FAIL'}")


# --- Monte Carlo of the continuous rule ---------------------------------------------------------------------------------------------
def load_jobs():
    done = {}
    if os.path.exists(JOBS):
        for l in open(JOBS):
            if l.strip(): r = json.loads(l); done[r['key']] = r
    return done

def run_tasks(tasks, out, label):
    """tasks: list of (key, delta, var, R, seed); runs them on WORKERS processes; yields raw outputs."""
    from multiprocessing import Pool
    t0 = time.perf_counter(); n = 0
    with Pool(WORKERS) as pool:
        for o in pool.imap_unordered(m15.run_job, tasks, chunksize=1):
            n += 1
            if n % 100 == 0 or n == len(tasks):
                out(f"MC {label}: {n}/{len(tasks)} jobs, {time.perf_counter()-t0:.0f} s")
            yield o

def write_rec(f, key, model, vname, rec):
    rec = dict(rec, key=key, model=model, variant=vname)
    f.write(json.dumps(rec)+'\n'); f.flush()
    return rec

def select(done, V):
    """§4b: complies iff at most 5 % of the 40 models have E[N] > B at c = 1 and at c = 2; VM* / VF* = complying member with the largest median
    Delta I at c = 2; if none complies, the member with the smallest fraction over budget at c = 2."""
    def frac(vn, c): return float(np.mean([done[f"main_c{c:g}_d{d:02d}|{vn}"]['EN'] > B for d in range(40)]))
    def med(vn, c): return float(np.median([done[f"main_c{c:g}_d{d:02d}|{vn}"]['dI'] for d in range(40)]))
    tab = {vn: dict(frac_c1=frac(vn, 1), frac_c2=frac(vn, 2), med_dI_c2=med(vn, 2)) for vn in V}
    for vn, t in tab.items(): t['complies'] = bool(t['frac_c1'] <= 0.05 and t['frac_c2'] <= 0.05)
    sel = {}
    for fam, pref in (('VM', 'VM('), ('VF', 'VF(')):
        mem = [vn for vn in V if vn.startswith(pref)]; comp = [vn for vn in mem if tab[vn]['complies']]
        if comp: sel[fam] = dict(variant=max(comp, key=lambda v: tab[v]['med_dI_c2']), complying=True, complying_members=comp)
        else: sel[fam] = dict(variant=min(mem, key=lambda v: (tab[v]['frac_c2'], -tab[v]['med_dI_c2'])), complying=False, complying_members=[])
    return tab, sel

def stage_mc(out):
    e0, d14 = e0_round14()
    M = m15.models(); V = m15.variants(e0)
    jdump(dict(e0=e0, e0_source='round14/results/part1.json d.e_up', models=M, variants=V), os.path.join(RES, 'setup.json'))
    out(f"SETUP e0 = {e0!r} (Round 14 part 1d e_C rounded up); {len(M)} models; {len(V)} variants; R = {R_JOB} per job, seed {SEED} for every job")
    for k_, v_ in (('main', 'main'), ('ref', 'ref'), ('scaled', 'scaled')):
        rm = [m['rms'] for m in M.values() if m['set'] == v_]; mx = [m['max'] for m in M.values() if m['set'] == v_]
        out(f"MODELS {k_}: {len(rm)} models; rms prediction error {np.min(rm):.4f} .. {np.max(rm):.4f} sigma (median {np.median(rm):.4f}); "
            f"max {np.min(mx):.4f} .. {np.max(mx):.4f} sigma (median {np.median(mx):.4f})")
    for g in ('gross_plus', 'gross_minus'): out(f"MODELS {g}: rms {M[g]['rms']:.2f} sigma, max {M[g]['max']:.2f} sigma")
    done = load_jobs()
    f = open(JOBS, 'a')
    # 1. true model: every variant and check 4, R = 20000 in 10 batches
    Vtrue = dict(V); V4 = {f'V0@{E0_CHECK4}': dict(kind='const', e=E0_CHECK4)}
    for T in (1000, 300, 100): V4[f'VF({T},0)@{E0_CHECK4}'] = dict(kind='fb', e0=E0_CHECK4, T=T, m=0)
    Vtrue.update(V4)
    todo = [vn for vn in Vtrue if f"true|{vn}" not in done]
    tasks = [(f"true|{vn}|b{b}", [0.0]*9, Vtrue[vn], R_JOB, SEED+b) for vn in todo for b in range(R_TRUE_BATCHES)]
    buf = {}
    for o in run_tasks(tasks, out, 'true model'):
        k, vn, b = o['key'].split('|'); buf.setdefault(vn, []).append(o)
        if len(buf[vn]) == R_TRUE_BATCHES:
            rec = m15.aggregate(sorted(buf.pop(vn), key=lambda x: x['seed'])); done[f"true|{vn}"] = write_rec(f, f"true|{vn}", 'true', vn, rec)
            out(f"TRUE {vn}: EN {rec['EN']:.3f} ({rec['EN_se']:.3f}) sd N {rec['sd_N']:.2f} frac N>B {rec['frac_over']:.3f} S {rec['S']:.2f} logdet {rec['logdet']:.4f} "
                f"({rec.get('logdet_batch_se', float('nan')):.4f}) dI {rec['dI']:.4f}" + (f"; e mean {rec['e_mean']:.4f} sd {rec['e_sd']:.4f} clip lo {rec['frac_clip_lo']:.4f} hi {rec['frac_clip_hi']:.4f}" if 'e_mean' in rec else ''))
    # 2. main set x 18 variants, 3. reference set x 4 variants
    ref_v = ['V0', 'VF(1000,0)', 'VF(300,0)', 'VF(100,0)']
    tasks = [(f"{mn}|{vn}", M[mn]['delta'], V[vn], R_JOB, SEED) for mn in M if M[mn]['set'] == 'main' for vn in V if f"{mn}|{vn}" not in done]
    tasks += [(f"{mn}|{vn}", M[mn]['delta'], V[vn], R_JOB, SEED) for mn in M if M[mn]['set'] == 'ref' for vn in ref_v if f"{mn}|{vn}" not in done]
    for o in run_tasks(tasks, out, 'main + reference'):
        mn, vn = o['key'].split('|'); done[o['key']] = write_rec(f, o['key'], mn, vn, m15.aggregate([o]))
    # 4. selection, then scaled and gross-error models x V0, VM*, VF*
    tab, sel = select(done, V)
    jdump(dict(table=tab, selection=sel, rule='prompt §4b'), os.path.join(RES, 'selection.json'))
    for fam in ('VM', 'VF'):
        out(f"SELECT {fam}* = {sel[fam]['variant']} (complying: {sel[fam]['complying']}; complying members {sel[fam]['complying_members']})")
    for vn, t in tab.items(): out(f"COMPLY {vn}: frac E[N] > B c=1 {t['frac_c1']:.3f} c=2 {t['frac_c2']:.3f}; median dI c=2 {t['med_dI_c2']:.4f}; complies {t['complies']}")
    sv = ['V0', sel['VM']['variant'], sel['VF']['variant']]
    tasks = [(f"{mn}|{vn}", M[mn]['delta'], V[vn], R_JOB, SEED) for mn in M if M[mn]['set'] in ('scaled', 'gross') for vn in sv if f"{mn}|{vn}" not in done]
    for o in run_tasks(tasks, out, 'scaled + gross'):
        mn, vn = o['key'].split('|'); done[o['key']] = write_rec(f, o['key'], mn, vn, m15.aggregate([o]))
    f.close()
    bad = [k for k, r in done.items() if not r['check5']]
    out(f"MC done: {len(done)} job records; check 5 (finite dI, 0 <= W <= 1, N >= 1) failures: {bad}")
    if bad: raise Stop(f"check 5 failed for {bad[:5]}")


STAGES = ('check1', 'check2', 'mc', 'summary')

def main():
    numba.set_num_threads(4)
    for stg in sys.argv[1:]:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        out = Tee(os.path.join(LOGS, f'{stg}.log'))
        out(f"=== ROUND 15 stage {stg} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
            f"numba {numba.__version__}; B = {B!r}; input {e13.NAME}; TOL24 {e13.TOL24:g}")
        t0 = time.perf_counter()
        try:
            if stg == 'check1': stage_check1(out)
            elif stg == 'check2': stage_check2(out)
            elif stg == 'mc': stage_mc(out)
            elif stg == 'summary':
                import summary15; summary15.stage(out)
        except (Stop, e13.AssertionStop, AssertionError) as e:
            out(f"STOP {type(e).__name__}: {e}"); out(f"=== ROUND 15 stage {stg} STOPPED after {time.perf_counter()-t0:.1f} s"); cleanup_frozen(out); raise
        out(f"=== ROUND 15 stage {stg} done in {time.perf_counter()-t0:.1f} s")
        cleanup_frozen(out)


if __name__ == '__main__':
    main()
