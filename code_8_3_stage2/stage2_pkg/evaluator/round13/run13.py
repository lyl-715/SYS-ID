"""Round 13 driver (README reports_CN/README_round13.md; prompts reports_CN/prompt_round13_part0_infrastructure_EN.md, ..._partA_...,
..._partB_...). Rounds 4-11 are imported read-only; nothing there is changed.

One process per (input, stage):   R13_INPUT=<name> ROUND13_WORK=<dir> python3 run13.py <stage>
  gate          Part 0 gates for this input (results/gate_<name>.json, logs/gate_<name>.log):
                  U      G0 (a)-(d) (bitwise copies; the two-scan design reproduces round 11), G2 (constant SoD on the full grid), G4
                  P5514  G1 (input registration), G4
                  N      G1, G4
                  LO     G3 (refined grid for a non-U input), wall time of one DP run, G4
                  SCH    G4
                  HI     G4
  gate_summary  (any R13_INPUT) collect the six gate files into results/gate13.json with a pass flag per gate.
  partA         Part A for this input (prompt_round13_partA §4): I_full; constant SoD at Delta* (sigma/24, sigma/120); matched constant
                SoD; the two-scan design (edge set, 9x9 log det); shape statistics of T* only, after the design is released (DECISION
                13-17); packed T*; derived c_tab, c_sod, gain, information per transmission; checks 1-5 of §8. Writes
                results/partA_<name>.json (after every step), results/partA_<name>/ (scan jsons, J matrices), results/table_A_<name>.npz.
                Inputs run one at a time (DECISION 13-17).
  partB         not implemented (Part B).
DP tables live in $ROUND13_WORK/<input>/<label>/ (default round13/work/, not committed).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R7 = os.path.join(EV, 'round7'); R11 = os.path.join(EV, 'round11')
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
import common13 as c13
e13, dp13 = c13.e13, c13.dp13
ew, dr = e13.ew, e13.dr
s, K, B, G, NAME = e13.s, e13.K, e13.B, e13.G, e13.NAME
RES = c13.RES; LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
WORK = c13.WORK
Stop = c13.Stop; jdump = c13.jdump; relerr = c13.relerr
GATE_INPUTS = ('U', 'P5514', 'SCH', 'LO', 'HI', 'N')
# G4 identification values (prompt §3 G4 / README table): I_full, largest per-step motion (sigma), G, output range (sigma)
REF_G4 = {'U': (219.0085, 3.99, 35146, 1144.4), 'P5514': (219.0080, 3.18, 34951, 1136.2), 'SCH': (219.0085, 3.03, 34596, 1121.4),
          'LO': (218.0343, 4.28, 40015, 1347.2), 'HI': (218.0343, 3.44, 28540, 869.1), 'N': (218.4169, 4.04, 43262, 1482.5)}
G4_TOL_I = 1e-4
REL_9 = 1e-9; REL_12 = 1e-12


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                numba_threads=numba.get_num_threads())

FROZEN = ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10', 'round11')
FROZEN_FILES = ('config.json', 'density_recursion_v1.py', 'evaluator_fast.py')

def cleanup_frozen(out):
    """prompt §0: importing round4 modules may create empty folders under round4/; remove empty untracked folders under rounds 4-11 and
    confirm that git sees no change there nor in config.json, density_recursion_v1.py, evaluator_fast.py."""
    removed = []
    for rd in FROZEN:
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_):
                os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + [os.path.join(EV, r) for r in FROZEN] + [os.path.join(EV, f) for f in FROZEN_FILES],
                        capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on rounds 4-11, config.json, density_recursion_v1.py, evaluator_fast.py: "
        f"{'clean' if not st else st}")
    return removed, st == ''

def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


# --- G4: sanity of the input -------------------------------------------------------------------------------------------------
def gate_G4(out):
    d = c13.input_sanity()
    rI, rstep, rG, rrange = REF_G4[NAME]
    d.update(ref_I_full=rI, ref_max_step_sigma=rstep, ref_G=rG, ref_range_sigma=rrange,
             I_full_absdiff=abs(d['I_full']-rI), G_equal=bool(d['G'] == rG),
             step_agrees_at_2_decimals=bool(abs(d['max_step_sigma']-rstep) <= 0.005), range_agrees_at_1_decimal=bool(abs(d['range_sigma']-rrange) <= 0.05),
             step_le_4_6=bool(d['max_step_sigma'] <= 4.6), sum_a2_ok=bool(abs(d['sum_a2_minus_1']) <= 1e-12))
    d['pass_'] = bool(d['I_full_absdiff'] <= G4_TOL_I and d['G_equal'] and d['step_le_4_6'] and d['sum_a2_ok'])
    out(f"G4 {NAME}: sum a^2 - 1 = {d['sum_a2_minus_1']!r}; output range {d['range_sigma']!r} sigma (ref {rrange}); largest per-step motion "
        f"{d['max_step_sigma']!r} sigma (ref {rstep}; <= 4.6: {d['step_le_4_6']}); G = {d['G']} (ref {rG}, equal: {d['G_equal']}); "
        f"I_full = {d['I_full']!r} (ref {rI}, |diff| {d['I_full_absdiff']:.2e} <= 1e-4: {d['I_full_absdiff'] <= G4_TOL_I}); "
        f"a = {d['a']!r}; chi = {d['chi']!r} -> {'PASS' if d['pass_'] else 'FAIL'}")
    return d


# --- G0 and G2: input U ------------------------------------------------------------------------------------------------------
def gate_G0a(out):
    sys.path.append(R11)
    import eval11                                              # frozen round 11 (read-only), bound to 'U'
    assert eval11.ew is e13.ew
    d = {}
    for n in ('XK', 'HK', 'Z', 'KA', 'KB', 'FL', 'CL'):
        d[n] = bitwise(getattr(e13, n), getattr(eval11, n))
    for n in ('DZ', 'Z0', 'G', 'X_PAD', 'NAME', 'B', 's'):
        d[n] = bool(getattr(e13, n) == getattr(eval11, n))
    res = dict(bitwise=d, pass_=all(d.values()))
    out(f"G0(a) eval13 vs eval11 module globals (one process imports both), bitwise: {d} -> {'PASS' if res['pass_'] else 'FAIL'}")
    return res

def gate_G2(des, out):
    rec_sod, J_sod, W_sod = des.sod_reference()
    r24, J24, fw24 = c13.eval_const(480, 1)
    d = dict(EN=r24['EN'], logdet=r24['logdet'], S=r24['S'], EN_ref=rec_sod['EN'], logdet_ref=rec_sod['logdet'], S_ref=rec_sod['S'],
             ref_path=rec_sod.get('path'), EN_rel=relerr(r24['EN'], rec_sod['EN']), logdet_rel=relerr(r24['logdet'], rec_sod['logdet']),
             S_rel=relerr(r24['S'], rec_sod['S']), J_maxrel=float(np.max(np.abs(J24-J_sod))/np.max(np.abs(J_sod))),
             EN_vs_prompt_rel=relerr(r24['EN'], 316.8921168493721), logdet_vs_prompt_rel=relerr(r24['logdet'], 197.20097668224076),
             mass_dev_max=r24['mass_dev_max'], wall_s=r24['wall_s'], support=r24['support'])
    d['pass24'] = bool(d['EN_rel'] <= REL_9 and d['logdet_rel'] <= REL_9 and d['EN_vs_prompt_rel'] <= REL_9 and d['logdet_vs_prompt_rel'] <= REL_9)
    out(f"G2 sigma/24 eval13.evaluate(ConstRows(480)) EN {r24['EN']!r} logdet {r24['logdet']!r} S {r24['S']!r} vs accepted operator "
        f"({rec_sod.get('path')}) EN {rec_sod['EN']!r} logdet {rec_sod['logdet']!r}: rel {d['EN_rel']:.2e} / {d['logdet_rel']:.2e} (S {d['S_rel']:.2e}, "
        f"J max rel {d['J_maxrel']:.2e}); mass dev {r24['mass_dev_max']:.2e}; {r24['wall_s']:.1f} s -> {'PASS' if d['pass24'] else 'FAIL'}")
    # sigma/120 against round 7's archive (theta_0, M = 480, f = 5)
    rows = [json.loads(l) for l in open(os.path.join(R7, 'results', 'archive_match.jsonl'))]
    hit = [r for r in rows if r['point'] == 'theta_0' and r['M'] == 480 and r['f'] == 5]
    r120, J120, fw120 = c13.eval_const(480, 5)
    d120 = dict(EN=r120['EN'], logdet=r120['logdet'], S=r120['S'], G=r120['G'], mass_dev_max=r120['mass_dev_max'], wall_s=r120['wall_s'],
                archive_present=bool(hit))
    if hit:
        a = hit[0]
        d120.update(EN_ref=a['EN'], logdet_ref=a['logdet'], S_ref=a['S'], ref_route=a['route'], ref_source='round7/results/archive_match.jsonl (theta_0, M = 480, f = 5)',
                    EN_rel=relerr(r120['EN'], a['EN']), logdet_rel=relerr(r120['logdet'], a['logdet']), S_rel=relerr(r120['S'], a['S']),
                    n_entries=len(hit))
        d120['pass120'] = bool(d120['EN_rel'] <= REL_9 and d120['logdet_rel'] <= REL_9)
        out(f"G2 sigma/120 eval13.evaluate_refined(ConstRows(480), 5) EN {r120['EN']!r} logdet {r120['logdet']!r} S {r120['S']!r} vs round 7 archive "
            f"(route {a['route']}) EN {a['EN']!r} logdet {a['logdet']!r}: rel {d120['EN_rel']:.2e} / {d120['logdet_rel']:.2e} (S {d120['S_rel']:.2e}); "
            f"mass dev {r120['mass_dev_max']:.2e}; G {r120['G']}; {r120['wall_s']:.1f} s -> {'PASS' if d120['pass120'] else 'FAIL'}")
    else:
        d120['pass120'] = None
        out(f"G2 sigma/120 EN {r120['EN']!r} logdet {r120['logdet']!r}: no archived entry (theta_0, M = 480, f = 5) -> reported only")
    return dict(sigma24=d, sigma120=d120, pass_=bool(d['pass24'] and d120['pass120'] in (True, None)))

SCAN_FIELDS = ('lam', 'EN', 'S', 'F', 'dsum', 'dual', 'logdet', 'feasible')

def gate_G0bcd(des, out):
    outdir = os.path.join(RES, 'gate_U_design')
    res = c13.design(e13.edge_set(), dp13.Design11.gamma_from_J, c13.logdet9, 9, 'edge', des=des, outdir=outdir,
                     work=os.path.join(WORK, NAME, 'gate_edge'), log=out)
    r11 = json.load(open(os.path.join(R11, 'results', 'round11.json')))
    g = {}
    # (b) the two scans run by run
    ok_b = True; scans = {}
    for j in ('j0', 'j1'):
        mine = json.load(open(os.path.join(outdir, f'scan_edge_{j}.json')))
        ref = json.load(open(os.path.join(R11, 'results', f'scan_edge_{j}.json')))
        runs = []; ok = len(mine['runs']) == len(ref['runs'])
        for a, b in zip(mine['runs'], ref['runs']):
            c = {f: bool(a[f] == b[f]) for f in SCAN_FIELDS}; c['phase'] = bool(a['phase'] == b['phase']); c['run'] = a['run']
            runs.append(c); ok &= all(v for k, v in c.items() if k != 'run')
        top = dict(gbar=bool(mine['gbar'] == ref['gbar']), gamma1=bool(mine['gamma1'] == ref['gamma1']), final_run=bool(mine['final_run'] == ref['final_run']),
                   min_dual=bool(mine['min_dual'] == ref['min_dual']), n_runs=bool(len(mine['runs']) == len(ref['runs'])),
                   lam_bracket=bool(mine['lam_bracket'] == ref['lam_bracket']), status=bool(mine['status'] == ref['status']))
        ok &= all(top.values())
        scans[j] = dict(runs=runs, top=top, gbar=mine['gbar'], gamma1=mine['gamma1'], final_run=mine['final_run'], final_run_ref=ref['final_run'],
                        n_runs=len(mine['runs']), min_dual=mine['min_dual'], pass_=bool(ok),
                        dp_s=[r['dp_s'] for r in mine['runs']], fwd_s=[r['fwd_s'] for r in mine['runs']])
        ok_b &= ok
        nbit = sum(all(v for k, v in c.items() if k != 'run') for c in runs)
        out(f"G0(b) scan edge_{j}: {nbit}/{len(runs)} runs bitwise in {SCAN_FIELDS} + phase (round 11: {len(ref['runs'])} runs); gbar {mine['gbar']!r} "
            f"gamma1 {mine['gamma1']!r} final_run {mine['final_run']} (round 11 {ref['final_run']}); top-level {top} -> {'PASS' if ok else 'FAIL'}")
    g['b'] = dict(scans=scans, pass_=bool(ok_b))
    # (c) codes and table numbers
    ok_c = True; tabs = {}
    for k, lab, npz in (('T0', 'edge_j0', 'table_edge_A0.npz'), ('T1', 'edge_j1', 'table_edge_iter.npz')):
        t = res['tables'][k]
        with np.load(os.path.join(R11, 'results', npz)) as z: ref_codes = z['codes']
        mine = np.load(t['table'])
        cb = bitwise(mine, ref_codes)
        ref = r11['tables'][lab]
        num = dict(I=(t['logdet24'], ref['I']), EN24=(t['EN24'], ref['EN24']), I120=(t['logdet120'], ref['I120']), EN120=(t['EN120'], ref['EN120']),
                   S24=(t['S24'], ref['S']), S120=(t['S120'], ref['S120']))
        rels = {n: relerr(v, r) for n, (v, r) in num.items()}
        ok = cb and all(rels[n] <= REL_12 for n in ('I', 'EN24', 'I120', 'EN120'))
        tabs[k] = dict(label=lab, archive=f'round11/results/{npz}', codes_bitwise=cb, values={n: dict(value=v, ref=r, rel=rels[n]) for n, (v, r) in num.items()},
                       pass_=bool(ok))
        ok_c &= ok
        out(f"G0(c) {k} ({lab}): codes vs round11/results/{npz} bitwise {cb}; I {t['logdet24']!r} (ref {ref['I']!r}, rel {rels['I']:.1e}), EN24 "
            f"{t['EN24']!r} (rel {rels['EN24']:.1e}), I120 {t['logdet120']!r} (ref {ref['I120']!r}, rel {rels['I120']:.1e}), EN120 {t['EN120']!r} "
            f"(ref {ref['EN120']!r}, rel {rels['EN120']:.1e}) -> {'PASS' if ok else 'FAIL'}")
    g['c'] = dict(tables=tabs, pass_=bool(ok_c))
    # (d) certificate from scan j1
    cert = res['certificates'][0]
    ref_j1 = json.load(open(os.path.join(R11, 'results', 'scan_edge_j1.json')))
    UB_arch = r11['tables']['edge_j0']['I']-9+ref_j1['min_dual']
    dd = dict(UB=cert['UB'], UB_archived=UB_arch, I_T0=cert['crit_ref'], min_dual_j1=cert['min_dual'], I_T0_archived=r11['tables']['edge_j0']['I'],
              min_dual_j1_archived=ref_j1['min_dual'], abs_diff=abs(cert['UB']-UB_arch), rel=relerr(cert['UB'], UB_arch),
              prompt_value=207.10197960571716-9+9.458505981635199, lam_argmin=cert['lam_argmin'], UBp=cert['UBp'], lam_argmin_p=cert['lam_argmin_p'])
    dd['pass_'] = bool(dd['rel'] <= REL_12)
    g['d'] = dd
    out(f"G0(d) certificate from scan edge_j1 (tangent T0): UB = I(T0) - 9 + min dual(j1) = {cert['crit_ref']!r} - 9 + {cert['min_dual']!r} = {cert['UB']!r}; "
        f"from the archived fields {UB_arch!r} (|diff| {dd['abs_diff']:.1e}, rel {dd['rel']:.1e}) -> {'PASS' if dd['pass_'] else 'FAIL'}; "
        f"UB' (B + 0.25) {cert['UBp']!r}; argmin lam {cert['lam_argmin']!r} / {cert['lam_argmin_p']!r}")
    g['design'] = dict(T_star=res['T_star'], certificate=res['certificate'], checks=res['checks'], wall_s=res['wall_s'],
                       best=dict((k, res['best'][k]) for k in ('label', 'logdet24', 'EN24', 'logdet120', 'EN120', 'S24', 'S120')))
    g['pass_'] = bool(g['b']['pass_'] and g['c']['pass_'] and g['d']['pass_'])
    return g, res


# --- G1: registration ------------------------------------------------------------------------------------------------------------
def gate_G1(out):
    xk, Hk = dr.input_arrays(NAME)
    d = dict(XK_vs_input_arrays=bitwise(e13.XK, xk), HK_vs_input_arrays=bitwise(e13.HK, Hk))
    src = dict(input_arrays=f"dr.input_arrays('{NAME}')")
    if NAME == 'P5514':
        xd, Hd = dr.input_arrays('D')
        d.update(XK_vs_config_D=bitwise(e13.XK, xd), HK_vs_config_D=bitwise(e13.HK, Hd))
        cand = ew.CFG2A['candidates'][0]
        d.update(a_chi_from_config_recheck=bool(dr.CFG['inputs']['P5514']['a'] == cand['a'] and dr.CFG['inputs']['P5514']['chi'] == cand['chi']),
                 registered_by='round4/evaluator_window.py (in memory)', not_reregistered=bool('P5514' not in e13.REGISTERED13))
        src.update(config_D="dr.input_arrays('D') (config.json)")
    else:
        d.update(not_reregistered=bool(NAME not in e13.REGISTERED13), registered_by='config.json')
    res = dict(checks=d, sources=src, pass_=all(v for v in d.values() if isinstance(v, bool)))
    out(f"G1 {NAME}: eval13.XK, HK (module grid of R13_INPUT={NAME}) bitwise: {d} -> {'PASS' if res['pass_'] else 'FAIL'}")
    return res


# --- G3: refined grid for LO, and one DP run ------------------------------------------------------------------------------------
def gate_G3(out):
    t0 = time.perf_counter(); g = e13.fine_grid(5); tg = time.perf_counter()-t0      # raises BoundaryNode on a cell-boundary node
    cell = g['cell']
    d = dict(no_BoundaryNode=True, zf0_equals_Z0=bool(float(g['z'][0]) == e13.Z0), zf0=float(g['z'][0]), Z0=e13.Z0, Gf=int(g['G']), G=int(G),
             cell_min=int(cell.min()), cell_max=int(cell.max()), cell_in_range=bool(int(cell.min()) >= 0 and int(cell.max()) <= G-1),
             boundary_gap=float(g['gap']), fine_grid_s=tg,
             er_default_z0=None)
    sys.path.insert(0, R7); import evaluator_refine as er
    d['er_default_z0'] = float(er.Z0); d['er_default_h'] = float(er.H24); d['own_dz'] = float(e13.DZ)
    out(f"G3 {NAME}: fine_grid(5) built without BoundaryNode in {tg:.1f} s; zf[0] == Z0: {d['zf0_equals_Z0']} ({d['zf0']!r}; round 7 default z0 "
        f"(input U) {d['er_default_z0']!r}); Gf {d['Gf']}; coarse cells [{d['cell_min']}, {d['cell_max']}] within [0, {G-1}]: {d['cell_in_range']}; "
        f"smallest distance to a cell boundary {d['boundary_gap']!r} cell widths")
    r24, J24, fw24 = c13.eval_const(480, 1)
    r120, J120, fw120 = c13.eval_const(480, 5)
    d.update(EN24=r24['EN'], logdet24=r24['logdet'], S24=r24['S'], mass_dev24=r24['mass_dev_max'], min_m24=r24['min_m'], wall24_s=r24['wall_s'],
             EN120=r120['EN'], logdet120=r120['logdet'], S120=r120['S'], mass_dev120=r120['mass_dev_max'], min_m120=r120['min_m'], wall120_s=r120['wall_s'],
             mass24_ok=bool(r24['mass_dev_max'] <= 1e-12), mass120_ok=bool(r120['mass_dev_max'] <= 1e-10))
    d['pass_'] = bool(d['zf0_equals_Z0'] and d['cell_in_range'] and d['mass24_ok'] and d['mass120_ok'])
    out(f"G3 {NAME} ConstRows(480): sigma/24 EN {r24['EN']!r} logdet {r24['logdet']!r} S {r24['S']!r} mass dev {r24['mass_dev_max']:.2e} ({r24['wall_s']:.1f} s); "
        f"sigma/120 EN {r120['EN']!r} logdet {r120['logdet']!r} S {r120['S']!r} mass dev {r120['mass_dev_max']:.2e} ({r120['wall_s']:.1f} s) "
        f"-> {'PASS' if d['pass_'] else 'FAIL'}")
    return d, (r24, J24)

def timing_dp(des, out, const24=None):
    """wall time of one DP run for this input (edge action set, tangent at the constant SoD, lambda = 5 gbar) and the sigma/24 forward of
    its table (diagnostic); for a non-U input also the accepted operator against eval13 on ConstRows(480) (diagnostic)."""
    rec_sod, J_sod, W_sod = des.sod_reference()
    d = dict(sod_reference=dict(EN=rec_sod['EN'], logdet=rec_sod['logdet'], S=rec_sod['S'], path=rec_sod.get('path')))
    if const24 is not None:
        r24, J24 = const24
        d['sod_reference'].update(EN_rel_vs_eval13=relerr(r24['EN'], rec_sod['EN']), logdet_rel_vs_eval13=relerr(r24['logdet'], rec_sod['logdet']))
        out(f"DIAG {NAME}: accepted operator at Delta* EN {rec_sod['EN']!r} logdet {rec_sod['logdet']!r}; eval13 ConstRows(480) rel "
            f"{d['sod_reference']['EN_rel_vs_eval13']:.2e} / {d['sod_reference']['logdet_rel_vs_eval13']:.2e}")
    acts = e13.edge_set(); gam, gbar, g1 = des.gamma_from_J(J_sod); lam = 5.0*gbar
    codes = np.empty((K-1, G), np.uint16)
    t0 = time.perf_counter(); dsum, nbad = des.dp(acts, gam, lam, codes); tdp = time.perf_counter()-t0
    tab = e13.Table11(codes, acts, 'timing')
    t0 = time.perf_counter(); rec, J, fw = e13.evaluate(tab, gam=gam); tfw = time.perf_counter()-t0
    F = float(gam[1:]@fw['W'][1:])
    d.update(lam=lam, lam_over_gbar=5.0, gbar=gbar, gamma1=g1, dp_s=tdp, nonfinite_V=int(nbad), dsum=dsum, EN=rec['EN'], logdet=rec['logdet'], S=rec['S'],
             F=F, dp_minus_policy_rel=abs(dsum-(F-lam*(rec['EN']-1)))/abs(F-lam*(rec['EN']-1)), fwd24_s=tfw, mass_dev_max=rec['mass_dev_max'],
             numba_threads=numba.get_num_threads())
    out(f"TIMING {NAME}: one DP run (edge set, tangent at SoD, lam = 5 gbar = {lam!r}) {tdp:.1f} s ({numba.get_num_threads()} numba threads); "
        f"its table on sigma/24: EN {rec['EN']!r} logdet {rec['logdet']!r} ({tfw:.1f} s); DP value vs policy rel {d['dp_minus_policy_rel']:.1e}")
    return d


# --- stages ---------------------------------------------------------------------------------------------------------------------------
def stage_gate(out):
    path = os.path.join(RES, f'gate_{NAME}.json')
    res = dict(input=NAME, versions=versions(), registered13=e13.REGISTERED13, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    def save(): jdump(res, path)
    t0 = time.perf_counter(); dp13.warmup(); res['warmup_s'] = time.perf_counter()-t0
    out(f"WARMUP (JIT / cache) {res['warmup_s']:.1f} s; registered in memory from inputs13.json: {e13.REGISTERED13}")
    res['G4'] = gate_G4(out); save()
    if not res['G4']['pass_']: raise Stop(f"G4 failed for {NAME}")
    if NAME == 'U':
        res['G0a'] = gate_G0a(out); save()
        if not res['G0a']['pass_']: raise Stop("G0(a) failed")
        des = c13.get_design(out)
        res['G2'] = gate_G2(des, out); save()
        if not res['G2']['pass_']: raise Stop("G2 failed")
        g, dres = gate_G0bcd(des, out)
        res['G0bcd'] = g
        res['timing'] = dict(dp_s_runs=g['b']['scans']['j0']['dp_s']+g['b']['scans']['j1']['dp_s'],
                             dp_s_median=float(np.median(g['b']['scans']['j0']['dp_s']+g['b']['scans']['j1']['dp_s'])),
                             fwd24_s_scan_median=float(np.median(g['b']['scans']['j0']['fwd_s']+g['b']['scans']['j1']['fwd_s'])),
                             fwd24_const_s=res['G2']['sigma24']['wall_s'], fwd120_const_s=res['G2']['sigma120']['wall_s'],
                             fwd120_T0_s=dres['tables']['T0']['fwd120_s'], fwd120_T1_s=dres['tables']['T1']['fwd120_s'],
                             scan_j0_s=dres['scans']['j0']['wall_s'], scan_j1_s=dres['scans']['j1']['wall_s'], design_s=dres['wall_s'],
                             numba_threads=numba.get_num_threads())
        save()
        if not g['pass_']: raise Stop("G0(b)-(d) failed")
    elif NAME in ('P5514', 'N'):
        res['G1'] = gate_G1(out); save()
        if not res['G1']['pass_']: raise Stop(f"G1 failed for {NAME}")
    elif NAME == 'LO':
        d, c24 = gate_G3(out); res['G3'] = d; save()
        if not d['pass_']: raise Stop("G3 failed")
        des = c13.get_design(out)
        res['timing'] = timing_dp(des, out, c24)
        res['timing'].update(fwd24_const_s=d['wall24_s'], fwd120_const_s=d['wall120_s'])
        save()
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save()
    return res

def stage_gate_summary(out):
    per = {}
    for n in GATE_INPUTS:
        p = os.path.join(RES, f'gate_{n}.json')
        if not os.path.exists(p): raise Stop(f"missing {p}: run the gate stage for {n} first")
        per[n] = json.load(open(p))
    U = per['U']
    gates = {}
    gates['G0'] = dict(a=U['G0a']['pass_'], b=U['G0bcd']['b']['pass_'], c=U['G0bcd']['c']['pass_'], d=U['G0bcd']['d']['pass_'])
    gates['G0']['pass_'] = all(gates['G0'].values())
    gates['G1'] = dict(P5514=per['P5514']['G1']['pass_'], N=per['N']['G1']['pass_']); gates['G1']['pass_'] = all(gates['G1'].values())
    gates['G2'] = dict(sigma24=U['G2']['sigma24']['pass24'], sigma120=U['G2']['sigma120']['pass120'], pass_=U['G2']['pass_'])
    gates['G3'] = dict(LO=per['LO']['G3']['pass_'], pass_=per['LO']['G3']['pass_'])
    gates['G4'] = {n: per[n]['G4']['pass_'] for n in GATE_INPUTS}; gates['G4']['pass_'] = all(gates['G4'][n] for n in GATE_INPUTS)
    allp = all(gates[g]['pass_'] for g in ('G0', 'G1', 'G2', 'G3', 'G4'))
    out_ = dict(gates=gates, all_pass=allp, per_input=per, versions=versions(),
                g4_table={n: {k: per[n]['G4'][k] for k in ('sum_a2', 'sum_a2_minus_1', 'range_sigma', 'max_step_sigma', 'G', 'I_full', 'ref_I_full',
                                                             'I_full_absdiff', 'ref_G', 'G_equal', 'ref_max_step_sigma', 'ref_range_sigma', 'pass_')}
                          for n in GATE_INPUTS})
    jdump(out_, os.path.join(RES, 'gate13.json'))
    for g in ('G0', 'G1', 'G2', 'G3', 'G4'): out(f"SUMMARY {g}: {gates[g]}")
    out(f"SUMMARY all gates pass: {allp}")
    return out_


# --- DECISION 13-22: neutrality of the dp13 window-width change for U ----------------------------------------------------------------
def stage_dpfix(out):
    """(a) every Design11 constructor array of dp13 bitwise equal to dp11's (both built in this process, input U); (b) one DP run at round
    11's final lambda of scan j0 (run 10) reproduces round11/results/table_edge_A0.npz codes and the archived dsum bitwise."""
    if NAME != 'U': raise Stop("dpfix runs for input U only")
    res = dict(input=NAME, versions=versions())
    path = os.path.join(RES, 'dpfix13_U.json')
    dp13.warmup()
    sys.path.append(R11)
    import dp11                                                 # frozen round 11 (read-only)
    d13 = dp13.Design11(out)
    d11 = dp11.Design11(out)
    arr = {}
    for n in ('UC', 'PPC', 'PNC', 'PHC', 'GK', 'g1'):
        arr[n] = bitwise(getattr(d13, n), getattr(d11, n))
    for n in ('W88', 'NC', 'a1', 'b1'):
        arr[n] = bool(getattr(d13, n) == getattr(d11, n))
    arr['no_short_steps_U'] = bool(d13.short_steps == [])
    res['a'] = dict(bitwise=arr, pass_=all(arr.values()))
    out(f"DPFIX (a) dp13.Design11 vs dp11.Design11 (input U), bitwise: {arr} -> {'PASS' if res['a']['pass_'] else 'FAIL'}")
    del d11; import gc; gc.collect()
    jdump(res, path)
    sc = json.load(open(os.path.join(R11, 'results', 'scan_edge_j0.json')))
    r10 = [r for r in sc['runs'] if r['run'] == sc['final_run']][0]
    rec_sod, J_sod, W_sod = d13.sod_reference()
    gam, gbar, g1 = d13.gamma_from_J(J_sod)
    acts = e13.edge_set(); codes = np.empty((K-1, G), np.uint16)
    t0 = time.perf_counter(); dsum, nbad = d13.dp(acts, gam, r10['lam'], codes); tdp = time.perf_counter()-t0
    with np.load(os.path.join(R11, 'results', 'table_edge_A0.npz')) as z: ref = z['codes']; lam_ref = float(z['lam'])
    b = dict(lam=r10['lam'], lam_npz=lam_ref, run=r10['run'], gbar_equal=bool(gbar == sc['gbar']), gamma1_equal=bool(g1 == sc['gamma1']),
             codes_bitwise=bitwise(codes, ref), n_diff_nodes=int(np.count_nonzero(codes != ref)), dsum=dsum, dsum_ref=r10['dsum'],
             dsum_bitwise=bool(dsum == r10['dsum']), nonfinite_V=int(nbad), dp_s=tdp)
    res['b'] = dict(b, pass_=bool(b['gbar_equal'] and b['gamma1_equal'] and b['codes_bitwise'] and b['dsum_bitwise'] and lam_ref == r10['lam']))
    out(f"DPFIX (b) one DP at round 11's final lambda of scan j0 (run {r10['run']}, lam {r10['lam']!r}): codes vs table_edge_A0.npz bitwise "
        f"{b['codes_bitwise']} ({b['n_diff_nodes']} differing nodes); dsum {dsum!r} vs {r10['dsum']!r} bitwise {b['dsum_bitwise']}; gbar/gamma1 equal "
        f"{b['gbar_equal']}/{b['gamma1_equal']}; {tdp:.1f} s -> {'PASS' if res['b']['pass_'] else 'FAIL'}")
    res['pass_'] = bool(res['a']['pass_'] and res['b']['pass_'])
    jdump(res, path)
    if not res['pass_']: raise Stop("DECISION 13-22 neutrality for U failed")
    return res


# --- Part A -------------------------------------------------------------------------------------------------------------------------
I_TABLE_U_R11 = 207.35521730257756          # round11.json tables.edge_j1.I (sigma/24), Part A §4 / check 5

def info_per_send(S, EN): return (S-1.0)/(EN-1.0)

def stage_partA(out):
    import shutil
    outdir = os.path.join(RES, f'partA_{NAME}'); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(RES, f'partA_{NAME}.json')
    res = json.load(open(path)) if os.path.exists(path) else dict(input=NAME)
    res['versions'] = versions()
    def save(): jdump(res, path)
    t_all = time.perf_counter()
    t0 = time.perf_counter(); dp13.warmup(); out(f"WARMUP {time.perf_counter()-t0:.1f} s")
    # 1. full-data information and input sanity
    Jf, If = c13.full_information()
    res['I_full'] = If; res['sanity'] = c13.input_sanity(); res['J_full'] = c13.save_J(Jf, outdir, 'full')
    out(f"PARTA {NAME} I_full {If!r}; G {G}; output range {res['sanity']['range_sigma']:.3f} sigma; largest step {res['sanity']['max_step_sigma']:.4f} sigma")
    save()
    # 2. constant SoD at Delta* (accepted operator on sigma/24; eval13 ConstRows(480) on sigma/120)
    if 'sod_delta' not in res:
        r24, J24, W24 = c13.sod_delta()
        r120, J120, _ = c13.eval_const(ew.SOD_M, c13.F_REF)
        res['sod_delta'] = dict(M=ew.SOD_M, EN24=r24['EN'], S24=r24['S'], I24=r24['logdet'], path24=r24.get('path'),
                                EN120=r120['EN'], S120=r120['S'], I120=r120['logdet'], mass_dev_max120=r120['mass_dev_max'], wall120_s=r120['wall_s'],
                                J24=c13.save_J(J24, outdir, 'sod_delta_s24'), J120=c13.save_J(J120, outdir, 'sod_delta_s120'))
        save()
    sd = res['sod_delta']
    out(f"PARTA {NAME} constant SoD at Delta*: sigma/24 EN {sd['EN24']!r} S {sd['S24']!r} I {sd['I24']!r}; sigma/120 EN {sd['EN120']!r} S {sd['S120']!r} I {sd['I120']!r}")
    # 3. matched constant SoD
    if 'matched' not in res:
        m, _, _ = c13.matched_sod(B, log=out, outdir=outdir, tag='sod_matched')
        res['matched'] = m; save()
    m = res['matched']
    if not m['accepted']:
        res['status'] = 'stopped: matched SoD not accepted (|E[N]_120 - B| > 1)'; save()
        raise Stop(f"{NAME}: matched SoD not accepted, residual {m['residual120']!r}")
    # 4. designed table (checkpointed scans; for U the two completed scans of gate G0 are reused, DECISION 13-20)
    if NAME == 'U':
        for j in ('j0', 'j1'):
            src = os.path.join(RES, 'gate_U_design', f'scan_edge_{j}.json'); dst = os.path.join(outdir, f'scan_edge_{j}.json')
            if not os.path.exists(dst) and os.path.exists(src): shutil.copyfile(src, dst); out(f"PARTA U: scan edge_{j} taken from gate G0 ({src})")
    if 'design' not in res:
        des = c13.get_design(out)
        res['dp_window_fix'] = dict(decision='DECISION 13-22', W88=des.W88, short_steps=des.short_steps); save()
        dres = c13.design(e13.edge_set(), dp13.Design11.gamma_from_J, c13.logdet9, 9, 'edge', des=des, outdir=outdir,
                          work=os.path.join(WORK, NAME, 'edge'), log=out)
        del des
        res['design'] = dres; save()
    dres = res['design']
    # 5. shape statistics of T* only, after the design and its caches are released (DECISION 13-17, 13-18)
    c13.release_design()
    if 'shape' not in res or res['shape'].get('skipped') is None:
        J_sod = np.load(os.path.join(HERE, dres['sod']['J']))
        res['shape'] = c13.shape_stats(dres['best'], e13.edge_set(), J_sod, out); save()
    # 6. packed T*, derived quantities, checks
    best = dres['best']
    res['packed'] = c13.pack_table(best['table'], e13.edge_set(), os.path.join(RES, f'table_A_{NAME}.npz'), best['lam'],
                                   note=f"Round 13 Part A, input {NAME}, best table T* = {dres['T_star']} ({best['label']}).")
    It, Is = best['logdet120'], m['logdet120']
    res['derived'] = dict(grid='sigma/120', I_full=If, I_table=It, I_SoD=Is, c_tab=It-If, c_sod=Is-If, gain=It-Is,
                          S_table=best['S120'], EN_table=best['EN120'], S_SoD=m['S120'], EN_SoD=m['EN120'],
                          info_per_send_table=info_per_send(best['S120'], best['EN120']), info_per_send_SoD=info_per_send(m['S120'], m['EN120']),
                          sigma24=dict(I_table=best['logdet24'], I_SoD=m['logdet24'], c_tab=best['logdet24']-If, c_sod=m['logdet24']-If,
                                       gain=best['logdet24']-m['logdet24'], S_table=best['S24'], EN_table=best['EN24'], S_SoD=m['S24'], EN_SoD=m['EN24'],
                                       info_per_send_table=info_per_send(best['S24'], best['EN24']), info_per_send_SoD=info_per_send(m['S24'], m['EN24'])),
                          certificate_gap=dres['certificate']['gap'], certificate_gap_two_scans=dres['certificate_two_scans']['gap'], UBp=dres['certificate']['UBp'])
    ch = dres['checks']
    tabs = dres['tables']
    res['checks'] = {
        '1_forwards': dict(pass_=True, rule='asserted in every forward (conservation 1e-12 / 1e-10, min m >= 0, zero mass at the grid ends); a failure stops the run',
                           mass_dev24_max=max(ch['mass_dev24_max'], m['mass_dev_max24']), mass_dev120_max=max(ch['mass_dev120_max'], m['mass_dev_max120'], sd['mass_dev_max120'])),
        '2_scans': dict(pass_=bool(ch['scans_monotone'] and ch['forward_backward']), monotone=ch['scans_monotone'],
                        forward_backward={k: tabs[k]['check_fb'] for k in tabs}),
        '3_table': dict(pass_=bool(ch['budget120_best'] and ch['crit_le_UBp']), EN120_best=best['EN120'], budget120=ch['budget120_best'],
                        I_le_UBp=ch['crit_le_UBp'], UBp=dres['certificate']['UBp'], I_tables={k: tabs[k]['crit24'] for k in tabs}),
        '4_matched': dict(pass_=bool(m['accepted']), residual120=m['residual120']),
    }
    if NAME == 'U':
        res['checks']['5_U_round11'] = dict(pass_=bool(best['logdet24'] == I_TABLE_U_R11), I_table24=best['logdet24'], ref=I_TABLE_U_R11)
    res['all_checks_pass'] = all(v['pass_'] for v in res['checks'].values())
    res['status'] = 'done'; res['wall_s_this_run'] = time.perf_counter()-t_all
    save()
    d = res['derived']
    out(f"PARTA {NAME} DONE: I_full {If:.4f}; matched SoD M* {m['M_star']} I_SoD {Is:.4f} (EN120 {m['EN120']:.3f}); T* {best['label']} I_table {It:.4f} "
        f"(sigma/24 {best['logdet24']:.4f}, EN120 {best['EN120']:.3f}, S120 {best['S120']:.2f}); c_tab {d['c_tab']:.4f} c_sod {d['c_sod']:.4f} gain {d['gain']:.4f}; "
        f"gap {d['certificate_gap']:.4f} (two scans {d['certificate_gap_two_scans']:.4f}); shape {'skipped' if res['shape'].get('skipped') else 'done'}; "
        f"checks {'PASS' if res['all_checks_pass'] else 'FAIL'}")
    if not res['all_checks_pass']: raise Stop(f"{NAME}: a Part A check failed: {res['checks']}")
    return res


STAGES = ('gate', 'gate_summary', 'dpfix', 'partA', 'partA_summary', 'partB')

def main():
    args = sys.argv[1:]
    numba.set_num_threads(int(os.environ.get('R13_NUMBA_THREADS', '4')))
    for stg in args:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        if stg == 'partB': raise SystemExit("stage partB is not implemented (Part B)")
        out = Tee(os.path.join(LOGS, f'{stg}_{NAME}.log' if stg in ('gate', 'partA', 'dpfix') else f'{stg}.log'))
        out(f"=== ROUND 13 stage {stg} input {NAME} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, numba {numba.__version__} ({numba.get_num_threads()} threads); B = {B!r}; G = {G}; work {WORK}")
        t0 = time.perf_counter()
        try:
            if stg == 'gate': stage_gate(out)
            elif stg == 'gate_summary': stage_gate_summary(out)
            elif stg == 'partA': stage_partA(out)
            elif stg == 'dpfix': stage_dpfix(out)
            elif stg == 'partA_summary':
                import partA13; partA13.stage(out)
        except (Stop, e13.AssertionStop, AssertionError) as e:
            out(f"STOP {type(e).__name__}: {e}")
            out(f"=== ROUND 13 stage {stg} input {NAME} STOPPED after {time.perf_counter()-t0:.1f} s")
            cleanup_frozen(out)
            raise
        out(f"=== ROUND 13 stage {stg} input {NAME} done in {time.perf_counter()-t0:.1f} s")
        cleanup_frozen(out)


if __name__ == '__main__':
    main()
