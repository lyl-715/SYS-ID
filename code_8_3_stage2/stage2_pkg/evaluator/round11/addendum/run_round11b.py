"""Round 11b (addendum) driver: the dual bound UB_SoD over the larger SoD class at the round-11 tangent point A_1 = J(T_1)^-1, the
budget convention B' = B + 0.1, the primal table T' of the scan, and checks 1-6 (prompt "Round 11b", §3-§4).

Class: the per-node union S u S' (DECISION 11b-2, chosen by the user after the predictions commit: S' as listed in DECISION 11b-1
does not contain S). Rounds 4-10 and round11/dp11.py, eval11.py (and the other round-11 files) are read-only; new code is in
round11/addendum/ (sprime.py, this file). Run tables live in $ROUND11B_WORK (default /home/user/round11b_work, not committed).

Stages (each appends to logs/run_round11b.log):
  inputs   T_1 from round11/results/table_edge_iter.npz; I(T_1) and gamma (A_1 = J(T_1)^-1) recomputed with eval11 / dp11 and
           checked bit for bit against round11.json and scan_edge_j2.json (gbar, gamma_1); the union set reproduces T_1's windows
           with its round-11 codes; edge-node checks; share of T_1's mass on windows that S' alone cannot represent.
  regress  check 1: the S-block code path (round-11 edge set, 2 sigma) reproduces the first two runs of scan_edge_j2 bit for bit.
  scan     the price scan over S u S' (resumable from results/archive_union.jsonl).
  post     per-lambda table, UB_SoD for B and B' (and B + delta), T' (sigma/24, sigma/120, forward = backward), checks 1-6,
           predictions, results/round11b.json, table_edge_Sprime.npz if T' is reported.
Usage (from this directory):  ROUND11B_WORK=/path python3 run_round11b.py inputs regress scan post
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); R11 = os.path.dirname(HERE); EV = os.path.dirname(R11)
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
import sprime as sp
e11, dp11 = sp.e11, sp.dp11
s, K, B, G = sp.s, sp.K, sp.B, sp.G

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
for _d in (LOGS, RES): os.makedirs(_d, exist_ok=True)
WORK = os.environ.get('ROUND11B_WORK', '/home/user/round11b_work'); os.makedirs(WORK, exist_ok=True)
R11RES = os.path.join(R11, 'results')
P = 9
I_T1_PROMPT = 207.35521730257756
GAMMA1_PROMPT = 0.009447863192807387
DELTA_B = 0.1                     # B' = B + 0.1 (prompt §3 B)
DELTAS = (0.1, 0.25, 0.5)         # UB(B + delta) for the budget-convention discussion (DECISION 11b-6)
ACCEPT_T = 0.05                   # report T' as the new best table iff I(T') > I(T_1) + 0.05 (prompt §3 C)
FB_TOL = 1e-12                    # check 2
MONO_TOL = 1e-9                   # check 4 (D)


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

class Stop(RuntimeError):
    """a reproduction, regression or check failure: the run stops (no fallback)."""

def jdump(obj, path):
    tmp = path+'.tmp'; json.dump(obj, open(tmp, 'w'), indent=1, default=_jdef); os.replace(tmp, path)

def _jdef(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))

def state_path(stage): return os.path.join(RES, f'state_{stage}.json')

def frozen_clean(out):
    """remove empty untracked folders created by imports under round4..round10 and confirm git sees no change in the read-only
    directories (round4..round10, round11 outside addendum/)."""
    removed = []
    for rd in ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10'):
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_): os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    paths = [os.path.join(EV, r) for r in ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10')]
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + paths + [R11, ':(exclude)'+HERE], capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on round4..round10 and round11 (outside addendum/): {'clean' if not st else st}")
    return st == ''


# --- shared setup ------------------------------------------------------------------------------------------------------------
_DES = None
def design(out):
    global _DES
    if _DES is None:
        t0 = time.perf_counter(); sp.warmup(); out(f"WARMUP {time.perf_counter()-t0:.1f} s")
        _DES = sp.DesignG(out)
    return _DES

def archived():
    r11 = json.load(open(os.path.join(R11RES, 'round11.json')))
    sc = json.load(open(os.path.join(R11RES, 'scan_edge_j2.json')))
    return dict(I_T1=r11['tables']['edge_j1']['I'], EN24_T1=r11['tables']['edge_j1']['EN24'], EN120_T1=r11['tables']['edge_j1']['EN120'],
                gbar=sc['gbar'], gamma1=sc['gamma1'], min_dual_S=sc['min_dual'], ub_sod_S=r11['ub_sod']['UB_SoD'],
                runs_S=sc['runs'], r11=r11,
                src=dict(I_T1='round11/results/round11.json["tables"]["edge_j1"]["I"]', gbar='round11/results/scan_edge_j2.json["gbar"]',
                         gamma1='round11/results/scan_edge_j2.json["gamma1"]', min_dual_S='round11/results/scan_edge_j2.json["min_dual"]',
                         ub_sod_S='round11/results/round11.json["ub_sod"]["UB_SoD"]'))

def load_T1():
    f = np.load(os.path.join(R11RES, 'table_edge_iter.npz'))
    assert str(f['action_set']) == 'edge' and int(f['round']) == 1
    return f['codes'], float(f['lam'])

def gamma_A1(out, des):
    """gamma at A_1 = J(T_1)^-1: J(T_1) from eval11 on sigma/24 (T_1 as round-11 Table11), then dp11.Design11.gamma_from_J."""
    gpath = os.path.join(WORK, 'gamma_A1.npy')
    st = json.load(open(state_path('inputs')))
    gam = np.load(gpath)
    assert float(gam[1:].mean()) == st['gbar'] and float(gam[0]) == st['gamma1']
    return gam, st['gbar'], st['gamma1']


# --- stage inputs ------------------------------------------------------------------------------------------------------------
def stage_inputs(out):
    A = archived(); des = design(out)
    for k in ('I_T1', 'gbar', 'gamma1', 'min_dual_S', 'ub_sod_S'): out(f"INPUT {k} = {A[k]!r}   [{A['src'][k]}]")
    assert A['I_T1'] == I_T1_PROMPT and A['gamma1'] == GAMMA1_PROMPT
    codes, lam1 = load_T1()
    t1_r11 = e11.Table11(codes, e11.edge_set(), 'T1')
    t0 = time.perf_counter(); rec, J, fw = e11.evaluate(t1_r11); t_fw = time.perf_counter()-t0
    out(f"T1 eval11 sigma/24: E[N] {rec['EN']!r}, log det {rec['logdet']!r} (archived {A['I_T1']!r}, equal {rec['logdet'] == A['I_T1']}); "
        f"E[N] archived {A['EN24_T1']!r} (equal {rec['EN'] == A['EN24_T1']}); {t_fw:.1f} s")
    if not (rec['logdet'] == A['I_T1'] and rec['EN'] == A['EN24_T1']): raise Stop("I(T_1) / E[N](T_1) not reproduced bit for bit")
    gam, gbar, g1 = des.gamma_from_J(J)
    ok_g = (gbar == A['gbar'] and g1 == A['gamma1'])
    out(f"GAMMA A_1 = J(T_1)^-1: gbar {gbar!r} (scan_edge_j2 {A['gbar']!r}), gamma_1 {g1!r} (scan_edge_j2 {A['gamma1']!r}); bit for bit {ok_g}")
    if not ok_g: raise Stop("gamma at A_1 differs from the round-11 edge round-2 scan")
    np.save(os.path.join(WORK, 'gamma_A1.npy'), gam); np.save(os.path.join(WORK, 'J_T1.npy'), J)
    np.save(os.path.join(RES, 'gamma_A1.npy'), gam)

    # action sets: edge nodes, union reproduces T_1's windows with its round-11 codes
    acts = {'S': sp.set_S(), "S'": sp.set_Sprime(), 'union': sp.set_union()}
    ec = {k: a.edge_check() for k, a in acts.items()}
    for k, v in ec.items(): out(f"EDGES {k} ({acts[k].n} actions): {v}")
    L1 = np.empty(G, np.int64); U1 = np.empty(G, np.int64); L2 = np.empty(G, np.int64); U2 = np.empty(G, np.int64)
    L3 = np.empty(G, np.int64); U3 = np.empty(G, np.int64)
    tS = sp.TableG(codes, acts['S']); tU = sp.TableG(codes, acts['union']); nd = 0
    for p in range(1, K):
        t1_r11.rows_full(p, (L1, U1)); tS.rows_full(p, (L2, U2)); tU.rows_full(p, (L3, U3))
        nd += int((L1 != L2).sum()+(U1 != U2).sum()+(L1 != L3).sum()+(U1 != U3).sum())
    out(f"ROWS T_1 codes: round-11 Table11 vs TableG(S) vs TableG(S u S') half-widths at every (k, i) of the full grid: {nd} differences")
    if nd: raise Stop("the generic rows do not reproduce the round-11 rows")

    # share of T_1's mass on windows that S' alone cannot represent (S' L options at node i: 24 or i - LN'_k(e), e in E' finite)
    a2 = acts["S'"]; fin = a2.am == 1
    acc = dict(mass=0.0, mass_lo=0.0, mass_hi=0.0, mass_any=0.0, info=0.0, info_any=0.0)
    def stats(p, m, qn, Mm, Mp, lo, hi, c):
        i = np.arange(lo, hi+1); mm = m[lo:hi+1]; L = Mm[lo:hi+1]; U = Mp[lo:hi+1]
        okL = (L == 24) | np.isin(i-L, a2.LN[p][fin])
        okU = (U == 24) | np.isin(i+U, a2.UN[p][fin])
        acc['mass'] += float(mm.sum()); acc['mass_lo'] += float(mm[~okL].sum()); acc['mass_hi'] += float(mm[~okU].sum())
        bad = ~(okL & okU); acc['mass_any'] += float(mm[bad].sum()); acc['info'] += float(mm@c); acc['info_any'] += float(mm[bad]@c[bad])
    t0 = time.perf_counter(); rec2, J2, fw2 = e11.evaluate(t1_r11, stats=stats)
    assert rec2['logdet'] == rec['logdet']
    memb = dict(mass_share_lower_not_in_Sprime=acc['mass_lo']/acc['mass'], mass_share_upper_not_in_Sprime=acc['mass_hi']/acc['mass'],
                mass_share_window_not_in_Sprime=acc['mass_any']/acc['mass'], info_share_window_not_in_Sprime=acc['info_any']/acc['info'])
    out(f"T_1 vs S' alone: share of the mass (k >= 2) on windows S' cannot represent: lower edge {memb['mass_share_lower_not_in_Sprime']!r}, "
        f"upper edge {memb['mass_share_upper_not_in_Sprime']!r}, window {memb['mass_share_window_not_in_Sprime']!r}; information share "
        f"{memb['info_share_window_not_in_Sprime']!r}; {time.perf_counter()-t0:.1f} s")
    st = dict(I_T1=rec['logdet'], EN24_T1=rec['EN'], lam_T1=lam1, gbar=gbar, gamma1=g1, gamma_sum=float(gam.sum()), lam0=5*gbar,
              actions={k: a.describe() for k, a in acts.items()}, edge_checks=ec, rows_differences=nd, T1_vs_Sprime=memb,
              S_subset_union=True, Sprime_subset_union=True, archived={k: A[k] for k in ('I_T1', 'gbar', 'gamma1', 'min_dual_S', 'ub_sod_S')},
              sources=A['src'])
    jdump(st, state_path('inputs'))
    return st


# --- stage regress (check 1) ---------------------------------------------------------------------------------------------------
REG_FIELDS = ('run', 'phase', 'lam', 'EN', 'N', 'F', 'dsum', 'S', 'logdet', 'dual', 'feasible')

def stage_regress(out):
    A = archived(); des = design(out); gam, gbar, g1 = gamma_A1(out, des)
    arch = os.path.join(WORK, 'archive_regress_S.jsonl')
    if os.path.exists(arch): os.remove(arch)
    sc = sp.ScanG(des, sp.set_S(), gam, gbar, g1, 'regress_S', WORK, out, arch, max_runs=2)
    t0 = time.perf_counter(); st = sc.scan(); wall = time.perf_counter()-t0
    sc.cleanup(keep=[])
    rows = []; ok = True
    for r, ref in zip(st['runs'], A['runs_S'][:2]):
        eq = {f: bool(r[f] == ref[f]) for f in REG_FIELDS}
        ok &= all(eq.values()) and len(st['runs']) == 2
        rows.append(dict(run=r['run'], fields={f: dict(new=r[f], round11=ref[f]) for f in REG_FIELDS}, equal=eq, dp_s=r['dp_s']))
        out(f"CHECK 1 run {r['run']}: " + ", ".join(f"{f} {'=' if v else '!='}" for f, v in eq.items()) +
            f"  (lam {r['lam']!r}, EN {r['EN']!r}, F {r['F']!r}, V {r['dsum']!r})")
    res = dict(n_runs=len(st['runs']), status=st['status'], rows=rows, bit_for_bit=bool(ok), wall_s=wall,
               note='first two lambda of the round-11 edge round-2 scan only (prompt §4 item 1)')
    out(f"CHECK 1 regression (S block, 2 sigma, first two lambda of scan_edge_j2): {'PASS' if ok else 'FAIL'}; scan status {st['status']}; {wall:.1f} s")
    jdump(res, state_path('regress'))
    if not ok: raise Stop("check 1 (regression) failed")
    return res


# --- stage scan ------------------------------------------------------------------------------------------------------------------
def stage_scan(out):
    des = design(out); gam, gbar, g1 = gamma_A1(out, des)
    acts = sp.set_union()
    reg = json.load(open(state_path('regress')))
    if not reg['bit_for_bit']: raise Stop("check 1 has not passed: no scan")
    arch = os.path.join(RES, 'archive_union.jsonl')
    out(f"SCAN union: actions {acts.n} ({acts.describe()['blocks']}); gbar {gbar!r} gamma_1 {g1!r}; lambda_0 = 5 gbar = {5*gbar!r}; archive {arch}")
    sc = sp.ScanG(des, acts, gam, gbar, g1, 'union', WORK, out, arch)
    t0 = time.perf_counter(); st = sc.scan(); st['wall_s_this_invocation'] = time.perf_counter()-t0; st['n_resumed'] = sc.n_resumed
    fin = None
    if st.get('final_run'):
        fin = [r for r in st['runs'] if r['run'] == st['final_run']][0]
        assert sc.best is not None and sc.best[2] == fin['table'] and os.path.exists(fin['table'])
        sc.cleanup(keep=[fin['table']])
    st['actions_describe'] = acts.describe()
    jdump(st, os.path.join(RES, 'scan_union.json'))
    out(f"SCAN union done: status {st['status']}, {len(st['runs'])} runs ({sc.n_resumed} resumed), final run {st.get('final_run')}, "
        f"min dual {st.get('min_dual')!r}")
    return st


# --- stage post ---------------------------------------------------------------------------------------------------------------
def stage_post(out):
    A = archived(); des = design(out); gam, gbar, g1 = gamma_A1(out, des)
    inp = json.load(open(state_path('inputs'))); reg = json.load(open(state_path('regress')))
    st = json.load(open(os.path.join(RES, 'scan_union.json')))
    if st['status'] != 'done': raise Stop(f"scan status {st['status']}")
    acts = sp.set_union(); I1 = A['I_T1']; Bm1 = B-1.0
    res = dict(class_=dict(name="S u S'", description=acts.describe(), chosen='user, 2026-11-07, after the predictions commit (DECISION 11b-2)'),
               inputs=inp, check1=reg)

    # A: per-lambda table, UB for B, B' and B + delta
    rows = []
    for r in st['runs']:
        D = g1+r['dsum']+r['lam']*Bm1
        row = dict(run=r['run'], phase=r['phase'], lam=r['lam'], lam_over_gbar=r['lam_over_gbar'], EN=r['EN'], F=r['F'], V=r['dsum'],
                   D=D, D_Bprime=D+r['lam']*DELTA_B, dual_record=r['dual'], dual_absdiff=abs(D-r['dual']), feasible=r['feasible'],
                   logdet=r['logdet'], UB=I1-P+D, UB_Bprime=I1-P+D+r['lam']*DELTA_B)
        rows.append(row)
        out(f"PERLAM run {r['run']:02d} {r['phase']:<6} lam {r['lam']!r} lam/gbar {r['lam_over_gbar']!r} EN {r['EN']!r} V {r['dsum']!r} "
            f"D {D!r} D+0.1lam {row['D_Bprime']!r} feasible {r['feasible']} logdet {r['logdet']!r}")
    bB = min(rows, key=lambda x: x['D']); bBp = min(rows, key=lambda x: x['D_Bprime'])
    UB = I1-P+bB['D']; UBp = I1-P+bBp['D_Bprime']
    ubd = {}
    for d in DELTAS:
        bd = min(rows, key=lambda x: x['D']+x['lam']*d)
        ubd[str(d)] = dict(delta=d, UB=I1-P+bd['D']+bd['lam']*d, argmin_run=bd['run'], argmin_lam=bd['lam'], added=I1-P+bd['D']+bd['lam']*d-UB)
    gap = UB-I1; gapp = UBp-I1
    ub = dict(I_T1=I1, p=P, gamma1=g1, B=B, Bprime=B+DELTA_B, UB=UB, gap=gap, deff_lower=float(np.exp(-gap/P)), argmin_run=bB['run'],
              argmin_lam=bB['lam'], argmin_lam_over_gbar=bB['lam_over_gbar'], argmin_phase=bB['phase'],
              UB_Bprime=UBp, gap_Bprime=gapp, deff_lower_Bprime=float(np.exp(-gapp/P)), argmin_run_Bprime=bBp['run'],
              argmin_lam_Bprime=bBp['lam'], Bprime_added=UBp-UB, UB_B_plus_delta=ubd, n_runs=len(rows),
              formula_Bprime='UB(B + 0.1) = I(T_1) - p + min_lambda [gamma_1 + V_lambda + lambda (B - 1) + 0.1 lambda]',
              UB_S_round11=A['ub_sod_S'], UB_minus_UB_S=UB-A['ub_sod_S'])
    out(f"UB_SoD(S u S'; B) = {UB!r} (run {bB['run']}, {bB['phase']}, lam {bB['lam']!r}); gap {gap!r}; exp(-gap/9) {ub['deff_lower']!r}")
    out(f"UB_SoD(S u S'; B') = {UBp!r} (run {bBp['run']}, lam {bBp['lam']!r}); gap {gapp!r}; exp(-gap/9) {ub['deff_lower_Bprime']!r}; added {UBp-UB!r}")
    for k, v in ubd.items(): out(f"UB_SoD(S u S'; B + {k}) = {v['UB']!r} (run {v['argmin_run']}); added {v['added']!r}")
    out(f"round-11 UB_SoD(S) = {A['ub_sod_S']!r}; UB(S u S') - UB(S) = {UB-A['ub_sod_S']!r}")
    res['per_lambda'] = rows; res['ub'] = ub

    # C: primal table T'
    fin = [r for r in st['runs'] if r['run'] == st['final_run']][0]
    codes = np.load(fin['table'], mmap_mode='r'); tab = sp.TableG(codes, acts, 'Tprime')
    blk = dict(mass=0.0, mass_Sprime_block=0.0, mass_lowmargin=0.0)
    def stats(p, m, qn, Mm, Mp, lo, hi, c):
        mm = m[lo:hi+1]; cd = np.asarray(codes[p-1, lo:hi+1]); spb = cd >= acts.boff[1]
        blk['mass'] += float(mm.sum()); blk['mass_Sprime_block'] += float(mm[spb].sum())
        blk['mass_lowmargin'] += float(mm[(Mm[lo:hi+1] < 48) | (Mp[lo:hi+1] < 48)].sum())
    t0 = time.perf_counter(); rec, J, fw = e11.evaluate(tab, gam=gam, stats=stats); t24 = time.perf_counter()-t0
    if not (rec['EN'] == fin['EN'] and rec['logdet'] == fin['logdet']): raise Stop("T' sigma/24 re-evaluation differs from its scan record")
    F = float(gam[1:]@fw['W'][1:]); N = rec['EN']-1.0
    t0 = time.perf_counter(); bw = e11.backward(tab, gam); tbw = time.perf_counter()-t0
    c2 = dict(F_forward=F, F_backward=bw['Fc'], F_rel=abs(bw['Fc']-F)/abs(F), N_forward=N, N_backward=bw['Nq'], N_rel=abs(bw['Nq']-N)/abs(N),
              total_forward=F-fin['lam']*N, total_backward=bw['Fc']-fin['lam']*bw['Nq'], dp_value=fin['dsum'], bw_s=tbw)
    c2['total_rel'] = abs(c2['total_backward']-c2['total_forward'])/abs(c2['total_forward'])
    c2['dp_minus_forward_abs'] = fin['dsum']-c2['total_forward']
    c2['pass_'] = bool(c2['F_rel'] <= FB_TOL and c2['N_rel'] <= FB_TOL and c2['total_rel'] <= FB_TOL)
    out(f"CHECK 2 T' forward vs fixed-policy backward: F rel {c2['F_rel']!r}, N rel {c2['N_rel']!r}, F - lam N rel {c2['total_rel']!r} "
        f"(<= {FB_TOL}) -> {'PASS' if c2['pass_'] else 'FAIL'}; diagnostic DP value - forward total {c2['dp_minus_forward_abs']!r}; {tbw:.1f} s")
    t0 = time.perf_counter(); r120, J120, fw120 = e11.evaluate_refined(tab, 5); t120 = time.perf_counter()-t0
    Ip = rec['logdet']; dIp = Ip-I1
    tp = dict(run=fin['run'], lam=fin['lam'], lam_over_gbar=fin['lam_over_gbar'], EN24=rec['EN'], I24=Ip, S24=rec['S'], F=F,
              EN120=r120['EN'], I120=r120['logdet'], S120=r120['S'], d1=rec['EN']-r120['EN'], budget120_pass=bool(r120['EN'] <= B),
              slack120=B-r120['EN'], mass_dev_max24=rec['mass_dev_max'], min_m24=rec['min_m'], mass_dev_max120=r120['mass_dev_max'],
              min_m120=r120['min_m'], improvement_over_T1=dIp, reported=bool(Ip > I1+ACCEPT_T), gap_to_UB=UB-Ip,
              deff_lower_vs_UB=float(np.exp(-(UB-Ip)/P)), gap_to_UB_Bprime=UBp-Ip,
              mass_share_Sprime_block=blk['mass_Sprime_block']/blk['mass'], mass_share_margin_below_2sigma=blk['mass_lowmargin']/blk['mass'],
              wall24_s=t24, wall120_s=t120)
    out(f"T' (run {fin['run']}, lam {fin['lam']!r}): sigma/24 E[N] {rec['EN']!r} I {Ip!r}; sigma/120 E[N] {r120['EN']!r} I {r120['logdet']!r} "
        f"(d1 = E24 - E120 = {tp['d1']!r}); I(T') - I(T_1) = {dIp!r}; reported as new best (> +{ACCEPT_T}): {tp['reported']}; "
        f"mass share on S'-block actions {tp['mass_share_Sprime_block']!r}, on windows with an edge < 2 sigma from r {tp['mass_share_margin_below_2sigma']!r}")
    res['Tprime'] = tp; res['check2'] = c2
    if tp['reported']:
        path = os.path.join(RES, 'table_edge_Sprime.npz'); codes_a = np.asarray(codes)
        np.savez_compressed(path, codes=codes_a, action_set=np.array(acts.name), blocks=np.array(json.dumps(acts.describe()['blocks'])),
                            am=acts.am, bmin=acts.bmin, minv=acts.minv, blo=acts.blo, bhi=acts.bhi, boff=acts.boff, LN=acts.LN, UN=acts.UN,
                            side_labels=np.array(acts.side_labels), z0=sp.Z0, dz=sp.DZ, G=G, lam=fin['lam'], run=fin['run'],
                            note=np.array("codes[r, i]: action of step k = r + 2 (array index p = r + 1) at sigma/24 node i (full grid). Decode: block b "
                                          "= largest b with boff[b] <= code, n_b = bhi[b] - blo[b], j_lo = blo[b] + (code - boff[b]) // n_b, j_hi = "
                                          "blo[b] + (code - boff[b]) % n_b; L = max(am[j_lo] (i - LN[p, j_lo]) + bmin[j_lo], minv[j_lo]), U = "
                                          "max(am[j_hi] (UN[p, j_hi] - i) + bmin[j_hi], minv[j_hi]); window (r_i - L dz, r_i + U dz). Class S u S' "
                                          "(round 11b)."))
        chk = np.load(path); same = bool(np.array_equal(chk['codes'], codes_a))
        res['packed'] = dict(path=os.path.relpath(path, HERE), bytes=os.path.getsize(path), roundtrip_equal=same)
        out(f"PACK {res['packed']}")

    # D: monotonicity of the minimum dual, plus pointwise comparisons at identical lambda (supplementary)
    minD = min(r['D'] for r in rows); minDS = A['min_dual_S']
    c4 = dict(min_D_union=minD, min_D_S=minDS, diff=minD-minDS, pass_=bool(minD >= minDS-MONO_TOL))
    lamS = {r['lam']: r for r in A['runs_S']}
    pw = [dict(lam=r['lam'], run_union=r['run'], run_S=lamS[r['lam']]['run'], V_union=r['V'], V_S=lamS[r['lam']]['dsum'],
               diff=r['V']-lamS[r['lam']]['dsum'], ge=bool(r['V'] >= lamS[r['lam']]['dsum'])) for r in rows if r['lam'] in lamS]
    t0 = time.perf_counter(); VS_star, nb = des.dp_g(sp.set_S(), gam, bB['lam']); tsx = time.perf_counter()-t0
    DS_star = g1+VS_star+bB['lam']*Bm1
    c4['pointwise_shared_lambda'] = pw
    c4['pointwise_at_argmin'] = dict(lam=bB['lam'], V_union=bB['V'], V_S=VS_star, D_union=bB['D'], D_S=DS_star, diff=bB['D']-DS_star,
                                     ge=bool(bB['D'] >= DS_star), dp_s=tsx)
    c4['pointwise_all_ge'] = bool(all(x['ge'] for x in pw) and c4['pointwise_at_argmin']['ge'])
    out(f"CHECK 4 (D) min D(S u S') {minD!r} >= min D(S) {minDS!r} - {MONO_TOL}: diff {minD-minDS!r} -> {'PASS' if c4['pass_'] else 'FAIL'}")
    for x in pw: out(f"CHECK 4 supplementary: same lambda {x['lam']!r} (union run {x['run_union']}, S run {x['run_S']}): V(S u S') - V(S) = {x['diff']!r}")
    out(f"CHECK 4 supplementary: S-block DP at the union argmin lambda {bB['lam']!r}: D(S u S') - D(S) = {bB['D']-DS_star!r} ({tsx:.1f} s)")

    # checks
    allfw = [r for r in st['runs']]
    rarch = os.path.join(WORK, 'archive_regress_S.jsonl')
    regfw = [json.loads(l) for l in open(rarch)] if os.path.exists(rarch) else []
    recs24 = [r['mass_dev_max'] for r in allfw+regfw]+[tp['mass_dev_max24']]; mins24 = [r['min_m'] for r in allfw+regfw]+[tp['min_m24']]
    c3 = dict(max_mass_dev24=max(recs24), min_m24=min(mins24), max_mass_dev120=tp['mass_dev_max120'], min_m120=tp['min_m120'],
              n_forwards_recorded=dict(scan=len(allfw), regression=len(regfw), Tprime_sigma24=1, Tprime_sigma120=1),
              n_forwards_unrecorded='2 sigma/24 forwards of T_1 in stage inputs (asserted; not stored), 1 forward in each of the 12 '
                                    'scan runs is the recorded one',
              note='asserted at every step of every forward by eval11.forward (|sum m - 1| <= 1e-12 on sigma/24, 1e-10 on sigma/120, '
                   'min m >= 0, zero mass at the first and last node); any failure raises AssertionStop and stops the run')
    c3['pass_'] = bool(c3['max_mass_dev24'] <= 1e-12 and c3['min_m24'] >= 0 and c3['max_mass_dev120'] <= 1e-10 and c3['min_m120'] >= 0)
    c5 = dict(UB=UB, I_T1=I1, I_Tprime=Ip, UB_ge_I_T1=bool(UB >= I1), UB_ge_I_Tprime=bool(UB >= Ip), pass_=bool(UB >= I1 and UB >= Ip))
    c6 = dict(applies=tp['reported'], EN120=tp['EN120'], B=B, pass_=(bool(tp['budget120_pass']) if tp['reported'] else None))
    checks = {'1_regression': dict(pass_=reg['bit_for_bit'], detail='state_regress.json'), '2_forward_backward': c2, '3_conservation': c3,
              '4_min_dual_monotone': c4, '5_UB_ge_I': c5, '6_budget120_Tprime': c6,
              'edges': dict(pass_=all(v['pass_'] for v in inp['edge_checks'].values()), detail=inp['edge_checks'])}
    res['checks'] = checks
    res['all_checks_pass'] = bool(all(v['pass_'] for k, v in checks.items() if v['pass_'] is not None))
    for k, v in checks.items(): out(f"CHECK {k}: {'PASS' if v['pass_'] else ('n/a' if v['pass_'] is None else 'FAIL')}")

    # B: budget-convention evidence (E24 - E120)
    r11 = A['r11']
    d1s = {lab: t['EN24']-t['EN120'] for lab, t in r11['tables'].items()}
    d1s['Tprime'] = tp['d1']
    res['budget_convention'] = dict(d1_round11_tables=d1s, max_d1_round7_partC='0.078020655082355 (round7/results/partC_regrade.json["gate"]["max_d1"])',
                                    max_d1_edge_class=max(v for k, v in d1s.items() if k.startswith('edge') or k == 'Tprime'),
                                    covers_all_seen=bool(max(d1s.values()) < DELTA_B))
    out(f"BUDGET d1 = E24 - E120: {d1s}; max over edge-class tables {res['budget_convention']['max_d1_edge_class']!r}; all < 0.1: {res['budget_convention']['covers_all_seen']}")

    # predictions
    vals = [gap, UBp-UB, dIp, bB['run']]
    last = len(rows)
    pr = [dict(n=1, text="UB_SoD(S') - I(T_1) in [0.06, 0.40] nat", value=gap, hit=bool(0.06 <= gap <= 0.40)),
          dict(n=2, text="the B' = B + 0.1 correction adds < 0.005 nat", value=UBp-UB, hit=bool(UBp-UB < 0.005)),
          dict(n=3, text="T' improves on I(T_1) by < 0.20 nat", value=dIp, hit=bool(dIp < 0.20)),
          dict(n='3b', text="(consequence) certificate gap, whichever table, < 0.40", value=max(UB-I1, UB-Ip), hit=bool(max(UB-I1, UB-Ip) < 0.40)),
          dict(n=4, text="argmin lambda at or next to the final bisection step", value=dict(argmin_run=bB['run'], n_runs=last, phase=bB['phase']),
               hit=bool(bB['run'] >= last-1 and bB['phase'] == 'bisect'))]
    res['predictions'] = pr
    for x in pr: out(f"PREDICTION {x['n']}: {x['text']}: value {x['value']!r} -> {'hit' if x['hit'] else 'miss'}")
    res['scan'] = {k: v for k, v in st.items() if k != 'runs'}
    jdump(res, os.path.join(RES, 'round11b.json'))
    out(f"WROTE results/round11b.json; all checks {'PASS' if res['all_checks_pass'] else 'FAIL'}")
    return res


STAGES = ('inputs', 'regress', 'scan', 'post')

def main():
    args = sys.argv[1:]
    numba.set_num_threads(4)
    out = Tee(os.path.join(LOGS, 'run_round11b.log'))
    for stg in args:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        out(f"=== ROUND 11b stage {stg} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, numba {numba.__version__} ({numba.get_num_threads()} threads); B = {B!r}; work {WORK}")
        t0 = time.perf_counter()
        {'inputs': stage_inputs, 'regress': stage_regress, 'scan': stage_scan, 'post': stage_post}[stg](out)
        out(f"=== ROUND 11b stage {stg} done in {time.perf_counter()-t0:.1f} s")
        frozen_clean(out)


if __name__ == '__main__':
    main()
