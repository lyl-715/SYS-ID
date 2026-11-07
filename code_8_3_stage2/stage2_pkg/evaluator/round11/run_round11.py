"""Round 11 driver (spec reports_CN/plan_round11_edge_actions_EN.md, frozen 2026-11-09; execution prompt
reports_CN/prompt_round11_claude_code_EN.md). Rounds 4-10 are imported read-only; nothing there is changed.

Stages (each appends to logs/run_round11.log and writes its state to results/state_<stage>.json; tables of the DP runs live in
the work directory $ROUND11_WORK, default round11/work/, which is not committed):
  inputs  §1   archived inputs at full precision; reproduction of I_SoD, B (constant SoD, accepted operator), I_old and E[N]_old
               (round-4 table through the frozen general path), relative error <= 1e-9; eval11 on the round-4 table bitwise
               against the frozen path; the per-step DP arrays bitwise against full scipy arrays; sigma/120 eval11 on the round-4
               table against round 7's archived f = 5 value.
  gate   §2.4  edge-code path (dp11) with the OLD action set, A_0, nominal scan: every run against round4/results/scan_U_W.json,
               the final table's codes against round4/results/table_U_W.npz bit for bit, E[N] to 1e-12, I to 1e-9. Blocking.
  edge   §2.3  edge action set: A_0 scan, then the iteration of A (at most 3 rounds, acceptance I(T_j) > I(T_{j-1}) + 0.05).
  old    §2.3  old action set: iteration of A starting from the gate table (= old/A_0).
  post   §3-§5 per-table numbers, checks 1-7, Monte Carlo on the best table, figure, results/round11.json, packed tables.

Usage (from this directory):  ROUND11_WORK=/path python3 run_round11.py inputs gate edge old post
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R7 = os.path.join(EV, 'round7'); R10 = os.path.join(EV, 'round10')
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
import eval11 as e11
import dp11
ew, dr = e11.ew, e11.dr
s, K, B, G = e11.s, e11.K, e11.B, e11.G

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
for _d in (LOGS, RES): os.makedirs(_d, exist_ok=True)
WORK = os.environ.get('ROUND11_WORK', os.path.join(HERE, 'work')); os.makedirs(WORK, exist_ok=True)
REL_TOL = 1e-9                              # §1 reproduction tolerance
ACCEPT = 0.05                               # DECISION 11-2: acceptance margin (nat)
MAX_ROUNDS = 3                              # DECISION 11-2: at most 3 rounds of the iteration of A
P = 9


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

class Stop(RuntimeError):
    """a reproduction, gate or check failure: the run stops (no fallback)."""

def relerr(x, ref): return abs(float(x)-float(ref))/abs(float(ref))

def jdump(obj, path):
    tmp = path+'.tmp'; json.dump(obj, open(tmp, 'w'), indent=1, default=_jdef); os.replace(tmp, path)

def _jdef(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))

def state_path(stage): return os.path.join(RES, f'state_{stage}.json')
def load_state(stage): return json.load(open(state_path(stage)))

def cleanup_frozen(out):
    """importing round4 modules may create empty folders under round4/results (prompt §0): remove empty untracked folders under
    round4..round10 and confirm that git sees no change there."""
    removed = []
    for rd in ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10'):
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_):
                os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + [os.path.join(EV, r) for r in ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10')],
                        capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on round4..round10: {'clean' if not st else st}")
    return removed, st == ''


# --- constants read from the archives ------------------------------------------------------------------------------------
def archived_inputs():
    mc = json.load(open(os.path.join(R4, 'results', 'mc.json')))
    r10 = json.load(open(os.path.join(R10, 'results', 'round10.json')))
    return dict(B=mc['U/SoD']['EN_eval'], I_SoD=mc['U/SoD']['logdet_eval'], I_old=mc['U/W']['logdet_eval'], EN_old=mc['U/W']['EN_eval'],
                U_cert=r10['budgets']['B316']['u_cert'],
                src=dict(B='round4/results/mc.json["U/SoD"]["EN_eval"]', I_SoD='round4/results/mc.json["U/SoD"]["logdet_eval"]',
                         I_old='round4/results/mc.json["U/W"]["logdet_eval"]', EN_old='round4/results/mc.json["U/W"]["EN_eval"]',
                         U_cert='round10/results/round10.json["budgets"]["B316"]["u_cert"]'))

def round4_table():
    with np.load(os.path.join(R4, 'results', 'table_U_W.npz')) as f:
        return f['codes'], f['start'], f['width'], float(f['lam']), int(f['run'])


# --- stage inputs ------------------------------------------------------------------------------------------------------------
def stage_inputs(out):
    res = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__))
    A = archived_inputs(); res['archived'] = A
    for k in ('B', 'I_SoD', 'I_old', 'EN_old', 'U_cert'): out(f"INPUT {k} = {A[k]!r}   [{A['src'][k]}]")
    assert A['B'] == B, (A['B'], B)
    out(f"INPUT sigma = {s!r}, Delta* = {e11.D!r}, K = {K}, grid h = sigma/24 = {e11.DZ!r}, G = {G}, z_0 = {e11.Z0!r}; "
        f"x range {(e11.XK.max()-e11.XK.min())/s!r} sigma; Hk {e11.HK.shape} from round4/evaluator_window.design_grid('U')")
    t0 = time.perf_counter(); dp11.warmup(); out(f"WARMUP (JIT) {time.perf_counter()-t0:.1f} s")
    # reproduction with the frozen evaluator
    rs, Js, rrs = ew.evaluate_table('U', ew.ConstTable(ew.SOD_M, ew.SOD_M))
    codes_w, start, width, lam4, run4 = round4_table()
    sc = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json')))
    fr = [r for r in sc['runs'] if r['run'] == sc['final_run']][0]
    assert run4 == sc['final_run'] and lam4 == fr['lam']
    assert np.array_equal(start, e11.KA[1:]) and np.array_equal(width, (e11.KB-e11.KA+1)[1:])
    ro, Jo, rro = ew.evaluate_table('U', ew.CodeTable(codes_w, start, width))
    rep = dict(B=(rs['EN'], relerr(rs['EN'], A['B'])), I_SoD=(rs['logdet'], relerr(rs['logdet'], A['I_SoD'])),
               EN_old=(ro['EN'], relerr(ro['EN'], A['EN_old'])), I_old=(ro['logdet'], relerr(ro['logdet'], A['I_old'])))
    for k, (v, e) in rep.items(): out(f"REPRO {k} frozen evaluator {v!r} rel err {e!r} ({'PASS' if e <= REL_TOL else 'FAIL'})")
    res['reproduction'] = {k: dict(value=v, rel_err=e, pass_=bool(e <= REL_TOL)) for k, (v, e) in rep.items()}
    res['reproduction']['paths'] = dict(sod=rs['path'], old=ro['path'])
    if not all(e <= REL_TOL for _, e in rep.values()): raise Stop("reproduction of the archived inputs failed")
    # eval11 on the round-4 table vs the frozen general path, bitwise
    T4 = e11.from_round4(codes_w, start, width, e11.old_set())
    r11, J11, f11 = e11.evaluate(T4)
    bw = dict(EN=bool(r11['EN'] == ro['EN']), W=bool(np.array_equal(f11['W'], rro['W'])), S=bool(r11['S'] == ro['S']),
              logdet=bool(r11['logdet'] == ro['logdet']), J=bool(np.array_equal(J11, Jo)))
    out(f"EVAL11 round-4 table on the full grid vs frozen general path (88-sigma support): bitwise {bw}; mass_dev_max {r11['mass_dev_max']!r}")
    res['eval11_vs_frozen'] = dict(bitwise=bw, pass_=all(bw.values()), mass_dev_max=r11['mass_dev_max'], EN=r11['EN'], logdet=r11['logdet'])
    if not all(bw.values()): raise Stop("eval11 does not reproduce the frozen evaluator on the round-4 table")
    # per-step DP arrays vs full scipy arrays
    des = dp11.Design11(out); bufs = des.ext_buffers(); pb = {}
    for p in (1, 1250, 2500, 3750, K-1):
        des.prep(p, bufs)
        uE, PP, PN, PH = bufs[:4]
        j = np.arange(-dp11.X, G+dp11.X)
        zj = np.where((j >= 0) & (j < G), e11.Z[np.clip(j, 0, G-1)], e11.Z0+j*e11.DZ)
        u = (zj-e11.XK[p])/s
        pb[p] = bool(np.array_equal(u, uE) and np.array_equal(dr.ndtr(u), PP) and np.array_equal(dr.ndtr(-u), PN) and np.array_equal(dr.phi(u), PH))
    out(f"PREP per-step u, ndtr(u), ndtr(-u), phi(u) on [-X, G-1+X] bitwise equal to full scipy/numpy arrays at steps {list(pb)}: {pb}")
    res['prep_bitwise'] = dict(steps=list(pb), ok=pb, pass_=all(pb.values()))
    if not all(pb.values()): raise Stop("per-step DP arrays differ from scipy")
    # sigma/120 path on the round-4 table vs round 7's archive (f = 5, theta_0)
    r7 = json.load(open(os.path.join(R7, 'results', 'partC_regrade.json')))
    N7, L7 = r7['N']['m000']['theta_0']['5'], r7['L']['m000']['theta_0']['5']
    rf, Jf, ff = e11.evaluate_refined(T4, 5)
    d120 = dict(EN=rf['EN'], logdet=rf['logdet'], EN_round7=N7, logdet_round7=L7, EN_rel=relerr(rf['EN'], N7), logdet_diff=rf['logdet']-L7,
                bitwise=bool(rf['EN'] == N7 and rf['logdet'] == L7), mass_dev_max=rf['mass_dev_max'], G=rf['G'], wall_s=rf['wall_s'])
    out(f"SIGMA120 eval11 refined (f = 5, full fine grid) on the round-4 table: EN {rf['EN']!r} (round 7 {N7!r}), logdet {rf['logdet']!r} "
        f"(round 7 {L7!r}); bitwise {d120['bitwise']}; mass_dev_max {rf['mass_dev_max']!r}; {rf['wall_s']:.1f} s")
    res['sigma120_vs_round7'] = d120
    if not (d120['EN_rel'] <= 1e-12 and abs(d120['logdet_diff']) <= 1e-9): raise Stop("sigma/120 eval11 does not reproduce round 7")
    res['design'] = dict(G=G, X_pad=dp11.X, cache_range=des.NC, kernel_window=des.W88, fl_cl_equal_steps=int((e11.FL == e11.CL).sum()),
                         old_set=e11.old_set().labels(), edge_set=e11.edge_set().labels(),
                         edge_set_grid=dict(am=e11.edge_set().am, bm=e11.edge_set().bm))
    jdump(res, state_path('inputs'))
    return res


# --- scans and the iteration of A ----------------------------------------------------------------------------------------
def scan_record(st):
    return {k: v for k, v in st.items()}

def run_scan(des, acts, J_ref, label, out):
    gam, gbar, g1 = des.gamma_from_J(J_ref)
    out(f"SCAN {label}: actions {acts.name} ({acts.n}); gbar {gbar!r} gamma_1 {g1!r}; lambda_0 = 5 gbar = {5*gbar!r}")
    sc = dp11.Scan11(des, acts, gam, gbar, g1, label, WORK, out)
    t0 = time.perf_counter(); st = sc.scan(); st['wall_s'] = time.perf_counter()-t0
    fin = None
    if st.get('final_run'):
        fin = [r for r in st['runs'] if r['run'] == st['final_run']][0]
        sc.cleanup(keep=[fin['table']])
    else:
        sc.cleanup(keep=[])
    return st, fin, gam, gbar, g1

def table_from(path, acts, tag):
    return e11.Table11(np.load(path, mmap_mode='r'), acts, tag)

def round_entry(j, st, fin, gam, gbar, g1, I_ref, J_ref_label, acts, out):
    """evaluate the scan's final table: I, J (for the next tangent point), tangent overestimate relative to the current reference."""
    tab = table_from(fin['table'], acts, f"{st['label']}")
    rec, J, fw = e11.evaluate(tab, gam=gam)
    assert rec['EN'] == fin['EN'] and rec['logdet'] == fin['logdet']
    F = float(gam[1:]@fw['W'][1:])
    E = F+g1-P-(rec['logdet']-I_ref)
    np.save(fin['table'].replace('.npy', '_J.npy'), J)
    e = dict(round=j, label=st['label'], reference=J_ref_label, I_ref=I_ref, gbar=gbar, gamma1=g1, lam=fin['lam'], lam_over_gbar=fin['lam_over_gbar'],
             run=fin['run'], n_runs=len(st['runs']), EN=rec['EN'], S=rec['S'], I=rec['logdet'], F=F, tangent_overestimate=E,
             table=fin['table'], J=fin['table'].replace('.npy', '_J.npy'), status=st['status'], wall_s=st['wall_s'])
    out(f"ROUND {st['label']} j={j}: I {rec['logdet']!r} EN {rec['EN']!r} F_j {F!r} gamma_1j {g1!r} I_ref {I_ref!r} "
        f"E_j = F_j + gamma_1j - p - (I - I_ref) = {E!r}")
    return e, J

def iterate(des, acts, r0, J0, cell, out, rounds_done=None):
    """iteration of A (DECISION 11-2) from round 0 (dict r0 with I, table, J path); returns the list of round entries with acceptance."""
    rounds = rounds_done or [r0]
    last = [r for r in rounds if r.get('accepted', True)][-1]
    J_last = np.load(last['J']) if 'J' in last else J0
    for j in range(len(rounds), MAX_ROUNDS+1):
        label = f"{cell}_j{j}"
        st, fin, gam, gbar, g1 = run_scan(des, acts, J_last, label, out)
        jdump(st, os.path.join(RES, f'scan_{label}.json'))
        if fin is None:
            e = dict(round=j, label=label, status=st['status'], accepted=False, note='no feasible table'); rounds.append(e)
            out(f"ITER {cell} round {j}: no feasible table ({st['status']}); stop"); break
        e, J = round_entry(j, st, fin, gam, gbar, g1, last['I'], f"T_{last['round']} ({last['label']})", acts, out)
        e['accepted'] = bool(e['I'] > last['I']+ACCEPT)
        e['gain_over_previous'] = e['I']-last['I']
        rounds.append(e)
        out(f"ITER {cell} round {j}: I(T_{j}) = {e['I']!r}, I(T_{j-1}) = {last['I']!r}, gain {e['gain_over_previous']!r} "
            f"-> {'ACCEPTED' if e['accepted'] else 'REJECTED (stop; keep the previous table)'}")
        jdump(dict(cell=cell, rounds=rounds), state_path(cell))
        if not e['accepted']: break
        last = e; J_last = J
    return rounds


# --- stage gate -----------------------------------------------------------------------------------------------------------
def stage_gate(out):
    dp11.warmup(); des = dp11.Design11(out)
    rec_sod, J_sod, W_sod = des.sod_reference()
    old = e11.old_set()
    st, fin, gam, gbar, g1 = run_scan(des, old, J_sod, 'old_j0', out)
    jdump(st, os.path.join(RES, 'scan_old_j0.json'))
    sc4 = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json')))
    cmp_runs = []; ok_runs = len(st['runs']) == len(sc4['runs'])
    for r, q in zip(st['runs'], sc4['runs']):
        c = dict(run=r['run'], phase=(r['phase'], q['phase']), lam_equal=bool(r['lam'] == q['lam']), EN_equal=bool(r['EN'] == q['EN']),
                 F_equal=bool(r['F'] == q['F']), S_equal=bool(r['S'] == q['S']), logdet_equal=bool(r['logdet'] == q['logdet']),
                 feasible_equal=bool(r['feasible'] == q['feasible']), dual_rel=relerr(r['dual'], q['dual']))
        cmp_runs.append(c); ok_runs &= c['lam_equal'] and c['EN_equal'] and c['F_equal'] and c['feasible_equal'] and c['logdet_equal']
        out(f"GATE run {r['run']:02d}: lam {c['lam_equal']} EN {c['EN_equal']} F {c['F_equal']} S {c['S_equal']} logdet {c['logdet_equal']} "
            f"feasible {c['feasible_equal']} dual rel {c['dual_rel']!r}")
    codes_w, start, width, lam4, run4 = round4_table()
    A = archived_inputs()
    if fin is None: raise Stop("gate: no final table")
    codes = np.load(fin['table'], mmap_mode='r')
    eq_rows = 0; diff_nodes = 0
    for r in range(K-1):
        s0, w = int(start[r]), int(width[r])
        d = int(np.count_nonzero(np.asarray(codes[r, s0:s0+w]) != codes_w[r, :w])); diff_nodes += d; eq_rows += (d == 0)
    gate = dict(final_run=fin['run'], final_run_round4=run4, lam=fin['lam'], lam_round4=lam4, runs_compared=cmp_runs, all_runs_bitwise=ok_runs,
                codes_rows_equal=eq_rows, codes_rows=K-1, codes_differing_nodes=diff_nodes, codes_bitwise=bool(diff_nodes == 0),
                EN=fin['EN'], EN_ref=A['EN_old'], EN_absdiff=abs(fin['EN']-A['EN_old']), I=fin['logdet'], I_ref=A['I_old'],
                I_rel=relerr(fin['logdet'], A['I_old']), dmD=fin['logdet']-A['I_SoD'], scan_wall_s=st['wall_s'],
                dp_s_mean=float(np.mean([r['dp_s'] for r in st['runs']])))
    gate['pass_'] = bool(gate['codes_bitwise'] and fin['run'] == run4 and fin['lam'] == lam4 and gate['EN_absdiff'] <= 1e-12 and gate['I_rel'] <= 1e-9)
    out(f"GATE final run {fin['run']} (round 4: {run4}), lam {fin['lam']!r} (round 4: {lam4!r}); codes on the round-4 support windows: "
        f"{diff_nodes} differing nodes in {K-1} rows -> bitwise {gate['codes_bitwise']}; EN {fin['EN']!r} |diff| {gate['EN_absdiff']!r}; "
        f"I {fin['logdet']!r} rel {gate['I_rel']!r}; all runs bitwise {ok_runs}")
    out(f"GATE (check 1, blocking): {'PASS' if gate['pass_'] else 'FAIL'}")
    jdump(gate, state_path('gate'))
    if not gate['pass_']: raise Stop("regression gate failed: nothing else counts")
    # round 0 of the old cell = the gate table
    e, J = round_entry(0, st, fin, gam, gbar, g1, A['I_SoD'], 'constant SoD (A_0 = J_SoD^-1)', old, out)
    e['accepted'] = True
    jdump(dict(cell='old', rounds=[e]), state_path('old'))
    return gate


def stage_cell(cell, out):
    dp11.warmup(); des = dp11.Design11(out)
    A = archived_inputs(); rec_sod, J_sod, W_sod = des.sod_reference()
    if cell == 'edge':
        acts = e11.edge_set()
        if os.path.exists(state_path('edge')):
            rounds = load_state('edge')['rounds']
        else:
            st, fin, gam, gbar, g1 = run_scan(des, acts, J_sod, 'edge_j0', out)
            jdump(st, os.path.join(RES, 'scan_edge_j0.json'))
            if fin is None: raise Stop("edge/A_0: no feasible table")
            e, J = round_entry(0, st, fin, gam, gbar, g1, A['I_SoD'], 'constant SoD (A_0 = J_SoD^-1)', acts, out)
            e['accepted'] = True; rounds = [e]
            jdump(dict(cell='edge', rounds=rounds), state_path('edge'))
    else:
        acts = e11.old_set()
        if not os.path.exists(state_path('gate')) or not load_state('gate')['pass_']: raise Stop("old cell requires the passed gate")
        rounds = load_state('old')['rounds']
    if rounds[-1].get('accepted', True) and len(rounds) <= MAX_ROUNDS:
        rounds = iterate(des, acts, rounds[0], None, cell, out, rounds_done=rounds)
    jdump(dict(cell=cell, rounds=rounds), state_path(cell))
    return rounds


STAGES = ('inputs', 'gate', 'edge', 'old', 'post')

def main():
    args = [a for a in sys.argv[1:]]
    numba.set_num_threads(4)
    out = Tee(os.path.join(LOGS, 'run_round11.log'))
    for stg in args:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        out(f"=== ROUND 11 stage {stg} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, numba {numba.__version__} ({numba.get_num_threads()} threads); B = {B!r}; work {WORK}")
        t0 = time.perf_counter()
        if stg == 'inputs': stage_inputs(out)
        elif stg == 'gate': stage_gate(out)
        elif stg in ('edge', 'old'): stage_cell(stg, out)
        elif stg == 'post':
            import post11; post11.stage_post(out)
        out(f"=== ROUND 11 stage {stg} done in {time.perf_counter()-t0:.1f} s")
        cleanup_frozen(out)


if __name__ == '__main__':
    main()
