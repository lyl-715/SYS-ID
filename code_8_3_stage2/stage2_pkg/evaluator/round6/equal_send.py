"""Round 6, Part C: equal-send comparison for the selected table only (spec reports_CN/plan_round6_CN.md v3.2, frozen; §3).
固定设计、离散方案下 -- a constant-Delta SoD with integer half-width M (grid units of h = sigma/24) calibrated at the same true theta
so that its E[N] is closest to the table's E[N]_table(theta); 按规定搜索得到的匹配阈值, not a global optimum, not a bound.

Constant-Delta SoD at M (DECISION C-1): M = 480 -> round5/robustness.eval_sod (evaluator_fast, the accepted operator; the SoD of
Part A's Delta m_D); M != 480 -> round5/robustness.eval_table with the constant window (M, M) (the round-4 general window path,
R = 88 sigma, frozen sigma/24 grid, windows centred at x(theta)) -- round 4's dispatch rule, as round 5's scheme A.

Search (spec §3, frozen): the nominal curve M = 120..960 at theta_0 (841 forwards); points theta_0 and the 20 P4 points with
eps in {0.2, 0.5}; M_0 = the curve's M closest to E[N]_table(theta) (ties -> larger M); every integer M in [M_0 - 20, M_0 + 20]
cap [120, 960] evaluated at the true theta; minimiser of |E[N_M] - E[N]_table| (ties -> larger M); matched if the residual
<= 1 send and the minimiser is not at an endpoint of that window; otherwise a full rescan of [120, 960] at that theta,
matched if the residual <= 1 send after it, else 未匹配. Every forward is memoised by (point, M) (theta_0's window values are
its curve values) and appended to results/archive_partC.jsonl as it completes (checkpoint; a re-run resumes).
Gains: Delta m_D(theta) = log det J_table(theta) - log det J_SoD,480(theta) (Part A archive) and
Delta m_D^eq(theta) = log det J_table(theta) - log det J_SoD,M_eq(theta); their difference = log det J_SoD,M_eq - log det J_SoD,480.

Usage (from this directory): python3 equal_send.py run | figs   -> logs/partC_equal_send.log, results/partC_equal_send.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R5 = os.path.join(EV, 'round5')
sys.path.insert(0, R4); sys.path.insert(0, R5); sys.path.insert(0, HERE)
import numpy as np
import robustness as rb                                   # round 5, unchanged
import robustness_margin as rm
ew, ef, dr = rb.ew, rb.ef, rb.dr

s, K, B = rb.s, rb.K, rb.B
LOGS, RES = rm.LOGS, rm.RES
NAME = 'U'
M_LO, M_HI, HALF, RTOL = 120, 960, 20, 1.0
ARCH = os.path.join(RES, 'archive_partC.jsonl'); OUT = os.path.join(RES, 'partC_equal_send.json')
VERDICT = '离散方案下'


class ConstWin(ew.ConstTable):
    """the constant window (M, M) with the attributes round5/robustness.eval_table reads for its boundary-lookup diagnostic:
    every node of the grid counts as stored (start 0, width G), so the diagnostic is 0 (a constant table has no stored range)."""
    def __init__(self, M):
        super().__init__(M, M); G = rb.INP[NAME]['G']
        self.start = np.zeros(K-1, np.int64); self.width = np.full(K-1, G, np.int64)

def eval_M(theta, M):
    xth = rb.x_of(NAME, np.asarray(theta))
    if M == ew.SOD_M: return rb.eval_sod(NAME, xth)
    return rb.eval_table(NAME, ConstWin(M), xth)

_PTS = None
def _init(): ew._warmup()

def _job(args):
    pid, M = args
    t0 = time.perf_counter()
    try:
        rec, J, r = eval_M(_PTS[pid]['theta'], M)
        d = dict(point=pid, M=M, Delta_over_sigma=M/ew.N_GRID, status='ok', EN=float(rec['EN']), EN_P=float(rec['EN_P']), S=float(rec['S']),
                 logdet=float(rec['logdet']), mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']),
                 assert_fail=int((~r['assert_ok']).sum()), inv=bool(rec['inv_S'] and rec['inv_J'] and rec['inv_w']), path=rec['path'])
    except (ew.AssertionStop, ef.AssertionStop) as e:
        d = dict(point=pid, M=M, status='assert_stop', message=str(e))
    d['wall_s'] = time.perf_counter()-t0
    return d

class Store:
    """memo of (point, M) -> record, backed by the append-only archive (one line per evaluation)."""
    def __init__(self):
        self.d = {}
        if os.path.exists(ARCH):
            for l in open(ARCH):
                r = json.loads(l); self.d[(r['point'], r['M'])] = r
        self.n_resumed = len(self.d); self.wall = 0.0
    def run(self, pool, jobs, out, tag):
        todo = sorted({j for j in jobs if j not in self.d})
        t0 = time.perf_counter()
        with open(ARCH, 'a') as f:
            for r in pool.imap_unordered(_job, todo, chunksize=1):
                f.write(json.dumps(r, ensure_ascii=False)+'\n'); f.flush(); self.d[(r['point'], r['M'])] = r
        w = time.perf_counter()-t0; self.wall += w
        out(f"C-RUN {tag}: {len(set(jobs))} evaluations requested, {len(todo)} new, {len(set(jobs))-len(todo)} memoised; {w:.1f} s")
        return w
    def EN(self, pid, M):
        r = self.d[(pid, M)]
        if r['status'] != 'ok': raise RuntimeError(f"assertion stop at {pid} M = {M}: {r['message']}")
        return r['EN']

def argmin_resid(vals, target):
    """vals: list of (M, E[N]); minimiser of |E[N] - target|, ties -> larger M."""
    return min(vals, key=lambda v: (abs(v[1]-target), -v[0]))

def run(out):
    global _PTS
    import multiprocessing as mp
    S_ = json.load(open(rm.SUMMARY)); sel = S_['selected']
    A = rm.read_archive(); q = [r for r in A if r['table'] == sel]
    sod = {rm.point_id(r): r for r in A if r['table'] == 'SoD'}
    pts = [r for r in q if r['cls'] == 'theta0']+sorted([r for r in q if r['cls'] == 'P4' and r['eps_target'] in (0.2, 0.5)],
                                                          key=lambda r: (r['eps_target'], int(r['dir'][1:])))
    assert len(pts) == 21
    _PTS = {rm.point_id(r): r for r in pts}; P0 = rm.point_id(pts[0])
    out(f"C0 selected table {sel} ({S_['selected_label']}); 21 points (theta_0 + 20 P4); M in [{M_LO}, {M_HI}]; window +-{HALF}; tolerance {RTOL} send")
    st = Store(); out(f"C0 archive {os.path.relpath(ARCH, HERE)}: {st.n_resumed} evaluations resumed")
    with mp.get_context('fork').Pool(4, initializer=_init) as pool:
        # C2 step 1: the nominal curve
        curve_M = list(range(M_LO, M_HI+1))
        st.run(pool, [(P0, M) for M in curve_M], out, 'nominal curve M = 120..960 at theta_0')
        curve = [(M, st.EN(P0, M)) for M in curve_M]
        ENc = np.array([v for _, v in curve]); inc = int((np.diff(ENc) > 0).sum())
        out(f"C1 curve: E[N_M] at M = 120 {curve[0][1]!r}, 480 {dict(curve)[480]!r}, 960 {curve[-1][1]!r}; increasing steps (E[N_(M+1)] > E[N_M]) {inc}")
        # regressions: the curve at theta_0 vs round 5's scheme A (M = 240..480) and the Part A SoD at theta_0 (M = 480)
        sa = {r['M']: r for r in json.load(open(os.path.join(R5, 'results', 'schemeA.json')))['rows']}
        bad = [M for M in sa if not (st.d[(P0, M)]['EN'] == sa[M]['EN'] and st.d[(P0, M)]['S'] == sa[M]['S'] and st.d[(P0, M)]['logdet'] == sa[M]['logdet'])]
        out(f"C1-REG curve at theta_0 vs round5/results/schemeA.json for M = 240..480 (E[N], S, log det, exact): {len(sa)-len(bad)}/{len(sa)} equal; mismatches {bad}")
        s0 = sod[P0]; r480 = st.d[(P0, 480)]
        reg480 = r480['EN'] == s0['EN'] and r480['logdet'] == s0['logdet']
        out(f"C1-REG M = 480 at theta_0 equals the Part A SoD at theta_0 (E[N], log det): {reg480}")
        # C2 step 2: local windows
        plan = {}
        for pid, r in _PTS.items():
            Et = r['EN']; M0, _ = argmin_resid(curve, Et)
            lo, hi = max(M_LO, M0-HALF), min(M_HI, M0+HALF)
            plan[pid] = dict(E_table=Et, M0=M0, window=[lo, hi])
        st.run(pool, [(pid, M) for pid, p in plan.items() for M in range(p['window'][0], p['window'][1]+1)], out, 'local windows (41 per point)')
        resc = []
        for pid, p in plan.items():
            lo, hi = p['window']; vals = [(M, st.EN(pid, M)) for M in range(lo, hi+1)]
            Ml, El = argmin_resid(vals, p['E_table'])
            p.update(M_local=Ml, EN_local=El, resid_local=El-p['E_table'], at_endpoint=Ml in (lo, hi))
            p['accepted_local'] = bool(abs(El-p['E_table']) <= RTOL and not p['at_endpoint'])
            if not p['accepted_local']: resc.append(pid)
        out(f"C2 points needing a full rescan of [{M_LO}, {M_HI}]: {len(resc)} {resc}")
        if resc:
            st.run(pool, [(pid, M) for pid in resc for M in curve_M], out, f'full rescans ({len(resc)} points)')
    rows = []
    for pid, p in plan.items():
        r = _PTS[pid]
        if p['accepted_local']:
            Meq, Eeq, how = p['M_local'], p['EN_local'], 'local window'
        else:
            vals = [(M, st.EN(pid, M)) for M in range(M_LO, M_HI+1)]
            Meq, Eeq = argmin_resid(vals, p['E_table']); how = 'full rescan'
        resid = Eeq-p['E_table']; matched = bool(abs(resid) <= RTOL)
        reason = None
        if not matched:
            reason = (f"after the full rescan the closest M = {Meq} leaves |residual| = {abs(resid):.4f} > {RTOL} send"
                      + (' (minimiser at the range end; the required M lies outside [120, 960])' if Meq in (M_LO, M_HI) else ''))
        ld_eq = st.d[(pid, Meq)]['logdet']
        d = dict(point=pid, cls=r['cls'], dir=r['dir'], eps_target=r['eps_target'], eps_attained=r['eps_attained'], E_table=p['E_table'],
                 M0=p['M0'], window=p['window'], M_local=p['M_local'], resid_local=p['resid_local'], at_endpoint=p['at_endpoint'],
                 rescan=not p['accepted_local'], how=how, M_eq=Meq, Delta_eq_over_sigma=Meq/ew.N_GRID, EN_eq=Eeq, resid=resid, matched=matched,
                 reason=reason, logdet_table=r['logdet'], logdet_SoD480=r['logdet_SoD'], EN_SoD480=r['EN_SoD'], logdet_SoDeq=ld_eq,
                 dmD=r['dmD'], dmD_eq=r['logdet']-ld_eq, diff=r['dmD']-(r['logdet']-ld_eq))
        if Meq == 480: d['M480_same_as_partA'] = bool(ld_eq == r['logdet_SoD'])
        rows.append(d)
        for k, v in d.items(): out(f"C2 {pid} {k} {v!r}")
    out(f"C2-TAB {'point':22s} {'E[N]_table':>11s} {'M0':>4s} {'M_eq':>4s} {'Delta/sigma':>11s} {'E[N_Meq]':>10s} {'resid':>8s} {'how':>12s} "
        f"{'matched':>7s} {'dmD':>8s} {'dmD_eq':>8s} {'dmD-dmD_eq':>10s}")
    for d in rows:
        out(f"C2-TAB {d['point']:22s} {d['E_table']:11.4f} {d['M0']:4d} {d['M_eq']:4d} {d['Delta_eq_over_sigma']:11.4f} {d['EN_eq']:10.4f} "
            f"{d['resid']:+8.4f} {d['how']:>12s} {('yes' if d['matched'] else '未匹配'):>7s} {d['dmD']:8.4f} {d['dmD_eq']:8.4f} {d['diff']:+10.4f}")
    p4 = [d for d in rows if d['cls'] == 'P4']; m4 = [d for d in p4 if d['matched']]; n_m = len(m4)
    unm = [d for d in rows if not d['matched']]
    out(f"C3 matched {sum(d['matched'] for d in rows)}/21 (P4 {n_m}/20, theta_0 {rows[0]['matched']}); full rescans {sum(d['rescan'] for d in rows)}; "
        f"signed residual range [{min(d['resid'] for d in rows)!r}, {max(d['resid'] for d in rows)!r}]")
    for d in unm: out(f"C3 未匹配 {d['point']} (direction {d['dir']}, eps {d['eps_target']}): {d['reason']}")
    th0 = rows[0]
    out(f"C3 theta_0: E[N]_table {th0['E_table']!r}, M_eq {th0['M_eq']} (Delta_eq = {th0['Delta_eq_over_sigma']!r} sigma), residual {th0['resid']!r}; "
        f"Delta m_D {th0['dmD']!r}, Delta m_D^eq {th0['dmD_eq']!r}, difference {th0['diff']!r}")
    res = dict(selected=sel, selected_label=S_['selected_label'], rows=rows, n_matched=sum(d['matched'] for d in rows), n_matched_P4=n_m,
               n_rescan=sum(d['rescan'] for d in rows), unmatched=[d['point'] for d in unm], curve=curve, curve_increasing_steps=inc,
               regression_schemeA_mismatch=bad, regression_480=reg480, wall_s=st.wall, n_evals=len(st.d))
    if n_m == 20:
        res['median_dmD_eq_P4'] = float(np.median([d['dmD_eq'] for d in p4])); res['median_diff_P4'] = float(np.median([d['diff'] for d in p4]))
        res['median_dmD_P4'] = float(np.median([d['dmD'] for d in p4]))
        out(f"C3 all 20 P4 points matched: median Delta m_D^eq {res['median_dmD_eq_P4']!r} (verdict item (d) >= 5 nat: {res['median_dmD_eq_P4'] >= 5}); "
            f"median Delta m_D - Delta m_D^eq {res['median_diff_P4']!r}; median Delta m_D {res['median_dmD_P4']!r}")
    else:
        if m4:
            res['matched_subset'] = dict(n=n_m, median_dmD_eq=float(np.median([d['dmD_eq'] for d in m4])), min_dmD_eq=min(d['dmD_eq'] for d in m4),
                                         max_dmD_eq=max(d['dmD_eq'] for d in m4), median_diff=float(np.median([d['diff'] for d in m4])))
        out(f"C3 only {n_m}/20 P4 points matched: descriptive statistics of the matched subset {res.get('matched_subset')}; verdict items (c), (d) 未定")
    out(f"C3 Delta m_D^eq over the 21 points: min {min(d['dmD_eq'] for d in rows)!r}, max {max(d['dmD_eq'] for d in rows)!r}; "
        f"evaluations {len(st.d)} in the archive; Part C wall {st.wall:.1f} s")
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
    return res


def figs(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(OUT)); rows = R['rows']
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    C = {0.0: '#2a78d6', 0.2: '#eb6834', 0.5: '#1baf7a'}; MK = {0.0: 'D', 0.2: 'o', 0.5: 's'}
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 8.5, 'axes.titlesize': 9})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axs[0]
    for e, lab in ((0.2, r'P4 $\varepsilon=0.2$'), (0.5, r'P4 $\varepsilon=0.5$'), (0.0, r'$\theta_0$')):
        q = [d for d in rows if d['eps_target'] == e]
        ax.scatter([d['dmD'] for d in q], [d['dmD_eq'] for d in q], s=40, marker=MK[e], color=C[e], edgecolor=SURF, linewidth=1.2,
                   label=lab, zorder=3)
        um = [d for d in q if not d['matched']]
        if um: ax.scatter([d['dmD'] for d in um], [d['dmD_eq'] for d in um], s=110, marker='o', facecolor='none', edgecolor='#e34948', lw=1.2, zorder=4)
    lo = min(min(d['dmD'] for d in rows), min(d['dmD_eq'] for d in rows)); hi = max(max(d['dmD'] for d in rows), max(d['dmD_eq'] for d in rows))
    pad = 0.05*(hi-lo+1e-9); xx = np.array([lo-pad, hi+pad])
    ax.plot(xx, xx, color=INK2, lw=0.9, ls=':', zorder=1); ax.text(xx[1], xx[1], ' y = x', color=INK2, fontsize=7.5, va='center')
    ax.set_xlabel(r'$\Delta m_D(\theta)$ vs SoD at $\Delta^*$ ($M=480$) (nat)'); ax.set_ylabel(r'$\Delta m_D^{\rm eq}(\theta)$ vs SoD at $M_{\rm eq}$ (nat)')
    ax.set_title('Gain before and after matching the SoD send count (same theta)', loc='left', color=INK)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    ax = axs[1]
    Ms = [c[0] for c in R['curve']]; Es = [c[1] for c in R['curve']]
    ax.plot(Ms, Es, color='#2a78d6', lw=2, label=r'constant-$\Delta$ SoD at $\theta_0$')
    th0 = rows[0]
    ax.axhline(th0['E_table'], color=INK2, lw=0.9, ls='--'); ax.axvline(th0['M_eq'], color=INK2, lw=0.9, ls='--')
    ax.text(th0['M_eq'], max(Es), f"  table at theta_0: E[N] = {th0['E_table']:.2f}, M_eq = {th0['M_eq']}",
            color=INK, fontsize=7.5, va='top')
    ax.axvline(480, color='#e34948', lw=0.9, ls=':'); ax.text(480, min(Es), r'  $\Delta^*$ (M = 480)', color=INK2, fontsize=7.5, va='bottom')
    ax.set_yscale('log'); ax.set_yticks([150, 200, 300, 400, 600, 1000, 1500]); ax.set_yticklabels(['150', '200', '300', '400', '600', '1000', '1500'])
    ax.set_xlabel(r'half-width $M$ (grid units of $h=\sigma/24$)'); ax.set_ylabel(r'$\mathbb{E}[N_M]$ at $\theta_0$')
    ax.set_title('Nominal curve used for the initial values M_0 (841 forwards)', loc='left', color=INK)
    for a in axs:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
    fig.suptitle(f"Equal-send comparison, table {R['selected']} ({'m*' if R['selected_label'] == 'm*' else 'no margin passed; m = 0.20'}); fixed design, discrete evaluation", color=INK, fontsize=10, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_equal_send.png'), dpi=130); plt.close(fig)
    out("C-FIG fig_equal_send.png written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['run', 'figs']); A_ = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(LOGS, f'partC_{"equal_send" if A_.stage == "run" else "figs"}.log'))
    print(f"=== equal_send.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round6_CN.md v3.2 §3; B = {B!r}")
    t0 = time.perf_counter(); (run if A_.stage == 'run' else figs)(print); print(f"=== equal_send.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
