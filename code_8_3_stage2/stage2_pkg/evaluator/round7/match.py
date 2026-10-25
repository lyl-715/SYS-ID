"""Round 7, Part D: nominal SoD curve, nominal matching, SoD-referenced decomposition, P4 matching of the selected candidate
(spec reports_CN/plan_round7_CN.md v3.1, frozen; §2.2 items 4-5, §2.3; task D0-D3). 固定设计、离散方案下 -- 按规定搜索得到的
匹配阈值 on the reference grid; not a global optimum, not a bound; the decomposition is descriptive (no causal reading).

Runs only if Part C's gate passed (sigma/120 = the reference grid); otherwise D1-D3 are 不适用 and only that is recorded.

D0  acceptance: |E[N_SoD] - E[N_table]| <= 1 send (absolute); the signed residual E[N_SoD] - E[N_table] is reported. After a full
    rescan this condition alone decides (the minimiser may be a global endpoint).
D1  nominal curve: M = 120..960 at theta_0 on sigma/120 (841 forwards; M = 480 via the reuse path, every other M via the
    general path with the constant window (5M, 5M)), shared by all tables. Nominal match per table: the curve point closest to
    the table's E[N](theta_0) on sigma/120 (ties -> larger M), accepted per D0; Q_m = log det of that SoD.
D2  for every table m with an accepted nominal match (and Q_0 accepted):
        L_0 - L_m = (Q_0 - Q_m) + [(L_0 - Q_0) - (L_m - Q_m)],   L = log det J_table(theta_0) on sigma/120;
    both terms, their matching residuals, and the numerical residual of the identity.
D3  the selected candidate (Part C selection rule), 20 P4 points: M_0 = the nominal curve's M closest to the table's E[N] at that
    point (ties -> larger M); local search [M_0 - 20, M_0 + 20] cap [120, 960] at the true theta; accepted per D0 if the
    minimiser is not at a window end; else full rescan [120, 960], accepted per D0; else 未匹配. Delta m_D^eq = L_table(theta) -
    log det J_SoD,M_eq(theta); the median over the 20 is used only if all 20 matched.
Every forward is appended (fsync) to results/archive_match.jsonl keyed by (point, M) as it completes (checkpoint; a re-run
resumes; theta_0's values are the curve's).

Usage (from this directory): python3 match.py run | figs   -> logs/partD_match.log, results/partD_match.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_refine as er
ew, ef, dr, rb = er.ew, er.ef, er.dr, er.rb
B = er.B
F_REF = 5; M_LO, M_HI, HALF, RTOL = 120, 960, 20, 1.0
ARCH = os.path.join(er.RES, 'archive_match.jsonl'); OUTJ = os.path.join(er.RES, 'partD_match.json'); CJ = os.path.join(er.RES, 'partC_regrade.json')
CANDS = er.CANDIDATES; P0 = 'theta_0'
NA = '不适用'

_PTS = None
def _init():
    global _PTS
    er.warmup(); _PTS = {p['pid']: p for p in er.points27()}

def _job(args):
    pid, M = args
    t0 = time.perf_counter()
    try:
        rec, J, r = er.evaluate(er.SodTable(M), F_REF, _PTS[pid]['theta'])
        d = {k: (rec[k].item() if hasattr(rec[k], 'item') else rec[k]) for k in ('EN', 'EN_P', 'S', 'logdet', 'mass_dev_max', 'min_m', 'route', 'assert_fail')}
        d['inv'] = bool(rec['inv_S'] and rec['inv_J'] and rec['inv_w']); d['status'] = 'ok'
    except (ew.AssertionStop, ef.AssertionStop) as e:
        d = dict(status='assert_stop', message=str(e))
    d.update(point=pid, M=M, f=F_REF, wall_s=time.perf_counter()-t0)
    return d

class Store:
    def __init__(self):
        self.d = {}
        if os.path.exists(ARCH):
            for l in open(ARCH): r = json.loads(l); self.d[(r['point'], r['M'])] = r
        self.n_resumed = len(self.d); self.wall = {}; self.new = {}; self.cpu = {}
    def run(self, pool, jobs, out, tag):
        todo = sorted({j for j in jobs if j not in self.d})
        t0 = time.perf_counter(); n = 0; self.cpu[tag] = 0.0
        with open(ARCH, 'a') as f:
            for r in pool.imap_unordered(_job, todo, chunksize=1):
                self.cpu[tag] += r['wall_s']
                f.write(json.dumps(r, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno()); self.d[(r['point'], r['M'])] = r; n += 1
                if r['status'] != 'ok': out(f"D-RUN ASSERTION STOP {r['point']} M = {r['M']}: {r['message']}")
                if n % 50 == 0: out(f"D-RUN {tag}: {n}/{len(todo)} ({time.perf_counter()-t0:.0f} s)")
        w = time.perf_counter()-t0; self.wall[tag] = w; self.new[tag] = len(todo)
        out(f"D-RUN {tag}: {len(set(jobs))} forwards requested, {len(todo)} new, {len(set(jobs))-len(todo)} memoised; {w:.1f} s")
    def EN(self, pid, M):
        r = self.d[(pid, M)]
        if r['status'] != 'ok': raise RuntimeError(f"assertion stop at {pid} M = {M}: {r['message']}")
        return r['EN']

def argmin_resid(vals, target):
    """vals: list of (M, E[N]); minimiser of |E[N] - target|, ties -> larger M."""
    return min(vals, key=lambda v: (abs(v[1]-target), -v[0]))

def run(out):
    import multiprocessing as mp
    C = json.load(open(CJ)); sel = C['selection']
    if not C['gate']['pass_']:
        res = dict(applicable=False, reason='C3 gate failed (max|d2| > 0.05): 网格未收敛; D1-D3 不适用', D1=NA, D2=NA, D3=NA)
        out(f"D0 {res['reason']}"); json.dump(res, open(OUTJ, 'w'), indent=1, ensure_ascii=False); return res
    N = C['N']; L = C['L']; dmD = C['dmD']; lab = er.label
    pts = er.points27(); P4 = [p['pid'] for p in pts if p['cls'] == 'P4']; assert len(P4) == 20
    st = Store(); out(f"D0 reference grid sigma/120 (f = {F_REF}); archive {os.path.relpath(ARCH, HERE)}: {st.n_resumed} forwards resumed")
    res = dict(applicable=True)
    with mp.get_context('fork').Pool(4, initializer=_init) as pool:
        curve_M = list(range(M_LO, M_HI+1))
        st.run(pool, [(P0, M) for M in curve_M], out, 'D1 nominal curve M = 120..960 at theta_0')
        curve = [(M, st.EN(P0, M)) for M in curve_M]
        ENc = np.array([v for _, v in curve]); inc = int((np.diff(ENc) > 0).sum())
        r480 = st.d[(P0, 480)]
        reg480 = dict(EN=r480['EN'] == N['SoD'][P0]['5'], logdet=r480['logdet'] == L['SoD'][P0]['5'], route=r480['route'])
        out(f"D1 curve at theta_0: E[N] at M = 120 {curve[0][1]!r}, 480 {dict(curve)[480]!r}, 960 {curve[-1][1]!r}; increasing steps {inc}; "
            f"M = 480 equals Part C's SoD at theta_0 on sigma/120: {reg480}")
        for M, v in curve: out(f"D1-CURVE M={M} E[N] {v!r} logdet {st.d[(P0, M)]['logdet']!r}")
        # nominal matching per table
        nom = {}
        for c in CANDS:
            tgt = N[c][P0]['5']; M_, E_ = argmin_resid(curve, tgt)
            nom[c] = dict(E_table=tgt, M=M_, Delta_over_sigma=M_/24, EN_SoD=E_, resid=E_-tgt, matched=bool(abs(E_-tgt) <= RTOL),
                          L=L[c][P0]['5'], Q=st.d[(P0, M_)]['logdet'])
            for k, v in nom[c].items(): out(f"D1-NOM {c} {k} {v!r}")
        res.update(curve=curve, curve_increasing_steps=inc, reg480=reg480, nominal=nom)
        # D2 decomposition
        dec = {}
        if not nom['m000']['matched']:
            out("D2 m000's nominal matching failed: Q_0 has no equal-send meaning; every decomposition 未定")
            for c in CANDS: dec[c] = dict(status='未定 (Q_0 unmatched)')
        else:
            L0, Q0 = nom['m000']['L'], nom['m000']['Q']
            for c in CANDS:
                if not nom[c]['matched']:
                    dec[c] = dict(status=f"未定 (nominal match residual {nom[c]['resid']:+.4f} > 1)"); out(f"D2 {c} {dec[c]['status']}"); continue
                Lm, Qm = nom[c]['L'], nom[c]['Q']
                t1 = Q0-Qm; t2 = (L0-Q0)-(Lm-Qm); tot = L0-Lm
                dec[c] = dict(status='ok', total=tot, term1_SoD_ref=t1, term2_adv_change=t2, identity_residual=tot-(t1+t2), M_0=nom['m000']['M'], M_m=nom[c]['M'],
                              resid_0=nom['m000']['resid'], resid_m=nom[c]['resid'], adv_0=L0-Q0, adv_m=Lm-Qm)
                for k, v in dec[c].items(): out(f"D2 {c} {k} {v!r}")
        res['decomposition'] = dec
        # D3 P4 matching of the selected candidate
        s_ = sel.get('selected')
        if not s_:
            res['D3'] = dict(status=NA, reason='no selected candidate (case 2)'); out("D3 不适用: no selected candidate (case 2)")
        else:
            plan = {}
            for pid in P4:
                tgt = N[s_][pid]['5']; M0, _ = argmin_resid(curve, tgt)
                plan[pid] = dict(E_table=tgt, M0=M0, window=[max(M_LO, M0-HALF), min(M_HI, M0+HALF)])
            st.run(pool, [(pid, M) for pid, p in plan.items() for M in range(p['window'][0], p['window'][1]+1)], out, 'D3 local windows')
            resc = []
            for pid, p in plan.items():
                lo, hi = p['window']; Ml, El = argmin_resid([(M, st.EN(pid, M)) for M in range(lo, hi+1)], p['E_table'])
                p.update(M_local=Ml, EN_local=El, resid_local=El-p['E_table'], at_endpoint=Ml in (lo, hi))
                p['accepted_local'] = bool(abs(El-p['E_table']) <= RTOL and not p['at_endpoint'])
                if not p['accepted_local']: resc.append(pid)
            out(f"D3 points needing a full rescan of [{M_LO}, {M_HI}]: {len(resc)} {resc}")
            if resc: st.run(pool, [(pid, M) for pid in resc for M in curve_M], out, f'D3 full rescans ({len(resc)} points)')
            rows = []
            for pid, p in plan.items():
                if p['accepted_local']: Meq, Eeq, how = p['M_local'], p['EN_local'], 'local window'
                else:
                    Meq, Eeq = argmin_resid([(M, st.EN(pid, M)) for M in curve_M], p['E_table']); how = 'full rescan'
                resid = Eeq-p['E_table']; ok = bool(abs(resid) <= RTOL)
                ld_eq = st.d[(pid, Meq)]['logdet']; Lt = L[s_][pid]['5']
                d = dict(point=pid, label=lab(pid), E_table=p['E_table'], M0=p['M0'], window=p['window'], M_local=p['M_local'], resid_local=p['resid_local'],
                         at_endpoint=p['at_endpoint'], rescan=not p['accepted_local'], how=how, M_eq=Meq, Delta_eq_over_sigma=Meq/24, EN_eq=Eeq,
                         resid=resid, matched=ok, logdet_table=Lt, logdet_SoDeq=ld_eq, logdet_SoD480=L['SoD'][pid]['5'], dmD=dmD[s_][pid]['5'],
                         dmD_eq=Lt-ld_eq, diff=dmD[s_][pid]['5']-(Lt-ld_eq),
                         reason=(None if ok else f"after the full rescan the closest M = {Meq} leaves |residual| = {abs(resid):.4f} > {RTOL}"))
                rows.append(d)
                for k, v in d.items(): out(f"D3 {lab(pid)} {k} {v!r}")
            m = [d for d in rows if d['matched']]
            D3 = dict(status='ok', selected=s_, rows=rows, n_matched=len(m), n_rescan=sum(d['rescan'] for d in rows), unmatched=[d['label'] for d in rows if not d['matched']],
                      resid_range=[min(d['resid'] for d in rows), max(d['resid'] for d in rows)])
            if len(m) == 20:
                D3.update(median_dmD_eq=float(np.median([d['dmD_eq'] for d in rows])), median_dmD=float(np.median([d['dmD'] for d in rows])),
                          median_diff=float(np.median([d['diff'] for d in rows])), median_label='完整中位数（20/20 匹配）')
            else:
                D3.update(median_label='未定（未全部匹配）', matched_subset=dict(n=len(m), median_dmD_eq=(float(np.median([d['dmD_eq'] for d in m])) if m else None)))
            res['D3'] = D3
            out(f"D3 {s_}: matched {len(m)}/20; full rescans {D3['n_rescan']}; signed residual range {D3['resid_range']}; unmatched {D3['unmatched']}; "
                f"median Delta m_D^eq {D3.get('median_dmD_eq')!r} ({D3['median_label']}); median Delta m_D {D3.get('median_dmD')!r}; "
                f"median Delta m_D - Delta m_D^eq {D3.get('median_diff')!r}")
            out(f"D3-TAB {'point':16s} {'E[N]_table':>11s} {'M0':>4s} {'M_eq':>4s} {'E[N_Meq]':>10s} {'resid':>8s} {'how':>12s} {'matched':>7s} {'dmD':>8s} {'dmD_eq':>8s}")
            for d in rows:
                out(f"D3-TAB {d['label']:16s} {d['E_table']:11.4f} {d['M0']:4d} {d['M_eq']:4d} {d['EN_eq']:10.4f} {d['resid']:+8.4f} {d['how']:>12s} "
                    f"{('yes' if d['matched'] else '未匹配'):>7s} {d['dmD']:8.4f} {d['dmD_eq']:8.4f}")
    out(f"D2-TAB {'table':5s} {'E[N]_tab':>10s} {'M_eq':>4s} {'resid':>8s} {'L':>11s} {'Q':>11s} {'L0-Lm':>8s} {'Q0-Qm':>8s} {'adv chg':>8s}")
    for c in CANDS:
        n_, d_ = nom[c], dec[c]
        out(f"D2-TAB {c:5s} {n_['E_table']:10.4f} {n_['M']:4d} {n_['resid']:+8.4f} {n_['L']:11.5f} {n_['Q']:11.5f} "
            + (f"{d_['total']:8.4f} {d_['term1_SoD_ref']:8.4f} {d_['term2_adv_change']:+8.4f}" if d_['status'] == 'ok' else d_['status']))
    res['forwards'] = len(st.d); res['wall_s'] = st.wall; res['new'] = st.new; res['per_forward_sum_s'] = st.cpu
    walls = [r['wall_s'] for r in st.d.values()]
    res['wall_per_forward'] = dict(median=float(np.median(walls)), sum_curve=float(sum(st.d[(P0, M)]['wall_s'] for M in range(M_LO, M_HI+1))),
                                   sum_local=float(sum(r['wall_s'] for (p, M), r in st.d.items() if p != P0)))
    ok = [r for r in st.d.values() if r['status'] == 'ok']
    res['A4'] = dict(mass_dev_max=max(r['mass_dev_max'] for r in ok), min_m=min(r['min_m'] for r in ok), assert_fail=sum(r['assert_fail'] for r in ok),
                     inv_all=all(r['inv'] for r in ok), assert_stops=len(st.d)-len(ok))
    out(f"D-SUM forwards {len(st.d)}; wall {st.wall}; per-forward median {res['wall_per_forward']['median']:.1f} s; A4 {res['A4']}")
    json.dump(res, open(OUTJ, 'w'), indent=1, ensure_ascii=False)
    return res


def figs(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(OUTJ))
    if not R.get('applicable'): out("D-FIG 不适用 (no reference grid)"); return
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    COL = {'m000': '#2a78d6', 'm005': '#eb6834', 'm010': '#1baf7a', 'm020': '#eda100', 'T4': '#e87ba4'}   # reference slots 1-5, fixed order
    MK = {'m000': 'o', 'm005': 's', 'm010': '^', 'm020': 'D', 'T4': 'v'}
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 8.5, 'axes.titlesize': 9})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axs[0]; Ms = [c[0] for c in R['curve']]; Es = [c[1] for c in R['curve']]
    ax.plot(Ms, Es, color=INK2, lw=1.6, label=r'constant-$\Delta$ SoD at $\theta_0$, $\sigma/120$')
    for c, n in R['nominal'].items():
        ax.plot([n['M']], [n['E_table']], MK[c], color=COL[c], ms=8, mec=SURF, label=f"{c}: E[N] {n['E_table']:.2f}, M_eq {n['M']}")
    ax.axvline(480, color='#e34948', lw=0.9, ls=':')
    ax.set_xlim(420, 700); ax.set_ylim(220, 340)
    ax.set_xlabel(r'half-width $M$ ($\sigma/24$ units)'); ax.set_ylabel(r'$\mathbb{E}[N]$ at $\theta_0$')
    ax.set_title('Nominal curve and the five nominal matches (reference grid)', loc='left', color=INK)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    ax = axs[1]
    if R['D3'].get('status') == 'ok':
        rows = R['D3']['rows']
        for e, mk in ((0.2, 'o'), (0.5, 's')):
            q = [d for d in rows if d['point'].endswith(f'|{e}')]
            ax.scatter([d['dmD'] for d in q], [d['dmD_eq'] for d in q], marker=mk, s=40, color=COL[R['D3']['selected']], edgecolor=SURF, label=f'P4 eps = {e}')
        lo = min(min(d['dmD'] for d in rows), min(d['dmD_eq'] for d in rows))-0.2; hi = max(max(d['dmD'] for d in rows), max(d['dmD_eq'] for d in rows))+0.2
        ax.plot([lo, hi], [lo, hi], ':', color=INK2, lw=0.9)
        ax.set_xlabel(r'$\Delta m_D$ vs SoD at $\Delta^*$ (nat)'); ax.set_ylabel(r'$\Delta m_D^{\rm eq}$ vs SoD at $M_{\rm eq}$ (nat)')
        ax.set_title(f"P4 points, selected {R['D3']['selected']} (reference grid)", loc='left', color=INK)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    else:
        ax.text(0.5, 0.5, 'D3 not applicable', ha='center', va='center', color=INK2, transform=ax.transAxes); ax.set_axis_off()
    for a in axs:
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
    fig.suptitle('Equal-send matching on sigma/120; fixed design, discrete evaluation', color=INK, fontsize=10, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_match.png'), dpi=130); plt.close(fig)
    out("D-FIG fig_match.png written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['run', 'figs']); A_ = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(er.LOGS, f'partD_{"match" if A_.stage == "run" else "figs"}.log'), 'a' if A_.stage == 'run' else 'w')
    print(f"=== match.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round7_CN.md v3.1 §2.2-§2.3; B = {B!r}")
    t0 = time.perf_counter(); (run if A_.stage == 'run' else figs)(print); print(f"=== match.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
