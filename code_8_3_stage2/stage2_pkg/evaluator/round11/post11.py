"""Round 11, stage post (prompt §3-§5, plan §4-§7): per-table numbers, checks 1-7, Monte Carlo on the best table, figure,
packed tables, results/round11.json, prediction scoring, verdict case.

Tables analysed: the final table of every round of both cells (old: round 0 = the gate table = the round-4 table; edge: round 0
= edge/A_0), including rejected rounds (their I is reported; they are not cell tables). A cell's table is its last accepted
round (A_0 cells: round 0).
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R7 = os.path.join(EV, 'round7')
import numpy as np
import eval11 as e11
import dp11
import run_round11 as rr
ew, dr = e11.ew, e11.dr
s, K, B, G = e11.s, e11.K, e11.B, e11.G
P = 9
FB_TOL = 1e-12                # check 2 (prompt §4 item 2)
MC_Z = 3.0                    # check 4: |z| <= 3 (DECISION 11-7)


def acts_of(cell): return e11.edge_set() if cell == 'edge' else e11.old_set()

def gamma0(des):
    rec_sod, J_sod, W_sod = des.sod_reference()
    return des.gamma_from_J(J_sod)

def analyse(entry, cell, des, A, gam0, out, keep_edges=False):
    """all §3 numbers of one round table."""
    acts = acts_of(cell); tab = rr.table_from(entry['table'], acts, entry['label'])
    J_ref_gamma = None
    # tangent point of this round (A_j): gamma of the round's scan
    if entry['round'] == 0:
        gam, gbar, g1 = gam0
    else:
        J_prev = np.load(entry['J_prev'])
        gam, gbar, g1 = des.gamma_from_J(J_prev)
    assert gbar == entry['gbar'] and g1 == entry['gamma1']
    st = e11.Stats(gam0[0])
    t0 = time.perf_counter(); rec, J, fw = e11.evaluate(tab, gam=gam, stats=st); t_fw = time.perf_counter()-t0
    assert rec['EN'] == entry['EN'] and rec['logdet'] == entry['I']
    sres, ev, w = st.result()
    I = rec['logdet']; dI = I-A['I_SoD']; gp = A['U_cert']-I
    d = dict(cell=cell, round=entry['round'], label=entry['label'], accepted=entry.get('accepted'), lam=entry['lam'],
             lam_over_gbar=entry['lam_over_gbar'], gbar=gbar, gamma1=g1, EN24=rec['EN'], S=rec['S'], I=I, dI=dI, g_prime=gp,
             deff_lower=float(np.exp(-gp/P)), F=float(gam[1:]@fw['W'][1:]), I_ref=entry['I_ref'], reference=entry['reference'],
             tangent_overestimate=entry['tangent_overestimate'], mass_dev_max24=rec['mass_dev_max'], min_m24=rec['min_m'],
             support_lo_min=int(fw['sup_lo'].min()), support_hi_max=int(fw['sup_hi'].max()),
             mean_support_width_sigma=float((fw['sup_hi']-fw['sup_lo'])[1:].mean()/24), shape=sres, fwd_s=t_fw)
    # E_0 relative to A_0 for every table (extra; the cell's own E_j is tangent_overestimate)
    F0 = float(gam0[0][1:]@fw['W'][1:]); d['F0'] = F0; d['E0'] = F0+gam0[2]-P-(I-A['I_SoD'])
    # check 2: forward = backward
    t0 = time.perf_counter(); bw = e11.backward(tab, gam); t_bw = time.perf_counter()-t0
    Nf = rec['EN']-1.0; Ff = d['F']
    d['check2'] = dict(F_forward=Ff, F_backward=bw['Fc'], F_rel=abs(bw['Fc']-Ff)/abs(Ff), N_forward=Nf, N_backward=bw['Nq'],
                       N_rel=abs(bw['Nq']-Nf)/abs(Nf), total_forward=Ff-entry['lam']*Nf, total_backward=bw['Fc']-entry['lam']*bw['Nq'],
                       dp_value=entry['dsum'], bw_s=t_bw)
    tf, tb = d['check2']['total_forward'], d['check2']['total_backward']
    d['check2']['total_rel'] = abs(tb-tf)/abs(tf); d['check2']['dp_vs_forward_rel'] = abs(entry['dsum']-tf)/abs(tf)
    d['check2']['pass_'] = bool(d['check2']['F_rel'] <= FB_TOL and d['check2']['N_rel'] <= FB_TOL and d['check2']['total_rel'] <= FB_TOL
                                and d['check2']['dp_vs_forward_rel'] <= FB_TOL)
    # check 5
    d['check5'] = e11.check_edges(tab)
    # sigma/120
    rf, Jf, ff = e11.evaluate_refined(tab, 5)
    d['EN120'] = rf['EN']; d['I120'] = rf['logdet']; d['S120'] = rf['S']; d['mass_dev_max120'] = rf['mass_dev_max']; d['min_m120'] = rf['min_m']
    d['budget120_pass'] = bool(rf['EN'] <= B); d['slack120'] = B-rf['EN']; d['slack24'] = B-rec['EN']; d['wall120_s'] = rf['wall_s']
    # check 6 (clarification 2): applies only to tables whose sigma/120 E[N] <= B
    d['check6_applies'] = d['budget120_pass']; d['check6'] = bool(I <= A['U_cert']) if d['budget120_pass'] else None
    # old-set tables: zero mass outside the round-4 windows asserted at every step (clarification 3), then the frozen round-7 path at
    # f = 5 as a cross-check of the sigma/120 entry
    if cell == 'old':
        e11.evaluate(tab, window=True); d['zero_mass_outside_round4_windows'] = True
        sys.path.insert(0, R7); import evaluator_refine as er
        codes = np.asarray(tab.codes)
        cw = np.full((K-1, int(e11.KB[1]-e11.KA[1]+1)), 0xFFFF, np.uint16)
        for r in range(K-1):
            a_, b_ = int(e11.KA[r+1]), int(e11.KB[r+1]); cw[r, :b_-a_+1] = codes[r, a_:b_+1]
        lt = er.rb.LookupTable(cw, e11.KA[1:], (e11.KB-e11.KA+1)[1:])
        r7, J7, _ = er.evaluate(lt, 5, xth=e11.XK)
        d['round7_f5'] = dict(EN=r7['EN'], logdet=r7['logdet'], EN_rel=rr.relerr(r7['EN'], rf['EN']), logdet_diff=r7['logdet']-rf['logdet'],
                              bitwise=bool(r7['EN'] == rf['EN'] and r7['logdet'] == rf['logdet']), route=r7['route'])
    out(f"POST {entry['label']} ({cell} round {entry['round']}{'' if entry.get('accepted', True) else ', rejected'}): I {I!r} dI {dI!r} "
        f"g' {gp!r} D-eff {d['deff_lower']!r}; EN24 {rec['EN']!r} EN120 {rf['EN']!r} (<= B: {d['budget120_pass']}); I120 {rf['logdet']!r}; "
        f"E_j {d['tangent_overestimate']!r}, E_0 {d['E0']!r}; edge quantiles 10/25/50/75/90 {sres['edge_quantiles']!r}; "
        f"body send share {sres['body_send_share']!r} info/send {sres['body_info_per_send']!r}; check2 {d['check2']['pass_']} "
        f"(F {d['check2']['F_rel']:.1e}, N {d['check2']['N_rel']:.1e}, total {d['check2']['total_rel']:.1e}, DP {d['check2']['dp_vs_forward_rel']:.1e}); "
        f"check5 {d['check5']['pass_']} (min L {d['check5']['min_L']}, min U {d['check5']['min_U']}); check6 {d['check6']}"
        + (f"; round-7 f=5 path {d['round7_f5']}" if 'round7_f5' in d else ''))
    for b in sres['bins']:
        out(f"POST {entry['label']} bin {b['bin']}: mass share {b['mass_share']!r}, sends {b['sends']!r} (share {b['send_share']!r}), "
            f"info {b['info']!r}, info/send {b['info_per_send']!r}, gamma0-info/send {b['gamma_info_per_send']!r}, near {b['near']!r}, edge {b['edge']!r}")
    return d, ((ev.astype(np.float32), w.astype(np.float32)) if keep_edges else None)    # float32 copies for the figure only


def figure(edges_old, edges_new, lab_new, q_old, q_new, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'; C1, C2 = '#2a78d6', '#eb6834'
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 10})
    lo, hi, bw = -4.0, 8.0, 1/8
    bins = np.arange(lo, hi+bw/2, bw)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, sharey=True, constrained_layout=True)
    ymax = 0.0; hs = []
    for (ev, w) in (edges_old, edges_new):
        h, _ = np.histogram(np.clip(ev, lo, hi-1e-9), bins=bins, weights=w); h = h/w.sum()/bw; hs.append(h); ymax = max(ymax, h.max())
    for ax, (ev, w), h, col, lab, q in ((axs[0], edges_old, hs[0], C1, 'old action set, A$_0$ (round-4 table)', q_old),
                                        (axs[1], edges_new, hs[1], C2, lab_new, q_new)):
        ax.bar(bins[:-1], h, width=bw*0.92, align='edge', color=col, linewidth=0)
        for qq, v in zip(('10 %', '50 %', '90 %'), (q[0], q[2], q[4])):
            ax.axvline(v, color=INK2, lw=1.0, ls='--')
            ax.text(v, ymax*1.02, f"{qq}: {v:.2f}", rotation=0, ha='center', va='bottom', fontsize=9, color=INK)
        out_lo = float(w[ev < lo].sum()/w.sum()); out_hi = float(w[ev >= hi].sum()/w.sum())
        ax.set_title(f"{lab}: mass outside the axis range {out_lo:.1%} (< {lo:g}) and {out_hi:.1%} (>= {hi:g}), stacked in the end bins",
                     fontsize=10, color=INK, loc='left')
        ax.set_ylabel('mass density per sigma')
        ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axs[0].set_ylim(0, ymax*1.15)
    axs[1].set_xlabel(r'facing-edge offset: near-side half-width $-\ |r-\bar x_k|$ (sigma; > 0: $\bar x_k$ inside the window)')
    fig.suptitle('Where the facing window edge sits relative to the predicted output (mass-weighted over all steps k >= 2 and reference values r).\n'
                 'Look at the spread: the edge set concentrates the facing edge in a narrow band; dashed lines = 10/50/90 % quantiles.',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.savefig(path, dpi=130); plt.close(fig)


def pack(entry, cell, path, note_extra=''):
    acts = acts_of(cell); codes = np.load(entry['table'])
    np.savez_compressed(path, codes=codes, action_set=np.array(acts.name), am=acts.am, bm=acts.bm, ap=acts.ap, bp=acts.bp,
                        labels_lo=np.array(acts.lab_lo), labels_hi=np.array(acts.lab_hi), fl=e11.FL, cl=e11.CL, z0=e11.Z0, dz=e11.DZ, G=G,
                        lam=entry['lam'], round=entry['round'], run=entry['run'],
                        note=np.array("codes[r, i]: action of step k = r + 2 at sigma/24 grid node i (full grid, z_i = z0 + i dz); "
                                      "code = i_lo * N_hi + i_hi; half-widths in grid units L = max(am[i_lo] (i - fl[k-1]) + bm[i_lo], 48), "
                                      "U = max(ap[i_hi] (cl[k-1] - i) + bp[i_hi], 48); window (r_i - L dz, r_i + U dz). " + note_extra))
    chk = np.load(path); same = bool(np.array_equal(chk['codes'], codes))
    return dict(path=os.path.relpath(path, HERE), bytes=os.path.getsize(path), roundtrip_equal=same)


def ub_sod(cells, A, res, out):
    """addendum to the post-processing (no new run). For the edge scan of round 2 (tangent point A_1 = J(T_1)^-1, T_1 the accepted
    edge round-1 table), every evaluated lambda gives
        UB(lambda) = I(T_1) - p + gamma_1 + V_lambda + lambda (B - 1),   V_lambda = sum_i ghat_{1,i} V_2(i) (the DP value, 'dsum'),
    and UB_SoD = min over those lambda. For any table T of the class below with E[N] <= B: log det J(T) <= I(T_1) + tr(A_1 J(T)) - p
    (concavity of log det, tangent at J(T_1)), tr(A_1 J(T)) = gamma_1 + F(T), F(T) <= F(T) - lambda (N(T) - (B - 1)) <= V_lambda +
    lambda (B - 1) (the DP maximises F - lambda N over the class). Class: tables that assign to every (step k >= 2, sigma/24 node r_i)
    one of the 100 edge actions (E^2, edges on grid nodes, at least 2 sigma from r_i), evaluated by the sigma/24 discrete model at
    theta_0 with E[N] <= B (sigma/24)."""
    e1 = [e for e in cells['edge'] if e['round'] == 1][0]
    sc = json.load(open(os.path.join(rr.RES, 'scan_edge_j2.json')))
    I1 = e1['I']; g1 = sc['gamma1']; Bm1 = B-1.0
    rows = []
    for r in sc['runs']:
        v = I1-P+g1+r['dsum']+r['lam']*Bm1
        rows.append(dict(run=r['run'], phase=r['phase'], lam=r['lam'], lam_over_gbar=r['lam_over_gbar'], V_lambda=r['dsum'], EN=r['EN'],
                         dual=r['dual'], dual_recomputed_absdiff=abs(g1+r['dsum']+r['lam']*Bm1-r['dual']), UB=v))
        out(f"UB_SoD edge_j2 run {r['run']:02d} lam {r['lam']!r} V_lambda {r['dsum']!r} EN {r['EN']!r} -> I(T_1) - p + gamma_1 + V + lambda (B - 1) = {v!r}")
    best = min(rows, key=lambda x: x['UB'])
    ub = best['UB']; gap = ub-I1
    # consistency: every edge-class table of this round with sigma/24 E[N] <= B must lie below UB_SoD
    below = {e['label']: dict(I=e['I'], EN24=e['EN'], below=bool(e['I'] <= ub)) for e in cells['edge'] if 'table' in e and e['EN'] <= B}
    d = dict(scan='edge_j2', tangent='A_1 = J(T_1)^-1', T1=e1['label'], I_T1=I1, gamma1=g1, p=P, B=B, per_lambda=rows, UB_SoD=ub, argmin_run=best['run'],
             argmin_lam=best['lam'], gap=gap, deff_lower=float(np.exp(-gap/P)), U_cert=A['U_cert'], gap_U_cert=A['U_cert']-I1,
             deff_lower_U_cert=float(np.exp(-(A['U_cert']-I1)/P)), check_UB_ge_I_T1=bool(ub >= I1), edge_tables_below=below,
             class_='discrete SoD class with the edge action set: a table assigns to every (step k >= 2, sigma/24 grid node r_i) one of the '
                    '100 edge actions (E^2, edges on grid nodes, at least 2 sigma from r_i); E[N] <= B on the sigma/24 discrete model at theta_0',
             basis='Lemma 3 (tangent at J(T_1)), eq. (12), Lemma 1; V_lambda from the DP of the scan (floating point)')
    # supplementary: the same bound from every scan of this round (tangent at that scan's reference table), for context only
    sup = {}
    for c, rounds in cells.items():
        for j, e in enumerate(rounds):
            p_ = os.path.join(rr.RES, f"scan_{e['label']}.json")
            if not os.path.exists(p_): continue
            s2 = json.load(open(p_)); I_ref = A['I_SoD'] if j == 0 else rounds[j-1]['I']
            sup[e['label']] = dict(actions=c, I_ref=I_ref, UB=I_ref-P+min(r['dual'] for r in s2['runs']))
    d['supplementary_all_scans'] = sup
    out(f"UB_SoD = {ub!r} (edge_j2 run {best['run']}, lam {best['lam']!r}); I(T_1) = {I1!r}; gap {gap!r}; exp(-gap/9) = {d['deff_lower']!r}; "
        f"UB_SoD >= I(T_1): {d['check_UB_ge_I_T1']}; side by side: U_cert = {A['U_cert']!r}, U_cert - I(T_1) = {d['gap_U_cert']!r}, "
        f"exp(-./9) = {d['deff_lower_U_cert']!r}; edge tables with sigma/24 E[N] <= B below UB_SoD: {below}")
    for k, v in sup.items(): out(f"UB_SoD supplementary: bound from scan {k} (tangent at its reference, I_ref {v['I_ref']!r}): {v['UB']!r}")
    return d


def stage_post(out):
    A = rr.archived_inputs()
    dp11.warmup(); des = dp11.Design11(out); gam0 = gamma0(des)
    inp = rr.load_state('inputs'); gate = rr.load_state('gate')
    cells = {c: rr.load_state(c)['rounds'] for c in ('old', 'edge')}
    # J of the previous round for the tangent point of round j
    for c, rounds in cells.items():
        for j, e in enumerate(rounds):
            if j > 0 and 'table' in e: e['J_prev'] = rounds[j-1]['J']
    # scan records: dsum of the final run
    for c, rounds in cells.items():
        for e in rounds:
            if 'table' not in e: continue
            sc = json.load(open(os.path.join(rr.RES, f"scan_{e['label']}.json")))
            fr = [r for r in sc['runs'] if r['run'] == e['run']][0]; e['dsum'] = fr['dsum']
    res = dict(inputs=inp, gate=gate, cells={}, tables={})
    keep = {}
    for c, rounds in cells.items():
        for e in rounds:
            if 'table' not in e: continue
            kk = (c == 'old' and e['round'] == 0) or (c == 'edge' and e.get('accepted', True))
            d, ed = analyse(e, c, des, A, gam0, out, keep_edges=kk)
            res['tables'][e['label']] = d
            if ed is not None: keep[e['label']] = ed
    # the shape statistics of the round-4 table against round4/results/shape_U_W.json (float32 masses there, float64 here)
    sh4 = json.load(open(os.path.join(EV, 'round4', 'results', 'shape_U_W.json')))
    d0 = res['tables'][cells['old'][0]['label']]['shape']
    res['shape_vs_round4'] = dict(edge_quantiles_round4=sh4['edge_quantiles'], edge_quantiles=d0['edge_quantiles'],
                                  max_abs_diff_quantiles=float(np.max(np.abs(np.array(sh4['edge_quantiles'])-np.array(d0['edge_quantiles'])))),
                                  edge_mean_round4=sh4['edge_mean'], edge_mean=d0['edge_mean'],
                                  mass_share_round4=[b['mass'] for b in sh4['bins']], mass_share=[b['mass_share'] for b in d0['bins']])
    out(f"SHAPE round-4 table vs shape_U_W.json: quantiles {d0['edge_quantiles']!r} vs {sh4['edge_quantiles']!r} "
        f"(max |diff| {res['shape_vs_round4']['max_abs_diff_quantiles']!r}); mass shares {res['shape_vs_round4']['mass_share']!r} vs "
        f"{res['shape_vs_round4']['mass_share_round4']!r}")
    # cells
    def last_accepted(rounds): return [e for e in rounds if 'table' in e and e.get('accepted', True)][-1]
    cell_tab = {'old/A0': cells['old'][0]['label'], 'old/iterated': last_accepted(cells['old'])['label'],
                'edge/A0': cells['edge'][0]['label'], 'edge/iterated': last_accepted(cells['edge'])['label']}
    for cn, lab in cell_tab.items():
        d = res['tables'][lab]
        res['cells'][cn] = dict(table=lab, dI=d['dI'], I=d['I'], g_prime=d['g_prime'], deff_lower=d['deff_lower'],
                                body_send_share=d['shape']['body_send_share'], body_info_per_send=d['shape']['body_info_per_send'],
                                EN24=d['EN24'], EN120=d['EN120'], budget120_pass=d['budget120_pass'],
                                edge_median=d['shape']['edge_median'], edge_q10=d['shape']['edge_q10'], edge_q90=d['shape']['edge_q90'],
                                tangent_overestimate=d['tangent_overestimate'], reference=d['reference'])
        out(f"CELL {cn}: table {lab}; dI {d['dI']!r}; g' {d['g_prime']!r}; D-eff {d['deff_lower']!r}; body share {d['shape']['body_send_share']!r}; "
            f"EN24 {d['EN24']!r}; EN120 {d['EN120']!r}")
    res['iteration_log'] = {c: [dict(round=e['round'], label=e['label'], I=e.get('I'), accepted=e.get('accepted'),
                                     gain=e.get('gain_over_previous'), EN=e.get('EN'), lam=e.get('lam'), lam_over_gbar=e.get('lam_over_gbar'),
                                     gbar=e.get('gbar'), n_runs=e.get('n_runs'), tangent_overestimate=e.get('tangent_overestimate'),
                                     status=e.get('status'), note=e.get('note')) for e in rounds] for c, rounds in cells.items()}
    # checks
    all_tabs = list(res['tables'].values())
    scans = {}
    for c, rounds in cells.items():
        for e in rounds:
            p_ = os.path.join(rr.RES, f"scan_{e['label']}.json")
            if os.path.exists(p_): scans[e['label']] = json.load(open(p_))
    feas = [(lab, r['run'], r['logdet']) for lab, sc in scans.items() for r in sc['runs'] if r['feasible']]
    over = [(lab, r['run'], r['logdet']) for lab, sc in scans.items() for r in sc['runs'] if not r['feasible'] and r['logdet'] > A['U_cert']]
    applies = [d for d in all_tabs if d['check6_applies']]
    mdev = max([r['mass_dev_max'] for sc in scans.values() for r in sc['runs']])
    mdev120 = max(d['mass_dev_max120'] for d in all_tabs)
    minm = min([r['min_m'] for sc in scans.values() for r in sc['runs']]+[d['min_m120'] for d in all_tabs])
    checks = {
        '1_regression_gate': dict(pass_=gate['pass_'], codes_bitwise=gate['codes_bitwise'], all_runs_bitwise=gate['all_runs_bitwise'],
                                  EN_absdiff=gate['EN_absdiff'], I_rel=gate['I_rel']),
        '2_forward_backward': dict(pass_=all(d['check2']['pass_'] for d in all_tabs),
                                   worst_F_rel=max(d['check2']['F_rel'] for d in all_tabs), worst_N_rel=max(d['check2']['N_rel'] for d in all_tabs),
                                   worst_total_rel=max(d['check2']['total_rel'] for d in all_tabs),
                                   worst_dp_vs_forward_rel=max(d['check2']['dp_vs_forward_rel'] for d in all_tabs), tol=FB_TOL),
        '3_conservation': dict(pass_=bool(mdev <= e11.TOL24 and mdev120 <= e11.TOL120 and minm >= 0), max_mass_dev_sigma24_all_runs=mdev,
                               max_mass_dev_sigma120=mdev120, min_m=minm, n_dp_runs=sum(len(sc['runs']) for sc in scans.values()),
                               note='asserted at every step of every forward (sigma/24: 1e-12, sigma/120: 1e-10, DECISION 11-5), '
                                    'together with min m >= 0 and zero mass at the first and last grid node; a failure would have stopped the run'),
        '5_edges': dict(pass_=all(d['check5']['pass_'] for d in all_tabs), per_table={d['label']: d['check5'] for d in all_tabs}),
        '6_ucert': dict(pass_=all(d['I'] <= A['U_cert'] for d in applies), U_cert=A['U_cert'],
                        rule='applies only to tables with sigma/120 E[N] <= B (clarification 2): every reported table that passes the '
                             'sigma/120 budget; scan candidates are not re-evaluated on sigma/120 and are reported for information only',
                        per_table={d['label']: dict(I=d['I'], EN120=d['EN120'], applies=d['check6_applies'], margin=A['U_cert']-d['I']) for d in all_tabs},
                        max_I_applicable=max(d['I'] for d in applies) if applies else None,
                        info_scan_candidates_sigma24_feasible=dict(n=len(feas), max_I=max(x[2] for x in feas), at=max(feas, key=lambda x: x[2])[:2],
                                                                   n_above_Ucert=sum(x[2] > A['U_cert'] for x in feas)),
                        info_scan_candidates_over_budget_above_Ucert=len(over)),
        '7_budget120': dict(pass_=all(d['budget120_pass'] for d in all_tabs), per_table={d['label']: dict(EN120=d['EN120'], slack120=d['slack120'])
                                                                                       for d in all_tabs}),
    }
    if not checks['6_ucert']['pass_']:
        out("CHECK 6 FAILED: a table with sigma/120 E[N] <= B exceeds U_cert -- STOP (Round 10 or this round is wrong)")
        rr.jdump(dict(res, checks=checks), os.path.join(rr.RES, 'round11_stopped_check6.json'))
        raise rr.Stop("check 6 failed")
    # best cell: largest I among the cells whose table passes sigma/120 budget and the per-table checks
    elig = {cn: c for cn, c in res['cells'].items() if c['budget120_pass'] and res['tables'][c['table']]['check2']['pass_']
            and res['tables'][c['table']]['check5']['pass_'] and res['tables'][c['table']]['check6'] is True}
    best = max(elig, key=lambda cn: (elig[cn]['I'], -list(res['cells']).index(cn)))
    res['best_cell'] = best
    bt = res['tables'][res['cells'][best]['table']]
    # check 4: Monte Carlo on the best table
    bentry = [e for c, rounds in cells.items() for e in rounds if e.get('label') == bt['label']][0]
    t0 = time.perf_counter()
    mc = e11.monte_carlo(bentry['table'], acts_of(bt['cell']), dict(EN=bt['EN24'], S=bt['S'], logdet=bt['I']))
    mc['wall_s'] = time.perf_counter()-t0
    # z-scores against the sigma/120 re-evaluation of the same continuous policy (DECISION 11-7: pass criterion); sigma/24 alongside
    mc['EN_eval120'] = bt['EN120']; mc['S_eval120'] = bt['S120']; mc['logdet_eval120'] = bt['I120']
    mc['z120_EN'] = (bt['EN120']-mc['EN_mc'])/mc['EN_se']; mc['z120_S'] = (bt['S120']-mc['S_mc'])/mc['S_se']
    mc['z120_logdet'] = (bt['I120']-mc['logdet_mc'])/mc['logdet_batch_se']
    mc['pass_sigma24'] = bool(abs(mc['z_EN']) <= MC_Z and abs(mc['z_S']) <= MC_Z and abs(mc['z_logdet']) <= MC_Z)
    mc['pass_'] = bool(abs(mc['z120_EN']) <= MC_Z and abs(mc['z120_S']) <= MC_Z and abs(mc['z120_logdet']) <= MC_Z and mc['n_clipped'] == 0)
    mc['criterion'] = '|z| <= 3 for E[N], S, log det against the sigma/120 values (DECISION 11-7); sigma/24 z-scores reported as pass_sigma24'
    checks['4_monte_carlo'] = mc
    out(f"MC best table {bt['label']}: {mc}")
    res['checks'] = dict(sorted(checks.items())); res['all_checks_pass'] = all(v['pass_'] for v in checks.values())
    for k, v in res['checks'].items(): out(f"CHECK {k}: {'PASS' if v['pass_'] else 'FAIL'}")
    # addendum: certified bound within the discrete edge SoD class from the edge round-2 scan (tangent A_1 = J(T_1)^-1), no new run
    res['ub_sod'] = ub_sod(cells, A, res, out)
    # verdict (plan §7)
    dI_best = bt['I']-A['I_old']; gp = bt['g_prime']
    case = 1 if dI_best < 0.5 else (2 if gp < 2 else 3)
    res['verdict'] = dict(delta_I=dI_best, I_best=bt['I'], I_old=A['I_old'], best_cell=best, g_prime=gp, deff_lower=bt['deff_lower'], case=case,
                          text={1: 'delta I < 0.5 nat: report the actual gain; this round did not reach the pre-set 0.5 nat improvement threshold; main line unchanged',
                                2: 'delta I >= 0.5 and g\' < 2 nat: Round 12 (margin design and two-stage confirmation with the new action set)',
                                3: 'delta I >= 0.5 and g\' >= 2 nat: supplementary result in IV-E; main line unchanged'}[case])
    out(f"VERDICT case {case}: delta I = I_best - I_old = {dI_best!r} (best cell {best}), g' = {gp!r}, D-eff >= {bt['deff_lower']!r}")
    # predictions
    ea = res['cells']['edge/A0']; ta = res['tables'][ea['table']]
    oi = res['cells']['old/iterated']; ei = res['cells']['edge/iterated']
    r3 = [e for rounds in cells.values() for e in rounds if e.get('round') == 3 and e.get('accepted')]
    pr = [
        ('edge/A0 dI in [8.5, 10.0]', ea['dI'], 8.5 <= ea['dI'] <= 10.0),
        ('edge/A0 body-transmission share < 5 %', ea['body_send_share'], ea['body_send_share'] < 0.05),
        ('edge/A0 facing-edge 10-90 % range within [1.5, 2.3] sigma', [ea['edge_q10'], ea['edge_q90']], 1.5 <= ea['edge_q10'] and ea['edge_q90'] <= 2.3),
        ('iteration of A, old: +0 to +0.5 nat', oi['I']-res['cells']['old/A0']['I'], 0 <= oi['I']-res['cells']['old/A0']['I'] <= 0.5),
        ('iteration of A, edge: +0 to +0.5 nat', ei['I']-ea['I'], 0 <= ei['I']-ea['I'] <= 0.5),
        ('no acceptance after round 2', len(r3), len(r3) == 0),
        ('old/iterated dI in [7.8, 8.3]', oi['dI'], 7.8 <= oi['dI'] <= 8.3),
        ("best cell g' in [2.0, 3.4] nat", gp, 2.0 <= gp <= 3.4),
        ('best cell D-eff in [69 %, 80 %]', bt['deff_lower'], 0.69 <= bt['deff_lower'] <= 0.80),
        ('tangent overestimate E_0 under edge > 4 nat', ta['tangent_overestimate'], ta['tangent_overestimate'] > 4),
    ]
    res['predictions'] = [dict(item=a, value=v, hit=bool(h)) for a, v, h in pr]
    for p_ in res['predictions']: out(f"PREDICTION {p_['item']}: value {p_['value']!r} -> {'HIT' if p_['hit'] else 'MISS'}")
    # figure: old table vs the best edge table (largest I among the edge rounds' cell tables)
    edge_best_lab = max((res['cells']['edge/A0']['table'], res['cells']['edge/iterated']['table']), key=lambda l: res['tables'][l]['I'])
    fig_path = os.path.join(rr.RES, 'fig_edge_positions.png')
    lab_new = f"edge action set, {'A$_0$' if edge_best_lab == res['cells']['edge/A0']['table'] else 'iterated A'} ({edge_best_lab})"
    figure(keep[res['cells']['old/A0']['table']], keep[edge_best_lab], lab_new, res['tables'][res['cells']['old/A0']['table']]['shape']['edge_quantiles'],
           res['tables'][edge_best_lab]['shape']['edge_quantiles'], fig_path)
    res['figure'] = dict(path='results/fig_edge_positions.png', old=res['cells']['old/A0']['table'], edge=edge_best_lab)
    # packed tables
    res['packed'] = {}
    for name, cn in (('table_edge_A0.npz', 'edge/A0'), ('table_edge_iter.npz', 'edge/iterated'), ('table_old_iter.npz', 'old/iterated')):
        lab = res['cells'][cn]['table']; e = [x for rounds in cells.values() for x in rounds if x.get('label') == lab][0]
        res['packed'][name] = pack(e, 'edge' if cn.startswith('edge') else 'old', os.path.join(rr.RES, name), note_extra=f"cell {cn}, table {lab}.")
        out(f"PACKED {name}: {res['packed'][name]}")
    # json (drop paths into the work directory)
    res['decisions_ref'] = 'reports_CN/report_round11_CN.md §7'
    for d in res['tables'].values(): d.pop('table', None)
    rr.jdump(res, os.path.join(rr.RES, 'round11.json'))
    out(f"WROTE results/round11.json; all checks pass: {res['all_checks_pass']}")
    return res
