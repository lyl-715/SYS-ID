"""Round 4, Part C: evaluation of the final tables, sigma/48 refinement check, comparison table, figures
(spec reports_CN/plan_round4_CN.md v3.1, frozen; §1.4, §5), plus two diagnostics the spec does not ask for (DECISION C-5,
C-6): a Monte Carlo cross-check of the final tables at the runtime lookup rule, and descriptive statistics of the table
shape for V4.

  C1  final table of each case through Part A (general path, record and J from the accepted dr.evaluate): E[N], S, log det J,
      Delta m_D vs the same input's constant-Delta SoD (same grid), m_E and nine lambda_i vs J_C (uniform input, constant-Delta
      SoD).
  C2  the frozen table moved to the sigma/48 grid (evaluator_window.FineTable: nearest coarse node, ties -> smaller index,
      outside the stored range -> boundary node; physical half-widths, grid counts doubled, window centred at the actual r) and
      re-evaluated; the reference SoD at sigma/48 is evaluator_fast.evaluate_fast(name, 48) (DECISION C-2). Changes of Delta m_D
      and E[N]; budget verdict E[N] <= B (frozen B) on each grid; a flipped verdict -> 预算可行性对网格敏感／未定; a change
      > 0.01 nat or > 0.05 sends -> 精度未定.
  C3  table {U, P5514} x {constant Delta, S table, weighted table}; heat maps of Delta^-, Delta^+ over (k, r) for the uniform
      weighted design; E[q_k] vs k with tr(h_k h_k^T A).

Usage (from this directory): python3 part_c.py c1 | c2 | c3 | mc | shape | all
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_window as ew
import evaluator_fast as ef
import trigger_dp as td
dr = ew.dr
s, K, B = ew.s, ew.K, ew.B_BUDGET
CASES = td.CASES
OBJ_NAME = {'S': 'S 表', 'W': '加权表'}


def final_record(case):
    tag, jpath, _ = td.case_paths(case)
    st = json.load(open(jpath)); fr = st['final_run']
    return st, [x for x in st['runs'] if x['run'] == fr][0]

def load_table(case, des):
    st, r = final_record(case)
    return des.table(np.load(r['table'], mmap_mode='r')), r, st

def log_rec(out, pre, rec, keys=('EN', 'EN_P', 'S', 'logdet', 'dmD', 'mE', 'mass_dev_max', 'min_m', 'inv_S', 'inv_J', 'inv_w', 'path')):
    for k in keys:
        if k in rec: out(f"{pre} {k} {rec[k]!r}")
    for i, v in enumerate(rec.get('lam', [])): out(f"{pre} lam{i+1} {v!r}")


# --- C1 ----------------------------------------------------------------------------------------------------------------
def c1(out):
    ew._warmup()
    recU, JC, _ = ew.evaluate_table('U', ew.ConstTable(ew.SOD_M, ew.SOD_M))
    res = {}; sod = {}
    for name in ew.INPUTS:
        rs, Js, rr = ew.evaluate_table(name, ew.ConstTable(ew.SOD_M, ew.SOD_M))
        ew.rel_record(rs, Js, Js, JC); sod[name] = (rs, Js)
        np.save(os.path.join(ew.RES, f'J_{name}_SoD.npy'), Js)
        res[f'{name}/SoD'] = {k: rs[k] for k in ('EN', 'S', 'logdet', 'dmD', 'mE', 'lam', 'path')}
        res[f'{name}/SoD']['dmD_vs_U'] = rs['logdet']-recU['logdet']
        log_rec(out, f"C1 {name}/SoD", rs); out(f"C1 {name}/SoD dmD_vs_uniform_SoD {rs['logdet']-recU['logdet']!r}")
    for case in CASES:
        name, obj = case.split('/')
        des = td.Design(name, out); tab, r, st = load_table(case, des)
        t0 = time.perf_counter(); rec, J, rr = ew.evaluate_table(name, tab); t1 = time.perf_counter()-t0
        ew.rel_record(rec, J, sod[name][1], JC)
        np.save(os.path.join(ew.RES, f"J_{case.replace('/', '_')}.npy"), J)
        gam, gbar, g1 = des.gamma(obj)
        d = {k: rec[k] for k in ('EN', 'EN_P', 'S', 'logdet', 'dmD', 'mE', 'lam', 'mass_dev_max', 'min_m', 'inv_S', 'inv_J', 'inv_w', 'path')}
        d.update(run=r['run'], lam_dp=r['lam'], lam_over_gbar=r['lam_over_gbar'], F=float(gam[1:]@rr['W'][1:]), slack=B-rec['EN'],
                 feasible=bool(rec['EN'] <= B), assert_fail=int((~rr['assert_ok']).sum()), dmD_vs_U=rec['logdet']-recU['logdet'],
                 EN_equals_scan=bool(rec['EN'] == r['EN']), eval_s=t1, notes=st.get('notes'))
        res[case] = d
        log_rec(out, f"C1 {case}", rec)
        for k in ('run', 'lam_dp', 'lam_over_gbar', 'F', 'slack', 'feasible', 'assert_fail', 'dmD_vs_U', 'EN_equals_scan', 'eval_s', 'notes'):
            out(f"C1 {case} {k} {d[k]!r}")
    td._atomic_json(res, os.path.join(ew.RES, 'c1.json'))
    return res


# --- C2 ----------------------------------------------------------------------------------------------------------------
def c2(out):
    ew._warmup()
    c1r = json.load(open(os.path.join(ew.RES, 'c1.json')))
    res = {}; sod48 = {}
    for name in ew.INPUTS:
        t0 = time.perf_counter(); r48, J48, rr48 = ef.evaluate_fast(name, 48); t1 = time.perf_counter()-t0
        sod48[name] = (r48, J48)
        out(f"C2 {name}/SoD n=48 (evaluator_fast, accepted operator) EN {r48['EN']!r} S {r48['S']!r} logdet {r48['logdet']!r} "
            f"assert_fail {int((~rr48['assert_ok']).sum())} runtime {t1:.1f} s")
        rg, Jg, rrg = ew.evaluate_table(name, ew.ConstTable(2*ew.SOD_M, 2*ew.SOD_M), n=48)
        out(f"C2 {name}/SoD n=48 general path (960, 960) diagnostic: EN rel {abs(rg['EN']-r48['EN'])/r48['EN']!r} "
            f"S rel {abs(rg['S']-r48['S'])/r48['S']!r} logdet diff {rg['logdet']-r48['logdet']!r}")
        res[f'{name}/SoD'] = dict(EN48=r48['EN'], S48=r48['S'], logdet48=r48['logdet'], EN24=c1r[f'{name}/SoD']['EN'],
                                  general_vs_accepted_EN_rel=abs(rg['EN']-r48['EN'])/r48['EN'])
    for case in CASES:
        name, obj = case.split('/')
        des = td.Design(name, out); tab, r, st = load_table(case, des)
        ft = ew.FineTable(tab)
        # nearest-node rule (DECISION C-3): the grids are nested exactly (fine node 2i == coarse node i bitwise), so fine node
        # 2i + 1 is an exact-arithmetic tie between coarse i and i + 1 -> smaller index i = j // 2; verified on the positions:
        # even nodes coincide bitwise, odd nodes are equidistant to within 4 ulp of the node values
        xk, Hk = dr.input_arrays(name); zf, dzf = dr.make_grid(xk, 48, 8*ew.D, 'B'); zc = des.z
        n2 = min(len(zf[0::2]), len(zc)); even_ok = bool(np.array_equal(zf[0::2][:n2], zc[:n2]))
        jo = np.arange(1, len(zf), 2); ic = jo//2; sel = ic+1 < len(zc); jo, ic = jo[sel], ic[sel]
        tie_dev = float(np.max(np.abs(np.abs(zf[jo]-zc[ic])-np.abs(zc[ic+1]-zf[jo]))))
        near_ok = bool(even_ok and tie_dev <= 4*np.spacing(np.abs(zc).max()))
        t0 = time.perf_counter(); rec, J, rr = ew.evaluate_table(name, ft, n=48); t1 = time.perf_counter()-t0
        dmD48 = float(rec['logdet']-np.linalg.slogdet(sod48[name][1])[1])
        c = c1r[case]
        d = dict(EN24=c['EN'], EN48=rec['EN'], dEN=rec['EN']-c['EN'], dmD24=c['dmD'], dmD48=dmD48, ddmD=dmD48-c['dmD'],
                 S24=c['S'], S48=rec['S'], logdet24=c['logdet'], logdet48=rec['logdet'],
                 verdict24=('可行' if c['EN'] <= B else '不可行'), verdict48=('可行' if rec['EN'] <= B else '不可行'),
                 slack24=B-c['EN'], slack48=B-rec['EN'], assert_fail=int((~rr['assert_ok']).sum()), mass_dev_max=rec['mass_dev_max'],
                 min_m=rec['min_m'], nearest_rule_ok=near_ok, even_nodes_bitwise=even_ok, odd_tie_max_dev=tie_dev, G48=rec['G'], eval_s=t1)
        flags = []
        if d['verdict24'] != d['verdict48']: flags.append('预算可行性对网格敏感／未定')
        if abs(d['ddmD']) > 0.01 or abs(d['dEN']) > 0.05: flags.append('精度未定')
        d['flags'] = flags
        res[case] = d
        for k, v in d.items(): out(f"C2 {case} {k} {v!r}")
    td._atomic_json(res, os.path.join(ew.RES, 'c2.json'))
    return res


# --- MC diagnostic (DECISION C-5) --------------------------------------------------------------------------------------
MC_R, MC_BATCH, MC_SEED = 10000, 20, 20261025

def _mc_batch(args):
    name, kind, tpath, start, width, R, seed = args
    xk, Hk = dr.input_arrays(name); z, dz = dr.make_grid(xk, ew.N_GRID, 8*ew.D, 'B'); z0 = z[0]
    rng = np.random.default_rng(seed)
    codes = np.load(tpath, mmap_mode='r') if tpath else None
    r = xk[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)   # Wk = sum over runs of c_k (c_1 = 1)
    for p in range(1, K):
        if codes is None:
            Dm = Dp = ew.SOD_M*dz
        else:
            i = np.ceil((r-z0)/dz-0.5).astype(np.int64)                     # nearest node, ties -> smaller index
            o = np.clip(i, start[p-1], start[p-1]+width[p-1]-1)-start[p-1]  # outside the stored range -> boundary node
            mm, mp = ew.code_to_m(codes[p-1][o]); Dm = mm*dz; Dp = mp*dz
        lo = r-Dm; hi = r+Dp
        cev, cno = dr.cfun((lo-xk[p])/s, (hi-xk[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = xk[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S

def mc(out):
    from multiprocessing import Pool
    jobs = []; labels = []
    for name in ew.INPUTS:
        labels.append((f'{name}/SoD', name, None, None, None))
    for case in CASES:
        name, obj = case.split('/'); st, r = final_record(case)
        xk, Hk, z, dz, a, b = ew.design_grid(name)
        labels.append((case, name, r['table'], a[1:], (b-a+1)[1:]))
    for li, (lab, name, tp, st_, wd) in enumerate(labels):
        for bi in range(MC_BATCH):
            jobs.append((name, lab, tp, st_, wd, MC_R//MC_BATCH, MC_SEED+1000*li+bi))
    t0 = time.perf_counter()
    with Pool(4) as pool: outs = pool.map(_mc_batch, jobs)
    out(f"MC R = {MC_R} per row in {MC_BATCH} batches, seed {MC_SEED} + 1000 row + batch; runtime lookup: nearest node (ties -> smaller), "
        f"outside the stored range -> boundary node, window centred at the continuous r; {time.perf_counter()-t0:.1f} s")
    c1r = json.load(open(os.path.join(ew.RES, 'c1.json'))); res = {}
    for li, (lab, name, tp, st_, wd) in enumerate(labels):
        bs = outs[li*MC_BATCH:(li+1)*MC_BATCH]
        Ns = np.concatenate([x[1] for x in bs]); Ss = np.concatenate([x[2] for x in bs])
        Wk = sum(x[0] for x in bs)/MC_R
        xk, Hk = dr.input_arrays(name)
        J = (Hk*Wk[:, None]).T@Hk/s**2; ld = float(np.linalg.slogdet(J)[1])
        ldb = np.array([np.linalg.slogdet((Hk*(x[0]/(MC_R//MC_BATCH))[:, None]).T@Hk/s**2)[1] for x in bs])
        ev = c1r[lab]
        d = dict(EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(MC_R)), S_mc=float(Ss.mean()), S_se=float(Ss.std(ddof=1)/np.sqrt(MC_R)),
                 logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(MC_BATCH)), EN_eval=ev['EN'], S_eval=ev['S'], logdet_eval=ev['logdet'])
        d['z_EN'] = (d['EN_eval']-d['EN_mc'])/d['EN_se']; d['z_S'] = (d['S_eval']-d['S_mc'])/d['S_se']
        d['z_logdet'] = (d['logdet_eval']-d['logdet_mc'])/d['logdet_batch_se']
        res[lab] = d
        for k, v in d.items(): out(f"MC {lab} {k} {v!r}")
    td._atomic_json(res, os.path.join(ew.RES, 'mc.json'))
    return res


# --- shape statistics for V4 (DECISION C-6) and the C3 figures ------------------------------------------------------------
def table_fields(case):
    """mass m_k on the support window (float32), half-widths (sigma units) and offsets (r - x_k)/sigma for every step."""
    name, obj = case.split('/')
    des = td.Design(name, print); tab, r, st = load_table(case, des)
    fw = ew.forward(des.z, des.dz, s, des.xk, des.a, des.b, tab, record_window=True)
    return des, tab, fw

def shape(out, des=None, tab=None, fw=None, case='U/W'):
    if des is None: des, tab, fw = table_fields(case)
    name, obj = case.split('/'); gam, gbar, g1 = des.gamma('W')
    meanw = np.full(K, np.nan); near_bins = [0, 2, 5, 10, 20, 40, 88]
    acc_n = np.zeros(len(near_bins)-1); acc_f = np.zeros(len(near_bins)-1); acc_m = np.zeros(len(near_bins)-1)
    acc_e = np.zeros(len(near_bins)-1); emar = []; ewt = []
    for p in range(1, K):
        A, Bb = int(des.a[p]), int(des.b[p]); m = fw['m_window'][p].astype(float)
        Mm, Mp = tab.rows(p, A, Bb); dlo = Mm/24.0; dhi = Mp/24.0
        off = (des.z[A:Bb+1]-des.xk[p])/s
        tot = m.sum(); meanw[p] = float(m@(dlo+dhi))/tot
        near = np.where(off > 0, dlo, dhi); far = np.where(off > 0, dhi, dlo)    # near side: the side of x_k as seen from r
        bi = np.clip(np.searchsorted(near_bins, np.abs(off), side='right')-1, 0, len(near_bins)-2)
        acc_n += np.bincount(bi, m*near, len(near_bins)-1); acc_f += np.bincount(bi, m*far, len(near_bins)-1)
        acc_m += np.bincount(bi, m, len(near_bins)-1)
        edge = near-np.abs(off)                                                # distance from x_k to the near edge (sigma)
        acc_e += np.bincount(bi, m*edge, len(near_bins)-1); sel = m > 0; emar.append(edge[sel]); ewt.append(m[sel])
    from scipy.stats import spearmanr
    rho = spearmanr(meanw[1:], gam[1:]).statistic
    out(f"SHAPE {case} Spearman rank correlation over k = 2..K of the mass-weighted mean full width (Delta^- + Delta^+) with "
        f"gamma_k = sigma^-2 h_k^T A h_k: {float(rho)!r}")
    q = np.quantile(gam[1:], [0.25, 0.75])
    lo_w = float(np.nanmean(meanw[1:][gam[1:] <= q[0]])); hi_w = float(np.nanmean(meanw[1:][gam[1:] >= q[1]]))
    out(f"SHAPE {case} mean full width (sigma) over steps with gamma_k in the lowest quarter {lo_w!r}, highest quarter {hi_w!r}")
    emar = np.concatenate(emar); ewt = np.concatenate(ewt); o_ = np.argsort(emar); cw = np.cumsum(ewt[o_])/ewt.sum()
    eq = [float(emar[o_][np.searchsorted(cw, qq)]) for qq in (0.1, 0.25, 0.5, 0.75, 0.9)]
    out(f"SHAPE {case} distance from x_k to the near window edge (near-side half-width - |r - x_k|), mass-weighted over all "
        f"(k, i): mean {float(ewt@emar/ewt.sum())!r} sigma; quantiles 10/25/50/75/90 % {eq!r} sigma")
    rows = []
    for j in range(len(near_bins)-1):
        if acc_m[j] > 0:
            rows.append(dict(bin=f"[{near_bins[j]}, {near_bins[j+1]})", mass=float(acc_m[j]/(K-1)), near=float(acc_n[j]/acc_m[j]), far=float(acc_f[j]/acc_m[j]),
                             edge=float(acc_e[j]/acc_m[j])))
            out(f"SHAPE {case} |r - x_k|/sigma in {rows[-1]['bin']}: mass share {rows[-1]['mass']!r}, mass-weighted near-side half-width "
                f"{rows[-1]['near']!r} sigma, far-side {rows[-1]['far']!r} sigma, near edge beyond x_k by {rows[-1]['edge']!r} sigma")
    return dict(spearman_width_gamma=float(rho), width_low_gamma=lo_w, width_high_gamma=hi_w, bins=rows, meanw=meanw,
                edge_mean=float(ewt@emar/ewt.sum()), edge_quantiles=eq)

def c3(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    c1r = json.load(open(os.path.join(ew.RES, 'c1.json')))
    hdr = '| 输入 | 触发 | $\\mathbb E[N]$ | $S$ | $\\log\\det J$ | $\\Delta m_D$（相对同输入常数 $\\Delta$） | $m_E$（相对均匀输入常数 $\\Delta$） |\n|---|---|---|---|---|---|---|\n'
    body = ''
    for name in ew.INPUTS:
        for lab, key in (('常数 $\\Delta$', f'{name}/SoD'), ('$S$ 表', f'{name}/S'), ('加权表', f'{name}/W')):
            d = c1r[key]
            body += f"| {name} | {lab} | {d['EN']:.4f} | {d['S']:.3f} | {d['logdet']:.5f} | {d['dmD']:+.5f} | {d['mE']:.5f} |\n"
            out(f"C3 {key} EN {d['EN']!r} S {d['S']!r} logdet {d['logdet']!r} dmD {d['dmD']!r} mE {d['mE']!r}")
    open(os.path.join(HERE, 'c3_table.md'), 'w').write(hdr+body)
    # figures for the uniform weighted design
    des, tab, fw = table_fields('U/W')
    sh = shape(out, des, tab, fw, 'U/W')
    json.dump({k: v for k, v in sh.items() if k != 'meanw'}, open(os.path.join(ew.RES, 'shape_U_W.json'), 'w'), indent=1)
    gam, gbar, g1 = des.gamma('W'); trA = gam*s**2                       # tr(h_k h_k^T A) = h_k^T A h_k
    ks = np.arange(2, K+1, 5); OFF = np.arange(-88*24, 88*24+1, 4)       # every 5th step, every 4th node offset
    img = {k_: np.full((OFF.size, ks.size), np.nan) for k_ in ('lo', 'hi')}
    for c_, k in enumerate(ks):
        p = k-1; A, Bb = int(des.a[p]), int(des.b[p]); Mm, Mp = tab.rows(p, A, Bb); m = fw['m_window'][p]
        i0 = int(round((des.xk[p]-des.z[A])/des.dz))                    # node nearest x_k (offset 0)
        idx = i0+OFF; okk = (idx >= 0) & (idx < Bb-A+1)
        sel = idx[okk]; reach = m[sel] > 0
        vlo = np.where(reach, Mm[sel]/24.0, np.nan); vhi = np.where(reach, Mp[sel]/24.0, np.nan)
        img['lo'][okk, c_] = vlo; img['hi'][okk, c_] = vhi
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'; C1_, C2_ = '#2a78d6', '#eb6834'
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 9})
    # heat maps: sequential single hue (light = narrow, dark = wide), grey = nodes with zero forward mass
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues', plt.get_cmap('Blues')(np.linspace(0.12, 1.0, 256)))
    cmap.set_bad('#d9d6cf')
    for ax, k_, lab in ((axs[0], 'lo', r'$\Delta^-/\sigma$ (lower half-width)'), (axs[1], 'hi', r'$\Delta^+/\sigma$ (upper half-width)')):
        im = ax.imshow(np.ma.masked_invalid(img[k_]), aspect='auto', origin='lower', cmap=cmap, vmin=2, vmax=40,
                       extent=[ks[0]*1e-3, ks[-1]*1e-3, OFF[0]/24, OFF[-1]/24], interpolation='nearest')
        ax.set_ylabel(r'offset $(r-\bar x_k)/\sigma$'); cb = fig.colorbar(im, ax=ax); cb.set_label(lab)
        ax.set_title(f"Uniform input, weighted table: {lab}; grey = zero mass under the table", fontsize=10, color=INK, loc='left')
    axs[1].set_xlabel('time t = k dt (s)')
    fig.savefig(os.path.join(HERE, 'fig_window_heatmap.png'), dpi=130); plt.close(fig)
    # E[q_k] and tr(h_k h_k^T A): one axis per measure (no dual axis); the overlay indexes both to their mean over k >= 2
    t = np.arange(1, K+1)*1e-3; q = fw['Q'][1:]; tr_ = trA[1:]
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axs[0].plot(t[1:], q/q.mean(), lw=0.8, color=C1_, label='E[q_k] / mean')
    axs[0].plot(t[1:], tr_/tr_.mean(), lw=1.2, color=C2_, label='tr(h_k h_k^T A) / mean')
    axs[0].set_ylim(0, 1.25*max(float((q/q.mean()).max()), float((tr_/tr_.mean()).max())))
    axs[0].legend(frameon=False, loc='upper center', ncol=2, labelcolor=INK)
    axs[0].set_ylabel('indexed to mean (k >= 2)')
    axs[0].set_title('Uniform input, weighted table: per-step send probability E[q_k] overlaid with tr(h_k h_k^T A), A = J_SoD^-1',
                     fontsize=10, color=INK, loc='left')
    axs[1].plot(t[1:], q, lw=0.8, color=C1_); axs[1].set_ylabel('E[q_k]')
    axs[2].plot(t[1:], tr_, lw=1.2, color=C2_); axs[2].set_ylabel('tr(h_k h_k^T A)'); axs[2].set_xlabel('time t = k dt (s)')
    for ax in axs:
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(HERE, 'fig_sendprob_trA.png'), dpi=130); plt.close(fig)
    from scipy.stats import spearmanr
    rho = spearmanr(fw['Q'][1:], trA[1:]).statistic
    out(f"C3 U/W Spearman rank correlation of E[q_k] with tr(h_k h_k^T A) over k = 2..K: {float(rho)!r}")
    out(f"C3 figures fig_window_heatmap.png, fig_sendprob_trA.png; table c3_table.md")
    # context for the report (DECISION C-6): how much information a non-sending step carries under the constant threshold
    xk, Hk, z, dz, a, b = ew.design_grid('U')
    fs = ew.forward(z, dz, s, xk, a, b, ew.ConstTable(ew.SOD_M, ew.SOD_M), record_window=True)
    vals, wts = [], []
    for p in range(1, K):
        A, Bb = int(a[p]), int(b[p]); m = fs['m_window'][p].astype(float); sel = m > 0; zz = z[A:Bb+1][sel]
        cev, cno = dr.cfun((zz-ew.SOD_M*dz-xk[p])/s, (zz+ew.SOD_M*dz-xk[p])/s); vals.append(cev+cno); wts.append(m[sel])
    v = np.concatenate(vals); w = np.concatenate(wts)
    out(f"C3 U/SoD mass-weighted share of (k, node) with c < 0.01: {float(w[v < 0.01].sum()/w.sum())!r}; with c < 0.05: {float(w[v < 0.05].sum()/w.sum())!r}")
    cev, cno = dr.cfun(np.array([-2.0]), np.array([30.0]))
    out(f"C3 single step, near edge 2 sigma beyond x_k (alpha = -2, beta = 30): c {float(cev[0]+cno[0])!r}, q {float(1-(dr.ndtr(30.0)-dr.ndtr(-2.0)))!r}")
    for case in CASES:
        if case == 'U/W': continue
        d2, t2, f2 = table_fields(case); sh2 = shape(out, d2, t2, f2, case)
        json.dump({k: v for k, v in sh2.items() if k != 'meanw'}, open(os.path.join(ew.RES, f"shape_{case.replace('/', '_')}.json"), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['c1', 'c2', 'c3', 'mc', 'all'])
    A_ = ap.parse_args()
    stages = ['c1', 'c2', 'c3', 'mc'] if A_.stage == 'all' else [A_.stage]
    for stg in stages:
        sys.stdout = dr.Tee(os.path.join(ew.LOGS, f'part_{stg}.log'))
        print(f"=== part_c.py {stg} started {time.strftime('%Y-%m-%d %H:%M:%S')}; B = {B!r}")
        t0 = time.perf_counter(); {'c1': c1, 'c2': c2, 'c3': c3, 'mc': mc}[stg](print)
        print(f"=== part_c.py {stg} done in {time.perf_counter()-t0:.1f} s")
        sys.stdout = sys.__stdout__
