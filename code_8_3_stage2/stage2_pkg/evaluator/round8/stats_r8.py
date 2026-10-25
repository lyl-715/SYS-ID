"""Round 8, Part C: statistics, paired bootstrap, verdict, figures, prediction scoring (spec reports_CN/plan_round8_CN.md
v3.1, frozen; §2, §3 items 5-6, §5). 本场景、固定设计规则下的序贯仿真结果.

Reads results/archive_round8.jsonl (R lines), results/sod_reference.json (b_1, J_1), results/partA_checks.json. For each
analysis (P = frozen acceptance rule, primary, binding; S = supplementary rule, DECISION E-1, not binding):
  final accuracy (C1)   all R and the "both succeeded" subset (stage-1 estimate, design, both joint estimates converged, no
                        fallback; subset normalised by its own size: covariance n - 1, MSE n): sample covariance C, MSE matrix
                        M = (1/n) sum (theta - theta_0)(theta - theta_0)^T, bias, D = -log det M (primary), -log det C, per-parameter
                        bias, RMSE, |bias|/RMSE; Delta D = D_seq - D_SoD+SoD.
  information (C2)      J_seq = J_1 + (1/R) sum J_table(T^(r); theta_0), J_SoD+SoD = 2 J_1; |-log det M - log det J| ("经验精度与信息
                        参考的差异"); stage 1 |-log det M_1 - log det J_1|.
  transmissions (C3)    N_1, N_2, totals; Pr{E[N_2 | T, theta_0] > B}; Pr{N_2 > B}; b_1; b_seq = b_1 + mean E[N_2 | T, theta_0];
                        2 b_1; both against 2B.
  stage 1 (C4)          eps_1 quantiles 50/90/99 % (numpy linear), fraction > 0.5; failure rates.
  diagnostics (C5)      Hessian-FIM residual D_r = H_r - J_table: mean, element-wise se, z, |z| > 4 flags, whitened spectral norm
                        ||J^-1/2 Dbar J^-1/2||_2 (J = mean J_table); paired d_r = N_2 - E[N_2 | T, theta_0]: dbar, s_d/sqrt(R), flag
                        |dbar| > 3 s_d/sqrt(R). Also for the control's stage 2 against (J_1, b_1) (extra).
  bootstrap (C6)        B_boot = 2000, default_rng(20281028), idx = rng.integers(0, R, R) per resample (both schemes together):
                        95 % percentile intervals of Delta D, b_seq, both proportions; subset bootstrap with the same idx filtered
                        by the success flags ("不可计算" counted).
  verdict (C7)          spec §5, cases 1-4 in order, interval rules with the point estimate.
Outputs: results/stats_round8.json, results/bootstrap_round8.npz, results/verdict.json, logs/partC_stats.log, logs/verdict.log,
fig_eps1_hist.png, fig_EN2_vs_eps1.png, fig_final_error_box.png.

Usage (from this directory): python3 stats_r8.py            (python3 stats_r8.py --dir results/runs/probe: smoke test on the probe
                                                            archive; every output then goes to that directory)
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np

RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs'); FIG = HERE; PROBE = False
if '--dir' in sys.argv:
    RES = LOGS = FIG = os.path.abspath(sys.argv[sys.argv.index('--dir')+1]); PROBE = True
B_BOOT, SEED_BOOT = 2000, 20281028
NAMES = ['G0', 'Re1', 'Im1', 'Re2', 'Im2', 'Re3', 'Im3', 'Re4', 'Im4']
B = 316.8921168493721


class Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()

def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return jl(v.tolist())
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, float) and not np.isfinite(v): return None if np.isnan(v) else (1e308 if v > 0 else -1e308)
    return v

def logdet(M):
    sg, ld = np.linalg.slogdet(M)
    return float(ld) if sg > 0 and np.isfinite(ld) else None

def acc_stats(th, th0):
    """MSE matrix (normalised by n), sample covariance (n - 1), bias, D = -log det M, -log det C, per-parameter rows."""
    n = th.shape[0]; e = th-th0
    M = e.T@e/n; C = np.cov(th.T, ddof=1) if n > 1 else np.full((9, 9), np.nan); b = e.mean(0)
    ldM = logdet(M); ldC = logdet(C) if n > 1 else None
    rmse = np.sqrt(np.diag(M))
    return dict(n=n, M=M, C=C, bias=b, rmse=rmse, sd=np.sqrt(np.maximum(np.diag(C), 0)) if n > 1 else None, bias_over_rmse=np.abs(b)/rmse,
                D=(-ldM if ldM is not None else None), D_C=(-ldC if ldC is not None else None), logdet_M=ldM, logdet_C=ldC)

def quant(x, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)):
    x = np.asarray(x, float); return {f"q{int(round(100*q)):02d}": float(np.quantile(x, q)) for q in qs} | dict(mean=float(x.mean()), sd=float(x.std(ddof=1)))

def ind_reading(name, pt, lo, hi, sup, fail):
    """spec §5 interval rule: supports if the CI and the point satisfy `sup`; clearly fails if the CI and the point satisfy
    `fail`; otherwise 未定 (a crossing CI never triggers 'fails' by its point estimate)."""
    if sup(lo, hi) and sup(pt, pt): return '支持达标'
    if fail(lo, hi) and fail(pt, pt): return '明确未达'
    return '未定'


def load():
    A = [json.loads(l) for l in open(os.path.join(RES, 'archive_round8.jsonl')) if l.strip()]
    A = sorted(A, key=lambda o: o['r'])
    rs = [o['r'] for o in A]
    assert PROBE or rs == list(range(1, len(A)+1)), 'archive must hold r = 1..R exactly once'
    assert len(set(rs)) == len(rs)
    ref = json.load(open(os.path.join(RES, 'sod_reference.json')))
    chk = json.load(open(os.path.join(HERE, 'results', 'partA_checks.json')))
    return A, ref, chk

def arrays(A, an):
    g = lambda f: np.array([f(o) for o in A])
    a = lambda o: o['analyses'][an]
    return dict(
        th1=g(lambda o: a(o)['theta1']), th_seq=g(lambda o: a(o)['seq']['theta_final']), th_sod=g(lambda o: a(o)['sod']['theta_final']),
        eps1=g(lambda o: a(o)['eps1']), conv1=g(lambda o: a(o)['stage1_converged']), dfail=g(lambda o: a(o)['design_failed']),
        conv_seq=g(lambda o: a(o)['seq']['converged']), conv_sod=g(lambda o: a(o)['sod']['converged']),
        succ=g(lambda o: a(o)['success_all']), EN2=g(lambda o: a(o)['posthoc']['EN2']),
        EN_hat=g(lambda o: a(o)['design'].get('EN_hat') if a(o)['design'].get('EN_hat') is not None else np.nan),
        N1=g(lambda o: o['N1']), N2=g(lambda o: a(o)['N2']), N2c=g(lambda o: o['control']['N2']),
        Jt=g(lambda o: a(o)['posthoc']['J_table']), Hr=g(lambda o: a(o)['H_r']), Hrc=g(lambda o: o['control']['H_r']),
        fallback1=g(lambda o: a(o)['stage1_fallback'] or ''), shared=g(lambda o: 'shared_with' in a(o)),
        lam=g(lambda o: a(o)['design'].get('lam') if a(o)['design'].get('lam') is not None else np.nan),
        frac_asym=g(lambda o: a(o)['frac_asym'] if a(o)['frac_asym'] is not None else np.nan),
        bsteps=g(lambda o: a(o)['posthoc'].get('boundary_steps', 0)))

def analyse(A, ref, chk, an, P):
    th0 = np.array(json.load(open(os.path.join(RES, 'theta0.json')))['theta0'])
    b1 = ref['b1']; J1 = np.array(ref['J1']); R = len(A)
    X = arrays(A, an); out = dict(analysis=an, R=R)
    # ---- C1 final accuracy, all R and subset
    acc = {sc: acc_stats(X['th_'+sc], th0) for sc in ('seq', 'sod')}
    acc1 = acc_stats(X['th1'], th0)
    dD = acc['seq']['D']-acc['sod']['D'] if acc['seq']['D'] is not None and acc['sod']['D'] is not None else None
    dDC = acc['seq']['D_C']-acc['sod']['D_C'] if acc['seq']['D_C'] is not None and acc['sod']['D_C'] is not None else None
    sub = X['succ']; nsub = int(sub.sum())
    if nsub >= 10:
        accs = {sc: acc_stats(X['th_'+sc][sub], th0) for sc in ('seq', 'sod')}
        dDs = accs['seq']['D']-accs['sod']['D'] if accs['seq']['D'] is not None and accs['sod']['D'] is not None else None
    else:
        accs, dDs = None, None
    out.update(acc_all=acc, acc_stage1=acc1, dD=dD, dD_C=dDC, subset_n=nsub, acc_subset=accs, dD_subset=dDs)
    # ---- C2 information reference
    Jseq = J1+X['Jt'].mean(0); Jsod = 2*J1
    ld = dict(J1=logdet(J1), Jseq=logdet(Jseq), Jsod=logdet(Jsod))
    out['info'] = dict(logdet=ld, diff_seq=abs(-acc['seq']['logdet_M']-ld['Jseq']) if acc['seq']['logdet_M'] is not None else None,
                       diff_sod=abs(-acc['sod']['logdet_M']-ld['Jsod']) if acc['sod']['logdet_M'] is not None else None,
                       diff_stage1=abs(-acc1['logdet_M']-ld['J1']) if acc1['logdet_M'] is not None else None,
                       signed_seq=(-acc['seq']['logdet_M']-ld['Jseq']) if acc['seq']['logdet_M'] is not None else None,
                       signed_sod=(-acc['sod']['logdet_M']-ld['Jsod']) if acc['sod']['logdet_M'] is not None else None,
                       signed_stage1=(-acc1['logdet_M']-ld['J1']) if acc1['logdet_M'] is not None else None)
    # ---- C3 transmissions
    tot_seq = X['N1']+X['N2']; tot_sod = X['N1']+X['N2c']
    out['tx'] = dict(b1=b1, B=B, twoB=2*B, b_seq=b1+float(X['EN2'].mean()), two_b1=2*b1,
                     p_EN2_gt_B=float(np.mean(X['EN2'] > B)), p_N2_gt_B=float(np.mean(X['N2'] > B)), p_N2c_gt_B=float(np.mean(X['N2c'] > B)),
                     N1=quant(X['N1']), N2_seq=quant(X['N2']), N2_sod=quant(X['N2c']), total_seq=quant(tot_seq), total_sod=quant(tot_sod),
                     EN2=quant(X['EN2']), EN_hat=quant(X['EN_hat'][np.isfinite(X['EN_hat'])]) if np.isfinite(X['EN_hat']).any() else None,
                     EN_hat_minus_EN2=quant((X['EN_hat']-X['EN2'])[np.isfinite(X['EN_hat'])]) if np.isfinite(X['EN_hat']).any() else None,
                     p_total_seq_gt_2B=float(np.mean(tot_seq > 2*B)), p_total_sod_gt_2B=float(np.mean(tot_sod > 2*B)),
                     lam_over_gbar=None)
    # ---- C4 stage 1, failures
    out['stage1'] = dict(eps1=quant(X['eps1'], (0.5, 0.9, 0.99)), eps1_full=quant(X['eps1']), frac_eps1_gt_05=float(np.mean(X['eps1'] > 0.5)),
                         fallback_kinds={k: int(np.sum(X['fallback1'] == k)) for k in sorted(set(X['fallback1'].tolist()))})
    out['fail'] = dict(design=float(np.mean(X['dfail'])), stage1=float(np.mean(~X['conv1'])), joint_seq=float(np.mean(~X['conv_seq'])),
                       joint_sod=float(np.mean(~X['conv_sod'])), n_design=int(X['dfail'].sum()), n_stage1=int((~X['conv1']).sum()),
                       n_joint_seq=int((~X['conv_seq']).sum()), n_joint_sod=int((~X['conv_sod']).sum()), shared_designs=int(X['shared'].sum()),
                       boundary_steps_max=int(X['bsteps'].max()))
    # ---- C5 diagnostics
    def hfim(Hr, Jt):
        Dr = Hr-Jt; Db = Dr.mean(0); se = Dr.std(0, ddof=1)/np.sqrt(R)
        z = np.where(se > 0, Db/np.where(se > 0, se, 1), 0.0)
        Jm = Jt.mean(0); w, V = np.linalg.eigh(Jm); Jmh = V@np.diag(w**-0.5)@V.T
        Wn = float(np.linalg.norm(Jmh@Db@Jmh, 2))
        flags = [(NAMES[i], NAMES[j], float(z[i, j])) for i in range(9) for j in range(i, 9) if abs(z[i, j]) > 4]
        return dict(Dbar=Db, se=se, z=z, max_abs_z=float(np.max(np.abs(z))), whitened_norm=Wn, flags=flags)
    out['hfim'] = hfim(X['Hr'], X['Jt']); out['hfim_control'] = hfim(X['Hrc'], np.broadcast_to(J1, X['Hrc'].shape))
    d = X['N2']-X['EN2']; dc = X['N2c']-b1
    dd = lambda v: dict(dbar=float(v.mean()), se=float(v.std(ddof=1)/np.sqrt(R)), flag=bool(abs(v.mean()) > 3*v.std(ddof=1)/np.sqrt(R)))
    out['paired_d'] = dd(d); out['paired_d_control'] = dd(dc)
    # ---- C6 bootstrap
    rng = np.random.default_rng(SEED_BOOT)
    bs = dict(dD=np.full(B_BOOT, np.nan), b_seq=np.empty(B_BOOT), p_EN2=np.empty(B_BOOT), p_N2=np.empty(B_BOOT),
              dD_sub=np.full(B_BOOT, np.nan), nsub=np.zeros(B_BOOT, int))
    nc_all = 0; nc_sub = 0
    E = {sc: X['th_'+sc]-th0 for sc in ('seq', 'sod')}
    for b in range(B_BOOT):
        idx = rng.integers(0, R, R)
        lds = [logdet(E[sc][idx].T@E[sc][idx]/R) for sc in ('seq', 'sod')]
        if None in lds: nc_all += 1
        else: bs['dD'][b] = -lds[0]+lds[1]
        bs['b_seq'][b] = b1+X['EN2'][idx].mean(); bs['p_EN2'][b] = np.mean(X['EN2'][idx] > B); bs['p_N2'][b] = np.mean(X['N2'][idx] > B)
        js = idx[sub[idx]]; n = js.size; bs['nsub'][b] = n
        if n < 10: nc_sub += 1; continue
        lds = [logdet(E[sc][js].T@E[sc][js]/n) for sc in ('seq', 'sod')]
        if None in lds: nc_sub += 1; continue
        bs['dD_sub'][b] = -lds[0]+lds[1]
    ci = lambda v: [float(np.quantile(v[np.isfinite(v)], 0.025)), float(np.quantile(v[np.isfinite(v)], 0.975))] if np.isfinite(v).any() else [None, None]
    out['boot'] = dict(B_boot=B_BOOT, seed=SEED_BOOT, dD=ci(bs['dD']), b_seq=ci(bs['b_seq']), p_EN2=ci(bs['p_EN2']), p_N2=ci(bs['p_N2']),
                       dD_sub=ci(bs['dD_sub']), not_computable_all=nc_all, not_computable_sub=nc_sub, nsub_quant=quant(bs['nsub']))
    np.savez_compressed(os.path.join(RES, f'bootstrap_round8_{an}.npz'), **bs)
    # ---- C7 verdict readings (binding only for P)
    pt_dD = dD if dD is not None else np.nan
    rd = dict(dD=ind_reading('dD', pt_dD, *out['boot']['dD'], lambda lo, hi: lo is not None and lo >= 2, lambda lo, hi: hi is not None and hi < 1)
              if out['boot']['dD'][0] is not None and dD is not None else '未定（不可计算）',
              b_seq=ind_reading('b_seq', out['tx']['b_seq'], *out['boot']['b_seq'], lambda lo, hi: hi <= 2*B, lambda lo, hi: lo > 2*B),
              p_EN2=ind_reading('p_EN2', out['tx']['p_EN2_gt_B'], *out['boot']['p_EN2'], lambda lo, hi: hi <= 0.20, lambda lo, hi: lo > 0.40))
    f = out['fail']; rates = [f['design'], f['stage1'], f['joint_seq'], f['joint_sod']]
    bor = max(float(np.max(acc[sc]['bias_over_rmse'])) for sc in ('seq', 'sod'))
    a3_ok = bool(chk['all_required_pass']) if an == 'P' else bool(chk['A3_1']['pass_supp_reading'] and chk['A3_2']['pass'] and chk['A3_3']['pass'] and chk['A3_4']['pass'])
    items = []
    if f['stage1'] > 0.10 or not a3_ok:
        case = 1
        if f['stage1'] > 0.10: items.append(f"第一段估计失败率 {f['stage1']:.1%} > 10%")
        if not a3_ok: items.append('正确性检查未全部通过（' + ', '.join(k for k in ('A3_1', 'A3_2', 'A3_3', 'A3_4') if not (chk[k]['pass'] if not (an == 'S' and k == 'A3_1') else chk[k]['pass_supp_reading'])) + '）')
    elif all(v == '支持达标' for v in rd.values()) and max(rates) <= 0.05 and bor <= 0.3:
        case = 2
    elif any(v == '明确未达' for v in rd.values()) or max(rates) > 0.10:
        case = 3
        items += [f"{k}: {v}" for k, v in rd.items() if v == '明确未达']
        if max(rates) > 0.10: items.append(f"失败率 > 10%: {rates}")
    else:
        case = 4
        items += [f"{k}: {v}" for k, v in rd.items() if v != '支持达标']
        if max(rates) > 0.05: items.append(f"失败率 > 5%: {rates}")
        if bor > 0.3: items.append(f"max |偏差|/RMSE = {bor:.3f} > 0.3")
    out['verdict'] = dict(case=case, readings=rd, items=items, max_bias_over_rmse=bor, failure_rates=rates, a3_ok=a3_ok,
                          X=dD, X_ci=out['boot']['dD'], Y=float(np.median(X['eps1'])), Z=out['tx']['p_EN2_gt_B'])
    return out, X


def log_analysis(o, P):
    an = o['analysis']; t = f"[{an}]"
    P(f"\n######## analysis {an} ({'frozen acceptance rule, primary, binding' if an == 'P' else 'supplementary rule (DECISION E-1), not binding'}) R = {o['R']}")
    P(f"C4 {t} eps_1 median {o['stage1']['eps1']['q50']!r} q90 {o['stage1']['eps1']['q90']!r} q99 {o['stage1']['eps1']['q99']!r} "
      f"mean {o['stage1']['eps1']['mean']!r}; fraction > 0.5 {o['stage1']['frac_eps1_gt_05']!r}; stage-1 fallback kinds {o['stage1']['fallback_kinds']}")
    f = o['fail']
    P(f"C4 {t} failure rates: design {f['design']!r} ({f['n_design']}), stage-1 estimation {f['stage1']!r} ({f['n_stage1']}), joint seq "
      f"{f['joint_seq']!r} ({f['n_joint_seq']}), joint SoD+SoD {f['joint_sod']!r} ({f['n_joint_sod']}); shared designs {f['shared_designs']}; "
      f"max boundary steps {f['boundary_steps_max']}")
    x = o['tx']
    P(f"C3 {t} b_1 {x['b1']!r}; b_seq = b_1 + mean E[N_2|T,theta_0] = {x['b_seq']!r}; 2 b_1 {x['two_b1']!r}; 2B {x['twoB']!r}")
    P(f"C3 {t} Pr{{E[N_2|T,theta_0] > B}} (生成的表在条件期望上超预算的比例) {x['p_EN2_gt_B']!r}; Pr{{N_2 > B}} (实际实验超预算的比例) {x['p_N2_gt_B']!r}; "
      f"control Pr{{N_2,SoD > B}} {x['p_N2c_gt_B']!r}; Pr{{N_1+N_2 > 2B}} seq {x['p_total_seq_gt_2B']!r} control {x['p_total_sod_gt_2B']!r}")
    for k in ('N1', 'N2_seq', 'N2_sod', 'total_seq', 'total_sod', 'EN2', 'EN_hat', 'EN_hat_minus_EN2'):
        if x[k] is not None: P(f"C3 {t} {k} " + ' '.join(f"{kk} {v:.3f}" for kk, v in x[k].items()))
    for sc in ('seq', 'sod'):
        a = o['acc_all'][sc]
        P(f"C1 {t} all R {sc}: D = -log det M {a['D']!r}; -log det C {a['D_C']!r}; log det M {a['logdet_M']!r}")
        P(f"C1 {t} all R {sc} per parameter " + ' | '.join(f"{n}: bias {a['bias'][i]:+.3e} rmse {a['rmse'][i]:.3e} |b|/rmse {a['bias_over_rmse'][i]:.3f}"
                                                         for i, n in enumerate(NAMES)))
    P(f"C1 {t} Delta D (all R) = D_seq - D_SoD+SoD = {o['dD']!r}; Delta of -log det C {o['dD_C']!r}")
    a1 = o['acc_stage1']
    P(f"C1 {t} stage 1: -log det M_1 {a1['D']!r}; per parameter |bias|/rmse max {float(np.max(a1['bias_over_rmse'])):.3f}")
    P(f"C1 {t} subset (both succeeded) n = {o['subset_n']}; Delta D subset {o['dD_subset']!r}" + ('' if o['acc_subset'] is None else
      f"; D_seq {o['acc_subset']['seq']['D']!r}, D_SoD {o['acc_subset']['sod']['D']!r}"))
    i_ = o['info']
    P(f"C2 {t} log det J_1 {i_['logdet']['J1']!r}, J_seq {i_['logdet']['Jseq']!r}, 2 J_1 {i_['logdet']['Jsod']!r}; 经验精度与信息参考的差异 "
      f"seq {i_['diff_seq']!r} (signed {i_['signed_seq']!r}), SoD+SoD {i_['diff_sod']!r} (signed {i_['signed_sod']!r}), stage 1 {i_['diff_stage1']!r} (signed {i_['signed_stage1']!r})")
    h = o['hfim']; hc = o['hfim_control']
    P(f"C5 {t} Hessian-FIM (seq stage 2): whitened ||J^-1/2 Dbar J^-1/2||_2 {h['whitened_norm']!r}; max |z| {h['max_abs_z']!r}; flags |z| > 4: {h['flags']}")
    P(f"C5 {t} Hessian-FIM (control stage 2 vs J_1, extra): whitened {hc['whitened_norm']!r}; max |z| {hc['max_abs_z']!r}; flags {hc['flags']}")
    P(f"C5 {t} paired d = N_2 - E[N_2|T,theta_0]: dbar {o['paired_d']['dbar']!r}, s_d/sqrt(R) {o['paired_d']['se']!r}, flag {o['paired_d']['flag']}; "
      f"control (N_2,SoD - b_1): dbar {o['paired_d_control']['dbar']!r}, se {o['paired_d_control']['se']!r}, flag {o['paired_d_control']['flag']}")
    b = o['boot']
    P(f"C6 {t} paired bootstrap ({b['B_boot']}, seed {b['seed']}): Delta D {b['dD']} (不可计算 {b['not_computable_all']}); b_seq {b['b_seq']}; "
      f"Pr{{E[N_2|T,theta_0] > B}} {b['p_EN2']}; Pr{{N_2 > B}} {b['p_N2']}; subset Delta D {b['dD_sub']} (不可计算 {b['not_computable_sub']} of {b['B_boot']})")
    v = o['verdict']
    P(f"C7 {t} readings {v['readings']}; failure rates {v['failure_rates']}; max |bias|/RMSE {v['max_bias_over_rmse']:.3f}; A3 ok {v['a3_ok']}; "
      f"case {v['case']}; items {v['items']}")


def figures(XP, XS, oP, oS, th0, J1):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C1, C2, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})
    # eps_1 histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, max(XP['eps1'].max(), XS['eps1'].max())*1.02, 31)
    ax.hist(XP['eps1'], bins=bins, histtype='step', lw=2, color=C1, label='P: frozen rule (theta_1 = fallback theta_LS)')
    ax.hist(XS['eps1'], bins=bins, histtype='step', lw=2, color=C2, label='S: supplementary rule (converged ML)')
    ax.axvline(0.5, color=MUTED, lw=1, ls='--'); ax.text(0.5, ax.get_ylim()[1]*0.95, ' 0.5', color=MUTED, va='top')
    ax.set_xlabel('eps_1 = max_k |h_k^T (theta_1 - theta_0)| / sigma'); ax.set_ylabel('replications')
    ax.set_title(f'Stage-1 estimation error, R = {len(XP["eps1"])}', loc='left', color=INK); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_eps1_hist.png'), dpi=150); plt.close(fig)
    # E[N_2 | T, theta_0] vs eps_1
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(XP['eps1'], XP['EN2'], s=16, color=C1, edgecolor='white', lw=0.4, label='P: frozen rule')
    ax.scatter(XS['eps1'], XS['EN2'], s=16, color=C2, edgecolor='white', lw=0.4, label='S: supplementary rule')
    ax.axhline(B, color=MUTED, lw=1, ls='--'); ax.text(ax.get_xlim()[1], B, ' B', color=MUTED, va='center', ha='left')
    ax.set_xlabel('eps_1'); ax.set_ylabel('E[N_2 | T, theta_0]  (sigma/120)')
    ax.set_title('Expected stage-2 transmissions of the generated table vs stage-1 error', loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_EN2_vs_eps1.png'), dpi=150); plt.close(fig)
    # final-error box plots, both schemes, both analyses, in units of the control's CRB standard deviation
    sdc = np.sqrt(np.diag(np.linalg.inv(2*J1)))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for ax, X, lab in ((axes[0], XP, 'P: frozen rule (all estimates are the fallback theta_LS; seq = control)'), (axes[1], XS, 'S: supplementary rule')):
        pos = np.arange(9)
        for k, (sc, col, dx) in enumerate((('seq', C1, -0.18), ('sod', C2, 0.18))):
            e = (X['th_'+sc]-th0)/sdc
            bp = ax.boxplot([e[:, i] for i in range(9)], positions=pos+dx, widths=0.3, patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', ms=2.5, mfc=col, mec='none', alpha=0.6), medianprops=dict(color=INK, lw=1.2),
                            whiskerprops=dict(color=col, lw=1.2), capprops=dict(color=col, lw=1.2), boxprops=dict(facecolor=col, edgecolor=col, alpha=0.35))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel('error / CRB sd of SoD+SoD'); ax.set_title(lab, loc='left', color=INK, fontsize=9)
    axes[1].set_xticks(np.arange(9)); axes[1].set_xticklabels(NAMES)
    fig.suptitle('Final joint-estimate errors, R = %d' % len(XP['eps1']), x=0.01, ha='left', color=INK, fontsize=10)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=C1, ec=C1, alpha=0.5), plt.Rectangle((0, 0), 1, 1, fc=C2, ec=C2, alpha=0.5)]
    fig.legend(hs, ['sequential (SoD + table)', 'SoD + SoD'], loc='upper right', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(FIG, 'fig_final_error_box.png'), dpi=150); plt.close(fig)


def main():
    t0 = time.perf_counter()
    sys.stdout = Tee(os.path.join(LOGS, 'partC_stats.log')); P = print
    A, ref, chk = load()
    sys.path.insert(0, HERE)
    import truth_r8
    json.dump(dict(theta0=truth_r8.THETA0.tolist(), note='theta_0 for the post-hoc statistics (truth_r8.THETA0)'), open(os.path.join(RES, 'theta0.json'), 'w'), indent=1)
    P(f"=== stats_r8.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; R = {len(A)}; b_1 {ref['b1']!r}; B {B!r}; A3 all required pass {chk['all_required_pass']}")
    res = {}; XX = {}
    for an in ('P', 'S'):
        o, X = analyse(A, ref, chk, an, P); res[an] = o; XX[an] = X; log_analysis(o, P)
    # equality of P and S where they should coincide (diagnostic)
    P(f"\nC0 control stage-2 records identical in both analyses by construction; designs shared P/S in {res['S']['fail']['shared_designs']} replications")
    figures(XX['P'], XX['S'], res['P'], res['S'], truth_r8.THETA0, np.array(ref['J1']))
    tm = [o['times'] for o in A]
    timing = dict(primary_work_max=float(max(t['primary_work_s'] for t in tm)), primary_work_median=float(np.median([t['primary_work_s'] for t in tm])),
                  total_max=float(max(t['total_s'] for t in tm)), total_median=float(np.median([t['total_s'] for t in tm])),
                  primary_work_sum=float(sum(t['primary_work_s'] for t in tm)), total_sum=float(sum(t['total_s'] for t in tm)),
                  partC_s=time.perf_counter()-t0)
    P(f"\nT timing per replication (worker wall, 1 numba thread, 4 workers): primary-analysis work median {timing['primary_work_median']:.1f} s max "
      f"{timing['primary_work_max']:.1f} s; whole replication (both analyses) median {timing['total_median']:.1f} s max {timing['total_max']:.1f} s; "
      f"sums {timing['primary_work_sum']:.0f} s / {timing['total_sum']:.0f} s; Part C {timing['partC_s']:.1f} s")
    res['timing'] = timing
    json.dump(jl(res), open(os.path.join(RES, 'stats_round8.json'), 'w'), indent=1)
    P(f"=== stats_r8.py done in {time.perf_counter()-t0:.1f} s")
    return res


if __name__ == '__main__':
    main()
