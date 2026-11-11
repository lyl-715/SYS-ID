"""Round 18 summary (prompt §5-§7): Q1-Q5, prediction scoring (predictions_round18.md, DECISION 18-1), checks -> results/round18.json.

Reads results/partA.json, partB.json, partB_estimates.npz, partB_newton.json (supplementary), partC.json; computes nothing new except
  * the Part B statistics of LS restricted to the runs whose CML was accepted (acc18.acc_stats on the archived estimates), and a
    recomputation of the archived Part B statistics from partB_estimates.npz (consistency check);
  * descriptive context for D-hat (DECISION 18-12, not used for any check or score): a bootstrap standard error of D-hat over the runs
    (2000 resamples, seed 20261118) and the reference distribution of D-hat - I for errors that attain the Cramer-Rao bound exactly
    (zero mean, covariance J^-1): n M-hat ~ Wishart(J^-1, n), so D-hat - I = -sum_{i=1..p} log(chi2_{n-i+1}/n) with independent
    chi-squares; mean -sum psi((n-i+1)/2) - p log(2/n), variance sum psi'((n-i+1)/2); P(|D-hat - I| <= 0.3) by 10^6 draws (seed 20261119).
"""
import os, json
import numpy as np
from scipy.special import digamma, polygamma
import acc18 as a

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
B_NREP = 500
TRIGS_B = ('SOD', 'RULE', 'PRED')
BOOT_N, BOOT_SEED, REF_DRAWS, REF_SEED = 2000, 20261118, 10**6, 20261119


def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return jl(v.tolist())
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def boot_se_Dhat(th, rng):
    e = th-a.THETA_STAR[None, :]; n = e.shape[0]
    d = np.empty(BOOT_N)
    for b in range(BOOT_N):
        x = e[rng.integers(0, n, n)]; d[b] = -np.linalg.slogdet(x.T@x/n)[1]
    return float(d.std(ddof=1))


def wishart_ref(n, p=a.P):
    i = np.arange(1, p+1); df = n-i+1
    mean = float(-(digamma(df/2).sum()+p*np.log(2.0/n))); sd = float(np.sqrt(polygamma(1, df/2).sum()))
    rng = np.random.default_rng(REF_SEED)
    x = -sum(np.log(rng.chisquare(k, REF_DRAWS)/n) for k in df)
    return dict(n=int(n), mean=mean, sd=sd, mc_mean=float(x.mean()), mc_sd=float(x.std(ddof=1)), p_within_03=float((np.abs(x) <= 0.3).mean()),
                p_below_minus03=float((x < -0.3).mean()))


def bias_z(st):
    z = np.array(st['bias_cr'])/np.array(st['bias_cr_se'])
    return float(np.abs(z).max()), int(np.abs(z).argmax())


def brief(st, I, se_boot=None):
    zmax, zarg = bias_z(st)
    return dict(n=st['n'], bias_cr=st['bias_cr'], bias_cr_se=st['bias_cr_se'], bias_max_abs=st['bias_max_abs'], bias_argmax=st['bias_argmax']+1,
                bias_z_max=zmax, bias_z_argmax=zarg+1, bias_mahalanobis=st['bias_mahalanobis'], rmse_cr=st['rmse_cr'], rmse_cr_min=st['rmse_cr_min'],
                rmse_cr_max=st['rmse_cr_max'], D_hat=st['D_hat'], D_hat_minus_I=st['D_hat']-I, D_hat_boot_se=se_boot,
                neg_logdet_cov=st['neg_logdet_cov'], neg_logdet_cov_minus_I=st['neg_logdet_cov']-I)


def stage_summary(out):
    out("=== ROUND 18 summary start")
    A = json.load(open(os.path.join(RES, 'partA.json'))); Bj = json.load(open(os.path.join(RES, 'partB.json')))
    Bn = json.load(open(os.path.join(RES, 'partB_newton.json'))); C = json.load(open(os.path.join(RES, 'partC.json')))
    est = np.load(os.path.join(RES, 'partB_estimates.npz'))
    TA = A['triggers']; res = dict(partA={}, partB={}, partC={}, questions={}, predictions={}, checks={})

    # --- Part A --------------------------------------------------------------------------------------------------------------
    for t in a.TRIGGERS:
        r = TA[t]
        res['partA'][t] = {k: r[k] for k in ('I', 'dI', 'I_batch_se', 'ld_run_mean', 'ld_run_sd', 'ld_run_p05', 'ld_run_p95', 'I_minus_mean', 'corr_ld_N',
                                              'ld_sent_of_mean', 'value_nontx', 'value_nontx_batch_se', 'S', 'EN', 'EN_se', 'sd_N')}
        out(f"A {t}: I {r['I']:.4f} dI {r['dI']:+.4f}; per-run log det mean {r['ld_run_mean']:.4f} sd {r['ld_run_sd']:.4f} P5 {r['ld_run_p05']:.4f} "
            f"P95 {r['ld_run_p95']:.4f}; I - mean {r['I_minus_mean']:.5f}; corr {r['corr_ld_N']}; log det mean(J_sent) {r['ld_sent_of_mean']:.4f}; "
            f"value of the non-transmissions {r['value_nontx']:.4f} nat; S {r['S']:.2f}; E[N] {r['EN']:.3f} sd N {r['sd_N']:.3f}")

    # --- Part B --------------------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED); recomputed_ok = True
    refs = {}
    for t in TRIGS_B:
        J = np.load(os.path.join(RES, f'J_A_{t}.npy')); I = TA[t]['I']; R = Bj['triggers'][t]
        thL = est[f'{t}_theta_LS']; thC = est[f'{t}_theta_CML']; conv = est[f'{t}_converged']
        thN = thC.copy()
        for run in Bn['runs']:
            if run['trig'] == t and run['newton_converged']: thN[run['i']] = np.array(run['theta_newton'])
        # consistency: archived statistics recomputed from the archived estimates
        for th, arch in ((thL, R['LS']), (thC, R['CML']), (thC[conv], R['CML_accepted_only']), (thN, Bn['triggers'][t]['CML_newton_substituted'])):
            st = a.acc_stats(th, J); recomputed_ok &= all(st[k] == arch[k] for k in ('bias_cr', 'rmse_cr', 'D_hat', 'neg_logdet_cov', 'bias_mahalanobis'))
        sets = dict(LS_all=(thL, R['LS']), LS_accepted=(thL[conv], a.acc_stats(thL[conv], J)), CML_all=(thC, R['CML']),
                    CML_accepted=(thC[conv], R['CML_accepted_only']), CML_newton_supp=(thN, Bn['triggers'][t]['CML_newton_substituted']))
        rec = dict(I=I, N_mean=R['N_mean'], N_sd=R['N_sd'], n_not_accepted=R['CML_not_accepted'], not_accepted_runs=R['not_accepted_runs'],
                   first_call_not_accepted=R['CML_first_call_not_accepted'], restart_used=R['CML_restart_used'], fallback=R['CML_fallback'],
                   fallback_estimate='theta_LS (least squares on the transmitted pairs of the same run)',
                   newton_converged=Bn['triggers'][t]['n_newton_converged'], sets={})
        for lab, (th, st) in sets.items():
            rec['sets'][lab] = brief(st, I, boot_se_Dhat(th, rng))
            b = rec['sets'][lab]
            out(f"B {t} {lab} (n {b['n']}): largest |bias|/CRsd {b['bias_max_abs']:.3f} (param {b['bias_argmax']}); largest |bias|/MC-SE "
                f"{b['bias_z_max']:.1f} (param {b['bias_z_argmax']}); Mahalanobis {b['bias_mahalanobis']:.3f}; RMSE/CRsd min {b['rmse_cr_min']:.3f} "
                f"max {b['rmse_cr_max']:.3f}; D-hat {b['D_hat']:.3f} (- I {b['D_hat_minus_I']:+.3f}, bootstrap SE {b['D_hat_boot_se']:.3f}); "
                f"-log det cov - I {b['neg_logdet_cov_minus_I']:+.3f}")
        for n in sorted({rec['sets'][k]['n'] for k in rec['sets']}):
            if n not in refs: refs[n] = wishart_ref(n)
        res['partB'][t] = rec
    for n, r in sorted(refs.items()):
        out(f"B reference (errors attaining the CR bound exactly, n = {n}): D-hat - I mean {r['mean']:+.4f} sd {r['sd']:.4f} (MC {r['mc_mean']:+.4f} / "
            f"{r['mc_sd']:.4f}); P(|D-hat - I| <= 0.3) = {r['p_within_03']:.4f}; P(D-hat - I < -0.3) = {r['p_below_minus03']:.4f}")
    res['partB_reference'] = {str(n): r for n, r in refs.items()}
    out(f"CHECK summary: archived Part B statistics (LS, CML, CML accepted only, CML Newton-substituted) recomputed from partB_estimates.npz / "
        f"partB_newton.json equal: {recomputed_ok} -> {'PASS' if recomputed_ok else 'FAIL'}")

    # --- Part C --------------------------------------------------------------------------------------------------------------
    for v in ('C0', 'C1', 'C2'):
        b = C['tuning'][v]['best']
        res['partC'][v] = {k: b.get(k) for k in ('par', 'EN', 'EN_se', 'sd_N', 'I', 'dI', 'ld_run_mean', 'ld_run_sd', 'S', 'e_p05', 'e_p50', 'e_p95',
                                                  'e_mean', 'e_frac_lo', 'e_frac_hi', 'rel_mean', 'rho_min', 'rho_max')}
        res['partC'][v].update(n_runs=C['tuning'][v]['n_runs'], sd_I_over_seeds=C['reps']['sd_I'][v], I_mean_over_seeds=float(np.mean(C['reps']['I'][v])),
                               EN_mean_over_seeds=float(np.mean(C['reps']['EN'][v])))
    res['partC']['diffs'] = {k: dict(mean=d['mean'], se=d['se'], tuned_seed=d['tuned_seed'], per_seed=d['per_seed']) for k, d in C['diffs'].items()}
    for k, d in C['diffs'].items():
        out(f"C {k}: mean over 10 seeds {d['mean']:+.4f} nat, paired SE {d['se']:.4f}; at the tuning seed {d['tuned_seed']:+.4f}")

    # --- questions -----------------------------------------------------------------------------------------------------------
    P_ = res['partB']; Q = res['questions']
    Q['Q1'] = {t: dict(sd=TA[t]['ld_run_sd'], I_minus_mean=TA[t]['I_minus_mean'], p05=TA[t]['ld_run_p05'], p95=TA[t]['ld_run_p95']) for t in a.TRIGGERS}
    Q['Q2'] = {t: dict(value_nontx=TA[t]['value_nontx'], batch_se=TA[t]['value_nontx_batch_se']) for t in a.TRIGGERS}
    Q['Q3'] = {t: {lab: dict(bias_max_abs=P_[t]['sets'][lab]['bias_max_abs'], bias_z_max=P_[t]['sets'][lab]['bias_z_max'],
                             I_minus_D_hat=-P_[t]['sets'][lab]['D_hat_minus_I']) for lab in ('LS_all', 'LS_accepted')} for t in TRIGS_B}
    def attains(b): return bool(b['bias_max_abs'] < 0.2 and abs(b['D_hat_minus_I']) <= 0.3)
    Q['Q4'] = {t: {lab: dict(bias_max_abs=P_[t]['sets'][lab]['bias_max_abs'], D_hat_minus_I=P_[t]['sets'][lab]['D_hat_minus_I'],
                             attains=attains(P_[t]['sets'][lab])) for lab in ('CML_all', 'CML_accepted', 'CML_newton_supp')} for t in ('RULE', 'PRED', 'SOD')}
    Q['Q5'] = {k: dict(mean=C['diffs'][k]['mean'], se=C['diffs'][k]['se']) for k in ('C1-C0', 'C2-C1', 'C2-C0')}
    for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'): out(f"{q}: {json.dumps(jl(Q[q]))}")

    # --- predictions (DECISION 18-1: six physical lines; pre-registered statistics = all 500 runs, fallback included) ----------
    T = TA; Lr, Ls = P_['RULE']['sets']['LS_all'], P_['SOD']['sets']['LS_all']
    Cr, Cp = P_['RULE']['sets']['CML_all'], P_['PRED']['sets']['CML_all']
    d10, d21 = C['diffs']['C1-C0']['mean'], C['diffs']['C2-C1']['mean']
    lines = dict(
        A1=(0.04 <= T['RULE']['ld_run_sd'] <= 0.09 and 0.20 <= T['SOD']['ld_run_sd'] <= 0.33 and T['RULE']['I_minus_mean'] <= 0.005
            and 0.02 <= T['SOD']['I_minus_mean'] <= 0.05,
            f"sd RULE {T['RULE']['ld_run_sd']:.4f}, SOD {T['SOD']['ld_run_sd']:.4f}; I - mean RULE {T['RULE']['I_minus_mean']:.5f}, SOD {T['SOD']['I_minus_mean']:.5f}"),
        A2=(12.8 <= T['RULE']['value_nontx'] <= 13.8 and 3.3 <= T['SOD']['value_nontx'] <= 4.1 and 14.0 <= T['PRED']['value_nontx'] <= 15.5,
            f"RULE {T['RULE']['value_nontx']:.3f}, SOD {T['SOD']['value_nontx']:.3f}, PRED {T['PRED']['value_nontx']:.3f} nat"),
        B1=(Lr['bias_max_abs'] > 3 and -Lr['D_hat_minus_I'] > 10, f"largest bias {Lr['bias_max_abs']:.3f}; I - D-hat {-Lr['D_hat_minus_I']:.3f}"),
        B2=(Ls['bias_max_abs'] > 0.5, f"largest bias {Ls['bias_max_abs']:.3f}"),
        B3=(Cr['bias_max_abs'] < 0.2 and Cp['bias_max_abs'] < 0.2 and abs(Cr['D_hat_minus_I']) <= 0.3 and abs(Cp['D_hat_minus_I']) <= 0.3
            and P_['RULE']['n_not_accepted'] <= 10 and P_['PRED']['n_not_accepted'] <= 10,
            f"RULE: bias {Cr['bias_max_abs']:.3f}, D-hat - I {Cr['D_hat_minus_I']:+.3f}, not accepted {P_['RULE']['n_not_accepted']}; PRED: bias "
            f"{Cp['bias_max_abs']:.3f}, D-hat - I {Cp['D_hat_minus_I']:+.3f}, not accepted {P_['PRED']['n_not_accepted']}"),
        C=(0.0 <= d10 <= 0.20 and -0.20 <= d21 <= 0.05, f"I(C1) - I(C0) {d10:+.4f}; I(C2) - I(C1) {d21:+.4f}"))
    hits = 0
    for k, (h, txt) in lines.items():
        h = bool(h); hits += h; res['predictions'][k] = dict(hit=h, values=txt)
        out(f"PREDICTION {k}: {'HIT' if h else 'MISS'} ({txt})")
    bul = dict(PartA=lines['A1'][0] and lines['A2'][0], PartB=lines['B1'][0] and lines['B2'][0] and lines['B3'][0], PartC=lines['C'][0])
    res['predictions']['tally_lines'] = f"{hits} of 6"; res['predictions']['bullets'] = {k: bool(v) for k, v in bul.items()}
    out(f"PREDICTIONS: {hits} of 6 lines hit; bullets: " + ', '.join(f"{k} {'hit' if v else 'miss'}" for k, v in bul.items()))

    # --- checks ----------------------------------------------------------------------------------------------------------------
    res['checks'] = dict(partA=A['checks'], partB=Bj['checks'], partB_newton=Bn['checks'], partC=C['checks'], summary_recomputed=recomputed_ok)
    out(f"CHECKS: A {A['checks']}; B {Bj['checks']}; Bn {Bn['checks']}; C {C['checks']}")
    path = os.path.join(RES, 'round18.json'); tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(jl(res), f, indent=1)
    os.replace(tmp, path)
    out("=== ROUND 18 summary done")
