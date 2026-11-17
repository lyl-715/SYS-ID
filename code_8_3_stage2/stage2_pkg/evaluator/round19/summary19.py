"""Round 19 summary (prompt §6, §7; scoring conventions of round19/predictions.md §2-§3). Reads results/rows.csv, main.npz,
checks.json, main.json; writes results/round19.json and the text tables to logs/summary.log."""
import os, json, csv
import numpy as np
import two_stage19 as t
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
B, P, K = t.B, t.P, t.K
NAMES = ('primary', 'K1_1250', 'K1_3750', 'control', 'baseline')


def load_rows():
    rows = {n: [] for n in NAMES}
    with open(os.path.join(RES, 'rows.csv')) as f:
        for r in csv.DictReader(f): rows[r['config']].append(r)
    for n in NAMES: rows[n].sort(key=lambda r: int(r['i']))
    return rows

def col(rows, k, typ=float):
    return np.array([typ(r[k]) for r in rows])

def q(x, p): return float(np.percentile(x, p))

def boot_ci(v):
    return [q(v, 2.5), q(v, 97.5)]


def run(out):
    rows = load_rows(); A = np.load(os.path.join(RES, 'main.npz'))
    chk = json.load(open(os.path.join(RES, 'checks.json'))); mj = json.load(open(os.path.join(RES, 'main.json')))
    res = dict(checks=chk['checks'], configs={}, gains={}, predictions={}, earlier={})
    Jc = {n: A[f'{n}_Jc'] for n in NAMES}
    TH = {n: A[f'{n}_theta_final'] for n in NAMES}
    R_of = {n: Jc[n].shape[0] for n in NAMES}
    cnt = {R: t.boot_counts(R) for R in sorted(set(R_of.values()))}

    # --- information and gains ------------------------------------------------------------------------------------------
    out("INFORMATION (log det J, J = s^-2 sum_k wbar_k h_k h_k^T over the configuration's replications; gain = paired difference "
        "with constant SoD throughout on the same replications; 95 % paired bootstrap, 2000 resamples, seed 194000000)")
    out(f"{'config':>9} {'K1':>5} {'R':>5} {'log det J':>10} {'base (same reps)':>16} {'gain':>8} {'boot SE':>8} {'95% interval':>18} "
        f"{'vs archive SoD':>14} {'ldJ(W)-ldJ(mean Jc)':>20}")
    for n in NAMES:
        R = R_of[n]; K1 = int(rows[n][0]['K1'])
        J = Jc[n].mean(0).reshape(P, P); I = float(t.logdet(J))
        Iw = float(t.logdet(t.J_of_weights(A[f'{n}_W'])))
        Jb = Jc['baseline'][:R]; Ib = float(t.logdet(Jb.mean(0).reshape(P, P)))
        C = cnt[R]
        ld_c = t.logdet_rows(C@Jc[n]/R); ld_b = t.logdet_rows(C@Jb/R); g = ld_c-ld_b
        rec = dict(K1=K1, R=R, I=I, I_from_W=Iw, I_base_same_reps=Ib, gain=I-Ib, gain_boot_se=float(g.std(ddof=1)), gain_ci=boot_ci(g),
                   I_boot_se=float(ld_c.std(ddof=1)), gain_vs_archive=I-t.I_SOD, cond_J=float(np.linalg.cond(J)))
        res['configs'][n] = rec
        out(f"{n:>9} {K1:>5} {R:>5} {I:>10.4f} {Ib:>16.4f} {I-Ib:>+8.4f} {g.std(ddof=1):>8.4f} "
            f"[{rec['gain_ci'][0]:+.4f}, {rec['gain_ci'][1]:+.4f}] {I-t.I_SOD:>+14.4f} {Iw-I:>20.1e}")
    # control minus primary on the paired replications 0..299
    R = R_of['control']; C = cnt[R]
    d = t.logdet_rows(C@Jc['control']/R)-t.logdet_rows(C@Jc['primary'][:R]/R)
    Ip300 = float(t.logdet(Jc['primary'][:R].mean(0).reshape(P, P)))
    cm = res['configs']['control']['I']-Ip300
    res['gains']['control_minus_primary_paired300'] = dict(diff=cm, boot_se=float(d.std(ddof=1)), ci=boot_ci(d), I_primary_300=Ip300)
    out(f"control minus primary on the paired replications 0..299 (cost of model error at K1 = 2500): {cm:+.4f} nat "
        f"(boot SE {d.std(ddof=1):.4f}, 95 % [{q(d, 2.5):+.4f}, {q(d, 97.5):+.4f}]); primary on 0..299: log det J {Ip300:.4f}, "
        f"gain {Ip300-res['configs']['control']['I_base_same_reps']:+.4f}")
    # proportional references (prompt §7 basis), context only
    for K1 in (1250, 2500, 3750):
        f = 1-K1/K
        res['gains'][f'proportional_reference_K1_{K1}'] = float(9*np.log((1-f)+f*np.exp(10.30/9)))
    out("proportional reference 9 ln((K1/K) + (1 - K1/K) exp(10.30/9)) (prompt §7 basis, context only): "
        + ', '.join(f"K1 = {K1}: {res['gains'][f'proportional_reference_K1_{K1}']:.3f}" for K1 in (1250, 2500, 3750)))

    # --- empirical measure D-hat (round 8b) -------------------------------------------------------------------------------
    out("")
    out("EMPIRICAL MEASURE (round 8b): D-hat = -log det M-hat, M-hat = mean (theta_hat - theta*)(theta_hat - theta*)^T over all "
        "replications (fallbacks included); Delta D-hat = paired difference with the baseline on the same replications")
    out(f"{'config':>9} {'R':>5} {'D-hat':>9} {'D-hat base':>10} {'Delta D-hat':>11} {'boot SE':>8} {'95% interval':>18} {'gain logdet J':>13}")
    for n in NAMES:
        if n == 'baseline': continue
        R = R_of[n]; C = cnt[R]
        Ec = TH[n]-t.THETA_STAR; Eb = TH['baseline'][:R]-t.THETA_STAR
        Dc = t.D_hat_of(Ec); Db = t.D_hat_of(Eb)
        EEc = (Ec[:, :, None]*Ec[:, None, :]).reshape(R, P*P); EEb = (Eb[:, :, None]*Eb[:, None, :]).reshape(R, P*P)
        dd = -t.logdet_rows(C@EEc/R)+t.logdet_rows(C@EEb/R)
        rec = res['configs'][n]; rec.update(D_hat=Dc, D_hat_base=Db, dD_hat=Dc-Db, dD_hat_boot_se=float(dd.std(ddof=1)), dD_hat_ci=boot_ci(dd))
        out(f"{n:>9} {R:>5} {Dc:>9.4f} {Db:>10.4f} {Dc-Db:>+11.4f} {dd.std(ddof=1):>8.4f} [{q(dd, 2.5):+.4f}, {q(dd, 97.5):+.4f}] "
            f"{rec['gain']:>+13.4f}")
    try:
        sb = json.load(open(os.path.join(EV, 'round8b', 'results', 'stats_round8b.json')))['P']
        res['earlier'] = dict(source='round8b/results/stats_round8b.json (P)', dD=sb['dD'], dD_ci=sb['boot']['dD'],
                              logdet_Jseq=sb['info']['logdet']['Jseq'], logdet_Jsod=sb['info']['logdet']['Jsod'],
                              info_gain=sb['info']['logdet']['Jseq']-sb['info']['logdet']['Jsod'], R=sb['R'])
        e = res['earlier']
        out(f"EARLIER two-stage experiment (round 8b, old table, two 5-s stages, total budget 2B, R = {e['R']}): Delta D-hat {e['dD']:.4f} "
            f"[{e['dD_ci'][0]:.3f}, {e['dD_ci'][1]:.3f}] (the measure of its headline 3.67); information measure log det J_seq - log det "
            f"J_SoD+SoD = {e['logdet_Jseq']:.4f} - {e['logdet_Jsod']:.4f} = {e['info_gain']:.4f}")
    except (OSError, KeyError) as ex:
        out(f"EARLIER: round 8b archive not read ({ex})")

    # --- budget -----------------------------------------------------------------------------------------------------------
    out("")
    out("BUDGET (N = all transmissions incl. the forced first; N_1 steps 1..K1, N_2 steps K1+1..K; plan_2 = B - 3 - N_1; 'N > B' = N >= 317)")
    out(f"{'config':>9} {'R':>5} {'mean N':>8} {'sd N':>6} {'min':>4} {'max':>4} {'frac>B':>7} {'#>B':>4} {'mean N_1':>9} {'sd N_1':>7} "
        f"{'mean N_2':>9} {'sd N_2':>7} {'mean plan_2':>11} {'mean N_2-plan_2':>15}")
    for n in NAMES:
        rr = rows[n]; N = col(rr, 'N'); N1 = col(rr, 'N_1'); N2 = col(rr, 'N_2'); pl = col(rr, 'plan2')
        rec = res['configs'][n]
        rec.update(N_mean=float(N.mean()), N_sd=float(N.std(ddof=1)), N_min=int(N.min()), N_max=int(N.max()), frac_over_B=float((N > B).mean()),
                   n_over_B=int((N > B).sum()), N1_mean=float(N1.mean()), N1_sd=float(N1.std(ddof=1)), N2_mean=float(N2.mean()),
                   N2_sd=float(N2.std(ddof=1)), plan2_mean=float(pl.mean()), N2_minus_plan2_mean=float((N2-pl).mean()),
                   N2_minus_plan2_sd=float((N2-pl).std(ddof=1)), N_hist={int(v): int((N == v).sum()) for v in np.unique(N)})
        out(f"{n:>9} {len(rr):>5} {N.mean():>8.3f} {N.std(ddof=1):>6.3f} {N.min():>4.0f} {N.max():>4.0f} {(N > B).mean():>7.4f} "
            f"{(N > B).sum():>4d} {N1.mean():>9.3f} {N1.std(ddof=1):>7.3f} {N2.mean():>9.3f} {N2.std(ddof=1):>7.3f} {pl.mean():>11.3f} "
            f"{(N2-pl).mean():>+15.3f}")
    for n in NAMES:
        out(f"  N histogram {n}: " + ' '.join(f"{k}:{v}" for k, v in res['configs'][n]['N_hist'].items()))
    f2500_300 = float((col(rows['primary'][:300], 'N') > B).mean())
    res['gains']['frac_over_B_primary_reps0_299'] = f2500_300
    out(f"  fraction N > B, primary on the replications 0..299 (paired with K1 = 1250): {f2500_300:.4f}")

    # --- stage 2 threshold e_k -----------------------------------------------------------------------------------------------
    out("")
    out("STAGE-2 THRESHOLD e_k (sigma units) and FAILURE CASES")
    out(f"{'config':>9} {'mean e':>7} {'min e':>6} {'max e':>6} {'#u<=0.5':>8} {'#u>=4':>6} {'#abnormal N_2':>13} "
        f"{'stage-1 CML not acc.':>20} {'final CML not acc.':>18}")
    for n in NAMES:
        rr = rows[n]; rec = res['configs'][n]
        s1na = [int(r['i']) for r in rr if r['s1_converged'] == '0']; fna = [int(r['i']) for r in rr if r['fin_converged'] == '0']
        rec.update(stage1_not_accepted=s1na, final_not_accepted=fna,
                   stage1_first_call_not_accepted=int(sum(r['s1_used'] != 'first' for r in rr)),
                   final_first_call_not_accepted=int(sum(r['fin_used'] != 'first' for r in rr)),
                   stage1_messages=sorted(set(r['s1_msg1'] for r in rr if r['s1_converged'] == '0')),
                   final_messages=sorted(set(r['fin_msg1'] for r in rr if r['fin_converged'] == '0')))
        if n != 'baseline':
            em = col(rr, 'e_mean2'); emin = col(rr, 'e_min2'); emax = col(rr, 'e_max2')
            hl = [int(r['i']) for r in rr if r['u_hit_lo'] == '1']; hh = [int(r['i']) for r in rr if r['u_hit_hi'] == '1']
            ab = [int(r['i']) for r in rr if r['abnormal'] == '1']
            rec.update(e_mean=float(em.mean()), e_mean_sd=float(em.std(ddof=1)), e_min=float(emin.min()), e_max=float(emax.max()),
                       u_hit_lo=hl, u_hit_hi=hh, abnormal=ab)
            out(f"{n:>9} {em.mean():>7.4f} {emin.min():>6.3f} {emax.max():>6.3f} {len(hl):>8d} {len(hh):>6d} {len(ab):>13d} "
                f"{len(s1na):>20d} {len(fna):>18d}")
        else:
            out(f"{n:>9} {'-':>7} {'-':>6} {'-':>6} {'-':>8} {'-':>6} {'-':>13} {len(s1na):>20d} {len(fna):>18d}")
    for n in NAMES:
        rec = res['configs'][n]
        out(f"  {n}: stage-1 CML not accepted in replications {rec['stage1_not_accepted']} (theta_tilde = fallback theta_LS); "
            f"final CML not accepted in {rec['final_not_accepted']} (final = fallback theta_tilde); first call not accepted: stage 1 "
            f"{rec['stage1_first_call_not_accepted']}, final {rec['final_first_call_not_accepted']}; messages {rec['stage1_messages']} / "
            f"{rec['final_messages']}")

    # --- model: prediction error of theta_tilde ---------------------------------------------------------------------------------
    out("")
    out("MODEL: prediction error of theta_tilde, RMS over all K steps of (h_k^T theta_tilde - x_k)/sigma (mismatch15.pred_error), "
        "the RMS over stage 2 only, and the max |.|")
    out(f"{'config':>9} {'K1':>5} {'mean':>7} {'P10':>7} {'median':>7} {'P90':>7} {'max':>7} | {'stage2 median':>13} {'P90':>7} | "
        f"{'max|.| median':>13} {'P90':>7}")
    for n in ('primary', 'K1_1250', 'K1_3750'):
        rr = rows[n]; pe = col(rr, 'pe_rms'); pe2 = col(rr, 'pe_rms_stage2'); pm = col(rr, 'pe_max')
        res['configs'][n].update(pe_rms=dict(mean=float(pe.mean()), p10=q(pe, 10), median=q(pe, 50), p90=q(pe, 90), max=float(pe.max())),
                                 pe_rms_stage2=dict(median=q(pe2, 50), p90=q(pe2, 90), max=float(pe2.max())),
                                 pe_max=dict(median=q(pm, 50), p90=q(pm, 90), max=float(pm.max())))
        out(f"{n:>9} {rr[0]['K1']:>5} {pe.mean():>7.4f} {q(pe, 10):>7.4f} {q(pe, 50):>7.4f} {q(pe, 90):>7.4f} {pe.max():>7.4f} | "
            f"{q(pe2, 50):>13.4f} {q(pe2, 90):>7.4f} | {q(pm, 50):>13.4f} {q(pm, 90):>7.4f}")
    rr = rows['primary']; pe = col(rr, 'pe_rms'); s1c = col(rr, 's1_converged', int)
    if (s1c == 0).any():
        out(f"  primary, stage-1 CML not accepted (theta_tilde = theta_LS): pe_rms {np.round(pe[s1c == 0], 4).tolist()}")

    # --- estimator ----------------------------------------------------------------------------------------------------------------
    out("")
    out("ESTIMATOR: final estimate (estimate_joint), CR sd = sqrt(diag J^-1) with the configuration's own J; I - D-hat = log det(M-hat J) "
        "(> 0: MSE above the CR bound); var ratio = exp((I - D-hat)/9), sd ratio = exp((I - D-hat)/18)")
    out(f"{'config':>9} {'runs':>15} {'n':>5} {'largest|bias|':>13} {'(param)':>7} {'Mahal.':>7} {'I - D-hat':>10} {'var ratio':>9} "
        f"{'sd ratio':>8} {'RMSE/CRsd min-max':>18}")
    for n in NAMES:
        rr = rows[n]; J = Jc[n].mean(0).reshape(P, P); th = TH[n]
        fc = col(rr, 'fin_converged', int) == 1; s1c = col(rr, 's1_converged', int) == 1
        sets = [('all', np.ones(len(rr), bool)), ('final accepted', fc), ('both accepted', fc & s1c)]
        res['configs'][n]['estimator'] = {}
        for lab, m in sets:
            if lab != 'all' and m.all(): continue
            st = t.acc_stats(th[m], J)
            res['configs'][n]['estimator'][lab] = st
            out(f"{n:>9} {lab:>15} {int(m.sum()):>5} {st['bias_max_abs']:>13.4f} {st['bias_argmax']+1:>7d} {st['bias_mahalanobis']:>7.3f} "
                f"{st['I_minus_Dhat']:>+10.4f} {st['var_ratio_gm']:>9.4f} {st['sd_ratio_gm']:>8.4f} "
                f"{st['rmse_cr_min']:>8.3f}-{st['rmse_cr_max']:.3f}")
        st = res['configs'][n]['estimator']['all']
        out(f"  {n} bias/CRsd: [{', '.join(f'{v:+.3f}' for v in st['bias_cr'])}] (MC SE avg {np.mean(st['bias_cr_se']):.3f})")
    # stage-1 estimate (theta_tilde) against its own information J_1 (context)
    for n in ('primary', 'K1_1250', 'K1_3750'):
        K1 = int(rows[n][0]['K1'])
        # J_1 = s^-2 sum_{k <= K1} wbar_k h_k h_k^T of the stage-1 SoD steps of this configuration
        W1 = A[f'{n}_W'][:K1]; J1 = (t.HK[:K1]*W1[:, None]).T@t.HK[:K1]/t.s**2
        st = t.acc_stats(A[f'{n}_theta_tilde'], J1)
        res['configs'][n]['stage1_estimator'] = st
        out(f"  stage-1 estimate theta_tilde, {n} (K1 = {K1}), against J_1 of the stage-1 SoD steps (log det {st['I']:.4f}): largest |bias| "
            f"{st['bias_max_abs']:.4f}, I_1 - D-hat {st['I_minus_Dhat']:+.4f}")

    # --- predictions ---------------------------------------------------------------------------------------------------------------
    out("")
    out("PREDICTIONS (scoring conventions of predictions.md §2, §3; closed intervals)")
    C = res['configs']; Pd = {}
    def score(key, hit, text):
        Pd[key] = dict(hit=bool(hit), text=text); out(f"  {key}: {'HIT' if hit else 'MISS'} - {text}")
    g = {n: C[n]['gain'] for n in NAMES}
    score('A1 K1=2500 gain in [5.3, 6.6]', 5.3 <= g['primary'] <= 6.6, f"gain {g['primary']:.4f}")
    score('A2 K1=1250 gain in [7.3, 8.6]', 7.3 <= g['K1_1250'] <= 8.6, f"gain {g['K1_1250']:.4f}")
    score('A3 K1=3750 gain in [3.2, 4.0]', 3.2 <= g['K1_3750'] <= 4.0, f"gain {g['K1_3750']:.4f}")
    ci = C['control']['gain_ci']
    score('A4 control rounds to 6.4', 6.35 <= g['control'] < 6.45, f"gain {g['control']:.4f}; 6.4 inside the 95 % interval "
          f"[{ci[0]:.4f}, {ci[1]:.4f}]: {ci[0] <= 6.4 <= ci[1]} (not scored)")
    Nm, fr = C['primary']['N_mean'], C['primary']['frac_over_B']
    score('A5 primary mean N in [312, 316] and frac N>B in [0.03, 0.25]', 312 <= Nm <= 316 and 0.03 <= fr <= 0.25,
          f"mean N {Nm:.3f} ({'in' if 312 <= Nm <= 316 else 'out'}); frac {fr:.4f} ({'in' if 0.03 <= fr <= 0.25 else 'out'})")
    f1250 = C['K1_1250']['frac_over_B']
    score('A6 frac N>B: K1=1250 > K1=2500', f1250 > fr, f"K1 = 1250 {f1250:.4f} vs K1 = 2500 {fr:.4f} (paired 0..299: {f2500_300:.4f})")
    st = C['primary']['estimator']['all']
    score('A7 primary largest bias < 0.3 and I - D-hat <= 9 ln 1.05', st['bias_max_abs'] < 0.3 and st['I_minus_Dhat'] <= 9*np.log(1.05),
          f"largest bias {st['bias_max_abs']:.4f}; I - D-hat {st['I_minus_Dhat']:.4f} (var ratio {st['var_ratio_gm']:.4f}; sd reading "
          f"{st['sd_ratio_gm']:.4f}: {'<=' if st['sd_ratio_gm'] <= 1.05 else '>'} 1.05)")
    # mine
    score('C1 K1=2500 gain in [5.8, 6.5]', 5.8 <= g['primary'] <= 6.5, f"{g['primary']:.4f}")
    score('C2 K1=1250 gain in [7.4, 8.3]', 7.4 <= g['K1_1250'] <= 8.3, f"{g['K1_1250']:.4f}")
    score('C3 K1=3750 gain in [3.3, 3.8]', 3.3 <= g['K1_3750'] <= 3.8, f"{g['K1_3750']:.4f}")
    cm = res['gains']['control_minus_primary_paired300']['diff']
    score('C4 control in [6.2, 6.6] and control - primary (0..299) in [0.05, 0.40]', 6.2 <= g['control'] <= 6.6 and 0.05 <= cm <= 0.40,
          f"control {g['control']:.4f}; control - primary {cm:+.4f}")
    sdN = C['primary']['N_sd']
    score('C5 primary mean N in [314.0, 315.5], sd in [1.6, 2.6], frac in [0.08, 0.25]',
          314.0 <= Nm <= 315.5 and 1.6 <= sdN <= 2.6 and 0.08 <= fr <= 0.25, f"mean {Nm:.3f}, sd {sdN:.3f}, frac {fr:.4f}")
    f3750 = C['K1_3750']['frac_over_B']
    score('C6 frac N>B: K1=1250 > K1=2500 > K1=3750', f1250 > fr > f3750, f"{f1250:.4f} / {fr:.4f} / {f3750:.4f}")
    n1 = len(C['primary']['stage1_not_accepted']); nf = len(C['primary']['final_not_accepted'])
    score('C7 primary stage-1 not accepted in [1, 15], final not accepted <= 25', 1 <= n1 <= 15 and nf <= 25, f"stage 1 {n1}, final {nf}")
    m = {n: C[n]['pe_rms']['median'] for n in ('primary', 'K1_1250', 'K1_3750')}
    score('C8 median pe_rms K1=2500 in [0.12, 0.25], ordered 1250 > 2500 > 3750',
          0.12 <= m['primary'] <= 0.25 and m['K1_1250'] > m['primary'] > m['K1_3750'],
          f"medians {m['K1_1250']:.4f} / {m['primary']:.4f} / {m['K1_3750']:.4f}")
    nh = len(set(C['primary']['u_hit_lo']) | set(C['primary']['u_hit_hi'])); na = len(C['primary']['abnormal'])
    score('C9 primary e_k clip reached in <= 2 replications, abnormal N_2 in 0', nh <= 2 and na == 0, f"clip {nh}, abnormal {na}")
    sta = C['primary']['estimator'].get('final accepted', st)
    score('C10 primary largest bias < 0.3, I - D-hat (all) in [0.3, 1.0], (accepted) in [0.15, 0.6]',
          st['bias_max_abs'] < 0.3 and 0.3 <= st['I_minus_Dhat'] <= 1.0 and 0.15 <= sta['I_minus_Dhat'] <= 0.6,
          f"bias {st['bias_max_abs']:.4f}; all {st['I_minus_Dhat']:.4f}; accepted {sta['I_minus_Dhat']:.4f}")
    dd = C['primary']['dD_hat']
    score('C11 primary |Delta D-hat - gain| <= 0.5', abs(dd-g['primary']) <= 0.5, f"Delta D-hat {dd:.4f}, gain {g['primary']:.4f}")
    res['predictions'] = Pd
    na_ = sum(v['hit'] for k, v in Pd.items() if k.startswith('A')); nc = sum(v['hit'] for k, v in Pd.items() if k.startswith('C'))
    out(f"  tally: prompt author {na_} of 7; mine {nc} of 11")
    res['main'] = mj
    t_path = os.path.join(RES, 'round19.json')
    import run_round19 as rr_
    rr_.jdump(res, t_path)
    out("=== ROUND 19 stage summary done")
