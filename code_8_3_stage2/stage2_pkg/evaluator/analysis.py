"""Step 5: analysis of the evaluator acceptance run (spec reports_CN/plan_evaluator_CN.md v4.1, §6, §7, §9).
Reads only stored outputs: results/rec_*.json, results/J_*.npy (run_all.py), results/mc/*.npz and mc_summary.json
(mc_reference.py). Writes logs/analysis.log (every number on a log line) and results/analysis.json.

  1 convergence  d1 = Q(6)-Q(12), d2 = Q(12)-Q(24), p_obs = log2|d1/d2| for absolute (log det, E[N], S; E[N]_P and
                 E[N]_P - E[N]_Phat as diagnostics) and relative quantities (Delta m_D, lambda_1..9) of every version and
                 input. U's relative quantities: "not applicable". |d2| <= 1e-8 max(1, |Q(24)|): "order undeterminable".
  2 stability    version B, finest two grids (12/24, or 24/48 for a switched input): |Delta m_D|, |log det|, |E[N]|, |m_E|
                 differences and ||J_f^{-1/2} J_c J_f^{-1/2} - I||_2 against the §6 limits.
  3 tail         version B, n = 12: margin 8 Delta -> 16 Delta, change of every recorded quantity.
  4 MC           the frozen 199 formal items (per input: E[N], S, log det, tr(Dbar)/9, 45 upper-triangle elements of Dbar;
                 paired Delta m_D of D, E, N), z = (rec - MC)/se, threshold t_{19,1-0.05/404} (corrected for 202);
                 W = J_rec^{-1/2} (B, accepted grid), D_b = W Jhat_b W - I, point estimate Dbar = W Jbar W - I (pooled).
                 Sign diagnostics; nonlinearity diagnostic; display only: m_E, lambda_2 - lambda_1, lambda_i(J_rec, Jbar).
                 Appendix: E[N], S, log det, Delta m_D of every version and grid against the same MC.
  5 runtime      steady-state time per version and grid; the B runtime line; cost of one search round (3000 evaluations).
  6 §9 criteria and the evidence for P1-P12 (verdicts are written in the report).
Usage (from this directory): python3 analysis.py [--switch48 E,N]   (inputs switched to n = 48 by the §6/§7 rule)
"""
import os, sys, json, argparse
import numpy as np
from scipy.stats import t as tdist
import density_recursion_v1 as dr

INPUTS = ['U', 'D', 'E', 'N']; CANDS = ['D', 'E', 'N']; GRIDS = dr.CFG['grids']['n']; VERS = dr.VERSIONS
P0 = {'v0': 1, 'A': 1, 'B': 2}; NB = dr.CFG['mc']['B']
PN = ['G0', 'Re1', 'Im1', 'Re2', 'Im2', 'Re3', 'Im3', 'Re4', 'Im4']      # H column order (config conventions.H_columns)
LIM = dict(dmD=0.002, logdet=0.005, EN=0.02, mE=0.002, Jnorm=0.005)


def rec(v, nm, n, mm=8): return json.load(open(os.path.join(dr.RES, f'rec_{v}_{nm}_n{n}_m{mm}.json')))
def Jm(v, nm, n, mm=8): return np.load(os.path.join(dr.RES, f'J_{v}_{nm}_n{n}_m{mm}.npy'))
def isqrt(J): ev, V = np.linalg.eigh(J); return (V/np.sqrt(ev))@V.T
def se(v): return float(np.std(v, ddof=1)/np.sqrt(len(v)))
def ld(M): sg, v = np.linalg.slogdet(M); assert sg > 0; return float(v)

def conv(Q, absolute):
    """d1, d2, p_obs and status for one quantity on three grids."""
    d1 = Q[0]-Q[1]; d2 = Q[1]-Q[2]
    if abs(d2) <= 1e-8*max(1.0, abs(Q[2])): return d1, d2, None, 'order_undeterminable'
    p = float(np.log2(abs(d1/d2))) if d1 != 0 else float('-inf')
    if np.sign(d1) != np.sign(d2): return d1, d2, p, 'sign_change'
    return d1, d2, p, 'ok'


def main(switch48):
    out = dict(switch48=switch48)
    ngrid = {nm: (48 if nm in switch48 else 24) for nm in INPUTS}
    if switch48: ngrid['U'] = 24      # U itself is accepted on 24 unless switched; U at 48 is used for switched candidates
    print(f"=== analysis.py; accepted grid per input (B): {ngrid}; switched to n=48: {switch48 or 'none'}")

    # --- 1 convergence -------------------------------------------------------------------------------------------
    print("\n[1] CONVERGENCE columns: version input quantity Q6 Q12 Q24 d1 d2 p_obs status")
    CV = {}
    for v in VERS:
        R = {(nm, n): rec(v, nm, n) for nm in INPUTS for n in GRIDS}
        for nm in INPUTS:
            qs = ['logdet', 'EN', 'S', 'EN_P'] + (['ENP_minus_ENPhat'] if v != 'v0' else []) + ['dmD', 'mE'] + [f'lam{i}' for i in range(1, 10)]
            for q in qs:
                if q in ('dmD', 'mE') or q.startswith('lam'):
                    if nm == 'U':
                        print(f"CONV {v} {nm} {q} not_applicable (identity: Delta m_D(U) = 0, lambda_i(U) = 1)"); continue
                    Q = [R[(nm, n)]['lam'][int(q[3:])-1] if q.startswith('lam') else R[(nm, n)][q] for n in GRIDS]
                elif q == 'ENP_minus_ENPhat':
                    Q = [R[(nm, n)]['EN_P']-R[(nm, n)]['EN_Phat'] for n in GRIDS]
                else:
                    Q = [R[(nm, n)][q] for n in GRIDS]
                d1, d2, p, st = conv(Q, q in ('logdet', 'EN', 'S'))
                CV[(v, nm, q)] = dict(Q=Q, d1=d1, d2=d2, p=p, status=st)
                print(f"CONV {v} {nm} {q} {Q[0]!r} {Q[1]!r} {Q[2]!r} {d1!r} {d2!r} {'n/a' if p is None else repr(p)} {st}")
    trig = [(v, nm, q, CV[(v, nm, q)]) for v in VERS for nm in INPUTS for q in ('logdet', 'EN', 'S')
            if CV[(v, nm, q)]['status'] == 'sign_change' or (CV[(v, nm, q)]['p'] is not None and abs(CV[(v, nm, q)]['p']-P0[v]) > 0.5)]
    for v, nm, q, c in trig: print(f"CONV-TRIGGER §6 {v} {nm} {q} p_obs {c['p']} p0 {P0[v]} status {c['status']}")
    print(f"CONV-TRIGGER count {len(trig)}")
    out['triggers_s6'] = [(v, nm, q, c['status'], c['p']) for v, nm, q, c in trig]

    # --- 2 stability (B, finest two grids) ------------------------------------------------------------------------
    print("\n[2] STABILITY version B, finest two grids; limits " + str(LIM))
    stab_ok = True; out['stability'] = {}
    for nm in INPUTS:
        nf = ngrid[nm]; nc = nf//2
        rc, rf = rec('B', nm, nc), rec('B', nm, nf)
        Wf = isqrt(Jm('B', nm, nf)); Jn = float(np.max(np.abs(np.linalg.eigvalsh(Wf@Jm('B', nm, nc)@Wf-np.eye(9)))))
        st = dict(logdet=abs(rc['logdet']-rf['logdet']), EN=abs(rc['EN']-rf['EN']), Jnorm=Jn)
        if nm == 'U': st.update(dmD=0.0, mE=0.0)
        else: st.update(dmD=abs(rc['dmD']-rf['dmD']), mE=abs(rc['mE']-rf['mE']))
        okk = {k: st[k] <= LIM[k] for k in LIM}; stab_ok &= all(okk.values())
        for k in LIM:
            note = ' (identity for U: Delta m_D = 0, m_E = 1 at every grid)' if nm == 'U' and k in ('dmD', 'mE') else ''
            print(f"STAB {nm} {nc}/{nf} {k} {st[k]!r} limit {LIM[k]} {'pass' if okk[k] else 'FAIL'}{note}")
        out['stability'][nm] = dict(grids=f'{nc}/{nf}', **st, ok=okk)
    print(f"STAB all four inputs within the §6 limits: {stab_ok}"); out['stability_ok'] = bool(stab_ok)

    # --- 3 tail ----------------------------------------------------------------------------------------------------
    print("\n[3] TAIL version B, n = 12, margin 8 Delta -> 16 Delta (change = 16 Delta value - 8 Delta value)")
    tmax = 0.0; out['tail'] = {}
    for nm in INPUTS:
        a, b = rec('B', nm, 12, 8), rec('B', nm, 12, 16)
        keys = ['EN', 'EN_P', 'S', 'logdet'] + ([] if nm == 'U' else ['dmD', 'mE'])
        ch = {k: b[k]-a[k] for k in keys}
        if nm != 'U': ch.update({f'lam{i+1}': b['lam'][i]-a['lam'][i] for i in range(9)})
        Ja, Jb_ = Jm('B', nm, 12, 8), Jm('B', nm, 12, 16); ch['J_max_rel_elem'] = float(np.max(np.abs(Jb_-Ja))/np.max(np.abs(Ja)))
        ch['G_8'] = a['G']; ch['G_16'] = b['G']
        for k, v in ch.items(): print(f"TAIL {nm} {k} {v!r}")
        m = max(abs(v) for k, v in ch.items() if not k.startswith('G_')); tmax = max(tmax, m)
        out['tail'][nm] = ch
    print(f"TAIL max |change| over all quantities and inputs {tmax!r}; <= 1e-6 (§9): {tmax <= 1e-6}; < 1e-8 (P6): {tmax < 1e-8}")
    out['tail_max'] = tmax

    # --- 4 MC comparison ---------------------------------------------------------------------------------------------
    MC = json.load(open(os.path.join(dr.RES, 'mc', 'mc_summary.json')))
    NPZ = {nm: np.load(os.path.join(dr.RES, 'mc', f'mc_{nm}.npz')) for nm in INPUTS}
    BS = np.load(os.path.join(dr.RES, 'mc', 'mc_batch_stats.npz'))
    thr = float(tdist.ppf(1-0.05/(2*202), NB-1))
    print(f"\n[4] MC COMPARISON (seed {dr.CFG['mc']['seed']}, R = {dr.CFG['mc']['R']}, {NB} batches); "
          f"threshold t_(19, 1-0.05/404) = {thr!r} (spec: 4.496)")
    THR = 4.496; assert abs(thr-THR) < 5e-4
    items = []; sign = {}
    for nm in INPUTS:
        n = ngrid[nm]; r = rec('B', nm, n); Jr = Jm('B', nm, n)
        for q, se_key in (('EN', 'EN_se'), ('S', 'S_se'), ('logdet', 'logdet_se')):
            items.append((nm, q, r[q], MC[nm][q], MC[nm][se_key]))
        W = isqrt(Jr); I9 = np.eye(9)
        Db = np.array([W@Jb@W-I9 for Jb in NPZ[nm]['Jbat']]); Dbar = W@NPZ[nm]['J']@W-I9
        tb = np.trace(Db, axis1=1, axis2=2)/9
        items.append((nm, 'tr(Dbar)/9', 0.0, float(np.trace(Dbar)/9), se(tb)))
        for i in range(9):
            for j in range(i, 9):
                items.append((nm, f'Dbar[{PN[i]},{PN[j]}]', 0.0, float(Dbar[i, j]), se(Db[:, i, j])))
        dg = np.diag(Dbar); npos = int((dg > 0).sum())
        sign[nm] = dict(tr=float(np.trace(Dbar)/9), n_pos_diag=npos, n_neg_diag=9-npos)
        lam_rm = dr.eigh(Jr, NPZ[nm]['J'], eigvals_only=True)
        for i, v in enumerate(lam_rm): print(f"DISPLAY {nm} lambda{i+1}(J_rec,Jbar_MC) {float(v)!r}")
    for nm in CANDS:
        items.append((nm, 'dmD_paired_vs_U', rec('B', nm, ngrid[nm])['dmD'], MC[nm]['dmD'], MC[nm]['dmD_se_paired']))
    assert len(items) == 199, len(items)
    print("ITEM columns: index input item rec MC se z |z|<=thr")
    nfail = 0; zs = []
    for k, (nm, q, rv, mv, s_) in enumerate(items, 1):
        z = (rv-mv)/s_; zs.append(z); ok = abs(z) <= THR; nfail += not ok
        print(f"ITEM {k:3d} {nm} {q} {rv!r} {mv!r} {s_!r} {z!r} {'pass' if ok else 'FAIL'}")
    zs = np.array(zs)
    print(f"MC formal items {len(items)}; |z| > {THR}: {nfail}; max |z| {float(np.abs(zs).max())!r}; |z| > 2: {int((np.abs(zs) > 2).sum())}; "
          f"|z| > 3: {int((np.abs(zs) > 3).sum())}")
    out['mc_items'] = [dict(input=a, item=b, rec=c, mc=d, se=e, z=float(z)) for (a, b, c, d, e), z in zip(items, zs)]
    out['mc_nfail'] = nfail; out['mc_max_absz'] = float(np.abs(zs).max())
    # z of the whitened elements per input (summary)
    for nm in INPUTS:
        zz = np.array([it['z'] for it in out['mc_items'] if it['input'] == nm and it['item'].startswith('Dbar[')])
        print(f"WHITENED {nm} 45 elements: max |z| {float(np.abs(zz).max())!r}, mean z {float(zz.mean())!r}, n(z>0) {int((zz > 0).sum())}, "
              f"n(z<0) {int((zz < 0).sum())}, |z| > 2: {int((np.abs(zz) > 2).sum())}")
    # sign diagnostics
    trz = {nm: next(it['z'] for it in out['mc_items'] if it['input'] == nm and it['item'] == 'tr(Dbar)/9') for nm in INPUTS}
    for nm in INPUTS:
        sg = sign[nm]
        print(f"SIGN {nm} tr(Dbar)/9 {sg['tr']!r} sign {'+' if sg['tr'] > 0 else '-'} z {trz[nm]!r}; diagonal of Dbar: "
              f"{sg['n_pos_diag']} positive, {sg['n_neg_diag']} negative (max same-sign {max(sg['n_pos_diag'], sg['n_neg_diag'])} of 9)")
    same_sign = len({np.sign(sign[nm]['tr']) for nm in INPUTS}) == 1
    susp = same_sign and all(abs(trz[nm]) > 2 for nm in INPUTS)
    print(f"SIGN tr(Dbar)/9 same sign for all four inputs: {same_sign}; all |z| > 2: {all(abs(trz[nm]) > 2 for nm in INPUTS)}; "
          f"'suspicious scale bias' flag (§9): {susp}")
    out['sign'] = sign; out['sign_same'] = bool(same_sign); out['suspicious_scale_bias'] = bool(susp)
    # nonlinearity diagnostic
    for nm in INPUTS:
        m = MC[nm]
        print(f"NONLIN {nm} D {m['Dnl']!r} -D/19 {m['minus_Dnl_over_19']!r} se(log det) {m['logdet_se']!r} "
              f"D>0 {m['Dnl'] > 0} |D/19|<se {abs(m['minus_Dnl_over_19']) < m['logdet_se']}")
    # display only: m_E and lambda_2 - lambda_1
    for nm in CANDS:
        r = rec('B', nm, ngrid[nm])
        print(f"DISPLAY {nm} m_E rec {r['mE']!r} MC {MC[nm]['mE']!r} diff {r['mE']-MC[nm]['mE']!r} paired se {MC[nm]['mE_se_paired']!r} "
              f"lambda2-lambda1 rec {r['lam'][1]-r['lam'][0]!r} MC {MC[nm]['lam2_minus_lam1']!r} (display only, not a test)")
    # appendix: every version and grid on the absolute quantities and Delta m_D
    print("APPX columns: version grid input quantity rec MC se z")
    for v in VERS:
        for n in GRIDS:
            for nm in INPUTS:
                r = rec(v, nm, n)
                for q, sk in (('EN', 'EN_se'), ('S', 'S_se'), ('logdet', 'logdet_se')) + ((('dmD', 'dmD_se_paired'),) if nm != 'U' else ()):
                    z = (r[q]-MC[nm][q])/MC[nm][sk]
                    print(f"APPX {v} {n} {nm} {q} {r[q]!r} {MC[nm][q]!r} {MC[nm][sk]!r} {z!r}")
    # §7: Richardson extrapolation (diagnostic) only if B fails
    if nfail:
        print("RICHARDSON needed: B/accepted grid failed the MC comparison; see report (diagnostic only)")

    # --- 5 runtime -----------------------------------------------------------------------------------------------------
    print("\n[5] RUNTIME steady state (s), import and JIT excluded")
    out['runtime'] = {}
    for v in VERS:
        for n in GRIDS:
            ts = [rec(v, nm, n)['runtime_s'] for nm in INPUTS]
            print(f"RUNTIME {v} n={n} mean {float(np.mean(ts))!r} min {min(ts)!r} max {max(ts)!r} per input {dict(zip(INPUTS, ts))}")
            out['runtime'][f'{v}_{n}'] = ts
    tB = [rec('B', nm, ngrid[nm])['runtime_s'] for nm in INPUTS]
    print(f"RUNTIME-LINE version B accepted grid: max {max(tB)!r} s, limit 15 s (numba): {'pass' if max(tB) <= 15 else 'FAIL'}")
    for lab, tv in (('mean', float(np.mean(tB))), ('max', max(tB))):
        print(f"SEARCH-ROUND 3000 evaluations x {lab} B time {tv!r} s = {3000*tv!r} s = {3000*tv/3600!r} h (serial, one core)")
    for v in VERS:
        print(f"RUNTIME-RATIO {v} t(24)/t(12) {float(np.mean(out['runtime'][f'{v}_24'])/np.mean(out['runtime'][f'{v}_12']))!r}")
    for v in ('A', 'B'):
        print(f"RUNTIME-RATIO numpy v0 / numba {v} at n=12 {float(np.mean(out['runtime']['v0_12'])/np.mean(out['runtime'][f'{v}_12']))!r}")
    out['runtime_line_ok'] = bool(max(tB) <= 15)

    # --- 6 §9 criteria and prediction evidence ------------------------------------------------------------------------
    print("\n[6] §9 CRITERIA")
    AB = [rec(v, nm, n) for v in ('A', 'B') for nm in INPUTS for n in GRIDS]
    V0 = [rec('v0', nm, n) for nm in INPUTS for n in GRIDS]
    crit = {}
    crit['nonneg'] = min(r['min_m'] for r in AB) >= 0
    print(f"CRIT nonnegativity A,B min m over 24 evaluations {min(r['min_m'] for r in AB)!r} -> {crit['nonneg']}; "
          f"v0 (record only) min m {min(r['min_m'] for r in V0)!r}")
    md = max(r['mass_dev_max'] for r in AB); crit['mass'] = md <= 1e-10
    print(f"CRIT mass conservation A,B max_k |sum m - 1| over 24 evaluations {md!r} (<= 1e-10) -> {crit['mass']}")
    kd = max(r['kernel_mass_dev_max'] for r in AB); kf = max(r['kernel_factor_dev_max'] for r in AB); crit['kernel'] = kd <= 1e-12
    print(f"CRIT kernel truncation A,B max |kernel mass - 1| {kd!r} (<= 1e-12), max |factor - 1| {kf!r} -> {crit['kernel']}")
    inv = all(r['inv_S'] and r['inv_J'] and r['inv_w'] and r['inv_c'] for r in AB); crit['invariants'] = inv
    print(f"CRIT invariants A,B (24 evaluations): 0<=c<=1 {all(r['inv_c'] for r in AB)}, 1<=S<=K {all(r['inv_S'] for r in AB)}, "
          f"0<=J<=Jfull {all(r['inv_J'] for r in AB)} -> {inv}; v0 (record only): c {all(r['inv_c'] for r in V0)}, "
          f"S {all(r['inv_S'] for r in V0)}, J {all(r['inv_J'] for r in V0)}")
    print(f"CRIT c on all nodes: A/B min {min(r['c_min'] for r in AB)!r} max {max(r['c_max'] for r in AB)!r} NaN {sum(r['c_nan'] for r in AB)}; "
          f"tails min {min(r['tail_c_min'] for r in AB)!r} max {max(r['tail_c_max'] for r in AB)!r} NaN {sum(r['tail_nan'] for r in AB)} "
          f"({sum(r['tail_nodes'] for r in AB)} tail node-steps, counted once per evaluation); v0 min {min(r['c_min'] for r in V0)!r} "
          f"max {max(r['c_max'] for r in V0)!r} NaN {sum(r['c_nan'] for r in V0)} tail NaN {sum(r['tail_nan'] for r in V0)}")
    crit['stability'] = out['stability_ok']; crit['tail'] = out['tail_max'] <= 1e-6
    crit['mc'] = nfail == 0; crit['sign'] = not susp; crit['runtime'] = out['runtime_line_ok']
    for k in ('stability', 'tail', 'mc', 'sign', 'runtime'): print(f"CRIT {k} -> {crit[k]}")
    num_ok = all(crit[k] for k in ('nonneg', 'mass', 'kernel', 'invariants', 'stability', 'tail', 'mc', 'sign'))
    print(f"CRIT numerical acceptance (discrete implementation, convergence, MC consistency): {'passed' if num_ok else 'NOT passed'}; "
          f"runtime line: {'passed' if crit['runtime'] else 'NOT passed'}")
    out['criteria'] = crit; out['numerical_ok'] = bool(num_ok)

    print("\n[7] PREDICTION EVIDENCE")
    for n in GRIDS:
        for nm in INPUTS:
            r = rec('v0', nm, n)
            ms = np.load(os.path.join(dr.RES, f'mass_v0_{nm}_n{n}_m8.npy')); kx = int(np.argmax(np.abs(ms-1)))
            print(f"P1 v0 {nm} n={n} final mass - 1 {r['mass_last']-1!r} max_k |mass - 1| {r['mass_dev_max']!r} at step index {kx}; "
                  f"mass - 1 at the prototype's checkpoints k = 1000..4000 {[float(v) for v in ms[1000:5000:1000]-1]}")
    print(f"P1 A,B max drift {md!r}")
    for v in VERS:
        for nm in INPUTS:
            for q in ('logdet', 'EN'):
                c = CV[(v, nm, q)]; print(f"P2 {v} {nm} {q} p_obs {c['p']!r} status {c['status']} (predicted {P0[v]})")
    for nm in INPUTS:
        bA = [rec('A', nm, n)['EN']-MC[nm]['EN'] for n in GRIDS]; b0 = rec('v0', nm, 12)['EN']-MC[nm]['EN']
        print(f"P3 {nm} A E[N] - MC at n=6,12,24 {bA[0]!r} {bA[1]!r} {bA[2]!r}; ratios {bA[0]/bA[1]!r} {bA[1]/bA[2]!r}; "
              f"v0 E[N] - MC at n=12 {b0!r}; A/v0 at n=12 {bA[1]/b0!r}")
    for nm in INPUTS:
        zE = next(it['z'] for it in out['mc_items'] if it['input'] == nm and it['item'] == 'EN')
        zL = next(it['z'] for it in out['mc_items'] if it['input'] == nm and it['item'] == 'logdet')
        print(f"P4/P8 {nm} z(E[N]) {zE!r} z(log det) {zL!r}")
    for v in ('A', 'B'):
        for nm in INPUTS:
            c = CV[(v, nm, 'ENP_minus_ENPhat')]
            print(f"P5 {v} {nm} E[N]_P - E[N]_Phat at n=6,12,24 {c['Q'][0]!r} {c['Q'][1]!r} {c['Q'][2]!r} "
                  f"ratios {c['Q'][0]/c['Q'][1]!r} {c['Q'][1]/c['Q'][2]!r}")
    for nm in CANDS:
        a_, b_ = CV[('B', nm, 'dmD')], CV[('B', nm, 'logdet')]
        print(f"P9 B {nm} |d2(dmD)| {abs(a_['d2'])!r} |d2(logdet)| {abs(b_['d2'])!r} smaller {abs(a_['d2']) < abs(b_['d2'])}; "
              f"p_obs dmD {a_['p']!r} ({a_['status']}), log det {b_['p']!r} ({b_['status']})")
    json.dump(out, open(os.path.join(dr.RES, 'analysis.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--switch48', default='')
    a = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'analysis.log'))
    main([x for x in a.switch48.split(',') if x])
