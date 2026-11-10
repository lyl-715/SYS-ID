"""Round 13 helpers used by Parts A and B (prompt reports_CN/prompt_round13_part0_infrastructure_EN.md §2.4).

One process serves one input: eval13 binds the scenario at import to the input named by R13_INPUT (default 'U'); this module
imports eval13 and dp13 and therefore inherits that binding.

  full_information()   J_full = HK^T HK / sigma^2 (every sample transmitted) and I_full = slogdet(J_full).
  ConstRows(M)         a constant-threshold SoD of half-width M grid units (threshold M sigma/24) on the full grid, usable by
                       eval13.evaluate and eval13.evaluate_refined: L_i = min(M, i + X_PAD - 2), U_i = min(M, G - 1 - i + X_PAD - 2).
                       For M <= X_PAD - 2 = 958 nothing is clipped. For M > 958, eval_const() asserts after every forward that no
                       clipped node carries mass (support strictly inside (M - X_PAD + 2, G - 1 - (M - X_PAD + 2)), scaled by f on
                       the fine grid); a violation stops the run.
  eval_const(M, f)     one constant-SoD forward on sigma/24 (f = 1) or sigma/(24 f).
  matched_sod(B)       the constant SoD whose sigma/120 E[N] is closest to B (search rule of §2.4, steps 1-3).
  design(...)          the two-scan design of §2.4 (scans j0, j1, optional j2), certificate for B and B + 0.25, sigma/120
                       re-evaluation, forward/backward check, shape statistics; scans are checkpointed.
  certificate(...)     UB = crit(T_ref) - p + min_lambda dual(lambda) and UB' = crit(T_ref) - p + min_lambda [dual + 0.25 lambda].

Every reported evaluation saves its 9x9 J as .npy next to the json it is reported in (`save_J`).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import eval13 as e13
import dp13
ew, dr = e13.ew, e13.dr
s, K, B, G = e13.s, e13.K, e13.B, e13.G
NAME = e13.NAME
X_PAD = e13.X_PAD
RES = os.path.join(HERE, 'results')
WORK = os.environ.get('ROUND13_WORK', os.path.join(HERE, 'work'))
F_REF = 5                                    # sigma/120 = refinement factor 5 of the sigma/24 grid
DB_CERT = 0.25                               # B' = B + 0.25
GAP_MAX = 0.5                                # third scan if the certificate gap exceeds 0.5 nat
FB_TOL = 1e-12                               # forward vs fixed-policy backward (Part A §8 item 2)
MATCH_TOL = 1.0                              # |E[N]_120(M*) - B| <= 1 (round 7's D0 rule)


class Stop(RuntimeError):
    """a check of round 13 failed: the run stops (no fallback)."""


def jdump(obj, path):
    tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, default=_jdef)
    os.replace(tmp, path)

def _jdef(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))

def relerr(x, ref): return abs(float(x)-float(ref))/abs(float(ref))

def save_J(J, outdir, tag):
    """the 9x9 J of a reported evaluation, next to the json (Part B reuses them)."""
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f'J_{tag}.npy'); np.save(p, np.asarray(J, np.float64)); return os.path.relpath(p, HERE)

def logdet9(J): return float(np.linalg.slogdet(J)[1])


# --- full-data information ---------------------------------------------------------------------------------------------
def full_information():
    J = e13.HK.T@e13.HK/s**2
    sgn, ld = np.linalg.slogdet(J)
    assert sgn > 0
    return J, float(ld)

def input_sanity():
    """Part 0 G4 line: sum a^2, output range and largest per-step motion in sigma, G, I_full."""
    e = dr.CFG['inputs'][NAME]; a = np.asarray(e['a'], np.float64)
    J, I = full_information()
    return dict(input=NAME, a=[float(x) for x in a], chi=[float(x) for x in e['chi']], sum_a2=float(np.sum(a**2)),
                sum_a2_minus_1=float(np.sum(a**2)-1.0), range_sigma=float((e13.XK.max()-e13.XK.min())/s),
                max_step_sigma=float(np.abs(np.diff(e13.XK)).max()/s), G=int(G), I_full=I, z0=e13.Z0,
                min_KA_minus_X24=int((e13.KA[1:]-ew.X24).min()), max_KB_plus_X24=int((e13.KB[1:]+ew.X24).max()))


# --- constant SoD on the full grid ----------------------------------------------------------------------------------------
class ConstRows:
    """constant half-width M (grid units) at every node and step, clipped near the grid ends so that every window edge stays in
    the padded index range of eval13 (prompt §2.4)."""
    def __init__(self, M):
        M = int(M); assert 48 <= M <= 1920, M
        self.M = M; self.X = X_PAD
        i = np.arange(G, dtype=np.int64)
        self._L = np.minimum(M, i+X_PAD-2).astype(np.int64)
        self._U = np.minimum(M, G-1-i+X_PAD-2).astype(np.int64)
        self.clipped = M > X_PAD-2
        self.n_clipped = int((self._L < M).sum()+(self._U < M).sum())

    def rows_full(self, p, out=None):
        if out is None: return self._L, self._U
        out[0][:] = self._L; out[1][:] = self._U; return out

    def is_sod(self): return False

    def check_support(self, fw, f=1):
        """clipped nodes never carry mass (only checked, and only needed, for M > X_PAD - 2)."""
        if not self.clipped: return dict(clipped=False)
        c = f*(self.M-X_PAD+2); Gn = int(fw['G'])
        lo, hi = int(fw['sup_lo'].min()), int(fw['sup_hi'].max())
        ok = lo > c and hi < Gn-1-c
        if not ok:
            raise e13.AssertionStop(f"ConstRows({self.M}), f = {f}: mass on a clipped node (support [{lo}, {hi}], clipped below index "
                                    f"{c} and above {Gn-1-c})")
        return dict(clipped=True, n_clipped_coarse=self.n_clipped, support_lo_min=lo, support_hi_max=hi, clip_lo=c, clip_hi=Gn-1-c)

def eval_const(M, f=1):
    """constant SoD of half-width M (coarse grid units) on sigma/24 (f = 1) or on sigma/(24 f) (eval13.evaluate_refined)."""
    cr = ConstRows(M)
    t0 = time.perf_counter()
    if f == 1:
        rec, J, fw = e13.evaluate(cr); rec['G'] = G; rec['f'] = 1
    else:
        rec, J, fw = e13.evaluate_refined(cr, f)
    rec['wall_s'] = time.perf_counter()-t0
    rec['M'] = cr.M; rec['support'] = cr.check_support(fw, f)
    return rec, J, fw

def matched_sod(Bt=B, log=print, outdir=None, tag='sod_matched'):
    """the constant SoD whose expected number of transmissions is closest to Bt (prompt §2.4):
      1. sigma/24: E[N](M) from M = 480 (Delta*) in steps of +-20 towards Bt until Bt is bracketed, then integer bisection until two
         consecutive M bracket Bt; M_hat = the one with the smaller |E[N] - Bt| (ties -> larger M, DECISION 13-6);
      2. sigma/120 at M_hat - 1, M_hat, M_hat + 1, extended by one on the side of the minimiser while the minimiser of |E[N]_120 - Bt|
         is at an end; M* = the minimiser (ties -> larger M);
      3. accepted iff |E[N]_120(M*) - Bt| <= 1."""
    c24, c120 = {}, {}
    def ev24(M):
        if M not in c24:
            rec, J, fw = eval_const(M, 1); c24[M] = (rec, J)
            log(f"MATCH {NAME} sigma/24 M={M} ({M/24:.4f} sigma) EN={rec['EN']!r} EN-B={rec['EN']-Bt!r} S={rec['S']!r} logdet={rec['logdet']!r} "
                f"mass_dev_max={rec['mass_dev_max']!r} wall={rec['wall_s']:.1f}s")
        return c24[M][0]['EN']-Bt
    def ev120(M):
        if M not in c120:
            rec, J, fw = eval_const(M, F_REF); c120[M] = (rec, J)
            log(f"MATCH {NAME} sigma/120 M={M} EN={rec['EN']!r} EN-B={rec['EN']-Bt!r} S={rec['S']!r} logdet={rec['logdet']!r} "
                f"mass_dev_max={rec['mass_dev_max']!r} G={rec['G']} wall={rec['wall_s']:.1f}s")
        return c120[M][0]['EN']-Bt
    M0 = 480; d0 = ev24(M0)
    step = 20 if d0 > 0 else -20                       # E[N] decreases with M: above B -> larger M
    Ma, da = M0, d0
    while True:
        Mb = Ma+step
        if not (48 <= Mb <= 1920): raise Stop(f"matched SoD: no bracket of B for M in [48, 1920]")
        db = ev24(Mb)
        if da*db <= 0: break
        Ma, da = Mb, db
    lo, hi = sorted((Ma, Mb))
    while hi-lo > 1:
        mid = (lo+hi)//2; dm = ev24(mid)
        if ev24(lo)*dm <= 0: hi = mid
        else: lo = mid
    M_hat = lo if abs(ev24(lo)) < abs(ev24(hi)) else hi
    Ms = [M_hat-1, M_hat, M_hat+1]
    while True:
        for M in Ms: ev120(M)
        key = sorted(Ms, key=lambda M: (abs(c120[M][0]['EN']-Bt), -M))
        Mst = key[0]
        if Mst == Ms[0]: Ms = [Ms[0]-1]+Ms
        elif Mst == Ms[-1]: Ms = Ms+[Ms[-1]+1]
        else: break
    ev24(Mst)
    r24, J24 = c24[Mst]; r120, J120 = c120[Mst]
    res = dict(input=NAME, B=Bt, M_hat=M_hat, M_star=Mst, M_star_sigma=Mst/24, threshold=Mst*s/24,
               EN24=r24['EN'], S24=r24['S'], logdet24=r24['logdet'], EN120=r120['EN'], S120=r120['S'], logdet120=r120['logdet'],
               residual120=r120['EN']-Bt, residual24=r24['EN']-Bt, accepted=bool(abs(r120['EN']-Bt) <= MATCH_TOL),
               mass_dev_max24=r24['mass_dev_max'], mass_dev_max120=r120['mass_dev_max'], support24=r24['support'], support120=r120['support'],
               bracket=[lo, hi], evaluated24={int(M): c24[M][0]['EN'] for M in sorted(c24)}, evaluated120={int(M): c120[M][0]['EN'] for M in sorted(c120)},
               wall24_s=r24['wall_s'], wall120_s=r120['wall_s'])
    if outdir is not None:
        res['J24'] = save_J(J24, outdir, f'{tag}_s24'); res['J120'] = save_J(J120, outdir, f'{tag}_s120')
    log(f"MATCHED {NAME} M*={Mst} ({Mst/24:.4f} sigma) EN120={r120['EN']!r} residual {r120['EN']-Bt!r} accepted={res['accepted']} "
        f"logdet24={r24['logdet']!r} logdet120={r120['logdet']!r}")
    return res, J24, J120


# --- the two-scan design --------------------------------------------------------------------------------------------------
_DES = {}
def get_design(log=print):
    if 'des' not in _DES:
        dp13.warmup(); _DES['des'] = dp13.Design11(log)
    return _DES['des']

def certificate(st, crit_ref, p_dim, dB=DB_CERT):
    """bound from a scan whose tangent point has criterion value crit_ref: UB for B, UB' for B + dB."""
    runs = st['runs']
    d = np.array([r['dual'] for r in runs]); lam = np.array([r['lam'] for r in runs])
    i = int(np.argmin(d)); dp = d+dB*lam; ip = int(np.argmin(dp))
    assert float(d[i]) == st['min_dual']
    return dict(scan=st['label'], crit_ref=crit_ref, p_dim=p_dim, min_dual=float(d[i]), lam_argmin=float(lam[i]), run_argmin=runs[i]['run'],
                UB=float(crit_ref-p_dim+d[i]), dB=dB, min_dual_p=float(dp[ip]), lam_argmin_p=float(lam[ip]), run_argmin_p=runs[ip]['run'],
                UBp=float(crit_ref-p_dim+dp[ip]))

def run_scan(des, acts, gtuple, label, outdir, work, log):
    """one Scan11 with checkpoint: a completed scan (state json with a final table whose .npy still exists, same tangent) is skipped."""
    gam, gbar, g1 = gtuple
    path = os.path.join(outdir, f'scan_{label}.json')
    if os.path.exists(path):
        st = json.load(open(path))
        if st.get('final_run') and st['gbar'] == gbar and st['gamma1'] == g1:
            fin = [r for r in st['runs'] if r['run'] == st['final_run']][0]
            if os.path.exists(fin['table']):
                log(f"SCAN {label}: completed scan found ({path}), skipped"); return st, fin
    log(f"SCAN {label}: actions {acts.name} ({acts.n}); gbar {gbar!r} gamma_1 {g1!r}; lambda_0 = 5 gbar = {5*gbar!r}")
    sc = dp13.Scan11(des, acts, gam, gbar, g1, label, work, log)
    t0 = time.perf_counter(); st = sc.scan(); st['wall_s'] = time.perf_counter()-t0
    fin = None
    if st.get('final_run'):
        fin = [r for r in st['runs'] if r['run'] == st['final_run']][0]
        sc.cleanup(keep=[fin['table']])
    else:
        sc.cleanup(keep=[])
    jdump(st, path)
    if st['status'] == 'monotonicity violation': raise Stop(f"scan {label}: monotonicity violation {st.get('message')}")
    if fin is None: raise Stop(f"scan {label}: no feasible table ({st['status']})")
    return st, fin

def eval_scan_table(fin, acts, label, gtuple, gam_stats, crit_fn, outdir, log, f=F_REF):
    """the final table of a scan: sigma/24 (record, J, F of the scan's gamma, shape statistics), fixed-policy backward check,
    sigma/120 (record, J). Both J are saved."""
    gam, gbar, g1 = gtuple
    tab = e13.Table11(np.load(fin['table'], mmap_mode='r'), acts, label)
    st = e13.Stats(gam_stats)
    t0 = time.perf_counter(); rec, J, fw = e13.evaluate(tab, gam=gam, stats=st); t24 = time.perf_counter()-t0
    if not (rec['EN'] == fin['EN'] and rec['logdet'] == fin['logdet']): raise Stop(f"{label}: re-evaluation differs from the scan record")
    sres, ev, w = st.result()
    F = float(gam[1:]@fw['W'][1:]); N = rec['EN']-1.0; lam = fin['lam']
    t0 = time.perf_counter(); bw = e13.backward(tab, gam); tbw = time.perf_counter()-t0
    tf, tb = F-lam*N, bw['Fc']-lam*bw['Nq']
    chk = dict(F_forward=F, F_backward=bw['Fc'], F_rel=abs(bw['Fc']-F)/abs(F), N_forward=N, N_backward=bw['Nq'], N_rel=abs(bw['Nq']-N)/abs(N),
               total_forward=tf, total_backward=tb, total_rel=abs(tb-tf)/abs(tf), bw_s=tbw)
    chk['pass_'] = bool(chk['F_rel'] <= FB_TOL and chk['N_rel'] <= FB_TOL and chk['total_rel'] <= FB_TOL)
    rf, Jf, ff = e13.evaluate_refined(tab, f)
    d = dict(label=label, run=fin['run'], lam=lam, lam_over_gbar=fin['lam_over_gbar'], gbar=gbar, gamma1=g1, table=fin['table'],
             EN24=rec['EN'], S24=rec['S'], logdet24=rec['logdet'], crit24=crit_fn(J), F=F, mass_dev_max24=rec['mass_dev_max'], min_m24=rec['min_m'],
             support_lo_min=int(fw['sup_lo'].min()), support_hi_max=int(fw['sup_hi'].max()), fwd24_s=t24,
             EN120=rf['EN'], S120=rf['S'], logdet120=rf['logdet'], crit120=crit_fn(Jf), mass_dev_max120=rf['mass_dev_max'], min_m120=rf['min_m'],
             G120=rf['G'], fwd120_s=rf['wall_s'], budget120_pass=bool(rf['EN'] <= B), slack120=B-rf['EN'], slack24=B-rec['EN'],
             check_fb=chk, shape=sres, J24=save_J(J, outdir, f'{label}_s24'), J120=save_J(Jf, outdir, f'{label}_s120'))
    log(f"TABLE {label}: crit24 {d['crit24']!r} logdet24 {rec['logdet']!r} EN24 {rec['EN']!r} S24 {rec['S']!r}; crit120 {d['crit120']!r} "
        f"logdet120 {rf['logdet']!r} EN120 {rf['EN']!r} (<= B: {d['budget120_pass']}) S120 {rf['S']!r}; mass dev {rec['mass_dev_max']:.2e} / "
        f"{rf['mass_dev_max']:.2e}; forward-backward F {chk['F_rel']:.1e} N {chk['N_rel']:.1e} total {chk['total_rel']:.1e} "
        f"({'PASS' if chk['pass_'] else 'FAIL'}); edge quantiles 10/50/90 {sres['edge_q10']:.4f} / {sres['edge_median']:.4f} / {sres['edge_q90']:.4f}; "
        f"walls {t24:.1f} / {tbw:.1f} / {rf['wall_s']:.1f} s")
    return d, J

def design(acts, gamma_fn, crit_fn, p_dim, label, des=None, outdir=None, work=None, log=print):
    """the two-scan design of prompt §2.4. Returns a dict with the scans, tables T0, T1 (T2), T*, certificate and checks."""
    des = get_design(log) if des is None else des
    outdir = os.path.join(RES, f'design_{NAME}_{label}') if outdir is None else outdir
    work = os.path.join(WORK, NAME, label) if work is None else work
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    t_all = time.perf_counter()
    rec_sod, J_sod, W_sod = des.sod_reference()
    g0 = gamma_fn(J_sod)
    res = dict(input=NAME, label=label, p_dim=p_dim, B=B, dB=DB_CERT, actions=acts.name, n_actions=acts.n,
               sod=dict(EN=rec_sod['EN'], S=rec_sod['S'], logdet=rec_sod['logdet'], crit=crit_fn(J_sod), path=rec_sod.get('path'),
                        J=save_J(J_sod, outdir, 'sod_reference_s24')),
               scans={}, tables={})
    # scan j0: tangent at the constant SoD
    st0, fin0 = run_scan(des, acts, g0, f'{label}_j0', outdir, work, log)
    d0, J0 = eval_scan_table(fin0, acts, f'{label}_j0', g0, g0[0], crit_fn, outdir, log)
    res['scans']['j0'] = dict(gbar=g0[1], gamma1=g0[2], final_run=st0['final_run'], n_runs=len(st0['runs']), min_dual=st0['min_dual'],
                              status=st0['status'], mono_worst=st0.get('mono_worst'), wall_s=st0.get('wall_s'), tangent='constant SoD (Delta*)')
    res['tables']['T0'] = d0
    # scan j1: tangent at J(T0)
    g1t = gamma_fn(J0)
    st1, fin1 = run_scan(des, acts, g1t, f'{label}_j1', outdir, work, log)
    d1, J1 = eval_scan_table(fin1, acts, f'{label}_j1', g1t, g0[0], crit_fn, outdir, log)
    res['scans']['j1'] = dict(gbar=g1t[1], gamma1=g1t[2], final_run=st1['final_run'], n_runs=len(st1['runs']), min_dual=st1['min_dual'],
                              status=st1['status'], mono_worst=st1.get('mono_worst'), wall_s=st1.get('wall_s'), tangent='J(T0)')
    res['tables']['T1'] = d1
    cands = {'T0': (d0, J0), 'T1': (d1, J1)}
    best = max(('T0', 'T1'), key=lambda k: cands[k][0]['crit24'])      # ties -> T0 (first)
    cert = certificate(st1, d0['crit24'], p_dim)
    certs = [cert]
    gap = cert['UBp']-cands[best][0]['crit24']
    log(f"CERT {label} from scan j1 (tangent T0): UB {cert['UB']!r} (min dual {cert['min_dual']!r} at lam {cert['lam_argmin']!r}), "
        f"UB' {cert['UBp']!r} (lam {cert['lam_argmin_p']!r}); T* = {best} crit24 {cands[best][0]['crit24']!r}; gap {gap!r}")
    if gap > GAP_MAX:
        g2t = gamma_fn(cands[best][1])
        st2, fin2 = run_scan(des, acts, g2t, f'{label}_j2', outdir, work, log)
        d2, J2 = eval_scan_table(fin2, acts, f'{label}_j2', g2t, g0[0], crit_fn, outdir, log)
        res['scans']['j2'] = dict(gbar=g2t[1], gamma1=g2t[2], final_run=st2['final_run'], n_runs=len(st2['runs']), min_dual=st2['min_dual'],
                                  status=st2['status'], mono_worst=st2.get('mono_worst'), wall_s=st2.get('wall_s'), tangent=f'J({best})')
        res['tables']['T2'] = d2
        cert2 = certificate(st2, cands[best][0]['crit24'], p_dim); certs.append(cert2)
        cands['T2'] = (d2, J2)
        best = max(cands, key=lambda k: (cands[k][0]['crit24'], -int(k[1])))
    UB = min(c['UB'] for c in certs); UBp = min(c['UBp'] for c in certs)
    db = cands[best][0]
    gap = UBp-db['crit24']
    res['certificates'] = certs
    res['certificate'] = dict(UB=UB, UBp=UBp, gap=gap, gap_le_0_5=bool(gap <= GAP_MAX), from_scan=[c['scan'] for c in certs if c['UBp'] == UBp][0],
                              lam_argmin=[c['lam_argmin'] for c in certs if c['UB'] == UB][0],
                              lam_argmin_p=[c['lam_argmin_p'] for c in certs if c['UBp'] == UBp][0])
    res['T_star'] = best
    res['best'] = {k: db[k] for k in ('label', 'run', 'lam', 'EN24', 'S24', 'logdet24', 'crit24', 'EN120', 'S120', 'logdet120', 'crit120',
                                      'budget120_pass', 'slack120', 'J24', 'J120', 'table')}
    res['best']['shape'] = {k: db['shape'][k] for k in ('edge_quantiles', 'edge_q10', 'edge_median', 'edge_q90', 'edge_mean', 'bins')}
    res['checks'] = dict(
        scans_monotone=all(res['scans'][j]['status'] == 'done' for j in res['scans']),
        forward_backward=all(cands[k][0]['check_fb']['pass_'] for k in cands),
        budget120_best=bool(db['EN120'] <= B),
        crit_le_UBp=all(cands[k][0]['crit24'] <= UBp for k in cands),
        mass_dev24_max=max(cands[k][0]['mass_dev_max24'] for k in cands), mass_dev120_max=max(cands[k][0]['mass_dev_max120'] for k in cands))
    res['wall_s'] = time.perf_counter()-t_all
    log(f"DESIGN {NAME}/{label}: T* = {best} ({db['label']}) crit24 {db['crit24']!r} crit120 {db['crit120']!r} EN120 {db['EN120']!r}; "
        f"UB {UB!r} UB' {UBp!r} gap {gap!r} (<= 0.5: {gap <= GAP_MAX}); checks {res['checks']}")
    jdump(res, os.path.join(outdir, f'design_{label}.json'))
    if not res['checks']['budget120_best']: raise Stop(f"design {label}: sigma/120 E[N] of T* {db['EN120']!r} > B")
    if not res['checks']['crit_le_UBp']: raise Stop(f"design {label}: a table's criterion exceeds UB'")
    if not res['checks']['forward_backward']: raise Stop(f"design {label}: forward/backward check failed")
    return res


def pack_table(path_npy, acts, out_npz, lam, note=''):
    """packed final table (np.savez_compressed), as round11/post11.pack; returns size and round-trip equality."""
    codes = np.load(path_npy)
    np.savez_compressed(out_npz, codes=codes, action_set=np.array(acts.name), am=acts.am, bm=acts.bm, ap=acts.ap, bp=acts.bp,
                        labels_lo=np.array(acts.lab_lo), labels_hi=np.array(acts.lab_hi), fl=e13.FL, cl=e13.CL, z0=e13.Z0, dz=e13.DZ, G=G,
                        input=np.array(NAME), lam=lam,
                        note=np.array("codes[r, i]: action of step k = r + 2 at sigma/24 grid node i of input " + NAME + " (full grid, z_i = z0 + i dz); "
                                      "code = i_lo * N_hi + i_hi; half-widths in grid units L = max(am[i_lo] (i - fl[k-1]) + bm[i_lo], 48), "
                                      "U = max(ap[i_hi] (cl[k-1] - i) + bp[i_hi], 48); window (r_i - L dz, r_i + U dz). " + note))
    chk = np.load(out_npz)
    return dict(path=os.path.relpath(out_npz, HERE), bytes=os.path.getsize(out_npz), roundtrip_equal=bool(np.array_equal(chk['codes'], codes)))
