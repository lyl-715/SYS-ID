"""Round 8b, Part C: statistics, paired bootstrap, verdict readings and figures (round 8 v3.1 §2, §3 items 5-6, §5; spec
reports_CN/plan_round8b_CN.md v2). 本场景、固定设计规则下的序贯仿真结果.

The computation is round 8's stats_r8.analyse, imported unchanged and called with this round's parameters (DECISION C-1):
the output directory (theta0.json in, bootstrap draws out) and the bootstrap seed are the module parameters stats_r8.RES and
stats_r8.SEED_BOOT, set before the call. So C1 (accuracy, all R and the "both succeeded" subset), C2 (information reference),
C3 (transmissions, b_1, b_seq, both proportions), C4 (stage 1, failures), C5 (Hessian-FIM, paired d), C6 (paired bootstrap,
B_boot = 2000, idx = rng.integers(0, R, R)) and C7 (verdict readings, cases 1-4 in order) are round 8's code. The round-8 log
lines are printed by stats_r8.log_analysis; only its header label of the analysis is replaced (round 8 calls analysis 'P' its
frozen rule; here 'P' is the primary analysis under the round-8b rule).

Added in round 8b: both over-budget proportions as "n/R observed" with Clopper-Pearson 95 % intervals (two-sided, beta
quantiles), next to the bootstrap intervals, and a flag for a degenerate bootstrap interval (lower = upper). The CP interval
supplements the frozen verdict rule; it never enters it. Figures as round 8 (one analysis).

Usage (from this directory):
  python3 stats_r8b.py                       # official: results/archive_round8b.jsonl, bootstrap seed 50281028
  python3 stats_r8b.py --e3                  # E3 engineering test: round 8's archive, analysis S, bootstrap seed 20281028,
                                             # compared with round8/results/stats_round8.json['S']; and a smoke run on the E2
                                             # output (results/runs/repro/); outputs under results/runs/e3*/
"""
import os, sys, json, time, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R8 = os.path.join(EV, 'round8')
for _p in (R8, HERE):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)
import numpy as np
from scipy.stats import beta as beta_dist
import stats_r8 as s8                                     # round 8, unchanged

RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
SEED_BOOT_OFFICIAL, SEED_BOOT_ROUND8 = 50281028, 20281028
B = s8.B
NAMES = s8.NAMES
LABEL_R8 = 'frozen acceptance rule, primary, binding'
LABEL_R8B = 'round-8b acceptance rule (Newton decrement <= 1e-5 with result.success and Cholesky), primary, binding'


class Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()

def cp_interval(x, n, alpha=0.05):
    """two-sided Clopper-Pearson interval for x successes out of n."""
    lo = 0.0 if x == 0 else float(beta_dist.ppf(alpha/2, x, n-x+1))
    hi = 1.0 if x == n else float(beta_dist.ppf(1-alpha/2, x+1, n-x))
    return [lo, hi]

def load_archive(path, require_full):
    A = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda o: o['r'])
    rs = [o['r'] for o in A]
    assert len(set(rs)) == len(rs), 'duplicate replication'
    if require_full: assert rs == list(range(1, len(A)+1)), 'archive must hold r = 1..R exactly once'
    return A

def run_analysis(A, ref, chk, an, seed, outdir, tag, th0, P):
    """round 8's analyse with (RES, SEED_BOOT) = (outdir, seed); then the CP additions."""
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(theta0=np.asarray(th0).tolist(), note='theta_0 for the post-hoc statistics (truth_r8.THETA0)'),
              open(os.path.join(outdir, 'theta0.json'), 'w'), indent=1)
    s8.RES = outdir; s8.SEED_BOOT = seed
    o, X = s8.analyse(A, ref, chk, an, P)
    assert o['boot']['seed'] == seed and o['boot']['B_boot'] == 2000
    shutil.move(os.path.join(outdir, f'bootstrap_round8_{an}.npz'), os.path.join(outdir, f'bootstrap_{tag}.npz'))
    R = o['R']; cnt = {}
    for k, v, bk in (('EN2_gt_B', X['EN2'] > B, 'p_EN2'), ('N2_gt_B', X['N2'] > B, 'p_N2'), ('N2c_gt_B', X['N2c'] > B, None)):
        n = int(np.sum(v)); row = dict(n=n, R=R, p=n/R, cp95=cp_interval(n, R))
        if bk: row.update(boot95=o['boot'][bk], boot_degenerate=bool(o['boot'][bk][0] == o['boot'][bk][1]))
        cnt[k] = row
    o['tx_counts'] = cnt
    o['fail_counts'] = {k: dict(n=o['fail'][f'n_{k}'], R=R, cp95=cp_interval(o['fail'][f'n_{k}'], R)) for k in ('design', 'stage1', 'joint_seq', 'joint_sod')}
    o['fail_reps'] = dict(stage1=[A[i]['r'] for i in np.flatnonzero(~X['conv1'])], design=[A[i]['r'] for i in np.flatnonzero(X['dfail'])],
                          joint_seq=[A[i]['r'] for i in np.flatnonzero(~X['conv_seq'])], joint_sod=[A[i]['r'] for i in np.flatnonzero(~X['conv_sod'])])
    return o, X

def log_counts(o, P, an):
    t = f"[{an}]"
    for k, lab in (('EN2_gt_B', 'Pr{E[N_2|T,theta_0] > B} (生成的表在条件期望上超预算的比例)'), ('N2_gt_B', 'Pr{N_2 > B} (实际实验超预算的比例)'),
                   ('N2c_gt_B', 'control Pr{N_2,SoD > B} (extra)')):
        c = o['tx_counts'][k]
        P(f"C3 {t} {lab}: {c['n']}/{c['R']} observed; Clopper-Pearson 95 % {c['cp95']}" +
          (f"; bootstrap 95 % {c['boot95']}{' (degenerate)' if c['boot_degenerate'] else ''}" if 'boot95' in c else ''))
    P(f"C4 {t} failure counts with CP 95 %: " + '; '.join(f"{k} {v['n']}/{v['R']} {v['cp95']}" for k, v in o['fail_counts'].items())
      + f"; replications: {o['fail_reps']}")
    a = o['acc_stage1']
    P(f"C4 {t} stage 1 per parameter " + ' | '.join(f"{n}: bias {a['bias'][i]:+.3e} rmse {a['rmse'][i]:.3e}" for i, n in enumerate(NAMES)))
    if o['acc_subset'] is not None:
        for sc in ('seq', 'sod'):
            q = o['acc_subset'][sc]
            P(f"C1 {t} subset {sc}: D {q['D']!r}; -log det C {q['D_C']!r}; max |bias|/rmse {float(np.max(q['bias_over_rmse'])):.3f}; per parameter "
              + ' | '.join(f"{n}: bias {q['bias'][i]:+.3e} rmse {q['rmse'][i]:.3e}" for i, n in enumerate(NAMES)))


def figures(X, o, th0, J1, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C1, C2, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})
    R = len(X['eps1'])
    # eps_1 histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, X['eps1'].max()*1.02, 31)
    ax.hist(X['eps1'], bins=bins, color=C1, edgecolor='white', lw=0.8, alpha=0.9)
    ax.axvline(0.5, color=MUTED, lw=1, ls='--'); ax.text(0.5, ax.get_ylim()[1]*0.95, ' 0.5', color=MUTED, va='top')
    med = float(np.median(X['eps1'])); ax.axvline(med, color=INK, lw=1); ax.text(med, ax.get_ylim()[1]*0.85, f' median {med:.3f}', color=INK, va='top', bbox=dict(fc='white', ec='none', pad=1.5))
    ax.set_xlabel('eps_1 = max_k |h_k^T (theta_1 - theta_0)| / sigma'); ax.set_ylabel('replications')
    ax.set_title(f'Stage-1 estimation error (round 8b, Newton-decrement acceptance), R = {R}', loc='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_eps1_hist.png'), dpi=150); plt.close(fig)
    # E[N_2 | T, theta_0] vs eps_1
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(X['eps1'], X['EN2'], s=16, color=C1, edgecolor='white', lw=0.4)                    # one series: the title names it
    ax.axhline(B, color=MUTED, lw=1, ls='--'); ax.text(ax.get_xlim()[1], B, ' B', color=MUTED, va='center', ha='left')
    ax.set_xlabel('eps_1'); ax.set_ylabel('E[N_2 | T, theta_0]  (sigma/120)')
    c = o['tx_counts']['EN2_gt_B']
    ax.set_title(f'E[N_2 | T, theta_0] of the generated tables vs stage-1 error ({c["n"]}/{c["R"]} above B)', loc='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_EN2_vs_eps1.png'), dpi=150); plt.close(fig)
    # final-error box plots, both schemes, in units of the control's CRB standard deviation
    sdc = np.sqrt(np.diag(np.linalg.inv(2*J1)))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pos = np.arange(9)
    for sc, col, dx in (('seq', C1, -0.18), ('sod', C2, 0.18)):
        e = (X['th_'+sc]-th0)/sdc
        ax.boxplot([e[:, i] for i in range(9)], positions=pos+dx, widths=0.3, patch_artist=True, showfliers=True,
                   flierprops=dict(marker='o', ms=2.5, mfc=col, mec='none', alpha=0.6), medianprops=dict(color=INK, lw=1.2),
                   whiskerprops=dict(color=col, lw=1.2), capprops=dict(color=col, lw=1.2), boxprops=dict(facecolor=col, edgecolor=col, alpha=0.35))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(NAMES); ax.set_ylabel('error / CRB sd of SoD+SoD')
    ax.set_title(f'Final joint-estimate errors, R = {R} (round 8b)', loc='left', color=INK)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=C1, ec=C1, alpha=0.5), plt.Rectangle((0, 0), 1, 1, fc=C2, ec=C2, alpha=0.5)]
    ax.legend(hs, ['sequential (SoD + table)', 'SoD + SoD'], loc='upper right', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_final_error_box.png'), dpi=150); plt.close(fig)


def timing_of(A, t0):
    tm = [q['times'] for q in A]
    return dict(primary_work_max=float(max(t['primary_work_s'] for t in tm)), primary_work_median=float(np.median([t['primary_work_s'] for t in tm])),
                total_max=float(max(t['total_s'] for t in tm)), total_median=float(np.median([t['total_s'] for t in tm])),
                primary_work_sum=float(sum(t['primary_work_s'] for t in tm)), total_sum=float(sum(t['total_s'] for t in tm)),
                design_median=float(np.median([q['analyses']['P']['times']['design'] for q in A])),
                dp_median=float(np.median([q['analyses']['P']['design'].get('t_dp_s') or np.nan for q in A])),
                posthoc_median=float(np.median([q['analyses']['P']['times']['posthoc'] for q in A])),
                partC_s=time.perf_counter()-t0)


def official():
    t0 = time.perf_counter()
    sys.stdout = Tee(os.path.join(LOGS, 'partC_stats.log')); P = print
    A = load_archive(os.path.join(RES, 'archive_round8b.jsonl'), True)
    assert all(o['family'] == 'official' for o in A) and len(A) == 300
    ref = json.load(open(os.path.join(RES, 'sod_reference.json'))); chk = json.load(open(os.path.join(RES, 'partA_checks.json')))
    import truth_r8
    P(f"=== stats_r8b.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; R = {len(A)}; b_1 {ref['b1']!r}; B {B!r}; bootstrap seed "
      f"{SEED_BOOT_OFFICIAL}; A3 all required pass {chk['all_required_pass']}")
    o, X = run_analysis(A, ref, chk, 'P', SEED_BOOT_OFFICIAL, RES, 'round8b', truth_r8.THETA0, P)
    s8.log_analysis(o, lambda m: P(m.replace(LABEL_R8, LABEL_R8B)))
    log_counts(o, P, 'P')
    figures(X, o, truth_r8.THETA0, np.array(ref['J1']), HERE)
    timing = timing_of(A, t0)
    P(f"\nT timing per replication (worker wall, 1 numba thread, 4 workers): primary work median {timing['primary_work_median']:.1f} s max "
      f"{timing['primary_work_max']:.1f} s (design median {timing['design_median']:.1f} s, of which DP {timing['dp_median']:.1f} s; post-hoc "
      f"{timing['posthoc_median']:.1f} s); sum {timing['primary_work_sum']:.0f} s; Part C {timing['partC_s']:.1f} s")
    res = dict(P=o, timing=timing)
    json.dump(s8.jl(res), open(os.path.join(RES, 'stats_round8b.json'), 'w'), indent=1)
    P(f"=== stats_r8b.py done in {time.perf_counter()-t0:.1f} s")


def e3():
    """E3: (i) round 8's archive, analysis S, bootstrap seed 20281028 -> must equal round 8's stats_round8.json['S'] on every
    number compared; (ii) smoke run on the E2 output (4 replications, round-8 seeds, round-8b archive layout)."""
    sys.stdout = Tee(os.path.join(LOGS, 'engineering_stats_E3.log')); P = print
    import truth_r8
    t0 = time.perf_counter()
    A8 = load_archive(os.path.join(R8, 'results', 'archive_round8.jsonl'), True)
    ref8 = json.load(open(os.path.join(R8, 'results', 'sod_reference.json'))); chk8 = json.load(open(os.path.join(R8, 'results', 'partA_checks.json')))
    old = json.load(open(os.path.join(R8, 'results', 'stats_round8.json')))['S']
    out = os.path.join(RES, 'runs', 'e3_round8S')
    o, X = run_analysis(A8, ref8, chk8, 'S', SEED_BOOT_ROUND8, out, 'e3_round8S', truth_r8.THETA0, lambda m: None)
    new = json.loads(json.dumps(s8.jl(o)))
    keys = [('dD',), ('dD_C',), ('dD_subset',), ('subset_n',), ('boot', 'dD'), ('boot', 'b_seq'), ('boot', 'p_EN2'), ('boot', 'p_N2'), ('boot', 'dD_sub'),
            ('boot', 'not_computable_all'), ('boot', 'not_computable_sub'), ('tx', 'b_seq'), ('tx', 'p_EN2_gt_B'), ('tx', 'p_N2_gt_B'),
            ('stage1', 'eps1'), ('stage1', 'frac_eps1_gt_05'), ('fail',), ('info',), ('hfim', 'whitened_norm'), ('hfim', 'max_abs_z'),
            ('paired_d',), ('paired_d_control',), ('verdict', 'case'), ('verdict', 'readings'), ('verdict', 'max_bias_over_rmse'),
            ('acc_all', 'seq', 'M'), ('acc_all', 'sod', 'M'), ('acc_all', 'seq', 'bias'), ('acc_subset', 'seq', 'M')]
    rows = []
    for k in keys:
        a, b = new, old
        for kk in k: a, b = a[kk], b[kk]
        rows.append((k, a == b))
        P(f"E3-i {'/'.join(k)}: {'equal' if a == b else 'DIFFERENT'}" + ('' if isinstance(a, (list, dict)) and len(str(a)) > 120 else f" ({a!r})"))
    ok1 = all(v for _, v in rows)
    P(f"E3-i round 8's analysis S re-computed through stats_r8b (seed {SEED_BOOT_ROUND8}): {len(rows)} quantities, "
      f"{'ALL EQUAL' if ok1 else 'DIFFERENCES'}; CP additions: " + '; '.join(f"{k} {v['n']}/{v['R']} CP {v['cp95']} boot {v.get('boot95')}"
                                                                   for k, v in o['tx_counts'].items()))
    # (ii) smoke run on the E2 output
    rd = os.path.join(RES, 'runs', 'repro'); out2 = os.path.join(RES, 'runs', 'e3_smoke')
    A2 = load_archive(os.path.join(rd, 'archive_round8b.jsonl'), False)
    ref2 = json.load(open(os.path.join(rd, 'sod_reference.json'))); chk2 = json.load(open(os.path.join(RES, 'partA_checks.json')))
    o2, X2 = run_analysis(A2, ref2, chk2, 'P', SEED_BOOT_ROUND8, out2, 'e3_smoke', truth_r8.THETA0, lambda m: None)
    s8.log_analysis(o2, lambda m: P('E3-ii ' + m.replace(LABEL_R8, LABEL_R8B).lstrip('\n')))
    log_counts(o2, lambda m: P('E3-ii ' + m), 'P')
    figures(X2, o2, truth_r8.THETA0, np.array(ref2['J1']), out2)
    P(f"E3-ii smoke run on the E2 output (R = {o2['R']}): completed; figures in {os.path.relpath(out2, HERE)}")
    json.dump(s8.jl(dict(P=o2, timing=timing_of(A2, t0))), open(os.path.join(out2, 'stats_round8b.json'), 'w'), indent=1)
    json.dump(dict(round8S_all_equal=ok1, compared=[['/'.join(k), v] for k, v in rows], smoke_R=o2['R'], seed=SEED_BOOT_ROUND8),
              open(os.path.join(RES, 'engineering_stats_E3.json'), 'w'), indent=1)
    P(f"=== stats_r8b.py --e3 done in {time.perf_counter()-t0:.1f} s")


if __name__ == '__main__':
    e3() if '--e3' in sys.argv else official()
