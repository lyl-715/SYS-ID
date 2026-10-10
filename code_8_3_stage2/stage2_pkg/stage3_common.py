"""§8.5 stage 3: shared setup, exec'd by stage3_*.py.  Run from code_8_3_stage2/stage2_pkg/:
    exec(open('stage3_common.py').read())
Reuses, without modification:
  - stage2_common.py: mc83 setup, qual (mc85_post.py), sod_batch + check_trigger (bitwise proof against mc83.sod),
    evaluate_crn, summarize_s2, pareto_front, eval_many, diff_se;
  - the Nelder-Mead block of stage2_local.py (x_to_input, input_to_x, eval_s, pen, pen_rec, nm_run), exec'd verbatim from its
    source. It reads the globals ZS, REFS, ldCs, RS, MU_B, MU_E, H_Z, H_CHI, MAXFEV, set here (stage-2 values except MAXFEV=200).
Outputs go to stage3/; stage2/ is only read (OUT is redirected before anything is written)."""
exec(open('stage2_common.py').read())
from scipy.optimize import minimize
OUT = 'stage3'; os.makedirs(OUT, exist_ok=True)
S2 = 'stage2'                                     # read-only stage-2 outputs
R200, SEED200 = 200, 20261010                     # screening draws (stage 1 / LHS)
RS, SEEDS = 500, 20261010                         # search draws (stage-2 convention: seed 20261010 extended to R=500)
R2, SEED2 = 2000, 20261012                        # refinement / ranking draws
MU_B, MU_E, H_Z, H_CHI, MAXFEV = 1.0, 20.0, 0.25, 0.25, 200
TARGETS = [('none', None), ('0.3', 0.3), ('0.5', 0.5), ('0.7', 0.7)]
REGIONS = [('A', lambda m: m < 0.1), ('B', lambda m: 0.1 <= m < 0.3), ('C', lambda m: m >= 0.3)]
DEG = np.pi/180
_src_local = open('stage2_local.py').read()
exec(_src_local[_src_local.index('def x_to_input(x):'):_src_local.index('# --- starting points:')])

def setup_search():
    """search draws and their uniform reference, exactly as in stage2_local.py"""
    global ZS, REFS, ldCs
    ZS = np.random.default_rng(SEEDS).standard_normal((RS, K))
    REFS = evaluate_crn(a_u, chi_u, ZS); ldCs = np.linalg.slogdet(REFS['J'])[1]
    print(f"search reference (uniform, R={RS}, seed {SEEDS}): B_u={REFS['cnt'].mean():.3f}, log det J_C={ldCs:.4f}")

def stage2_pool():
    P = json.load(open(f'{S2}/pool_R2000.json')); return P, {r['id']: r for r in P['records']}

def key_of(p, chi): return tuple(np.round(np.asarray(p, float), 7))+tuple(np.round(np.mod(np.asarray(chi, float), TWO_PI), 7))
