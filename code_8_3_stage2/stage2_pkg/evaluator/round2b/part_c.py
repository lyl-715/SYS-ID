"""Round 2b, Part C: reading the result (spec reports_CN/plan_round2b_CN.md v3.2, frozen, §5).

C1  primary: S~_1D (no acceleration constraint) - S~_acc (2-D, acceleration constraint), both outer-minimised on the same discrete model.
C2  rho = (S~_acc - S_found)/S_found. Before classifying: band = 4684 eps_lambda / S_found; if [rho - band, rho + band] meets a branch
    boundary (0, 0.10, 0.35), eps_lambda of the 2-D outer minimisation is divided by 4 and the 2-D golden section is re-run at the finer
    eps (cached lambdas replay), at most 3 times; still crossing -> 未定. Branches, 未定 first: rho < 0 -> check model consistency and numerical
    error first (not counted as support); 0 <= rho < 10 % -> supports; 10 % <= rho <= 35 % -> 未定; rho > 35 % -> does not support.
    If the refinement changes the finally selected lambda*, the policy memmap, forward S and E[N], the T3 statistics and the plots are
    regenerated for it (mdp2d_estimate.stage_policy). Grid error is not included in the band.
Writes logs/part_c.log, results/part_c.json. Usage (from this directory): nohup python3 part_c.py --threads 4 > logs/part_c.stdout 2>&1 &"""
import os, sys, json, time, argparse
import numpy as np
import grid30 as g
from grid30 import dr, LOGS, RES, HERE
import mdp2d_estimate as m2
from numba import set_num_threads, get_num_threads

SLOPE = 4684.0; BOUNDS = (0.0, 0.10, 0.35); EPS0 = 1e-2; MAX_REFINE = 3


def crossing(rho, band): return [b for b in BOUNDS if rho-band <= b <= rho+band]

def branch(rho):
    if rho < 0: return 'rho < 0: 先核查两边的模型口径和数值误差，不自动判为支持'
    if rho < 0.10: return '0 <= rho < 10%: 支持假设'
    if rho <= 0.35: return '10% <= rho <= 35%: 未定'
    return 'rho > 35%: 未支持'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=None); args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    sys.stdout = dr.Tee(os.path.join(LOGS, 'part_c.log'), 'a')
    def log(m): print(m, flush=True)
    T0 = time.time()
    log(f"\n=== part_c.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; numba threads {get_num_threads()}; load {g.loadavg()}")
    r1 = json.load(open(os.path.join(RES, 'mdp1d_result.json')))
    r2 = json.load(open(os.path.join(RES, f'mdp2d_outer_eps{EPS0:.6g}.json')))
    pol = json.load(open(os.path.join(RES, 'mdp2d_policy.json')))
    sf = json.load(open(os.path.join(HERE, 'search_maxS', 'S_found.json')))['S_found']
    S1, Sacc, Sfound = r1['S_dual'], r2['S_acc'], sf['S']
    log(f"C0 inputs: S~_1D = {S1!r} (lambda* {r1['lam_star_repr']}, eps {r1['eps_lambda']}; results/mdp1d_result.json); S~_acc = {Sacc!r} (lambda* "
        f"{r2['lam_star_repr']}, eps {r2['eps_lambda']}; results/mdp2d_outer_eps0.01.json); S_found = {Sfound!r} ({sf['tag']}; search_maxS/S_found.json); "
        f"policy memmap built at lambda {pol['lam_repr']}")
    d = S1-Sacc
    log(f"C1 same discrete model: S~_1D - S~_acc = {d!r} ({d/S1*100!r} % of S~_1D); 加入加速度限制后数值下降 {d:.3f}（两个对偶估算值之差，均为估算值）")
    model = None; eps = EPS0; steps = []; lam_s = r2['lam_star']; res2 = r2
    while True:
        rho = (res2['S_acc']-Sfound)/Sfound; band = SLOPE*eps/Sfound; cr = crossing(rho, band)
        log(f"C2 eps_lambda {eps!r}: S~_acc {res2['S_acc']!r} (lambda* {res2['lam_star_repr']}); rho = {rho!r} ({rho*100:.4f} %); band = 4684 eps / S_found = "
            f"{band!r} ({band*100:.4f} percentage points); [rho - band, rho + band] = [{rho-band!r}, {rho+band!r}]; boundaries met: {cr if cr else 'none'}")
        steps.append(dict(eps=eps, S_acc=res2['S_acc'], lam_star=res2['lam_star'], rho=rho, band=band, crossing=cr, n_evals=res2['n_evals']))
        if not cr: break
        if len(steps) > MAX_REFINE:
            log(f"C2 still crossing after {MAX_REFINE} refinements (eps_lambda = {eps!r}): 未定"); break
        eps = eps/4
        log(f"C2 refinement {len(steps)}: eps_lambda -> {eps!r} (÷4); re-running the 2-D outer minimisation (golden section from [0, 20], cached lambdas replay)")
        if model is None: model = g.mdp.model()
        res2 = m2.stage_outer(model, eps, log)
    final = steps[-1]
    label = '未定（细化 3 次后仍跨界）' if final['crossing'] else branch(final['rho'])
    log(f"C2 BRANCH: {label}; final eps_lambda (2-D) {final['eps']!r}; refinement steps taken {len(steps)-1}; final rho {final['rho']!r}, band {final['band']!r}; "
        f"grid error not included in the band")
    regen = res2['lam_star'] != pol['lam']
    log(f"C2 final lambda* (2-D) {res2['lam_star_repr']} vs lambda of the stored policy {pol['lam_repr']}: changed: {regen}")
    if regen:
        if model is None: model = g.mdp.model()
        log("C2 regenerating the policy memmap, forward S and E[N], T3 statistics and plots for the final lambda*")
        pol = m2.stage_policy(model, res2['lam_star'], log)
    t5 = res2['S_acc']-Sfound
    log(f"C3 T5 check: S~_acc - S_found = {t5!r} (prediction [100, 250]: {100 <= t5 <= 250})")
    out = dict(S_1D=S1, eps_1D=r1['eps_lambda'], lam_1D=r1['lam_star'], S_acc=res2['S_acc'], eps_2D=final['eps'], lam_2D=res2['lam_star'],
               S_found=Sfound, S_found_tag=sf['tag'], C1_diff=d, steps=steps, branch=label, policy_regenerated=bool(regen),
               policy=dict(lam=pol['lam'], S=pol['S'], EN=pol['EN'], EN_le_B=pol['EN_le_B'], T3=pol['T3']), T5_diff=t5)
    json.dump(out, open(os.path.join(RES, 'part_c.json'), 'w'), indent=1, ensure_ascii=False)
    log(f"=== part_c.py finished; wall {time.time()-T0:.1f} s")
