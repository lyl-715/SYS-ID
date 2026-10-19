"""Markdown tables of report_evaluator_CN.md, generated from the stored outputs (results/rec_*.json, results/mc/,
results/analysis.json) so that no number is retyped. Every value printed here is the rounded form of a value that
appears on a line of logs/run_all.log, logs/mc_reference.log or logs/analysis.log (named under each table).
Usage (from this directory): python3 report_tables.py > logs/report_tables.md"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
INPUTS = ['U', 'D', 'E', 'N']; GRIDS = [6, 12, 24]; VERS = ['v0', 'A', 'B']
PN = ['G0', 'Re1', 'Im1', 'Re2', 'Im2', 'Re3', 'Im3', 'Re4', 'Im4']
def rec(v, nm, n, mm=8): return json.load(open(os.path.join(RES, f'rec_{v}_{nm}_n{n}_m{mm}.json')))
MC = json.load(open(os.path.join(RES, 'mc', 'mc_summary.json')))
AN = json.load(open(os.path.join(RES, 'analysis.json')))
def f(v, d): return '—' if v is None else f'{v:.{d}f}'
def e(v): return f'{v:.1e}'

print('### 表 1　版本 × 输入 × 网格（裕量 8Δ；来源 logs/run_all.log 的 `EV` 行）\n')
print('| 版本 | 输入 | n | G | E[N] | E[N]（精确 P，诊断） | S | log det J | Δm_D | m_E | max\\|Σm−1\\| | min m | 耗时 s |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for v in VERS:
    for nm in INPUTS:
        for n in GRIDS:
            r = rec(v, nm, n)
            dm = '0（恒等）' if nm == 'U' else f(r['dmD'], 5); me = '1（恒等）' if nm == 'U' else f(r['mE'], 5)
            enp = '不适用' if v == 'v0' else f(r['EN_P'], 4)
            print(f"| {v} | {nm} | {n} | {r['G']} | {f(r['EN'], 4)} | {enp} | {f(r['S'], 3)} | {f(r['logdet'], 5)} | {dm} | {me} | "
                  f"{e(r['mass_dev_max'])} | {e(r['min_m']) if r['min_m'] != 0 else '0'} | {f(r['runtime_s'], 2)} |")
print('\nv0 的 E[N] 只有一个（原型用精确 P）；它没有 ω 形式的 \\hat P，"用 \\hat P 的 E[N]" 记为不适用，表中 E[N] 列即原型值。\n')

print('### 收敛：p_obs = log2|d1/d2|（来源 logs/analysis.log 的 `CONV` 行）\n')
print('| 版本 | 输入 | log det | E[N] | S | Δm_D | m_E | λ_2..λ_9 范围 |')
print('|---|---|---|---|---|---|---|---|')
CV = {}
for l in open(os.path.join(HERE, 'logs', 'analysis.log')):
    if l.startswith('CONV ') and 'not_applicable' not in l and 'columns' not in l:
        p = l.split(); CV[(p[1], p[2], p[3])] = (float(p[7]), float(p[8]), None if p[9] == 'n/a' else float(p[9]), p[10])
for v in VERS:
    for nm in INPUTS:
        def po(q): c = CV.get((v, nm, q)); return '不适用' if c is None else (c[3] if c[2] is None else f'{c[2]:.3f}')
        lam = '不适用' if nm == 'U' else f"{min(CV[(v, nm, f'lam{i}')][2] for i in range(2, 10)):.3f}–{max(CV[(v, nm, f'lam{i}')][2] for i in range(2, 10)):.3f}"
        print(f"| {v} | {nm} | {po('logdet')} | {po('EN')} | {po('S')} | {po('dmD')} | {po('mE')} | {lam} |")
print('\n全部状态为 ok；无相邻差值变号，无舍入量级的 d2（"阶数不可判定"未出现）；U 的相对量为恒等式，按 §6 标"不适用"。\n')

print('### 稳定性：版本 B，σ/12 对 σ/24（来源 logs/analysis.log 的 `STAB` 行）\n')
print('| 输入 | \\|ΔΔm_D\\| (≤0.002) | \\|Δlog det\\| (≤0.005) | \\|ΔE[N]\\| (≤0.02) | \\|Δm_E\\| (≤0.002) | ‖J24^-1/2 J12 J24^-1/2 − I‖₂ (≤0.005) |')
print('|---|---|---|---|---|---|')
for nm in INPUTS:
    s = AN['stability'][nm]
    dm = '0（恒等）' if nm == 'U' else e(s['dmD']); me = '0（恒等）' if nm == 'U' else e(s['mE'])
    print(f"| {nm} | {dm} | {e(s['logdet'])} | {e(s['EN'])} | {me} | {e(s['Jnorm'])} |")

print('\n### 尾部：版本 B，n = 12，裕量 8Δ → 16Δ（来源 `TAIL` 行）\n')
print('| 输入 | G (8Δ → 16Δ) | ΔE[N] | ΔS | Δlog det | max \\|Δλ_i\\| | J 元素最大相对变化 |')
print('|---|---|---|---|---|---|---|')
for nm in INPUTS:
    t = AN['tail'][nm]; lm = '—' if nm == 'U' else e(max(abs(t[f'lam{i}']) for i in range(1, 10)))
    print(f"| {nm} | {t['G_8']} → {t['G_16']} | {e(t['EN'])} | {e(t['S'])} | {e(t['logdet'])} | {lm} | {e(t['J_max_rel_elem'])} |")
print(f"\n全部量的最大绝对变化 {e(AN['tail_max'])}。\n")

print('### 表 2a　正式比较中的标量项（版本 B，n = 24；MC 种子 20261019，R = 10000，20 批；来源 `ITEM` 行）\n')
print('| # | 输入 | 项 | 评价器 | MC（汇总） | se（20 批） | z |')
print('|---|---|---|---|---|---|---|')
nm_cn = {'EN': 'E[N]', 'S': 'S', 'logdet': 'log det J', 'tr(Dbar)/9': 'tr(D̄)/9', 'dmD_paired_vs_U': 'Δm_D（对 U 配对）'}
for k, it in enumerate(AN['mc_items'], 1):
    if it['item'].startswith('Dbar['): continue
    d = 4 if it['item'] in ('EN', 'S') else (5 if it['item'] in ('logdet', 'dmD_paired_vs_U') else 6)
    rv = f"{it['rec']:.{d}f}" if it['item'] != 'tr(Dbar)/9' else '0'
    print(f"| {k} | {it['input']} | {nm_cn[it['item']]} | {rv} | {it['mc']:.{d}f} | {it['se']:.{d}f} | {it['z']:+.2f} |")

print('\n### 表 2b　白化元素 D̄ = W J̄_MC W − I 的 z（上三角 45 个/输入，W = J_B,24^{-1/2}；评价器一侧为 0；来源 `ITEM` 行）\n')
for nm in INPUTS:
    Z = {it['item']: it['z'] for it in AN['mc_items'] if it['input'] == nm and it['item'].startswith('Dbar[')}
    print(f'**{nm}**\n')
    print('| | ' + ' | '.join(PN) + ' |'); print('|---' * 10 + '|')
    for i in range(9):
        print(f'| {PN[i]} | ' + ' | '.join('' if j < i else f"{Z[f'Dbar[{PN[i]},{PN[j]}]']:+.2f}" for j in range(9)) + ' |')
    zz = np.array(list(Z.values()))
    print(f"\nmax|z| {np.abs(zz).max():.2f}；z>0 {int((zz > 0).sum())} 个，z<0 {int((zz < 0).sum())} 个；|z|>2 {int((np.abs(zz) > 2).sum())} 个。\n")
zs = np.array([it['z'] for it in AN['mc_items']])
print(f"199 项合计：|z| > 4.496 的 {int((np.abs(zs) > 4.496).sum())} 项；max|z| = {np.abs(zs).max():.3f}；|z| > 2 的 {int((np.abs(zs) > 2).sum())} 项；"
      f"|z| > 3 的 {int((np.abs(zs) > 3).sum())} 项。\n")

print('### 表 2c　符号诊断（来源 `SIGN` 行）\n')
print('| 输入 | tr(D̄)/9 | 符号 | z | D̄ 对角元 正/负 |')
print('|---|---|---|---|---|')
for nm in INPUTS:
    sg = AN['sign'][nm]; z = next(it['z'] for it in AN['mc_items'] if it['input'] == nm and it['item'] == 'tr(Dbar)/9')
    print(f"| {nm} | {sg['tr']:+.2e} | {'+' if sg['tr'] > 0 else '−'} | {z:+.2f} | {sg['n_pos_diag']}/{sg['n_neg_diag']} |")

print('\n### 表 2d　只展示、不检验：m_E 与 λ_2 − λ_1（来源 `DISPLAY` 行与 logs/mc_reference.log）\n')
print('| 输入 | m_E 评价器 (B, 24) | m_E MC | 差 | 配对 se | λ_2−λ_1 评价器 | λ_2−λ_1 MC |')
print('|---|---|---|---|---|---|---|')
for nm in ['D', 'E', 'N']:
    r = rec('B', nm, 24)
    print(f"| {nm} | {r['mE']:.5f} | {MC[nm]['mE']:.5f} | {r['mE']-MC[nm]['mE']:+.5f} | {MC[nm]['mE_se_paired']:.5f} | "
          f"{r['lam'][1]-r['lam'][0]:.5f} | {MC[nm]['lam2_minus_lam1']:.5f} |")

print('\n### 非线性偏差诊断（来源 `NONLIN` 行）\n')
print('| 输入 | D = log det J̄ − 批均值 | −D/19 | se(log det) |')
print('|---|---|---|---|')
for nm in INPUTS:
    print(f"| {nm} | {MC[nm]['Dnl']:.2e} | {MC[nm]['minus_Dnl_over_19']:.2e} | {MC[nm]['logdet_se']:.4f} |")

print('\n### 附表　各版本各网格对同一 MC 的 z（E[N]、S、log det、Δm_D；来源 `APPX` 行）\n')
print('| 版本 | n | U: E[N] / S / log det | D: E[N] / S / log det / Δm_D | E: 同左 | N: 同左 |')
print('|---|---|---|---|---|---|')
AP = {}
for l in open(os.path.join(HERE, 'logs', 'analysis.log')):
    if l.startswith('APPX ') and 'columns' not in l:
        p = l.split(); AP[(p[1], int(p[2]), p[3], p[4])] = float(p[8])
for v in VERS:
    for n in GRIDS:
        cells = []
        for nm in INPUTS:
            qs = ['EN', 'S', 'logdet'] + ([] if nm == 'U' else ['dmD'])
            cells.append(' / '.join(f'{AP[(v, n, nm, q)]:+.1f}' for q in qs))
        print(f"| {v} | {n} | " + ' | '.join(cells) + ' |')

print('\n### 耗时（稳态，去掉导入与 JIT；单进程单线程；来源 `RUNTIME` 行）\n')
print('| 版本 | n | 均值 s | 最小 s | 最大 s | U | D | E | N |')
print('|---|---|---|---|---|---|---|---|---|')
for v in VERS:
    for n in GRIDS:
        ts = AN['runtime'][f'{v}_{n}']
        print(f"| {v} | {n} | {np.mean(ts):.2f} | {min(ts):.2f} | {max(ts):.2f} | " + ' | '.join(f'{x:.2f}' for x in ts) + ' |')
