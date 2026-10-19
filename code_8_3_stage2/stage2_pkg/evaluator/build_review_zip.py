"""Review package in the layout of review_8_5_stage2.zip: one top folder review_pkg/ that mirrors stage2_pkg/ (read-only
dependencies and candidate sources at its root, the evaluator under review_pkg/evaluator/, the spec and the 8.3 report
under review_pkg/reports_CN/), REVIEW_GUIDE.md at the top. Usage (from this directory): python3 build_review_zip.py
-> <repo root>/review_evaluator_8_5.zip"""
import os, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(PKG))
OUT = os.path.join(ROOT, 'review_evaluator_8_5.zip')
pkg_files = ['README.md', 'mc83.py', 'mc85_scan.py', 'mc85_post.py', 'stage2_common.py', 'density_recursion.py', 'wk_d20_10000.npy',
             'amp_nm_P5514.json', 'stage2/final_batch1.json', 'stage2/pool_R2000.json', 'stage2/stage2_final.log',
             'stage3/dense_R2000.json', 'stage3/front_union_R2000.json', 'stage3/stage3_dense.log', 'stage3/unconstrained.json',
             'stage3/stage3_unconstrained.log']
root_files = ['reports_CN/plan_evaluator_CN.md', 'reports_CN/section_8_3_single_CN.md', 'requirements.txt']
skip_ext = ('.stdout', '.nbi', '.nbc', '.pyc')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(HERE, 'REVIEW_GUIDE.md'), 'review_pkg/REVIEW_GUIDE.md')
    for f in pkg_files: zf.write(os.path.join(PKG, f), f'review_pkg/{f}')
    for f in root_files: zf.write(os.path.join(ROOT, f), f'review_pkg/{f}')
    for d, _, fs in os.walk(HERE):
        if '__pycache__' in d: continue
        for f in sorted(fs):
            if f.endswith(skip_ext) or f == '.gitignore': continue
            p = os.path.join(d, f); zf.write(p, 'review_pkg/evaluator/' + os.path.relpath(p, HERE))
    n = len(zf.namelist())
print(f"{OUT}: {n} files, {os.path.getsize(OUT)/1e6:.1f} MB")
