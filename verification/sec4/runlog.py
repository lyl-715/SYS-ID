#!/usr/bin/env python3
"""runlog.py OUT CMD...  : run CMD with stdout -> OUT, stderr -> OUT.err; write OUT.meta with
start/end time, wall time, exit code, peak RSS of the child. Threads pinned to 1."""
import sys, subprocess, time, resource, os, datetime
out = sys.argv[1]; cmd = sys.argv[2:]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
t0 = time.time(); st = datetime.datetime.now().isoformat(timespec='seconds')
with open(out, "w") as fo, open(out + ".err", "w") as fe:
    rc = subprocess.call(cmd, stdout=fo, stderr=fe, env=env)
t1 = time.time()
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(out + ".meta", "w") as fm:
    fm.write(f"cmd: {' '.join(cmd)}\nstart: {st}\nend: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
             f"wall_s: {t1-t0:.1f}\ncpu_user_s: {ru.ru_utime:.1f}\ncpu_sys_s: {ru.ru_stime:.1f}\n"
             f"peak_rss_MB: {ru.ru_maxrss/1024:.1f}\nexit_code: {rc}\nthreads: OMP/OPENBLAS/MKL_NUM_THREADS=1\n")
