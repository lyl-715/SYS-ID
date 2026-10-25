"""Round 8: a designed stage-2 window table as an object (lookup by the actual reference value) and its packed copy.

The frozen continuous policy (round 7 §1.1): at step k the half-widths are constant within the cell of width h = sigma/24
centred at each coarse node of THIS table's grid (origin z0, generated from x_hat by the design); the window is centred at
the actual reference r: (r - M^- h, r + M^+ h). Cell of r: i = floor((r - z0)/h + 1/2) (round 7 evaluator_refine.cell_index,
without its grid-node boundary assertion: a real reference value lies on a cell boundary with probability 0; the floor rule
gives the upper cell); a cell outside the step's stored range [start, start + width - 1] takes the range's boundary node
(round 5 / round 7 rule). Codes -> half-widths by round 4's code_to_m (verbatim, r8k_ew).

Imports: numpy, bz2, io, declared, r8k_ew -- nothing that holds theta_0 (used by the estimator and the design side).

Packed copy: bz2 of an uncompressed npz holding the two uint8 planes i_- = code // 20 and i_+ = code % 20 (pad 255), start,
width, z0, h, G, lambda, gbar; unpack() rebuilds the uint16 codes exactly (pad 65535); round trip checked on every write.
"""
import os, sys, io, bz2
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np
import declared as dk
import r8k_ew

PAD = 0xFFFF


class SeqTable:
    def __init__(self, codes, start, width, z0, h, tag=''):
        self.codes = codes; self.start = np.asarray(start, np.int64); self.width = np.asarray(width, np.int64)
        self.z0 = float(z0); self.h = float(h); self.tag = tag
        self.X = r8k_ew.X24

    @classmethod
    def from_design(cls, d, tag=''):
        return cls(d['codes'], d['start'], d['width'], d['z0'], d['h'], tag)

    def cell(self, r):
        return int(np.floor((r-self.z0)/self.h+0.5))

    def lookup(self, k, r):
        """(M^-, M^+) in grid units at step k (1-based, k >= 2) for the actual reference value r."""
        row = k-2; s0 = self.start[row]; w = self.width[row]
        i = min(max(self.cell(r), s0), s0+w-1)
        c = int(self.codes[row, i-s0])
        return 48*(c//20+1), 48*(c % 20+1)

    def lookup_vec(self, k, r):
        """vectorised lookup for arrays of steps k (1-based, >= 2) and reference values r; returns int64 arrays (M^-, M^+)."""
        k = np.asarray(k, np.int64); r = np.asarray(r, float)
        row = k-2; s0 = self.start[row]; w = self.width[row]
        i = np.floor((r-self.z0)/self.h+0.5).astype(np.int64)
        i = np.minimum(np.maximum(i, s0), s0+w-1)
        c = np.asarray(self.codes[row, i-s0], np.int64)
        return r8k_ew.code_to_m(c)


def pack(d, path, extra=None):
    """write the packed copy of design d (dict with codes, start, width, z0, h, G, lam, gbar) to path; returns (bytes, sha256
    of the codes array)."""
    import hashlib
    c = np.asarray(d['codes'])
    pad = c == PAD
    lo = np.where(pad, 255, c//20).astype(np.uint8); hi = np.where(pad, 255, c % 20).astype(np.uint8)
    buf = io.BytesIO()
    meta = dict(z0=d['z0'], h=d['h'], G=d['G'], lam=d['lam'], gbar=d['gbar'])
    if extra: meta.update(extra)
    np.savez(buf, lo=lo, hi=hi, start=np.asarray(d['start'], np.int64), width=np.asarray(d['width'], np.int64),
             **{k: np.asarray(v) for k, v in meta.items()})
    data = bz2.compress(buf.getvalue(), 9)
    tmp = path+'.tmp'
    with open(tmp, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    back = unpack(path)
    assert np.array_equal(back['codes'], c) and np.array_equal(back['start'], d['start']) and back['z0'] == d['z0'], path
    return len(data), hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()

def unpack(path):
    with open(path, 'rb') as f: raw = bz2.decompress(f.read())
    with np.load(io.BytesIO(raw)) as z:
        lo, hi = z['lo'].astype(np.uint16), z['hi'].astype(np.uint16)
        codes = np.where(lo == 255, np.uint16(PAD), lo*20+hi).astype(np.uint16)
        out = dict(codes=codes, start=z['start'], width=z['width'])
        for k in z.files:
            if k not in ('lo', 'hi', 'start', 'width'): out[k] = z[k].item()
    return out
