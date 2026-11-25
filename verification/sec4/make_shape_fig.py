#!/usr/bin/env python3
"""make_shape_fig.py OUT.png NPZ [NPZ ...]

One panel per check_shape.py output (.npz): weighted distribution of the candidate value e that sets the
facing edge of the round-3 table (status 'e', all steps k >= 2), as a fraction of the total weight; the
dashed line is the one-parameter rule's e at the same MIN. Prints the plotted numbers as a text table."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, INK, INK2, GRID = "#257de0", "#191919", "#666666", "#e6e6e6"
out, files = sys.argv[1], sys.argv[2:]
fig, axes = plt.subplots(len(files), 1, figsize=(6.4, 2.3*len(files)), sharex=True, squeeze=False)
for ax, f in zip(axes[:, 0], files):
    z = np.load(f)
    E = z["E"]; F = z["face_T"]                      # (side, near, status, slot)
    W = F.sum()
    h = F[:2, :, 0, :len(E)].sum(axis=(0, 1))/W      # status 'e', all sides and distances
    other = 1 - h.sum()
    print(f"\n{f}: MIN = {float(z['MIN']):g}, rule e = {float(z['e_rule']):.4f}; facing edge set by e: {h.sum():.4f}, "
          f"otherwise (tie or MIN): {max(other, 0.0):.4f}")
    print("     e   fraction of weight")
    for e, v in zip(E, h):
        if v > 5e-5: print(f"  {e:5.2f}   {v:.4f}")
    ax.bar(E, h, width=0.04, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(float(z["e_rule"]), color=INK, ls="--", lw=1.0, zorder=4)
    ax.text(float(z["e_rule"]) + 0.03, 0.97*max(h.max(), 1e-9), f"one-parameter rule, e = {float(z['e_rule']):.4f}",
            color=INK2, fontsize=8, va="top")
    mean = float((E*h).sum()/h.sum())
    setname = {30: "full", 7: "small"}.get(len(E), f"{len(E)} values")
    ax.set_title(f"σ/{int(z['FAC'])}, {setname} set, MIN = {float(z['MIN']):g} σ: round-3 table, facing edge "
                 f"(weighted mean e = {mean:.3f})", fontsize=9, color=INK, loc="left")
    ax.set_ylabel("fraction of weight", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
axes[-1, 0].set_xlabel("candidate e that sets the facing edge (σ beyond the predicted output x_k)", fontsize=8, color=INK2)
axes[-1, 0].set_xlim(0.3, 3.2)
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f"\nwrote {out}")
