"""Figures for the model-known paper: the design table T' as an image, the facing-edge histogram (round-4 table vs T'), and the
information per transmission by offset bin. No new computation beyond sigma/24 forward evaluations of existing tables.

Inputs (read-only): T' = round11/addendum/results/table_edge_Sprime.npz (union S u S' action set: 31 edge values, r-side minimum
m sigma, m in {1, 2}); the round-4 table round4/results/table_U_W.npz (old action set); T_1 = round11/results/table_edge_iter.npz
(only for the reproduction check of the binned statistic). Forward evaluation: round11/eval11.evaluate on the full sigma/24 grid with
round 11's frozen statistic eval11.Stats (facing-edge quantiles, offset bins) plus the collector below (pooled offset distribution,
facing-edge histogram with |r - xbar_k| < 1 sigma excluded, eq. (12) recomputation of T''s edges, heatmap samples).

Definitions (round 11): offset = (r_i - xbar_k)/sigma; window (r_i - L h, r_i + U h), h = sigma/24; facing edge = near-side half-width -
|offset| (near side = the side of xbar_k as seen from r: lower edge if r > xbar_k), i.e. how far beyond the predicted output the facing
edge lies; masses m_{k,i} = pre-update reference mass of step k >= 2 on the sigma/24 grid.

Eq. (12) for T' (independent of the table code): action -> (e-, e+, m); lower edge node = min(floor(u_k - 24 e-), i - 24 m),
upper edge node = max(ceil(u_k + 24 e+), i + 24 m), u_k = (xbar_k - z_0)/h; e = inf -> i -/+ 24 m. Asserted equal to the table rows
(i - L, i + U) at every node of every step.

Usage (from anywhere): python3 make_figures.py   -> out/fig_*.pdf/.png, out/captions.md, out/checks.json, out/figures_log.txt
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, math
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); R11 = os.path.dirname(HERE); EV = os.path.dirname(R11)
ADD = os.path.join(R11, 'addendum'); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ADD)
import numpy as np
import sprime as sp                       # round 11b addendum (union action set, TableG)
e11, dp11, dr = sp.e11, sp.dp11, sp.dr
s, K, G, Z, XK, UK, DZ, Z0 = sp.s, sp.K, sp.G, sp.Z, sp.XK, sp.UK, sp.DZ, sp.Z0

STRIDE = 5                 # heatmap: every 5th step (5 ms)
DY = 0.25                  # heatmap: offset sampled every 0.25 sigma (6 grid nodes)
YW = 600.0                 # heatmap samples collected over |offset| <= 600 sigma, cropped to the 99.9 % mass interval afterwards
OFF_BIN = 0.05             # pooled offset histogram resolution (sigma), range +-1300 sigma
FE_RES = 0.001             # facing-edge fine histogram resolution (sigma), range [-100, 100) sigma
MASS_EPS = 1e-12           # white cells: reference mass below this
MASS_COVER = 0.999
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
LOG = []
def out(m=''):
    print(m, flush=True); LOG.append(m)


class Collect:
    """per-step collector (called by eval11.forward with the pre-update mass of step k = p + 1)."""
    def __init__(self, label, acts=None, codes=None, heat=False):
        self.label, self.acts, self.codes, self.heat = label, acts, codes, heat
        self.off_lo = -1300.0; self.noff = int(round(2600.0/OFF_BIN)); self.offh = np.zeros(self.noff)
        self.fe_lo = -100.0; self.nfe = int(round(200.0/FE_RES)); self.feh = np.zeros(self.nfe); self.fe_out = 0.0
        self.mass_tot = 0.0; self.mass_excl = 0.0; self.mass_fe_neg = 0.0
        self.eq_n = 0; self.eq_bad = 0
        if heat:
            self.ygrid = np.arange(-YW, YW+DY/2, DY); self.cols = []
            self.H_lo, self.H_hi, self.H_m, self.H_t = [], [], [], []
        if acts is not None:
            a = acts; E = []; mm = []
            for (Eb, mb) in a.blocks:
                for e in Eb: E.append(e); mm.append(mb//24)
            self.e24 = np.array([np.nan if e == 'inf' else float(Fraction(24)*Fraction(e)) for e in E])
            self.mside = np.array(mm, np.int64)

    def eq12(self, p, L, U):
        """recompute the window edges of every node of step p from the stored codes with eq. (12); compare with the table rows."""
        a = self.acts; cd = np.asarray(self.codes[p-1], np.int64); i = np.arange(G, dtype=np.int64)
        b = np.searchsorted(a.boff, cd, side='right')-1; nb = a.bhi[b]-a.blo[b]; rr = cd-a.boff[b]
        jl = a.blo[b]+rr//nb; jh = a.blo[b]+rr % nb
        el = self.e24[jl]; eh = self.e24[jh]; ml = 24*self.mside[jl]; mh = 24*self.mside[jh]
        with np.errstate(invalid='ignore'):
            lo_x = np.floor(UK[p]-el); hi_x = np.ceil(UK[p]+eh)
        lower = np.where(np.isnan(el), i-ml, np.minimum(np.where(np.isnan(lo_x), 0, lo_x).astype(np.int64), i-ml))
        upper = np.where(np.isnan(eh), i+mh, np.maximum(np.where(np.isnan(hi_x), 0, hi_x).astype(np.int64), i+mh))
        self.eq_n += G; self.eq_bad += int((lower != i-L).sum()+(upper != i+U).sum())

    def __call__(self, p, m, qn, Mm, Mp, lo, hi, c):
        mm = m[lo:hi+1]; off = (Z[lo:hi+1]-XK[p])/s
        L = Mm[lo:hi+1]; U = Mp[lo:hi+1]
        near = np.where(off > 0, L/24.0, U/24.0); fe = near-np.abs(off)
        self.mass_tot += float(mm.sum())
        io = np.floor((off-self.off_lo)/OFF_BIN).astype(np.int64); assert io.min() >= 0 and io.max() < self.noff
        self.offh += np.bincount(io, mm, self.noff)
        inc = np.abs(off) >= 1.0
        self.mass_excl += float(mm[~inc].sum())
        fi = np.floor((fe[inc]-self.fe_lo)/FE_RES).astype(np.int64); wi = mm[inc]
        ok = (fi >= 0) & (fi < self.nfe); self.fe_out += float(wi[~ok].sum())
        self.feh += np.bincount(fi[ok], wi[ok], self.nfe)
        self.mass_fe_neg += float(mm[fe < 0].sum())
        if self.acts is not None: self.eq12(p, Mm, Mp)
        if self.heat and (p-1) % STRIDE == 0:
            ii = np.rint(UK[p]+24.0*self.ygrid).astype(np.int64); valid = (ii >= 0) & (ii < G); ic = np.clip(ii, 0, G-1)
            zi = Z[ic]
            alo = (zi-Mm[ic]*DZ-XK[p])/s; bhi = (zi+Mp[ic]*DZ-XK[p])/s     # window edges relative to xbar_k (sigma), as eval11.forward
            mass = np.where(valid, m[ic], 0.0)
            self.H_lo.append(alo.astype(np.float32)); self.H_hi.append(bhi.astype(np.float32)); self.H_m.append(mass)
            self.H_t.append(float(dr.t[p]))


def weighted_quantiles(h, lo, res, qs):
    cw = np.cumsum(h)/h.sum()
    return [float(lo+res*np.searchsorted(cw, q)) for q in qs]

def shortest_interval(h, lo, res, cover):
    """shortest [a, b] (bin edges) with at least `cover` of the mass, by a two-pointer pass over the histogram."""
    tot = h.sum(); need = cover*tot; cs = np.concatenate([[0.0], np.cumsum(h)])
    best = (np.inf, 0, 0); j = 0
    for i in range(len(h)):
        while j < len(h) and cs[j+1]-cs[i] < need: j += 1
        if j >= len(h): break
        if j-i < best[0]: best = (j-i, i, j)
    _, i, j = best
    return lo+res*i, lo+res*(j+1), float((cs[j+1]-cs[i])/tot)


def evaluate(label, tab, gam0, collect, window=False):
    st = e11.Stats(gam0)
    def cb(*a):
        st(*a); collect(*a)
    t0 = time.perf_counter()
    rec, J, fw = e11.evaluate(tab, stats=cb, window=window)
    sres, ev, w = st.result(); del ev, w
    out(f"FORWARD {label}: E[N] {rec['EN']!r} log det {rec['logdet']!r} S {rec['S']!r}; sum m qhat over bins {sum(b['sends'] for b in sres['bins'])!r}; "
        f"facing-edge quantiles {sres['edge_quantiles']!r}; {time.perf_counter()-t0:.1f} s")
    return rec, sres, st


def main():
    t00 = time.perf_counter()
    out(f"make_figures.py {time.strftime('%Y-%m-%d %H:%M:%S')}; numpy {np.__version__}")
    import matplotlib
    out(f"matplotlib {matplotlib.__version__}")
    sp.warmup(); des = dp11.Design11(out)
    rec_sod, J_sod, W_sod = des.sod_reference(); gam0 = des.gamma_from_J(J_sod)[0]
    r11 = json.load(open(os.path.join(R11, 'results', 'round11.json')))
    r11b = json.load(open(os.path.join(ADD, 'results', 'round11b.json')))
    chk = {}

    # --- T_1: reproduction of the round-11 binned statistic --------------------------------------------------------------
    c1 = np.load(os.path.join(R11, 'results', 'table_edge_iter.npz'))['codes']
    rec1, sh1, _ = evaluate('T1 (edge_j1)', e11.Table11(c1, e11.edge_set(), 'T1'), gam0, Collect('T1'))
    ref1 = r11['tables']['edge_j1']['shape']
    keys = ('mass_share', 'sends', 'send_share', 'info', 'info_per_send')
    chk['T1_bins_equal_round11'] = bool(all(all(x[k] == y[k] for k in keys) for x, y in zip(sh1['bins'], ref1['bins'])))
    out(f"CHECK T1 binned statistic equals round11.json tables.edge_j1.shape.bins ({', '.join(keys)}): {chk['T1_bins_equal_round11']}")
    del c1

    # --- round-4 table (old action set) ------------------------------------------------------------------------------------
    with np.load(os.path.join(EV, 'round4', 'results', 'table_U_W.npz')) as f:
        cw, st0, wd0 = f['codes'], f['start'], f['width']
    told = e11.from_round4(cw, st0, wd0, e11.old_set())
    col_old = Collect('old')
    rec0, sh0, _ = evaluate('round-4 table (old action set)', told, gam0, col_old, window=True)
    ref0 = r11['tables']['old_j0']

    # --- T' ------------------------------------------------------------------------------------------------------------------------
    acts = sp.set_union()
    with np.load(os.path.join(ADD, 'results', 'table_edge_Sprime.npz')) as f: cP = np.array(f['codes'])
    tP = sp.TableG(cP, acts, 'Tprime')
    col_P = Collect('Tprime', acts=acts, codes=cP, heat=True)
    recP, shP, _ = evaluate("T' (S u S')", tP, gam0, col_P)
    refP = r11b['shape']['Tprime']; tpj = r11b['Tprime']
    chk['Tprime_record'] = bool(recP['EN'] == tpj['EN24'] and recP['logdet'] == tpj['I24'])
    chk['Tprime_eq12'] = dict(nodes_checked=col_P.eq_n, mismatches=col_P.eq_bad, pass_=bool(col_P.eq_bad == 0))
    out(f"CHECK T' eq. (12) recomputation vs table rows: {col_P.eq_bad} mismatches over {col_P.eq_n} (node, step) pairs")

    # checks: quantiles of the edge positions (round-11 definition, all cells) against the archived values and the prompt's values
    def qcheck(lab, q, arch, prompt):
        d = dict(quantiles_10_25_50_75_90=q, archived=arch, prompt_10_50_90=prompt,
                 max_abs_diff_archived=max(abs(a-b) for a, b in zip(q, arch)),
                 diff_prompt_10_50_90=[q[0]-prompt[0], q[2]-prompt[1], q[4]-prompt[2]])
        d['pass_archived_0p01'] = bool(d['max_abs_diff_archived'] <= 0.01)
        d['pass_prompt_0p01'] = bool(all(abs(x) <= 0.01 for x in d['diff_prompt_10_50_90']))
        out(f"CHECK quantiles {lab}: {['%.4f' % v for v in q]}; archived max |diff| {d['max_abs_diff_archived']:.2e} "
            f"(<= 0.01: {d['pass_archived_0p01']}); vs prompt 10/50/90 {prompt}: diffs {['%+.4f' % v for v in d['diff_prompt_10_50_90']]} "
            f"(<= 0.01: {d['pass_prompt_0p01']})")
        return d
    chk['quantiles_old'] = qcheck('old', sh0['edge_quantiles'], ref0['shape']['edge_quantiles'], (1.05, 1.9, 2.87))
    chk['quantiles_Tprime'] = qcheck("T'", shP['edge_quantiles'], refP['edge_quantiles'], (1.41, 1.62, 1.81))

    # check: Figure 3 totals = E[N] - 1
    for lab, sh, rec in (('old', sh0, rec0), ('Tprime', shP, recP)):
        tot = sum(b['sends'] for b in sh['bins']); d = abs(tot-(rec['EN']-1.0))
        chk[f'fig3_total_{lab}'] = dict(sum_sends=tot, EN_minus_1=rec['EN']-1.0, abs_diff=d, pass_=bool(d <= 1e-6))
        out(f"CHECK fig 3 totals {lab}: sum over bins of m qhat {tot!r}, E[N] - 1 {rec['EN']-1.0!r}, |diff| {d:.2e} (<= 1e-6: {d <= 1e-6})")
    chk['old_bins_equal_round11'] = bool(all(all(x[k] == y[k] for k in keys) for x, y in zip(sh0['bins'], ref0['shape']['bins'])))
    chk['Tprime_bins_equal_round11b'] = bool(all(all(x[k] == y[k] for k in keys) for x, y in zip(shP['bins'], refP['bins'])))
    out(f"CHECK binned statistic equals the archive: old (round11.json old_j0) {chk['old_bins_equal_round11']}; T' (round11b.json) "
        f"{chk['Tprime_bins_equal_round11b']}")

    # Figure-2 statistics (|offset| >= 1 sigma), heatmap y-range
    f2 = {}
    for lab, col in (('old', col_old), ('Tprime', col_P)):
        h = col.feh; tot_inc = h.sum()+col.fe_out
        x0, x1 = 0.5, 3.5; i0 = int(round((x0-col.fe_lo)/FE_RES)); i1 = int(round((x1-col.fe_lo)/FE_RES))
        q = weighted_quantiles(h, col.fe_lo, FE_RES, (0.1, 0.5, 0.9))
        f2[lab] = dict(excluded_mass_share=col.mass_excl/col.mass_tot, share_below_0p5=float(h[:i0].sum()/tot_inc),
                       share_above_3p5=float((h[i1:].sum()+col.fe_out)/tot_inc), share_fe_negative_all=col.mass_fe_neg/col.mass_tot,
                       quantiles_10_50_90=q, fe_out_of_fine_range=col.fe_out/tot_inc)
        out(f"FIG2 {lab}: excluded mass (|offset| < 1 sigma) {f2[lab]['excluded_mass_share']:.4f}; share < 0.5 sigma {f2[lab]['share_below_0p5']:.4f}, "
            f"> 3.5 sigma {f2[lab]['share_above_3p5']:.4f}; facing edge < 0 (x-bar outside the window, all cells) {f2[lab]['share_fe_negative_all']:.4f}; "
            f"quantiles 10/50/90 (|offset| >= 1, resolution {FE_RES}) {['%.3f' % v for v in q]}")
    a, b, cov = shortest_interval(col_P.offh, col_P.off_lo, OFF_BIN, MASS_COVER)
    out(f"HEATMAP y-range: shortest interval with >= {MASS_COVER} of T''s reference mass (pooled over k >= 2): [{a:.2f}, {b:.2f}] sigma (covers {cov:.5f})")
    chk['heatmap_interval'] = dict(lo=a, hi=b, covered=cov)

    # --- figures -------------------------------------------------------------------------------------------------------------------
    if os.environ.get('FIG_CACHE'):
        import pickle
        for c in (col_P, col_old): c.codes = None; c.acts = None
        pickle.dump(dict(col_P=col_P, yr=(a, b), col_old=col_old, f2=f2, sh0=sh0, shP=shP, rec0=rec0, recP=recP, chk=chk),
                    open(os.environ['FIG_CACHE'], 'wb'))
    figs = make_plots(col_P, (a, b), col_old, f2, sh0, shP, rec0, recP)
    res = dict(checks=chk, fig2=f2, fig3=dict(old=[dict(bin=x['bin'], info_per_send=x['info_per_send'], send_share=x['send_share']) for x in sh0['bins']],
                                              Tprime=[dict(bin=x['bin'], info_per_send=x['info_per_send'], send_share=x['send_share']) for x in shP['bins']],
                                              ref_lines=figs['ref_lines']),
               heatmap=dict(stride_steps=STRIDE, dy_sigma=DY, y_range=[a, b], clip=6.0), EN=dict(old=rec0['EN'], Tprime=recP['EN']),
               quantiles_all_cells=dict(old=sh0['edge_quantiles'], Tprime=shP['edge_quantiles']))
    json.dump(res, open(os.path.join(OUT, 'checks.json'), 'w'), indent=1)
    write_captions(res, f2, sh0, shP, (a, b), figs)
    out(f"DONE in {time.perf_counter()-t00:.1f} s")
    open(os.path.join(OUT, 'figures_log.txt'), 'w').write('\n'.join(LOG)+'\n')


# --- plotting ----------------------------------------------------------------------------------------------------------------------
W_IN = 8.8/2.54
BLUE, GREYL, INK, MUTED, GRID = '#2a78d6', '#8c8b86', '#0b0b0b', '#52514e', '#e4e3df'

def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': '#6b6a66', 'xtick.color': '#3a3936', 'ytick.color': '#3a3936', 'axes.labelcolor': INK,
                         'savefig.dpi': 300})
    return plt

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.02)

def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # blue arm (edge below the predicted output) - neutral grey - red arm (above), linear in sigma over [-6, 6]: stops at 0, +-3, +-6
    stops = [(0.0, '#0d366b'), (0.25, '#3987e5'), (0.5, '#f0efec'), (0.75, '#e34948'), (1.0, '#7a1b1b')]
    cm = LinearSegmentedColormap.from_list('edge_div', stops, N=1024)
    cm.set_over('#a8a7a1'); cm.set_under('#a8a7a1'); cm.set_bad('#ffffff')
    return cm

def make_plots(col, yr, col_old, f2, sh0, shP, rec0, recP):
    plt = style()
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    cm = diverging_cmap(); norm = Normalize(-6, 6)
    figs = {}

    # Figure 1 -------------------------------------------------------------------------------------------------------------------
    yg = col.ygrid; sel = (yg >= np.floor(yr[0])) & (yg <= np.ceil(yr[1]))
    T = np.array(col.H_t); Hm = np.array(col.H_m)[:, sel]
    Hlo = np.ma.array(np.array(col.H_lo)[:, sel], mask=Hm < MASS_EPS); Hhi = np.ma.array(np.array(col.H_hi)[:, sel], mask=Hm < MASS_EPS)
    y = yg[sel]
    te = np.concatenate([T-STRIDE*0.0005, [T[-1]+STRIDE*0.0005]]); ye = np.concatenate([y-DY/2, [y[-1]+DY/2]])
    fig = plt.figure(figsize=(W_IN, 4.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.20, 0.07], hspace=0.20, left=0.165, right=0.98)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    yt = [-300, -30, -4, 0, 4, 30, 300]
    for ax, H, tag in ((axs[0], Hlo, 'a  lower edge'), (axs[1], Hhi, 'b  upper edge')):
        im = ax.pcolormesh(te, ye, H.T, cmap=cm, norm=norm, shading='flat', rasterized=True)
        ax.set_yscale('symlog', linthresh=4, linscale=1.0)
        ax.set_ylim(ye[0], ye[-1]); ax.set_yticks(yt); ax.set_yticklabels([str(v).replace('-', '\u2212') for v in yt])
        ax.set_ylabel('Offset $(r-\\bar x_k)/\\sigma$')
        ax.text(0.0, 1.025, tag, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=INK)
        for yy in (-4, 4): ax.axhline(yy, color='#6b6a66', lw=0.3, ls=(0, (1, 2)))
        ax.axhline(0, color='#3a3936', lw=0.4, ls=(0, (2, 2)))
        ax.set_xlim(0, 5)
    axs[0].tick_params(labelbottom=False); axs[1].set_xlabel('Time $t$ (s)')
    cax = fig.add_subplot(gs[3])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both', ticks=[-6, -4, -2, 0, 2, 4, 6])
    cb.set_label('Edge relative to predicted output, $(\\mathrm{edge}-\\bar x_k)/\\sigma$', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    leg = [Patch(facecolor='#a8a7a1', edgecolor='none', label='beyond $\\pm6\\sigma$ (far edge)'),
           Patch(facecolor='#ffffff', edgecolor='#8c8b86', lw=0.5, label='no reference mass')]
    fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False, fontsize=7.5, handlelength=1.2)
    save(fig, 'fig_table_edge'); plt.close(fig)
    figs['fig1'] = dict(y=[float(y[0]), float(y[-1])], n_t=len(T), n_y=int(sel.sum()))

    # Figure 2 -------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(W_IN, 2.3))
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.19, top=0.97)
    edges = np.round(np.arange(0.5, 3.5+1e-9, 0.05), 10)
    for lab, c in (('old', col_old), ('Tprime', col)):
        h = c.feh; tot = h.sum()+c.fe_out
        idx = np.round((edges-c.fe_lo)/FE_RES).astype(np.int64)
        vals = np.array([h[idx[j]:idx[j+1]].sum() for j in range(len(edges)-1)])/tot
        if lab == 'old':
            ax.stairs(vals, edges, color='#3a3936', lw=1.1, label='Round-4 table', zorder=3)
        else:
            ax.stairs(vals, edges, fill=True, color=BLUE, alpha=0.6, lw=0, label="$T'$", zorder=2)
        figs[f'fig2_{lab}_max'] = float(vals.max())
    ax.axvline(1.9, color=INK, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
    ymax = max(figs['fig2_old_max'], figs['fig2_Tprime_max'])*1.08
    ax.set_ylim(0, ymax)
    ax.text(1.95, ymax*0.55, '1.9$\\sigma$\n(Remark 1)', fontsize=7.5, va='center', ha='left', color=INK)
    ax.set_xlim(0.5, 3.5); ax.set_xlabel('Facing edge beyond predicted output (σ)'); ax.set_ylabel('Mass-weighted share\nper 0.05σ bin')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.0)); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    save(fig, 'fig_edge_offset_hist'); plt.close(fig)

    # Figure 3 -------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(W_IN, 2.5))
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.17, top=0.80)
    labs = ['[0, 2)', '[2, 5)', '[5, 10)', '[10, 20)', '[20, 40)', '≥ 40']
    x = np.arange(6); wb = 0.38
    v0 = [b_['info_per_send'] for b_ in sh0['bins']]; vP = [b_['info_per_send'] for b_ in shP['bins']]
    s0 = [b_['send_share'] for b_ in sh0['bins']]; sP = [b_['send_share'] for b_ in shP['bins']]
    hb0 = ax.bar(x-wb/2-0.01, v0, wb, color='#a8a7a1', label='Round-4 table', zorder=3)
    hbP = ax.bar(x+wb/2+0.01, vP, wb, color=BLUE, label="$T'$", zorder=3)
    ref_sod = 1.4725209162672948; ref_one = 5.877546151510727
    def ly(v): return ref_sod+0.06 if (v+0.08 < ref_sod+0.02 and v+0.08+0.28 > ref_sod) else v+0.08   # keep labels off the SoD line
    for xi, v, sh_ in zip(x, v0, s0): ax.text(xi-wb/2-0.01, ly(v), f'{100*sh_:.0f}%', ha='center', va='bottom', fontsize=7, color=INK)
    for xi, v, sh_ in zip(x, vP, sP): ax.text(xi+wb/2+0.01, ly(v), f'{100*sh_:.0f}%', ha='center', va='bottom', fontsize=7, color=INK)
    hl1 = ax.axhline(ref_sod, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=4, label='constant SoD (1.47)')
    hl2 = ax.axhline(ref_one, color=INK, lw=0.7, ls=(0, (1, 1.2)), zorder=4, label='one-sided edge at 2σ (5.88)')
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlim(-0.6, 5.6); ax.set_ylim(0, 6.3)
    ax.set_xlabel('Offset bin $|r-\\bar x_k|/\\sigma$'); ax.set_ylabel('Information per transmission')
    ax.legend(handles=[hb0, hbP, hl1, hl2], frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=0.8, handlelength=1.8)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    save(fig, 'fig_edge_profile'); plt.close(fig)
    figs['ref_lines'] = dict(constant_SoD=ref_sod, one_sided_2sigma=ref_one,
                             sources=dict(constant_SoD='(S - 1)/(E[N] - 1) of round4/results/mc.json["U/SoD"] (S_eval, EN_eval)',
                                          one_sided_2sigma='c_one/q at alpha = -2: q = Phi(-2), c = q - alpha phi(alpha) + phi(alpha)^2/(1 - q)'))
    return figs


def write_captions(res, f2, sh0, shP, yr, figs):
    q0 = res['quantiles_all_cells']['old']; qP = res['quantiles_all_cells']['Tprime']
    o, p_ = f2['old'], f2['Tprime']
    last0, lastP = sh0['bins'][-1], shP['bins'][-1]
    M = lambda v, f='.0f': format(v, f).replace('-', '−')
    lines = ["# Figure captions (drafts)", "",
             "All quantities from the σ/24 discrete model (forward evaluation of the full σ/24 design grid, round11/eval11.py) at the "
             "nominal parameter. T′ = round11/addendum/results/table_edge_Sprime.npz; round-4 table = round4/results/table_U_W.npz.", "",
             "## Figure 1 — `fig_table_edge`", "",
             f"Window edges of the design table T′ on the σ/24 grid: (a) lower edge, (b) upper edge, each relative to the predicted output, "
             f"(edge − x̄ₖ)/σ, shown against time t and the offset of the reference value from the predicted output, (r − x̄ₖ)/σ, over the "
             f"shortest offset interval that holds 99.9 % of the reference mass ([{M(yr[0])}, {M(yr[1])}]σ, symmetric-log axis, linear within ±4σ; "
             f"image sampled every 5 ms and every 0.25σ). Colours run from blue (edge below the output) through grey-white (edge at the output) to red (above), clipped at ±6σ; "
             f"edges more than 6σ from the output — the far edge r ∓ mσ when r is far — are mid-grey, and cells without reference mass are white. "
             f"For r above the output the lower edge is the facing edge and forms one blue band (panel a, upper half), for r below the output the "
             f"upper edge forms one red band (panel b, lower half), and within 2σ of the output both edges are coloured. "
             f"The facing edge stays about 1.6σ beyond the output at every offset; there is no cap "
             f"(10/50/90 % of the facing-edge position {qP[0]:.2f}/{qP[2]:.2f}/{qP[4]:.2f}σ, mass-weighted over all cells).", "",
             "## Figure 2 — `fig_edge_offset_hist`", "",
             f"Position of the facing edge beyond the predicted output (σ; the edge on the side of x̄ₖ, measured outward from x̄ₖ) for the round-4 table "
             f"(grey line) and T′ (blue), weighted by the reference mass m_k,i over steps k ≥ 2 on the σ/24 grid and normalised to one per table, "
             f"in bins of 0.05σ. Cells with |r − x̄ₖ| < 1σ have no well-defined facing side and are excluded ({100*o['excluded_mass_share']:.1f} % of the "
             f"mass for the round-4 table, {100*p_['excluded_mass_share']:.1f} % for T′); outside the plotted 0.5–3.5σ lie "
             f"{100*(o['share_below_0p5']+o['share_above_3p5']):.1f} % (round-4) and {100*(p_['share_below_0p5']+p_['share_above_3p5']):.1f} % (T′) of the "
             f"remaining mass. 10/50/90 % quantiles: round-4 table {o['quantiles_10_50_90'][0]:.2f}/{o['quantiles_10_50_90'][1]:.2f}/{o['quantiles_10_50_90'][2]:.2f}σ, "
             f"T′ {p_['quantiles_10_50_90'][0]:.2f}/{p_['quantiles_10_50_90'][1]:.2f}/{p_['quantiles_10_50_90'][2]:.2f}σ (over all cells, without the "
             f"exclusion: {q0[0]:.2f}/{q0[2]:.2f}/{q0[4]:.2f}σ and {qP[0]:.2f}/{qP[2]:.2f}/{qP[4]:.2f}σ). "
             f"Look at the spread: the round-4 table's facing edge spreads over about ±1σ, as expected from half-widths that come in 2σ steps around r, "
             f"whereas T′ concentrates the facing edge in 1.4–1.8σ; the dotted line marks 1.9σ, the optimum of Remark 1 at a 6.3 % send rate.", "",
             "## Figure 3 — `fig_edge_profile`", "",
             f"Information bought per transmission, (Σ m·c)/(Σ m·q̂) over the cells of each offset bin |r − x̄ₖ|/σ (steps k ≥ 2, σ/24 grid), "
             f"for the round-4 table (old action set, grey) and T′ (blue); the number above each bar is the share of the table's expected transmissions "
             f"E[N] − 1 that fall in that bin. Reference lines: constant SoD (dashed, 1.47, the ratio over the whole experiment) and a one-sided "
             f"window with its edge 2σ beyond the output (dotted, 5.88, Remark 1). Look at the last bin: under the round-4 table "
             f"{100*last0['send_share']:.0f} % of the transmissions occur at offsets of 40σ or more and buy {last0['info_per_send']:.2f} units each; "
             f"there the 40σ cap on the half-width leaves the output outside the window (mean facing edge {M(last0['edge'], '.2f')}σ). T′ spends {100*lastP['send_share']:.0f} % of its "
             f"transmissions there at {lastP['info_per_send']:.2f} units each, close to the other bins.", "",
             "## Checks", "",
             f"- Facing-edge quantiles (10/25/50/75/90 %, all cells, round-11 definition) from the same edge arrays as Figure 1: T′ "
             f"{', '.join('%.4f' % v for v in qP)} (archived round11b.json: max |diff| {res['checks']['quantiles_Tprime']['max_abs_diff_archived']:.1e}); "
             f"round-4 table {', '.join('%.4f' % v for v in q0)} (archived round11.json old_j0: max |diff| {res['checks']['quantiles_old']['max_abs_diff_archived']:.1e}).",
             f"- Against the values given in the request: T′ 1.41/1.62/1.81 → diffs "
             f"{', '.join('%+.3f' % v for v in res['checks']['quantiles_Tprime']['diff_prompt_10_50_90'])} (within 0.01: "
             f"{res['checks']['quantiles_Tprime']['pass_prompt_0p01']}); round-4 table 1.05/1.9/2.87 → diffs "
             f"{', '.join('%+.3f' % v for v in res['checks']['quantiles_old']['diff_prompt_10_50_90'])} (within 0.01: "
             f"{res['checks']['quantiles_old']['pass_prompt_0p01']}).",
             f"- T′ edges recomputed from the stored actions with eq. (12): {res['checks']['Tprime_eq12']['mismatches']} mismatches over "
             f"{res['checks']['Tprime_eq12']['nodes_checked']} (node, step) pairs.",
             f"- Figure 3 totals Σ m·q̂ = E[N] − 1: round-4 |diff| {res['checks']['fig3_total_old']['abs_diff']:.1e}, T′ |diff| "
             f"{res['checks']['fig3_total_Tprime']['abs_diff']:.1e} (≤ 1e-6).",
             f"- Binned statistic reproduces the archive bit for bit: T₁ (round11.json edge_j1) {res['checks']['T1_bins_equal_round11']}, "
             f"round-4 table (old_j0) {res['checks']['old_bins_equal_round11']}, T′ (round11b.json) {res['checks']['Tprime_bins_equal_round11b']}.", ""]
    open(os.path.join(OUT, 'captions.md'), 'w').write('\n'.join(lines))


def replot(path):
    import pickle
    d = pickle.load(open(path, 'rb'))
    make_plots(d['col_P'], d['yr'], d['col_old'], d['f2'], d['sh0'], d['shP'], d['rec0'], d['recP'])


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--replot': replot(sys.argv[2])
    else: main()
