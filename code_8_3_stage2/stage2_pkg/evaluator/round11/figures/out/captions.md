# Figure captions (drafts)

All quantities from the σ/24 discrete model (forward evaluation of the full σ/24 design grid, round11/eval11.py) at the nominal parameter. T′ = round11/addendum/results/table_edge_Sprime.npz; round-4 table = round4/results/table_U_W.npz.

## Figure 1 — `fig_table_edge`

Window edges of the design table T′ on the σ/24 grid: (a) lower edge, (b) upper edge, each relative to the predicted output, (edge − x̄ₖ)/σ, shown against time t and the offset of the reference value from the predicted output, (r − x̄ₖ)/σ, over the shortest offset interval that holds 99.9 % of the reference mass ([−326, 459]σ, symmetric-log axis, linear within ±4σ; image sampled every 5 ms and every 0.25σ). Colours run from blue (edge below the output) through grey-white (edge at the output) to red (above), clipped at ±6σ; edges more than 6σ from the output — the far edge r ∓ mσ when r is far — are mid-grey, and cells without reference mass are white. For r above the output the lower edge is the facing edge and forms one blue band (panel a, upper half), for r below the output the upper edge forms one red band (panel b, lower half), and within 2σ of the output both edges are coloured. The facing edge stays about 1.6σ beyond the output at every offset; there is no cap (10/50/90 % of the facing-edge position 1.41/1.62/1.81σ, mass-weighted over all cells).

## Figure 2 — `fig_edge_offset_hist`

Position of the facing edge beyond the predicted output (σ; the edge on the side of x̄ₖ, measured outward from x̄ₖ) for the round-4 table (grey line) and T′ (blue), weighted by the reference mass m_k,i over steps k ≥ 2 on the σ/24 grid and normalised to one per table, in bins of 0.05σ. Cells with |r − x̄ₖ| < 1σ have no well-defined facing side and are excluded (10.9 % of the mass for the round-4 table, 12.4 % for T′); outside the plotted 0.5–3.5σ lie 3.1 % (round-4) and 0.0 % (T′) of the remaining mass. 10/50/90 % quantiles: round-4 table 1.04/1.99/2.85σ, T′ 1.40/1.61/1.79σ (over all cells, without the exclusion: 1.07/1.92/2.87σ and 1.41/1.62/1.81σ). Look at the spread: the round-4 table's facing edge spreads over about ±1σ, as expected from half-widths that come in 2σ steps around r, whereas T′ concentrates the facing edge in 1.4–1.8σ; the dotted line marks 1.9σ, the optimum of Remark 1 at a 6.3 % send rate.

## Figure 3 — `fig_edge_profile`

Information bought per transmission, (Σ m·c)/(Σ m·q̂) over the cells of each offset bin |r − x̄ₖ|/σ (steps k ≥ 2, σ/24 grid), for the round-4 table (old action set, grey) and T′ (blue); the number above each bar is the share of the table's expected transmissions E[N] − 1 that fall in that bin. Reference lines: constant SoD (dashed, 1.47, the ratio over the whole experiment) and a one-sided window with its edge 2σ beyond the output (dotted, 5.88, Remark 1). Look at the last bin: under the round-4 table 18 % of the transmissions occur at offsets of 40σ or more and buy 1.13 units each; there the 40σ cap on the half-width leaves the output outside the window (mean facing edge −1.56σ). T′ spends 15 % of its transmissions there at 4.55 units each, close to the other bins.

## Checks

- Facing-edge quantiles (10/25/50/75/90 %, all cells, round-11 definition) from the same edge arrays as Figure 1: T′ 1.4058, 1.4359, 1.6165, 1.7262, 1.8142 (archived round11b.json: max |diff| 0.0e+00); round-4 table 1.0709, 1.4395, 1.9201, 2.4846, 2.8674 (archived round11.json old_j0: max |diff| 0.0e+00).
- Against the values given in the request: T′ 1.41/1.62/1.81 → diffs -0.004, -0.004, +0.004 (within 0.01: True); round-4 table 1.05/1.9/2.87 → diffs +0.021, +0.020, -0.003 (within 0.01: False).
- T′ edges recomputed from the stored actions with eq. (12): 0 mismatches over 175694854 (node, step) pairs.
- Figure 3 totals Σ m·q̂ = E[N] − 1: round-4 |diff| 8.0e-13, T′ |diff| 2.8e-12 (≤ 1e-6).
- Binned statistic reproduces the archive bit for bit: T₁ (round11.json edge_j1) True, round-4 table (old_j0) True, T′ (round11b.json) True.
