# §8.5 stage-1 predictions (written before inspecting results; scan launched 2026-10-10)

Setting: single scenario (Delta_*, sigma=Delta_*/20), R=200 CRN, reference (C) = uniform input on the same draws.

A. Amplitude scan at chi=0 (84 points on the simplex p_m=a_m^2, step 0.1)
 1. Budget: output amplitude C_m = a_m|G(j w_m)| falls with m (|G| = 0.105, 0.022, 0.014, 0.013), so moving power into tone 1
    raises V_T and the send count; moving power into tones 3-4 lowers it. Expect D>0 (over budget) on the tone-1-heavy side,
    D<0 on the high-frequency side; the budget constraint binds on one side of the simplex only.
 2. Information: for periodic sampling the D-optimum is uniform (log det = C + 4 sum log a_m). Under SoD the optimum is expected
    to deviate from uniform but the direction is not predicted here (reviewer's point: w_k, trigger times and budget all move).
    Expect |Delta m_D^(C)| of order 1 nat or less over the grid, with a single broad maximum (log det is a smooth function of p
    when w_k is smooth), and the budget-feasible maximum not far from uniform.
 3. m_E^(C) < 1 for most candidates (any move from uniform weakens some direction).

B. Phase scan at uniform amplitudes (6^4 = 1296 points)
 1. Budget varies with phase (N^nf 321 -> 302 for chi_m = 0.2 w_m): expect spread of B_u of roughly +-5-10% and both signs of D.
 2. Delta m_D^(C) varies within about +-0.5 nat; the relative phase of the close pair (w_3, w_4) is expected to matter most
    (it changes the window energy of the difference-frequency term), but "most sensitive direction" is to be read off the results.
 3. Correlation between send count and log det across phase candidates expected positive but weak (more sends -> more weight,
    but the weight pattern, not the count, sets precision).
