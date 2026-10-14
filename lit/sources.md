# Sources for the literature comparison

Status (2026-10-14): **URLs located, full texts not yet read.**

This session's network policy allows only GitHub and package registries.
Every PDF host below was refused at the proxy (HTTP 403 on CONNECT), so no
PDF could be downloaded. The URLs come from web-search results and have
**not** been fetched or checked. `comparison.md` and `summary.md` are not
written yet: they need the full texts, and filling them from abstracts would
not meet the brief.

Hosts that must be allowed to continue:

| host | needed for |
|---|---|
| `arxiv.org` | Molin & Hirche; González, Tiels & Oomen; Dozzi, Oomen & González |
| `people.kth.se` | Godoy et al., ECC 2014 |
| `www3.diism.unisi.it` | Casini, Garulli & Vicino: TAC 2011, Automatica 2012 (and CDC 2007, if it is on the author page) |
| `skoge.folk.ntnu.no` | Casini, Garulli & Vicino: CDC 2008 (proceedings mirror) |
| `marco-campi.unibs.it` | Trimpe & Campi, EBCCSP 2015 |
| `www.jstage.jst.go.jp` | Kawaguchi et al., SICE Trans. 2017 |
| `research.tue.nl` | Dozzi et al. (alternative open-access copy) |

## Papers requested

| # | Paper | Open-access URL (unverified) | Access |
|---|---|---|---|
| 1 | B. I. Godoy, P. E. Valenzuela, C. R. Rojas, J. C. Agüero, B. Ninness, "A novel input design approach for systems with quantized output data", ECC 2014, pp. 1049–1054, doi:10.1109/ECC.2014.6862329 | https://people.kth.se/~pva/documents/publications/c4.pdf | Open access (author page), blocked here |
| 2 | M. Casini, A. Garulli, A. Vicino, "Time complexity and input design in worst-case identification using binary sensors", CDC 2007, pp. 5528–5533, doi:10.1109/CDC.2007.4434445 | None found. The Siena repository record (https://usiena-air.unisi.it/handle/11365/21792) has restricted access. Check https://www3.diism.unisi.it/~casini/ once that host is allowed. | No open-access copy found yet |
| 3 | M. Casini, A. Garulli, A. Vicino, "Optimal input design for identification of systems with quantized measurements", CDC 2008, pp. 5506–5512 | https://skoge.folk.ntnu.no/prost/proceedings/cdc-2008/data/papers/0998.pdf (search results connect this file to the paper; not confirmed) | Probably open access, blocked here |
| 4 | M. Casini, A. Garulli, A. Vicino, "Input design in worst-case system identification using binary sensors", IEEE TAC 56(5):1186–1191, 2011, doi:10.1109/TAC.2011.2107091 | https://www3.diism.unisi.it/~casini/pdf/TAC_2011.pdf | Open access (author page), blocked here |
| 5 | M. Casini, A. Garulli, A. Vicino, "Input design in worst-case system identification with quantized measurements", Automatica 48(12):2997–3007, 2012, doi:10.1016/j.automatica.2012.08.016 | https://www3.diism.unisi.it/~casini/pdf/Automatica_2012.pdf | Open access (author page), blocked here |
| 6 | L. Y. Wang, G. G. Yin, J.-F. Zhang, Y. Zhao, *System Identification with Quantized Observations*, Birkhäuser 2010, doi:10.1007/978-0-8176-4956-2 (Ch. 7, "Input Design for Identification in Connected Systems") | None. Springer offers only a chapter preview (https://link.springer.com/chapter/10.1007/978-0-8176-4956-2_7). | **Paywalled**: will be skipped |
| 7 | J.-D. Diao, J. Guo, C.-Y. Sun, "Event-triggered identification of FIR systems with binary-valued output observations", Automatica 98:95–102, 2018 (DOI per a citation listing: 10.1016/j.automatica.2018.09.024) | None found. No arXiv version; the XJTU record (https://scholar.xjtu.edu.cn/en/publications/event-triggered-identification-of-fir-systems-with-binary-valued-/) has the abstract only. | **Paywalled**: will be skipped unless an author copy turns up |
| 8 | J. Guo, Yu, Li, Song, Jing, "Optimal Period Input Design in FIR System Identification With Binary-Valued Observations and Event-Triggered Communication", IEEE TCAS-II 71(1), 2024. This is the input-design follow-up; the January 2024 issue date suggests 2023 early access. | None found (listed only on an author publication list). | **Paywalled**: will be skipped |
| 9 | E. Dozzi, T. Oomen, R. A. González, "D-Optimized Sampling Design for System Identification", arXiv:2605.13120 (accepted for IFAC 2026) | https://arxiv.org/pdf/2605.13120 ; TU/e portal: https://research.tue.nl/en/publications/ac9f2dd1-7bbf-45dc-acb2-cffe9fa0f52a | Open access, blocked here |
| 10 | S. Trimpe, M. C. Campi, "On the choice of the event trigger in event-based estimation", EBCCSP 2015, doi:10.1109/EBCCSP.2015.7300691 | https://marco-campi.unibs.it/pdf-pszip/EBCCSP-2016.pdf (the file name says 2016; search results say the content is the 2015 paper, which still needs checking) | Open access (author page), blocked here |
| 11 | A. Molin, S. Hirche, "Event-triggered estimation of linear systems: an iterative algorithm and optimality properties", arXiv:1203.4980 (journal version: IEEE TAC 62(11), 2017) | https://arxiv.org/pdf/1203.4980 | Open access, blocked here |
| 12 | R. A. González, K. Tiels, T. Oomen, "Kernel-based identification using Lebesgue-sampled data", arXiv:2303.06045 (Automatica 164:111648, 2024) | https://arxiv.org/pdf/2303.06045 | Open access, blocked here |
| 13 | T. Kawaguchi, S. Hikono, I. Maruta, S. Adachi, "System identification under Lebesgue sampling" (in Japanese), SICE Trans. 53(3):236–243, 2017 | https://www.jstage.jst.go.jp/article/sicetr/53/3/53_236/_article/-char/ja/ | Free on J-STAGE (per the listing), blocked here |

Related items found while searching. These are not input-design papers and
not yet assessed:

- R. A. González, K. Tiels, T. Oomen, "Identifying Lebesgue-sampled
  continuous-time impulse response models: a kernel-based approach",
  arXiv:2304.03312 (IFAC WC 2023). Its arXiv note says it overlaps
  substantially with #12.
- T. Kawaguchi et al., "System identification and state estimation under
  Lebesgue sampling: use of inter-sample information", CCTA 2017,
  doi:10.1109/CCTA.2017.8062656. Closed access; an English companion to #13.
- J. Guo et al., "Identification for FIR systems with scheduled binary-valued
  observations", IEEE Access 2018. Its abstract says it discusses input
  design under a scheduling (not event-triggered) policy. This is an IEEE
  Access paper, so it should be open access via ieeexplore.ieee.org (also
  blocked here).

## Search log: input design specifically for Lebesgue / send-on-delta identification

These searches ran through the web-search tool, which returns result
snippets, not full texts. None of them returned a paper that designs the
input (or experiment) for identification from send-on-delta or Lebesgue
sampled data. This is a **preliminary negative** result: it rests on search
snippets, not on reading the related-work sections of #9, #12 and #13, which
is where such a paper would most likely be cited. It will be rechecked once
those full texts are readable.

| query | mode | closest hits | input design for SoD / Lebesgue? |
|---|---|---|---|
| `input design system identification send-on-delta sampling` | extended | SSOD-based PI autotuning (relay / limit-cycle identification); event-based relay FRF experiment | No. These identify under SoD feedback but do not optimise an excitation signal. |
| `optimal input design Lebesgue sampling identification event-based sampling experiment design` | extended | #12; LQG event-based sampling co-design; Wang et al. binary-sensor work | No |
| `"input design" "event-triggered" OR "event-based" identification "Fisher information" level-crossing sampling` | extended | level-crossing signal-processing papers; level-triggered decentralized estimation (arXiv:1211.3720) | No |
| `excitation signal design for identification with Lebesgue sampled or level-crossing sampled output` | extended | #13 (abstract mentions asymptotic variance, no input design); level-crossing reconstruction | No |
| `"send-on-delta" identification "persistent excitation" OR "persistently exciting" input parameter estimation` | extended | generic persistent-excitation papers | No |
| `"event-based sampling" OR "Lebesgue sampling" "experiment design" parameter estimation Fisher information input signal optimization` | extended | Fisher-based sampling-interval design for stochastic models (periodic, not threshold) | No |
| `input design FIR identification event-triggered communication binary-valued observations 2023` | standard | #7, #8, the Franklin Institute 2023 "either-or communication" paper | #8 designs the input under event-triggered **binary** transmission. This is not send-on-delta on a continuous-valued output, but it is the closest match. |
| `arxiv 2605.13120 Dozzi Oomen González sampling design D-optimal continuous-time identification` | extended | #9 | #9 designs the **sampling schedule** (D-optimal, nonuniform) under multisine excitation. The schedule is chosen, not output-triggered. |
