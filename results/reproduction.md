# Phase 1: Reproduction on PU Cross-Bearing Instance

Runs use pdm-bench at commit `ca52408` with the paper's settings, via
`scripts/phase1_reproduce.sh` on Google Colab (Python 3.13, pinned libraries from
`requirements.txt`, T4 GPU for the CNN). Metric: macro-F1 on the held-out bearings
K001, KA30, KI21, KB23. Splits pass `lsbfd.splits` (no shared recording or bearing).

| Model | Paper | Ours | Gap | Runs |
|---|---|---|---|---|
| Logistic regression | 0.355 (single run, Table 3) | **0.388** (balanced acc. 0.391) | +0.033 | 1 (2026-10-15) |
| 1D CNN | 0.326 ± 0.008 (seeds 41–43, Table 4) | pending | | |

## Notes

- **LR does not reproduce exactly.** Same code, data, features, search budget and seed, but
  0.388 instead of 0.355. Candidate causes, not yet tested: different library or BLAS builds
  (the authors' Python and OS are not stated), and the solver hitting its 1,000-iteration cap
  during the search (scikit-learn printed many ConvergenceWarnings), which makes the chosen
  hyperparameters sensitive to numerical details. A gap of this size from one deterministic
  run means the paper's single-run 0.355 carries an unknown error bar.
- The direction of the paper's finding holds so far: LR is well above the paper's CNN number.
  Whether it beats our CNN depends on the pending runs.
