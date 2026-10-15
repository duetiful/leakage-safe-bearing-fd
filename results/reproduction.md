# Phase 1: Reproduction on PU Cross-Bearing Instance

Runs use pdm-bench at commit `ca52408` with the paper's settings, via
`scripts/phase1_reproduce.sh` on Google Colab (Python 3.13, pinned libraries from
`requirements.txt`, T4 GPU for the CNN). Metric: macro-F1 on the held-out bearings
K001, KA30, KI21, KB23. Splits pass `lsbfd.splits` (no shared recording or bearing).

| Model | Paper | Ours | Gap | Verdict |
|---|---|---|---|---|
| Logistic regression | 0.355 (single run, Table 3) | **0.388** (single run; balanced acc. 0.391) | +0.033 | Not reproduced exactly |
| 1D CNN | 0.326 ± 0.008 (seeds 41–43, Table 4) | **0.315 ± 0.032** (seeds 41–43) | −0.011 | Reproduced within our seed spread |

CNN per seed (macro-F1 / balanced accuracy): 41: 0.290 / 0.283, 42: 0.351 / 0.313,
43: 0.303 / 0.280. Raw table: `summary.csv` from the Colab runs, values copied above.

## Findings

1. **The paper's headline holds, and more strongly.** LR beats the CNN by 0.074 (0.388 vs
   0.315), against 0.029 in the paper, and LR beats every one of the three CNN seeds.
2. **The CNN's seed spread is four times the paper's.** Our standard deviation is 0.032 against
   the paper's 0.008, with seeds ranging from 0.290 to 0.351. Same code, seeds and settings;
   GPU nondeterminism and library builds are the likely source. Any Phase 2 gain smaller than
   about 0.03 cannot be told apart from seed noise with three seeds.
3. **LR is 0.033 above the paper with identical code and seed.** Candidate causes, not yet
   tested: library or BLAS builds (the authors' environment is not stated), and the solver
   hitting its 1,000-iteration cap during the search (many ConvergenceWarnings), which makes the
   chosen hyperparameters sensitive to numerical details. The paper's single-run 0.355 has no
   error bar.
4. **Inputs are 3 channels.** Both models used the two motor currents plus vibration, as the
   released code does (docs/benchmark-protocol.md §6.1). These are "as released" numbers.

## Implications for Phase 2

- The bar to beat is our own LR, **0.388**, not 0.355, since both are measured in the same
  environment.
- Use at least 3 seeds (5 if affordable) and report mean ± std; a claimed gain must exceed the
  CNN's 0.032 spread.
- Still to run for M1: the vibration-only rerun of both baselines, and the deliberately leaky
  random split for contrast.
