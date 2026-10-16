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

## Vibration-only rerun and leaky-split contrast (2026-10-16)

Same settings as above. "Vibration-only" keeps only `vibration_1` from the PU HostService raster
(`lsbfd.run_variant`). "Leaky" trains on every bearing and tests on measurements 1, 6, 11, 16 of
each bearing, so test recordings are new but every test bearing was seen in training.

| Variant | LR macro-F1 (single run) | CNN macro-F1 (seeds) |
|---|---|---|
| As released (currents + vibration) | 0.388 | 0.315 ± 0.032 (41–43) |
| **Vibration-only** | **0.508** (balanced acc. 0.535) | **0.436 ± 0.023** (41–43: 0.440, 0.457, 0.411) |
| Leaky (seen bearings, unseen recordings) | 0.841 | 0.999 (seed 41 only) |

5. **Dropping the motor currents helps both models a lot**: +0.120 for LR and +0.121 for the
   CNN. The currents appear to carry bearing-specific (or rig-specific) cues that do not transfer
   to new bearings; this is an inference from the scores, not tested directly. The paper's
   0.355 / 0.326 therefore understate what its own models reach on the vibration input it
   describes.
6. **LR still beats the CNN on vibration** (0.508 vs 0.436, above every CNN seed), so the paper's
   headline survives the input change.
7. **Bearing leakage inflates scores massively.** With the same models, letting test bearings
   appear in training lifts LR from 0.388 to 0.841 and the CNN from 0.315 to 0.999. On seen
   bearings the CNN wins; on unseen bearings it loses. A leaky protocol would have reversed the
   paper's conclusion.

## Implications for Phase 2 (updated)

- Phase 2 uses **vibration-only input**, matching the paper's description and scoring higher.
- The bar to beat is vibration-only LR, **0.508**; the deep baseline is the vibration-only CNN,
  0.436 ± 0.023. A claimed gain must exceed about 0.03 (seed spread).
- LR remains a single run; rerunning it with the BayesSearch seed varied would give it a spread.
