# Phase 2: Domain adaptation on PU Cross-Bearing Instance (vibration only)

Status: **validation grid done; test runs pending.** This file fixes the protocol before any result
is seen, so the selection cannot be tuned to the test bearings afterwards.

## Bars to beat (from [reproduction.md](reproduction.md), vibration-only)

| Baseline | Macro-F1 on K001, KA30, KB23, KI21 |
|---|---|
| Logistic regression (pdm-bench) | **0.508** (single run) |
| 1D CNN (pdm-bench) | 0.436 ± 0.023 (seeds 41–43) |

A method counts as better only if its 3-seed mean beats 0.508 by more than about 0.03 (the
CNN's seed spread).

## Protocol

- **Input and split:** the `vibration_1` channel of the PU HostService raster, windows of
  8192 samples, train overlap 0.75, test overlap 0.5, exactly as Phase 1. Train on the 28
  source bearings, score on the 4 held-out bearings. Whole bearings are held out, so no
  recording or window is shared (checked by `lsbfd.splits` and asserted in `lsbfd.adapt`).
- **Backbone and recipe:** pdm-bench's `CNN1D`, unchanged, with its Phase 1 recipe (AdamW 1e-3,
  weight decay 1e-4, plateau scheduler on train loss, label smoothing 0.05, batch 64,
  final epoch scored, no early stopping), but **20 epochs instead of 50** and mixed precision
  on the GPU: one 20-epoch run takes about 50 minutes on a Colab T4, so 50 epochs × 3 seeds
  would exceed 30 GPU hours. **ERM in the same trainer is therefore the control**, not the
  pdm-bench number.
- **Methods:**

  | Run | What it adds | Setting |
  |---|---|---|
  | `erm` | nothing (cross-entropy only) | control |
  | `dann-bearing` | gradient-reversal head that predicts the source bearing | DG |
  | `dann-bearing_in_class` | same, but only among bearings of the window's own class, so it removes bearing identity without fighting the fault label | DG |
  | `simclr` | SimCLR pretraining of the feature extractor on augmented source windows (gain, polarity, circular shift, noise, time mask; τ = 0.2, 10 epochs, batch 128), then full fine-tuning | DG |
  | `dann-target` | source vs. unlabeled target-window adversary (classic DANN) | UDA |

  DG (domain generalization) uses no target data at all, which is what the benchmark
  measures. UDA uses the *unlabeled* windows of the scored bearings during training
  (transductive); it is reported separately because it is not the benchmark's setting.
- **Hyperparameter selection, source only:** four source bearings (K006, KA22, KB27, KI18;
  one per class, real damage like the test bearings) are held out as validation. Each
  setting trains on the other 24 bearings (seed 41, 20 epochs) and is scored on those four;
  the test bearings are never loaded in this stage. Grid: DANN λ ∈ {0.1, 1} for each domain
  type. Rule (`lsbfd.select`): within each family (`erm`, DG-DANN, `simclr`, UDA-DANN) the
  highest validation macro-F1 wins, ties to the smaller λ. Only the winners run on the test
  bearings, with seeds 41, 42, 43.

## Results

### Validation grid (source-only hold-out K006, KA22, KB27, KI18; seed 41, 20 epochs)

Runs 1–2 on Colab (T4), runs 3–8 on Kaggle, same code and cache contents. One PU file,
`N15_M01_F10_KA08_2.mat`, cannot be read and is skipped, as pdm-bench skips it (2,239 of 2,240
source files).

| Run | Setting | Macro-F1 | Balanced acc. |
|---|---|---|---|
| erm | control | 0.294 | 0.358 |
| dann-bearing λ=0.1 | DG | 0.354 | 0.401 |
| dann-bearing λ=1 | DG | 0.337 | 0.356 |
| dann-bearing_in_class λ=0.1 | DG | 0.308 | 0.367 |
| **dann-bearing_in_class λ=1** | DG | **0.527** | **0.595** |
| simclr (10 pretraining epochs) | DG | 0.283 | 0.329 |
| dann-target λ=0.1 | UDA | 0.309 | 0.373 |
| dann-target λ=1 | UDA | 0.200 | 0.280 |

Chosen by the fixed rule (`lsbfd.select`): **erm**, **dann-bearing_in_class λ=1** (DG-DANN),
**simclr** (only setting), **dann-target λ=0.1** (UDA).

Reading, before any test result: every DG-DANN setting beats ERM on the validation bearings,
and the within-class adversary at λ=1 does so by 0.23. These are single-seed scores on four
bearings (one per class), so one bearing flipping changes macro-F1 by a lot; the test seeds
decide. SimCLR pretraining does not help here, and source-vs-target DANN at λ=1 hurts.

### Test bearings

Pending: `test 41`, `test 42`, `test 43` (then `uda 41–43`).
