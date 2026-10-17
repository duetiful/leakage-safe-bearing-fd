# Phase 2: Domain adaptation on PU Cross-Bearing Instance (vibration only)

Status: **code ready, runs pending** (Colab). This file fixes the protocol before any result
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
  50 epochs, final epoch scored, no early stopping). Phase 2 runs use mixed precision on the
  GPU for speed, so **ERM in the same trainer is the control**, not the pdm-bench number.
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

Pending. Tables to fill: validation grid; test macro-F1 (mean ± std over seeds) and balanced
accuracy per method against both baselines; per-class F1 and per-bearing accuracy.

Reproduce: `bash scripts/phase2_adapt.sh val`, then `test 41`, `test 42`, `test 43`
(optional `uda <seed>`), then `python -m lsbfd.summarize runs/phase2`.
