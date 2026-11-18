# Phase 3: methods that worked on neighbouring problems (PU Cross-Bearing Instance, vibration only)

Status: **feature models (10 folds) and envelope CNN (fold 0, 3 seeds) done; results below.** The protocol section was written and committed before any PU result. Code: `src/lsbfd/envelope.py`,
`src/lsbfd/novel.py`, `scripts/phase3_novel.sh`. This file states what will be run and how it
will be judged before any number exists, so nothing can be tuned to the test bearings afterwards.

## Why these methods

Phase 2 showed that changing the *training objective* of a raw-signal CNN (DANN, SimCLR) does
not help on unseen bearings. Work on related problems points to changing the *input* and the
*normalization* instead:

| Method | Where it worked | Why it might transfer here |
|---|---|---|
| Envelope-order features at the bearing's fault frequencies | Classical bearing diagnostics (envelope analysis); the original PU paper (Lessmeier et al., 2016) used envelope-based features | BPFO / BPFI / BSF are set by the 6203 geometry and the shaft speed, not by the individual bearing, so a feature read at those orders means the same thing on every bearing |
| CNN on the order-normalized envelope spectrum | Cross-domain bearing diagnosis with standard envelope spectra (2024, Sensors) | Same physics as above, but the model learns which peaks matter; order normalization removes the speed shift |
| AdaBN, test-time BatchNorm re-estimation | Li et al. (2016); WDCNN-AdaBN for bearings (Zhang et al., 2017); later bearing TTA work | Bearing-to-bearing shift looks like a change in feature statistics; re-estimating them on the new machine's own unlabeled data is free |

## Protocol

- **Data and split:** the Phase 2 vibration cache (`vibration_1`, 64 kHz), windows of 8192
  samples. Fold 0 is the benchmark split (train on 28 bearings, score K001, KA30, KB23, KI21).
- **Rotated folds:** folds 1–9 each hold out one random real-damage bearing per class plus one
  healthy bearing (fixed seed 2026, `lsbfd.novel.folds`) and train on the other 28. The
  benchmark score rests on four bearings, so a method only counts as better if it also wins
  across folds. Whole bearings are held out; an assertion checks no bearing or recording is on
  both sides.
- **Feature models:** logistic regression (standardized inputs, balanced class weights; C from
  {0.01, 0.1, 1, 10} by 4-fold leave-bearings-out CV *inside the training bearings*) and
  histogram gradient-boosted trees (fixed: 200 iterations, learning rate 0.1, 31 leaves, L2 1.0,
  balanced class weights). Windows at 50 % overlap for training and scoring, as the
  benchmark's LR. Three feature sets, identical otherwise:
  - `generic`: the 11 features of the pdm-bench LR (rms, std, kurt, cf, sf, clf, ii, mf, fc,
    rms_f, std_f). This re-creates the 0.508 baseline inside our pipeline and is the
    reference for every paired comparison.
  - `phys`: 85 envelope-order features (5 bands × [BSF, BPFO, BPFI harmonics 1–3, shaft 1×
    and 2×, BPFI ± 1× sidebands] in log units over the band's median, plus band energy share
    and envelope kurtosis). Shaft speed comes from the file name.
  - `generic+phys`: both.
- **Envelope CNN:** 3 conv layers on the 5-band × 395-order log envelope spectrum, keeping
  coarse order position; class-weighted cross-entropy, label smoothing 0.05, AdamW 1e-3,
  one-cycle schedule, 15 epochs, batch 256, band-dropout and ±2-bin order-jitter augmentation.
  Windows at 75 % overlap for training (as the Phase 2 CNN), 50 % for scoring. Seeds 41–43 on
  fold 0; seed 41 on all folds. No setting was tuned.
- **AdaBN (test-time adaptation):** the trained CNN is scored three ways: as is (pure DG);
  with BatchNorm statistics re-estimated on all scored windows pooled; and re-estimated per
  scored bearing. Both AdaBN variants read unlabeled test windows, so they are reported as
  test-time adaptation, not DG. The pooled variant also sees a class-balanced mix of four
  bearings, which a real machine would not, so the per-bearing variant is the realistic one.
  The same is done for the raw-signal ERM CNN of Phase 2 (`raw-adabn`, optional).

## How results will be judged (fixed now)

A method **works** if, on fold 0, its mean macro-F1 beats the pdm-bench LR (0.508) by more
than the seed spread (~0.03) **and**, across all 10 folds, it beats `lr / generic` on average
with a two-sided Wilcoxon signed-rank p < 0.05. A fold-0 win without the fold-level win is
reported as "not established". Losses are reported the same way.

## Results

Run in a cloud container (CPU only) on 2026-11-15, from the PU archives downloaded and
checksum-verified by `scripts/prepare_pu_cache.sh`. Full generated tables, including the
recording-level analysis: [phase3_tables.md](phase3_tables.md).

### Across all 10 folds (the main result)

Macro-F1 per fold, then mean ± std over folds. Paired against `lr / generic` (the baseline's
features in our pipeline); p is a two-sided Wilcoxon signed-rank test over the 10 folds.

| Method | Macro-F1 (10 folds) | Δ vs `lr / generic` | Folds better | p |
|---|---|---|---|---|
| LR, generic features (reference) | 0.355 ± 0.101 | — | — | — |
| Boosted trees, generic features | 0.381 ± 0.093 | +0.026 | 5 of 10 | 0.63 |
| LR, envelope-order features | 0.448 ± 0.098 | +0.093 | 8 of 10 | 0.027 |
| LR, generic + envelope | 0.442 ± 0.108 | +0.088 | 6 of 10 | 0.084 |
| **Boosted trees, envelope-order features** | **0.528 ± 0.100** | **+0.174** | **9 of 10** | **0.004** |
| Boosted trees, generic + envelope | 0.500 ± 0.112 | +0.145 | 8 of 10 | 0.020 |

Ablation, same model and only the features changed: trees on envelope vs generic features
+0.147 (8 of 10 folds, p = 0.010); LR on envelope vs generic +0.093 (p = 0.027).

Per-class F1 averaged over folds (healthy / inner ring / multiple / outer ring):

| Method | Healthy | Inner ring | Multiple | Outer ring | Balanced acc. |
|---|---|---|---|---|---|
| LR, generic | 0.75 | 0.24 | 0.31 | 0.12 | 0.404 |
| Boosted trees, generic | 0.68 | 0.44 | 0.23 | 0.18 | 0.451 |
| Boosted trees, envelope | 0.81 | 0.56 | 0.18 | 0.56 | 0.580 |

### Benchmark split only (fold 0: K001, KA30, KB23, KI21)

| Method | Macro-F1 |
|---|---|
| LR, pdm-bench (Phase 1, its own Bayesian search) | 0.508 (single run) |
| LR, generic features, our pipeline | 0.376 |
| Boosted trees, envelope-order features | 0.490 |
| Envelope CNN (DG) | 0.318 ± 0.003 (seeds 41–43) |
| Envelope CNN + AdaBN, pooled scored windows (TTA) | 0.331 ± 0.008 |
| Envelope CNN + AdaBN, per scored bearing (TTA) | 0.393 ± 0.009 |

## Findings

1. **Physics-informed envelope-order features transfer to unseen bearings.** Boosted trees on
   them average 0.528 macro-F1 over 10 held-out bearing sets, against 0.355 for LR on the
   generic features (+0.174, better on 9 of 10 folds, p = 0.004). Holding the model fixed and
   only swapping in the envelope features gives +0.147 (trees) and +0.093 (LR), so the
   features, not the classifier, carry most of the gain. Under the pre-registered rule this
   is the fold-level win.
2. **The gain is where the physics says it should be.** Outer-ring F1 rises from 0.12–0.18 to
   0.56 and inner-ring F1 from 0.24–0.44 to 0.56: these are the classes with a single
   characteristic frequency (BPFO, BPFI). The "multiple damage" class (KB, inner and outer
   at once) does not improve (0.18–0.31), consistent with its signature being a mix of both.
3. **On the benchmark's four bearings alone, the rule is not met.** 0.490 is 0.018 below the
   pdm-bench LR's 0.508, so "beats 0.508 on fold 0" fails. That split is fragile: our LR on the
   same 11 features scores anywhere from 0.36 to 0.47 on fold 0 just by changing C or class
   weighting (checked after the main run, not used for selection), and fold-0 scores rank the
   methods differently from the 10-fold average. We therefore report the 10-fold result as
   the main evidence and the fold-0 result as a stated failure of the stricter criterion.
4. **The envelope CNN does not work** (0.318 on fold 0). It fits the source bearings
   (training loss ~0.55 with label smoothing) but learns bearing-specific spectral detail;
   with ~300 k windows from only 28 bearings, a flexible model on the full spectrum overfits
   the bearings, while a few physically placed features cannot.
5. **Test-time BN adaptation helps the CNN consistently, but not enough.** AdaBN per scored
   bearing lifts the envelope CNN from 0.318 to 0.393 (+0.075, about eight times the seed
   spread, all 3 seeds), pooled AdaBN only +0.013. It uses unlabeled test data, so it is
   test-time adaptation, not DG, and it still stays below the feature models.

### Follow-up: tuned trees (Phase 3b, added after the results above)

To check whether the trees' fixed settings were lucky or leaving gains on the table, a
variant (`hgb-cv`) chooses max leaves {7, 15, 31}, L2 {0, 1, 10} and iterations {100, 300}
by the same leave-bearings-out CV *inside the training bearings* of each fold.

| Method | 10-fold macro-F1 | Fold 0 |
|---|---|---|
| Boosted trees, envelope features, fixed settings | 0.528 ± 0.100 | 0.490 |
| Boosted trees, envelope features, CV-tuned | 0.524 ± 0.100 | 0.467 |
| Boosted trees, generic features, CV-tuned | 0.382 ± 0.095 | 0.327 |

Tuning changes nothing (−0.004, 3 of 10 folds better, p = 0.38), and the envelope-vs-generic
gap holds under tuning (+0.142, 8 of 10 folds, p = 0.010). The CV mostly picked small, heavily
regularized trees. The result does not depend on a lucky hyperparameter choice, and tuning is
not a route to beating 0.508 on fold 0.

## Limitations

- Feature models are deterministic, so they have no seed spread; the fold-to-fold spread
  (~0.10) is the relevant uncertainty and is far larger than any seed effect seen in Phase 2.
- The rotated folds reuse bearings (only 3 KB bearings exist), so folds are not independent;
  the Wilcoxon p-values are indicative, not exact.
- The envelope CNN and AdaBN were run on fold 0 only (CPU time); `cnn-folds` and the raw-CNN
  AdaBN (`raw-adabn`) are scripted but not run.
- Windows are 0.128 s (8192 samples), as the benchmark fixes. That gives a 7.8 Hz envelope
  resolution, about 0.3–0.5 orders, which limits how sharply BPFO and BPFI separate; the
  recording-level vote (phase3_tables.md) is a secondary analysis added after the fold-0
  results.
- Nominal shaft speeds (900 / 1500 rpm) from the file names; slip is absorbed by the peak
  tolerance, not measured.
