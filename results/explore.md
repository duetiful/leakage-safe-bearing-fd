# Phase 4: what limits cross-bearing transfer on PU, and what fixes it

Status: done (2026-11-16). Code: `src/lsbfd/explore.py` (variants), `src/lsbfd/envelope.py`
(`despike`). Every number below is macro-F1 on the same 10 folds as Phase 3 (fold 0 = the
benchmark split K001, KA30, KB23, KI21; folds 1–9 rotate one real-damage bearing per class),
whole bearings held out. Full generated table: [explore_tables.md](explore_tables.md).

**Honesty note.** These variants were chosen *after* Phase 3, from a data audit that looked at
per-bearing signal statistics of all 32 bearings and at Phase 3's published per-bearing
accuracies. No variant's settings were tuned on scored bearings, but the choice of which ideas
to try was informed by the data, so the results below are exploratory, not pre-registered.

## 1. What the data audit found

### A non-bearing spike artifact, unevenly spread across classes

Counting impulses above 8 robust standard deviations in the 1500 rpm / 0.7 Nm / 1000 N recordings:

| Group | Spike rate (events/s) | Share of signal energy in spikes |
|---|---|---|
| Healthy K001, K002, K003, K006 | 74–225 | 0.38–0.77 |
| Healthy K004, K005 | 0.5–1.3 | ~0.01 |
| Artificial damage (KA05–KA09, KI05, KI07, KI08) | 7–69 | 0.04–0.37 |
| Real damage (KA04, KA15, KA22, KA30, KI14, KI16, KI17, KI21, KB27, ...) | 0–5 | 0.00–0.07 |

The spikes form a train at **100 Hz at 1500 rpm and 52.5 Hz at 900 rpm**: not proportional to
shaft speed (ratio 1.90 vs 1.67) and not at any bearing fault order, so they are not bearing
impacts. At 900 rpm they appear in almost every bearing (kurtosis 10–43). Genuine faults show
physical periodicities instead (KA16 at 3.07 orders = BPFO; KI18 and KB24 at the shaft order).

Consequence: a model can learn "large periodic spikes → healthy or artificial damage" instead
of fault physics. Generic statistics (kurtosis, crest factor, peak values) are dominated by
the spikes. The two healthy bearings without the artifact are exactly the hard ones
(K005 is right 18–23 % of the time); healthy K002, which has a weaker artifact, is called
inner-ring damage in almost every window.

### The "multiple damage" class is not a distinct physical class

The dataset authors (Lessmeier et al., 2016, Table 5 and Sec. 7.4) label KB23 and KB24 as
inner-ring dominant and KB27 as outer-ring dominant, and classify with three classes. Our
Phase 3 model predicts KB24 as inner-ring in 85 % of windows. With one KB bearing per test
fold, the benchmark's 4-class task asks the model to separate a mixture from its dominant
component on a bearing it has never seen.

### Damage mode differs between the benchmark's test bearing and its training set

KA30, the benchmark's outer-ring test bearing, has *plastic deformation (indentations),
distributed*; most outer-ring training bearings are pitting or artificial EDM / drilled /
engraved damage. KA30 was the worst-recognized outer-ring bearing in Phase 3 (21 %).

## 2. Fixes tested (boosted trees unless noted; 10 folds; Δ is paired against the same model in Phase 3)

| Variant | Features | 10-fold macro-F1 | Fold 0 | Δ vs Phase 3 | Folds better | p |
|---|---|---|---|---|---|---|
| Phase 3 (all 28 training bearings) | envelope | 0.528 ± 0.100 | 0.490 | — | — | — |
| **Train on healthy + real-damage bearings only** | envelope | **0.563 ± 0.093** | **0.560** | +0.035 | 9 of 10 | 0.010 |
| Despiked windows (clip at 4 robust std) | envelope | 0.524 ± 0.102 | 0.521 | −0.004 | 4 of 10 | 0.70 |
| **Despiked + real-damage-only training** | envelope | **0.570 ± 0.091** | **0.590** | +0.041 | 7 of 10 | 0.027 |
| Real-only + operating condition as input | envelope | 0.550 ± 0.101 | 0.537 | | | |
| Real-only + bagging over training bearings (15 bags) | envelope | 0.545 ± 0.087 | 0.545 | | | |
| Recording-level: averaged envelope spectrum per 4 s recording | envelope | 0.477 ± 0.127 | 0.384 | | | |
| Despiked windows | generic | 0.465 ± 0.100 | 0.376 | +0.084 | 9 of 10 | 0.014 |
| Real-damage-only training | generic | 0.462 ± 0.085 | 0.493 | +0.081 | 9 of 10 | 0.004 |

Under the dataset authors' 3-class labels (a different task, never compared with 4-class
numbers):

| Variant | LR, generic features (reference) | Trees, envelope features |
|---|---|---|
| All training bearings | 0.489 ± 0.098 | 0.718 ± 0.127 (fold 0: 0.622) |
| Real-damage-only training | 0.501 ± 0.110 | **0.756 ± 0.129** (fold 0: 0.730) |

## 3. Findings

1. **Training only on healthy and real-damage bearings is the single most useful change.**
   With the envelope features it reaches 0.563 over 10 folds and **0.560 on the benchmark
   split, above pdm-bench's 0.508** by more than the ~0.03 seed spread. With despiking
   added: 0.570 and 0.590. The 12 artificial-damage bearings (EDM, drilling, engraving) carry
   the spike artifact and a different damage geometry; removing them mostly helps KA30
   (outer ring, indentations): 21 % → 52 % (66 % despiked).
2. **The spike artifact explains much of what generic features learn.** Despiking alone lifts
   trees on generic features by +0.084 (9 of 10 folds), while the envelope features, which
   read periodicity at fault orders rather than amplitude, are already insensitive to it
   (−0.004). This is direct evidence that generic-feature models partly fit a recording
   artifact, and a reason the benchmark's LR baseline is less meaningful than its score.
3. **Label definition matters more than any model change.** With the authors' 3-class labels
   the same envelope-feature trees reach 0.756 vs 0.501 for the generic-feature LR on the same
   task. The 4-class "multiple" class is near-unlearnable across bearings (KB27 is never
   recognized in any variant).
4. **What did not help:** operating condition as an input, bagging over bearings, CV-tuned tree
   settings (Phase 3b), recording-averaged spectra (0.477, fewer and smoother training
   samples), the envelope CNN and DANN / SimCLR (Phases 2–3).
5. **What remains unsolved:** healthy bearings without the artifact (K005, and K002) and the KB
   class. No method here identifies them reliably on unseen bearings.

## 4. Recommended headline (with its caveats)

> Physics-informed envelope-order features with gradient-boosted trees, trained on healthy and
> real-damage bearings only (despiked input), reach macro-F1 0.590 on the PU Cross-Bearing
> Instance benchmark split (pdm-bench LR: 0.508; 1D CNN: 0.436) and 0.570 ± 0.091 across 10
> rotated bearing hold-outs (generic-feature LR: 0.355).

Caveats to state with it: the training-set and despiking choices came from a post-hoc data
audit; the feature models are deterministic (fold spread, not seed spread, is the uncertainty);
folds share bearings, so p-values are indicative; one dataset.

## Not tried, and why

- Foundation models (e.g. UniFault, 2025): no public weights found; general time-series models
  are not trained on 64 kHz vibration.
- MiniRocket / raw-signal CNN with the new training set: feasible on a Kaggle GPU, too slow on
  this CPU-only container; worth one run given finding 1.
