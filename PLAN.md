# Project Plan: Improving Cross-Domain Generalization in Leakage-Safe Bearing Fault Diagnosis

**Course:** ME G545 · **Author:** Aryan · **Source of truth:** `DNNpresentation.pptm` (slides 1 to 5)

## 1. Goal

Build an honest deep learning model for bearing fault diagnosis that generalizes across domains
(different bearings, loads, operating conditions, datasets) when evaluated under a **leakage-safe**
protocol, and report clearly where it does and does not beat simple baselines.

**Why (from the presentation, corrected after the Phase 0 check in `docs/literature.md`):** data leakage
has affected at least 294 papers across 17 fields (Kapoor & Narayanan, *Patterns*, 2023), and reviews of
bearing fault diagnosis report data-splitting issues in over 90% of papers (as cited by Vieira et al.,
MSSP 2026; primary source still to confirm). Knap, Jachymczyk & Lalik (PHM Society Europe, 2026) built a
leakage-safe cross-domain benchmark on CWRU + PU. On its PU Cross-Bearing Instance scenario their best
deep model reached **macro F1 = 0.326 ± 0.008** (3 seeds), while logistic regression reached **0.355**
(single run), the only one of six scenarios where the simple model wins. That gap is the target.

## 2. Research questions

1. **RQ1 (Reproduce):** Can the 0.326 deep baseline and the 0.355 logistic regression result be
   reproduced exactly with the same split, metric and seeds?
2. **RQ2 (Adapt):** Does domain-adaptation training (adversarial DANN and/or contrastive pretraining)
   push a deep model above the logistic regression baseline under the same leakage-safe protocol?
3. **RQ3 (Explain):** Which single change is responsible for any improvement, verified by
   one-change-at-a-time ablations?

**Success criterion:** mean macro F1 over seeds beats 0.355 on the benchmark's PU Cross-Bearing Instance scenario,
with the gain larger than the seed-to-seed spread. A negative result, honestly reported, is also a
valid outcome.

## 3. Phases and milestones

### Phase 0: Literature verification and setup
- Read and summarize the four core references (slide 5): Vieira et al. (MSSP, 2026); Knap et al.
  (PHM Europe, 2026); Chen et al. (IEEE TIM, 2023); Ganin & Lempitsky (ICML, 2015).
- Verify every number used in the presentation against the source papers (294 papers / 17 fields,
  >90% splitting issues, 0.326 vs 0.355), noting page and table for each.
- Record exactly how the benchmark defines its splits, domains, classes, windowing and metric.
- Check whether the benchmark code and split files are public.
- Set up the repo, environment file, and the CWRU and PU download scripts.
- **Milestone M0:** `docs/literature.md` with verified claims, and a working environment.

### Phase 1: Reproduce (slide 3, step 1)
- Implement the leakage-safe split: no bearing, recording or overlapping window appears in both
  train and test. Add an automated check that fails if any overlap exists.
- Reproduce the logistic regression baseline (0.355) and the deep baseline (0.326).
- Run several seeds; report mean ± std.
- Contrast with a deliberately leaky random split to show how much accuracy inflates.
- **Milestone M1:** `results/reproduction.md`; reproduced numbers within tolerance, or a documented
  explanation of any mismatch.

### Phase 2: Adapt (slide 3, step 2)
- Add a domain-adversarial head (DANN, gradient reversal) so features stop encoding bearing identity.
- Try contrastive pretraining (e.g. SimCLR-style augmentations on vibration windows) and fine-tuning.
- Domain labels and any target data used only in ways the benchmark protocol allows; state clearly
  whether a setting is unsupervised DA or pure domain generalization.
- Hyperparameters chosen on a validation split drawn from source domains only, never the test domain.
- **Milestone M2:** `results/adaptation.md` comparing each method with both baselines.

### Phase 3: Experiment and explain (slide 3, step 3)
- Ablations changing one thing at a time: input (raw vs FFT vs envelope spectrum), normalization,
  backbone size, DA loss weight, augmentations, pretraining on/off.
- Probe whether learned features still predict bearing identity (a linear probe), as evidence the
  adaptation worked for the stated reason.
- Report per-class and per-domain F1 and confusion matrices, not just the average.
- **Milestone M3:** ablation table showing what caused any improvement.

### Phase 4: Honest overview and future prospects
- Write up: what was reproduced, what improved, what didn't, limitations (datasets, class balance,
  real-world transfer), and future directions (more datasets such as Paderborn variants or XJTU-SY,
  test-time adaptation, physics-informed features).
- **Milestone M4:** final report and presentation, tagged release `v1.0` on GitHub.

## 4. Leakage-safety rules (apply to every experiment)
1. Split by bearing / recording first, then window. Never window first and split randomly.
2. Fit scalers, feature selection and any normalization on training data only.
3. No hyperparameter or early-stopping decision uses the test domain.
4. Every run logs its config, git commit hash, seed and split ID.
5. Results are reported over multiple seeds with spread.

## 5. GitHub workflow (so the professor can track every change)
- One repository; `main` always holds working code.
- Every change is committed with a clear message; work happens on short branches merged by pull
  request, so each step has a reviewable record.
- One GitHub Milestone per phase (M0 to M4) and one Issue per task above; commits reference issues.
- `docs/progress-log.md` gets a dated entry for each working session: what was done, results, next step.
- Tag each milestone (`m0` ... `m4`) so the professor can jump to any checkpoint.
- Large raw data stays out of git; download scripts and checksums are committed instead.

## 6. Proposed repository layout
```
README.md            project summary and how to reproduce
PLAN.md              this plan
docs/                literature.md, progress-log.md, presentation
data/                download scripts only (raw data git-ignored)
src/                 splits, datasets, models, training, evaluation
configs/             one YAML per experiment
results/             tables and figures per milestone
tests/               leakage checks and split tests
environment.yml
```

## 7. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Benchmark code or splits not public | Rebuild from the paper's description; document every assumption |
| Can't reproduce 0.326 / 0.355 exactly | Report the gap honestly, then compare methods on my own reproduced baselines |
| Domain adaptation doesn't beat 0.355 | Report as a negative result with ablations explaining why |
| Compute limits | Small 1D CNN backbones; CWRU and PU are small enough for a single GPU or Colab |

## 8. Timeline

Target finish: **October 31, 2026**. Hard deadline: **November 23, 2026** (the gap is buffer).

| Milestone | Phase | Due |
|---|---|---|
| M0 | Literature verification and setup | Oct 17 |
| M1 | Reproduce baselines | Oct 22 |
| M2 | Domain adaptation | Oct 26 |
| M3 | Ablations and explanation | Oct 29 |
| M4 | Honest overview and final report | Oct 31 |

If a phase slips, later phases shift into the buffer before Nov 23; M4 (the write-up) is never
cut, because an honest report of partial results beats unreported experiments.

## 9. Open items to confirm
- Whether the professor wants a specific report format or submission schedule.
