# Progress Log

One dated entry per working session: what was done, results, next step.

## 2026-10-15
- Done: Created the repository; added the project plan and presentation.
- Next: Phase 0, verify literature claims and set up the environment.
- Added timeline: target finish Oct 31, hard deadline Nov 23; milestone due dates set.

## 2026-10-15 (Phase 0 start)
- Done: Verified the presentation's numbers against the source papers and summarized the four core
  references in [literature.md](literature.md) (#1, #2).
- Results: 294 papers / 17 fields is correct but comes from Kapoor & Narayanan (2023), not the slide 5
  references. The ">90%" figure is about bearing-diagnosis papers (cited secondhand in Vieira et al.),
  not about the 294 papers; primary source not yet confirmed. 0.326 vs 0.355 is correct but only for
  PU Cross-Bearing Instance, and compares a 3-seed CNN mean with a single LR run. Vieira et al. is
  confirmed published in MSSP 2026. Benchmark code is public (1Sensor/pdm-bench, MIT).
- Updated PLAN.md motivation and success criterion to match the sources.
- Next: #3 and #4 (cross-check protocol against the pdm-bench configs), #5 (environment and download scripts).

## 2026-10-15 (Phase 0 continued)
- Done: Audited the benchmark code (1Sensor/pdm-bench, commit ca52408) against the paper and wrote
  [benchmark-protocol.md](benchmark-protocol.md) (#3, #4). Added `requirements.txt` pinned to the
  benchmark's own library versions, `environment.yml`, `.env.example`, and CWRU / PU download and
  checksum scripts (#5).
- Results: code is public and split definitions are config queries, so splits are exactly
  reproducible. The benchmark's 278 tests pass in the pinned Python 3.11 environment. Found in the
  code: PU models likely take motor currents as well as vibration (to confirm on download), DL gets
  no input normalization, and the per-scenario "winner" selection has no visible source-only
  validation step.
- Not yet tested: the dataset downloads themselves (the hosts were unreachable from the cloud
  machine); the scripts' logic and checksum handling were tested offline.
- Next: download both datasets, commit checksums, tag `m0`, then start Phase 1.

## 2026-10-15 (Phase 0 data)
- Done: Downloaded both datasets on Colab with the repo's scripts: CWRU 161 `.mat` files, PU 2,560
  (32 bearings × 4 conditions × 20 measurements, the full set) (#5).
- Results: Confirmed that PU models in the benchmark see two motor currents plus vibration, not
  vibration only. Colab runs Python 3.13, which pdm-bench formally excludes; installing it with
  `--ignore-requires-python` passes all 278 of its tests on 3.13, so `setup.sh` now does that.
- Committed SHA-256 checksums from the Colab download: `data/checksums/cwru.sha256` (161 files) and
  `data/checksums/pu.sha256` (2,560 files: all 32 PU bearings × 80 files, no duplicate hashes).
  Later downloads are verified against these automatically (#5).
- Milestone M0 reached and tagged `m0`. One open follow-up: the primary source for the ">90%"
  splitting figure (#2).
- Next: Phase 1, reproduce the 0.355 LR and 0.326 CNN baselines on PU Cross-Bearing Instance.

## 2026-10-15 (Phase 1 start)
- Done: Added `lsbfd` package with a leakage check over all six scenarios (`lsbfd.splits`,
  8 tests run on the real file lists), a run script with the paper's exact settings for LR and
  the 1D CNN (`scripts/phase1_reproduce.sh`), and a results summarizer. pdm-bench is now an
  editable clone in `third_party/` so its configs are found. Both pipelines ran end to end on
  synthetic data here; real training runs on Colab.
- Results: no recording is shared across any split, but CWRU Cross-Load, CWRU Cross-FS and PU
  Cross-Operating Condition test on bearings already seen in training
  (results/splits/leakage_report.md).
- Next: run LR and CNN seeds 41 to 43 on Colab, then write results/reproduction.md.
- LR on PU Cross-Bearing Instance (Colab): macro-F1 0.388 vs the paper's 0.355; recorded in
  [results/reproduction.md](../results/reproduction.md). CNN seeds 41 to 43 next.

## 2026-10-15 (Phase 1 baselines)
- Done: CNN seeds 41 to 43 on Colab. CNN 0.315 ± 0.032 vs paper 0.326 ± 0.008 (reproduced within
  spread); LR 0.388 vs 0.355 (not exact). LR beats every CNN seed. Details in
  [results/reproduction.md](../results/reproduction.md).
- Next: vibration-only rerun and leaky-split contrast to complete M1.

## 2026-10-16
- Done: Added the two remaining M1 variants to `scripts/phase1_reproduce.sh`: vibration-only
  (`lsbfd.run_variant` drops the motor currents from the PU loader; 18 instead of 54 LR features)
  and a bearing-leaky split (measurements 1, 6, 11, 16 of every bearing as test). Both smoke-tested
  end to end on synthetic data.
- Next: run them on Colab and record the results.
- Done: Ran both variants on Colab. Vibration-only: LR 0.508, CNN 0.436 ± 0.023 (both +0.12 over
  3-channel). Leaky split: LR 0.841, CNN 0.999. Phase 2 bar is now vibration-only LR, 0.508.
  M1 complete; details in [results/reproduction.md](../results/reproduction.md).
- Next: user creates the `m1` release; Phase 2 (domain adaptation) on vibration-only input.

## 2026-10-17 (Phase 2 start)
- Done: Added `lsbfd.adapt`, a trainer on pdm-bench's unchanged CNN1D and vibration-only PU
  input with four methods: ERM (control), DANN with a source-bearing adversary (plain and
  within-class), SimCLR pretraining plus fine-tuning, and classic source-vs-target DANN (UDA,
  reported separately). Hyperparameters are chosen on four held-out *source* bearings
  (K006, KA22, KB27, KI18) by a fixed rule (`lsbfd.select`); the test bearings are never
  loaded in that stage. `scripts/phase2_adapt.sh` runs the stages and resumes after a
  disconnect. Protocol written down before any result in
  [results/adaptation.md](../results/adaptation.md). 6 new tests; all 15 pass. Every stage
  smoke-tested end to end here on synthetic signals.
- Next: run the validation grid and the three test seeds on Colab, then fill in
  results/adaptation.md (M2, #6).
- Fix: the first Colab run stopped on a PU file that scipy cannot read ("Expecting matrix
  here"). pdm-bench's own loader skips unreadable files, so the Phase 1 baselines never saw it;
  `lsbfd.adapt` now skips them the same way, prints their names, and records the number of
  recordings loaded in each run's config.
- Validation stage started on Colab: one file skipped (`N15_M01_F10_KA08_2.mat`, 2,239 of 2,240
  loaded); ERM on the validation bearings scored macro-F1 0.294 (20 epochs, ~50 min on a T4).
  Test runs cut to 20 epochs (from 50) to fit Colab; ERM in the same trainer stays the control.
- Colab compute units ran out after 2 of 8 validation runs (ERM 0.294, DANN within-class
  λ=0.1 0.308 on the validation bearings). To finish the full plan on Kaggle's free GPU, the
  vibration cache is now one file per bearing, and `scripts/prepare_pu_cache.sh` builds it one
  bearing at a time (download, checksum, cache, delete), so the raw dataset never has to fit
  on disk. Runs read only the cache, and the leakage check falls back to the committed
  manifest when raw files are absent.
