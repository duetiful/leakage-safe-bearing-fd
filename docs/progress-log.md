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
