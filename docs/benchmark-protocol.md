# Benchmark Protocol (Knap, Jachymczyk & Lalik, PHME 2026)

Covers issue #3 (how the benchmark defines splits, domains, classes, windowing and metric) and
issue #4 (whether code and split files are public). Each fact is checked against both the paper
(see [literature.md](literature.md)) and the released code, **1Sensor/pdm-bench at commit
`ca52408`** (the only commit, MIT licence). Paths below are relative to that repo.

## 1. Is it public? (issue #4)

**Yes.** <https://github.com/1Sensor/pdm-bench> contains the loaders, both pipelines, all six scenario
configs, the run scripts that produced the paper's tables, a dataset download script, and a passing
test suite (278 tests pass in our pinned environment, see `requirements.txt`).

There are **no separate split files**. Each split is a pandas query over recording metadata in
`config/task/<scenario>.yaml`, so the split is fully defined by the config and the dataset's file
names. That is enough to reproduce it exactly, and Phase 1 should still export the resolved
recording lists so they can be checked into `results/`.

## 2. Splits and domains (issue #3)

Splits are queries over whole recordings, applied before windowing, so no recording contributes
windows to both sides. One PU recording is one `.mat` file (one operating condition, one bearing,
one measurement number); one CWRU recording is one `.mat` file.

| Scenario | Train (source) query | Test (target) query | Classes |
|---|---|---|---|
| CWRU Cross-Load | 12 kHz DE fault + normal, rpm 1797/1772/1750 | rpm 1730 | 4 |
| CWRU Cross-FS | 12 kHz DE fault, excluding 0.028 in faults | 48 kHz DE fault | 3 |
| CWRU Cross-Fault Instance | DE fault families 0.007-* and 0.014-* | 0.021-* | 3 |
| PU Cross-Operating Condition | N15_M01_F10, N15_M07_F04, N15_M07_F10 | N09_M07_F10 | 4 |
| PU Cross-Damage Provenance | healthy K002–K006 + artificial damage | healthy K001 + lifetime damage (excl. KB multiple) | 3 |
| **PU Cross-Bearing Instance** | **all bearings except K001, KA30, KI21, KB23** | **K001, KA30, KI21, KB23** | **4** |

PU classes come from the bearing code (`loaders/pu.py`): K0xx healthy, KAxx outer ring, KIxx inner
ring, KBxx multiple. In Cross-Bearing Instance each test class is therefore **one single held-out
bearing**, which is why the scenario is hard and why per-class F1 will be noisy.

Matches the paper's Table 1. One wording difference: the paper lists CWRU Cross-Load's source as
1797/1772/1750 rpm and target 1730; the code agrees. (The default `config/dataset/cwru.yaml` uses a
different query, but every scenario overrides it.)

## 3. Windowing

- Window length: CWRU **2048**, PU **8192** samples (task configs).
- Overlap: task configs set 0.5 / 0.5. The paper's "long" profile overrides train overlap to **0.75**
  (`scripts/run_phm_benchmark.sh`, `scripts/run_phm_followup_dl.sh`); test overlap stays 0.5.

## 4. Metric

Macro-F1 over the target classes (paper §2.4); the ML hyperparameter search also scores with
`f1_macro` (`training/ml_classifiers.py`).

## 5. Exact recipe behind the two target numbers

- **0.326 ± 0.008 (deep baseline):** `scripts/run_phm_followup_dl.sh --scope winners`, entry
  `pu_cross_bearing_instance_long_cnn1d`: task `pu_cross_bearing_instance/cnn1d`, 50 epochs, batch 64,
  AdamW lr 1e-3, weight decay 1e-4, ReduceLROnPlateau (factor 0.5, patience 3), train overlap 0.75,
  seeds 41, 42, 43 via `+train.random_state`.
- **0.355 (logistic regression):** `scripts/run_phm_benchmark.sh --long --pipelines ml --datasets pu`:
  14 time + 4 frequency features per channel, StandardScaler inside the sklearn pipeline (fit on
  train only), Bayesian search with 30 iterations and 5-fold CV, single run.

## 6. Things in the code that the paper does not make obvious

These matter for an honest reproduction. Each is read from the code; items marked *to confirm* need
the downloaded data.

1. **PU input may not be vibration-only (*to confirm*).** The paper says it uses the HostService
   vibration signal. The PU loader keeps **every channel** in the HostService raster and no PU task
   config filters channels. In the PU `.mat` files the 64 kHz raster also holds the two motor phase
   currents, so the models likely see 3 channels (2 currents + vibration). CWRU configs do filter to
   the drive-end sensor.
2. **No source validation split for DL.** No task config sets `val_query`, so DL training runs a fixed
   50 epochs with the LR scheduler stepping on training loss, and the final epoch is evaluated. That
   avoids test-based early stopping. However, the "winner" model and profile for each scenario
   (`run_phm_followup_dl.sh`) were picked from the first benchmark round, and the code does not show
   a source-only validation step for that choice. If it was picked on target scores, 0.326 is a
   slightly optimistic number for the best deep model. This is inferred, not confirmed.
3. **No input normalization for DL.** `Dataset.torch_dataset` defaults to `normalization=False` and
   the DL pipeline does not turn it on, so CNNs get raw-scale signals. Per-window normalization is a
   cheap, leakage-free change worth testing in Phase 3.
4. **ML hyperparameter CV is not grouped by recording.** `BayesSearchCV(cv=5)` splits windows, not
   recordings, so overlapping windows of one recording can land in both CV folds. This never touches
   the test set, but it can favour over-fitted hyperparameters.
5. **LR is a single run.** Its spread across seeds is unknown; Phase 1 should rerun it with the same
   seeds as the CNN.

## 7. Plan for Phase 1

Use pdm-bench as a pinned dependency for loading and splitting, rather than rewriting it, so the
reproduction runs on the same code as the paper. Our own code adds the leakage checks (no shared
recording or bearing across train and test), exports resolved split lists, and reruns both
baselines over seeds 41, 42, 43.
