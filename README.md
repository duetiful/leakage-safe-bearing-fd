# Leakage-Safe Bearing Fault Diagnosis: Cross-Domain Generalization

ME G545 project. The aim is to build an honest deep learning model for bearing fault diagnosis that
generalizes across domains under a leakage-safe evaluation protocol on CWRU and PU.

- Plan: [PLAN.md](PLAN.md)
- Progress log: [docs/progress-log.md](docs/progress-log.md)
- Presentation: [docs/DNNpresentation.pptm](docs/DNNpresentation.pptm)
- Literature check: [docs/literature.md](docs/literature.md)
- Benchmark protocol: [docs/benchmark-protocol.md](docs/benchmark-protocol.md)

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
bash setup.sh                            # pinned libraries + benchmark code
cp .env.example .env                     # set WORKSPACE to <repo>/data/raw
data/download_cwru.sh                    # needs KAGGLE_API_TOKEN set
data/download_pu.sh                      # needs unrar or 7z; aria2c optional
```

On Google Colab, run the same commands with a leading `!` (e.g. `!bash setup.sh`).

## Phase 1: reproduce the baselines

```bash
python -m lsbfd.splits                         # leakage report for all six scenarios
bash scripts/phase1_reproduce.sh lr            # logistic regression, PU Cross-Bearing Instance
bash scripts/phase1_reproduce.sh cnn 41        # 1D CNN, one seed (repeat for 42, 43); needs a GPU
bash scripts/phase1_reproduce.sh vib-lr        # same, vibration channel only (also vib-cnn <seed>)
bash scripts/phase1_reproduce.sh leaky-lr      # same, bearing-leaky split (also leaky-cnn <seed>)
python -m lsbfd.summarize runs/phase1          # macro-F1 table, mean ± std
```

## Phase 2: domain adaptation (vibration only)

```bash
bash scripts/phase2_adapt.sh val               # source-only validation grid (seed 41); needs a GPU
bash scripts/phase2_adapt.sh select            # settings chosen by the fixed rule
bash scripts/phase2_adapt.sh test 41           # ERM, DANN, SimCLR on the test bearings (repeat 42, 43)
bash scripts/phase2_adapt.sh uda 41            # optional: unsupervised DA variant
python -m lsbfd.summarize runs/phase2          # macro-F1 table, mean ± std
```

On a machine with little disk (e.g. Kaggle), build the vibration-only cache one bearing at a
time instead of downloading all of PU, then point the runs at it:

```bash
CACHE=/kaggle/working/cache bash scripts/prepare_pu_cache.sh   # ~2.6 GB, checksums verified
CACHE=/kaggle/working/cache OUT=/kaggle/working/phase2 bash scripts/phase2_adapt.sh val
```

Protocol and results: [results/adaptation.md](results/adaptation.md).

Raw data lands in `data/raw/` (git-ignored). Each download script records SHA-256 checksums in
`data/checksums/` on first run and verifies them afterwards.

Milestones M0 to M4 (see PLAN.md) are tracked as GitHub Milestones and tagged `m0` ... `m4`.
