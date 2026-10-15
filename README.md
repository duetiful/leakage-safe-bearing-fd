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

Raw data lands in `data/raw/` (git-ignored). Each download script records SHA-256 checksums in
`data/checksums/` on first run and verifies them afterwards.

Milestones M0 to M4 (see PLAN.md) are tracked as GitHub Milestones and tagged `m0` ... `m4`.
