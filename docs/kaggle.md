# Running Phases 2–4 on Kaggle (free GPU)

The full PU dataset does not fit on Kaggle's disk, so the runs read a vibration-only cache that
`scripts/prepare_pu_cache.sh` builds one bearing at a time (download, verify checksum, keep
`vibration_1`, delete the raw file). One notebook is reused for every phase.

## Notebook settings (once)

- **Accelerator:** GPU T4 or P100 for Phase 2 and the CNN steps of Phase 3; **None** is enough for
  the feature models (Phase 3 `features` / `ml`, Phase 4).
- **Internet:** On (phone-verified account).
- **Persistence:** Files only, so `/kaggle/working` (cache and results) survives between sessions.

## Cell 1: code and libraries (every session)

```
%cd /kaggle/working
!git clone https://github.com/duetiful/leakage-safe-bearing-fd 2>/dev/null || true
%cd /kaggle/working/leakage-safe-bearing-fd
!git pull
!bash setup.sh
!apt-get -qq install -y unrar > /dev/null
%env CACHE=/kaggle/working/cache
```

`setup.sh` must end with `ready; torch ... cuda True` (or `cuda False` on a CPU session).

## Cell 2: vibration cache (once, ~2.6 GB, resumes if interrupted)

```
!bash scripts/prepare_pu_cache.sh      # must end with "32 of 32 bearings"
```

## Phase 2 (GPU, ~25 GPU hours; each finished run is saved and skipped next time)

```
%env OUT=/kaggle/working/phase2
!bash scripts/phase2_adapt.sh val
!bash scripts/phase2_adapt.sh select
!bash scripts/phase2_adapt.sh test 41   # then 42, 43 (~4 h each)
```

## Phase 3

```
%env OUT=/kaggle/working/phase3
!bash scripts/phase3_novel.sh features  # CPU, 10-30 min
!bash scripts/phase3_novel.sh ml        # CPU, 30-60 min: the main result
!bash scripts/phase3_novel.sh cnn       # GPU, ~15 min per seed
```

## Phase 4 (CPU)

```
!python -m lsbfd.explore features --cache $CACHE --clip 4 --tag clip4
!python -m lsbfd.explore ml --env $CACHE/env_clip4 --train real --tag clip4_real
```

A session lasts at most 12 hours; if a step stops, run the same line again. Download
`/kaggle/working/phase*` from the Output panel as a backup before finishing for the day.
