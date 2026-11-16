| Labels | Variant | Method | Folds | Macro-F1 mean ± std | Fold 0 | Δ vs Phase 3 same method | Better folds | p |
|---|---|---|---|---|---|---|---|---|
| 3 | labels3 | hgb / generic | 10 | 0.519 ± 0.104 | 0.511 |  |  |  |
| 3 | labels3 | hgb / generic+phys | 10 | 0.649 ± 0.136 | 0.516 |  |  |  |
| 3 | labels3 | hgb / phys | 10 | 0.718 ± 0.127 | 0.622 |  |  |  |
| 3 | labels3 | lr / generic | 10 | 0.489 ± 0.098 | 0.489 |  |  |  |
| 3 | labels3 | lr / generic+phys | 10 | 0.619 ± 0.105 | 0.567 |  |  |  |
| 3 | labels3 | lr / phys | 10 | 0.630 ± 0.099 | 0.565 |  |  |  |
| 3 | real_labels3 | hgb / generic | 10 | 0.540 ± 0.109 | 0.558 |  |  |  |
| 3 | real_labels3 | hgb / phys | 10 | 0.756 ± 0.129 | 0.730 |  |  |  |
| 3 | real_labels3 | lr / generic | 10 | 0.501 ± 0.110 | 0.467 |  |  |  |
| 3 | real_labels3 | lr / phys | 10 | 0.639 ± 0.115 | 0.611 |  |  |  |
| 4 | base | hgb-bag / generic | 10 | 0.387 ± 0.089 | 0.341 |  |  |  |
| 4 | base | hgb-bag / phys | 10 | 0.532 ± 0.100 | 0.497 |  |  |  |
| 4 | clip4 | hgb / generic | 10 | 0.465 ± 0.100 | 0.376 | +0.084 | 9 of 10 | 0.014 |
| 4 | clip4 | hgb / generic+phys | 10 | 0.510 ± 0.095 | 0.374 | +0.010 | 7 of 10 | 0.557 |
| 4 | clip4 | hgb / phys | 10 | 0.524 ± 0.102 | 0.521 | -0.004 | 4 of 10 | 0.695 |
| 4 | clip4 | lr / generic | 10 | 0.373 ± 0.076 | 0.311 | +0.018 | 6 of 10 | 0.557 |
| 4 | clip4 | lr / generic+phys | 10 | 0.442 ± 0.099 | 0.341 | -0.000 | 4 of 10 | 0.922 |
| 4 | clip4 | lr / phys | 10 | 0.455 ± 0.089 | 0.374 | +0.007 | 7 of 10 | 0.322 |
| 4 | clip4_real | hgb / generic | 10 | 0.491 ± 0.106 | 0.391 | +0.110 | 9 of 10 | 0.006 |
| 4 | clip4_real | hgb / generic+phys | 10 | 0.554 ± 0.134 | 0.374 | +0.054 | 8 of 10 | 0.027 |
| 4 | clip4_real | hgb / phys | 10 | 0.570 ± 0.091 | 0.590 | +0.041 | 7 of 10 | 0.027 |
| 4 | phase3 | hgb / generic | 10 | 0.381 ± 0.093 | 0.336 |  |  |  |
| 4 | phase3 | hgb / generic+phys | 10 | 0.500 ± 0.112 | 0.371 |  |  |  |
| 4 | phase3 | hgb / phys | 10 | 0.528 ± 0.100 | 0.490 |  |  |  |
| 4 | phase3 | hgb-cv / generic | 10 | 0.382 ± 0.095 | 0.327 |  |  |  |
| 4 | phase3 | hgb-cv / phys | 10 | 0.524 ± 0.100 | 0.467 |  |  |  |
| 4 | phase3 | lr / generic | 10 | 0.355 ± 0.101 | 0.376 |  |  |  |
| 4 | phase3 | lr / generic+phys | 10 | 0.442 ± 0.108 | 0.363 |  |  |  |
| 4 | phase3 | lr / phys | 10 | 0.448 ± 0.098 | 0.360 |  |  |  |
| 4 | realonly | hgb / generic | 10 | 0.462 ± 0.085 | 0.493 | +0.081 | 9 of 10 | 0.004 |
| 4 | realonly | hgb / generic+cond | 10 | 0.483 ± 0.102 | 0.488 |  |  |  |
| 4 | realonly | hgb / generic+phys | 10 | 0.518 ± 0.113 | 0.411 | +0.018 | 7 of 10 | 0.084 |
| 4 | realonly | hgb / phys | 10 | 0.563 ± 0.093 | 0.560 | +0.035 | 9 of 10 | 0.010 |
| 4 | realonly | hgb / phys+cond | 10 | 0.550 ± 0.101 | 0.537 |  |  |  |
| 4 | realonly | hgb-bag / generic | 10 | 0.476 ± 0.081 | 0.499 |  |  |  |
| 4 | realonly | hgb-bag / phys | 10 | 0.545 ± 0.087 | 0.545 |  |  |  |
| 4 | realonly | lr / generic | 10 | 0.310 ± 0.110 | 0.379 | -0.044 | 4 of 10 | 0.322 |
| 4 | realonly | lr / generic+phys | 10 | 0.470 ± 0.119 | 0.437 | +0.028 | 8 of 10 | 0.131 |
| 4 | realonly | lr / phys | 10 | 0.452 ± 0.113 | 0.439 | +0.003 | 6 of 10 | 0.770 |
| 4 | rec_all | hgb / generic | 10 | 0.417 ± 0.068 | 0.325 | +0.036 | 6 of 10 | 0.275 |
| 4 | rec_all | hgb / generic+phys | 10 | 0.477 ± 0.111 | 0.399 | -0.023 | 6 of 10 | 0.922 |
| 4 | rec_all | hgb / phys | 10 | 0.477 ± 0.127 | 0.384 | -0.051 | 2 of 10 | 0.193 |
| 4 | rec_all | lr / generic | 10 | 0.301 ± 0.132 | 0.324 | -0.054 | 2 of 10 | 0.084 |
| 4 | rec_all | lr / generic+phys | 10 | 0.436 ± 0.139 | 0.561 | -0.007 | 5 of 10 | 0.846 |
| 4 | rec_all | lr / phys | 10 | 0.391 ± 0.136 | 0.359 | -0.057 | 2 of 10 | 0.160 |
| 4 | rec_real | hgb / generic | 10 | 0.468 ± 0.146 | 0.536 | +0.088 | 7 of 10 | 0.049 |
| 4 | rec_real | hgb / generic+phys | 10 | 0.409 ± 0.097 | 0.476 | -0.091 | 3 of 10 | 0.064 |
| 4 | rec_real | hgb / phys | 10 | 0.396 ± 0.155 | 0.347 | -0.132 | 1 of 10 | 0.037 |
