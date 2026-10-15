# Literature Verification (Phase 0, Milestone M0)

Covers issue #1 (summaries of the four core references) and issue #2 (checking every number used in
`docs/DNNpresentation.pptm` against its source). Checked on 2026-10-15 against the papers' online full
texts. Page and table numbers refer to the versions linked below.

Status key: **Verified** (number and attribution match the source), **Verified with caveats** (number
is right but the slide's framing needs correcting), **Not verified** (source not confirmed yet).

## 1. Claim check (issue #2)

| # | Slide claim | Status | Where it actually comes from |
|---|---|---|---|
| C1 | Data leakage "affected 294 papers across 17 fields" (slides 1–2) | **Verified with caveats** | Kapoor & Narayanan (2023), *Patterns* 4, 100804, Abstract: "17 fields where leakage has been found, collectively affecting 294 papers." It is a lower bound compiled from 22 field reviews, not a systematic count. This paper is **not one of the four references on slide 5**; Vieira et al. quote the figure and cite it. |
| C2 | "Within this problem, >90% papers show splitting problems" (slides 1–2) | **Not verified; wording is wrong** | Kapoor & Narayanan do **not** say that >90% of the 294 papers have splitting problems. The ">90%" comes from Vieira et al. (2026), §3.3, which says earlier works (their refs [4], [6]) report "data splitting issues in over 90% of published papers" in **bearing fault diagnosis**. I could not open Vieira's reference list to confirm those primary sources yet (follow-up below). Vieira's own small survey (§3.3, Table 1) found that of 18 papers from 2025, 8 used random splits, 9 condition-wise splits and 1 did not say. |
| C3 | Knap, Jachymczyk & Lalik built a leakage-safe cross-domain benchmark on CWRU + PU (slide 2) | **Verified** | Knap et al. (2026), Abstract and §3.1: six scenarios, three on CWRU and three on PU. |
| C4 | "Their model showed an F1 of 0.326 whereas a simple logistic regression performed better with 0.355" (slide 2); "reproduce the 0.326 baseline" (slide 3) | **Verified with caveats** | Both numbers are real, but they come from **one of six scenarios**, PU Cross-Bearing Instance. See caveats below. |
| C5 | F1 is "harmonic mean of precision and recall" (slide 2) | **Verified with caveats** | The benchmark reports **macro-F1**: per-class F1 averaged over classes (Knap et al. §2.4, Eqs. 3–4). The slide should say macro-F1. |
| C6 | CWRU: 12 kHz / 48 kHz, four loads 0–3 HP, multiple defect sizes (slide 2) | **Verified** | Consistent with Knap et al. Table 1 (1797/1772/1750/1730 rpm loads; 12 vs 48 kHz; defect families 0.007/0.014/0.021 in). |
| C7 | PU: multiple operating conditions (speed, torque, radial force) (slide 2) | **Verified** | Knap et al. Table 1: conditions such as N15 M07 F10 (speed, load torque, radial force). |
| C8 | References on slide 5 (authors, titles, venues, years) | **Verified** | See §2. Vieira et al. is now published in MSSP (DOI below). |

### Caveats on 0.326 vs 0.355 (C4)

1. **Scenario.** Both numbers are for **PU Cross-Bearing Instance** (target bearings K001, KA30, KI21,
   KB23 held out; 4 classes). Source: Knap et al. Table 3 (ML) and Table 4 (DL), p. 6, and §4.4.
2. **It is the exception, not the rule.** Knap et al. §4.4 call it "the only scenario in which the final
   ML result exceeds the final DL result." Deep models win the other five scenarios (Table 4 vs Table 3).
3. **Not like-for-like.** 0.326 ± 0.008 is the **1D CNN mean over seeds 41, 42, 43** (Table 4). 0.355 is
   a **single run** of logistic regression (Table 3; §5.1 says "the final ML results are based on single
   benchmark runs"). In the single-run heatmap (Figure 1, p. 5) the 1D CNN appears to score 0.343 on the
   same scenario (my reading of the figure; cells are not labelled in the extracted text). So the true
   DL–LR gap is between about 0.012 and 0.029, with no spread reported for LR.
4. **It is PU only.** The slide places the result "on CWRU + PU". The benchmark spans both, but this gap is
   a PU result.
5. **The other hard scenario.** CWRU Cross-FS (train 12 kHz, test 48 kHz) is also far from solved:
   best ML 0.233 (XGBoost, Table 3), best DL 0.363 ± 0.013 (STFT CNN, Table 4).

### Suggested slide fixes

- Slides 1–2: cite Kapoor & Narayanan (2023) for 294 / 17, and change ">90%" to "Reviews of bearing
  fault diagnosis report data-splitting issues in over 90% of papers (cited in Vieira et al., 2026)",
  once the primary source is confirmed.
- Slide 2: "On the PU cross-bearing scenario, their best deep model reached macro-F1 0.326 (mean of 3
  seeds); logistic regression reached 0.355 (single run). It is the only one of six scenarios where the
  simple model wins."

## 2. Summaries of the four core references (issue #1)

### 2.1 Vieira, Bauler, Rosa & Silva (2026). Towards a more realistic evaluation of machine learning models for bearing fault diagnosis. *Mechanical Systems and Signal Processing*.

- **Publication:** MSSP 2026, DOI [10.1016/j.ymssp.2026.114640](https://doi.org/10.1016/j.ymssp.2026.114640);
  preprint [arXiv:2509.22267](https://arxiv.org/abs/2509.22267) (v5, 2026-07-06). Code:
  <https://github.com/gama-ufsc/bearing-data-leakage>.
- **Problem:** bearing-level leakage (the same physical bearing in train and test) inflates reported
  accuracy. A toy example (§3.2) shows leaky test sets beating the theoretical maximum accuracy of 90.30%.
- **Method:** strict **bearing-wise** splits; fault diagnosis recast as **multi-label** binary
  classification; prevalence-independent metric (**macro AUROC**); 100 random bearing-wise splits per
  dataset.
- **Datasets:** CWRU, PU (Paderborn), UORED-VAFCLS (Ottawa).
- **Results (macro AUROC, mean ± SD over 100 splits):** UORED best 93.12 ± 4.26% (WDCNN, frequency input,
  Table 8); PU best 79.56 ± 13.07% (WDCNN, envelope spectrum, Table 9); CWRU best 85.06 ± 8.92% (Random
  Forest on time + envelope features, Table 10), where all WDCNN variants (63–75%) lost to RF.
- **Key finding:** performance rises with the **number of distinct training bearings** even at fixed
  training size (Figs. 10, 12, 14; PU 0.686 to 0.853). Bearing diversity, not model capacity, is the
  bottleneck on PU.
- **Relevance:** supports our split rules; warns that CWRU favours classic features over CNNs; uses a
  different metric (macro AUROC, multi-label) from Knap et al., so the two papers' numbers are not
  comparable.

### 2.2 Knap, Jachymczyk & Lalik (2026). Leakage-Safe, Reproducible Benchmarking for Vibration-Based Fault Diagnosis. *PHM Society European Conference 2026*.

- **Publication:** [papers.phmsociety.org/…/4924](https://papers.phmsociety.org/index.php/phme/article/view/4924)
  (volume/DOI not printed in the PDF). Code: <https://github.com/1Sensor/pdm-bench> (public, MIT; see issue #4).
- **Protocol:** splits are queries over **whole recordings**, and each recording is indivisible, so all
  its windows go to one side (§2.1–2.2). Preprocessing and hyperparameter search use source training data
  only, with source-side validation.
- **Scenarios (Table 1):** CWRU Cross-Load, Cross-FS, Cross-Fault Instance; PU Cross-Operating Condition,
  Cross-Damage Provenance, Cross-Bearing Instance.
- **Windowing (§3.2):** CWRU 2048 samples, PU 8192; test overlap 50%; training overlap 75% (long) /
  50% / 10%.
- **Metric:** macro-F1 (§2.4), with balanced accuracy and accuracy as support.
- **Models:** ML (LR, XGBoost, others) as single runs; DL (MLP, 1D CNN, STFT CNN) rerun with seeds
  {41, 42, 43}.
- **Results:** easy for cross-load (1.000), hard for CWRU Cross-FS (≤ 0.363) and PU Cross-Bearing
  Instance (≤ 0.355).
- **Relevance:** this is the benchmark we reproduce in Phase 1; the PU Cross-Bearing Instance gap is
  our target.

### 2.3 Chen, Yang, Xue, Huang, Ferrero & Wang (2023). Deep Transfer Learning for Bearing Fault Diagnosis: A Systematic Review Since 2016. *IEEE Transactions on Instrumentation and Measurement* 72, 3508221.

- **Publication:** DOI [10.1109/TIM.2023.3244237](https://doi.org/10.1109/TIM.2023.3244237); open
  manuscripts at Brunel and Liverpool repositories.
- **Content:** a review of deep transfer learning (DTL) for bearing diagnosis since 2016, motivated by the
  train/test distribution shift between operating conditions and machines. Proposes a taxonomy based on
  the **target domain**: whether target labels are available, whether the target is a different machine,
  and whether the fault sets differ. Covers the full DTL life cycle and lists open challenges.
- **Relevance:** places our work (labelled source bearings, unseen target bearings of the same machine
  type) in the field's taxonomy and gives the menu of adaptation methods for Phase 2.
- **Not yet checked:** the review's internal tables (I read the abstract and metadata, not the full
  text). Nothing on the slides depends on them.

### 2.4 Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML 2015*, PMLR 37.

- **Publication:** [proceedings.mlr.press/v37/ganin15](https://proceedings.mlr.press/v37/ganin15.html);
  [arXiv:1409.7495](https://arxiv.org/abs/1409.7495).
- **Method (DANN):** a feature extractor feeds both a label classifier and a domain classifier; a
  **gradient reversal layer** flips the domain classifier's gradient so features become discriminative for
  the task but invariant across domains. Trained with ordinary backpropagation using labelled source and
  **unlabelled target** data.
- **Results:** strong adaptation under large domain shifts in image classification, state of the art on
  Office at the time.
- **Relevance:** the adversarial method for Phase 2. Caution: DANN needs unlabelled target data during
  training. Knap et al.'s protocol does not use target data in training, so a DANN run is **unsupervised
  domain adaptation**, not domain generalization, and must be reported as such (PLAN.md Phase 2). A
  domain-generalization variant would use the source bearings themselves as the domains.

## 3. Implications for the plan

- The success target "beat 0.355" applies to **PU Cross-Bearing Instance** specifically; PLAN.md now says so.
- Because LR's 0.355 is a single run, Phase 1 should rerun LR over the same seeds so the comparison has
  a spread on both sides.
- CWRU Cross-FS is a second worthwhile target (best 0.363).

## 4. Open follow-ups

- Confirm the primary sources behind the ">90%" figure (Vieira et al. refs [4], [6]); the reference list
  did not load through the tools available here.
- Confirm the Figure 1 single-run value (0.343) by reading the PDF figure directly.
- Issues #3 and #4: cross-checked against the pdm-bench code in
  [benchmark-protocol.md](benchmark-protocol.md).
