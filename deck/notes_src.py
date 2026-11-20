"""Speaker notes for the new/changed v4 slides. {fs_*} fields are filled from fs_metrics.json."""
import json
import sys
from pathlib import Path

FIG = Path(sys.argv[1])
fs = json.loads((FIG / "fs_metrics.json").read_text()) if (FIG / "fs_metrics.json").exists() else {}

N = {}
N["problem"] = """[1:10–1:45] 35 s
"First, the problem as the benchmark defines it. Each scenario has a source domain, which is labelled training data, and a target domain, which is shifted data used only for scoring. No target data is used in training, not even unlabelled, so strictly this is domain generalisation. The six scenarios shift different things: the load, the sampling rate, the fault size, the speed, artificial versus real damage, and, in the last row, the bearing itself.

The last column is my own audit. No scenario shares a recording, but three of them reuse every test bearing in training. So a high score there may just mean the model recognises the bearing."

If asked: the scenario definitions are pandas queries in pdm-bench's config/task/*.yaml (commit ca52408). CWRU loads 0, 1, 2, 3 hp correspond to about 1797, 1772, 1750, 1730 rpm. The audit is python -m lsbfd.splits; the report is results/splits/leakage_report.md. For Cross-FS, 13 bearings assumes the 12 kHz and 48 kHz drive-end recordings come from the same seeded bearings."""

N["old4"] = """[1:45–2:10] 25 s
"So I chose Cross-Bearing Instance. Four whole bearings are held out, one per class, shown in blue, and the model trains on the other 28. To show why leakage matters, I ran the same models with the test bearings also in training. The CNN jumps from 0.315 to 0.999. It is excellent at recognising bearings it has seen and poor at new ones: it learns bearing identity, not the fault."

If asked: in the leaky split, recordings 1, 6, 11 and 16 of every bearing are the test set, so the recordings are new but the bearings are not. Numbers are with inputs as released (results/reproduction.md)."""

N["task"] = """[2:10–2:40] 30 s
"To be precise about my task. I picked this scenario for two reasons: the audit showed the cross-load and cross-condition splits reuse test bearings, and this is the scenario where logistic regression beat the best deep model. The source is 28 bearings under all four operating conditions; the target is four unseen bearings, and I never use even their unlabelled windows.

For DANN, a domain is one training bearing, so there are 28 domains, and the target bearing is a new, unseen domain. That makes it domain generalisation, not adaptation.

One honest limitation, which came up in feedback: domain adaptation is usually studied across operating conditions or machines. Here both sides contain all four conditions from one rig, so only the bearing changes. A leakage-safe version of the condition and machine shifts is in future work."

If asked: the λ grid and the 4 validation bearings are in results/adaptation.md. In the test stage there are 28 domains, in the validation stage 24."""

N["waterfall0"] = """[2:40–3:20] 40 s
"Here is the whole project in order, on the benchmark split. Grey is the published logistic regression, 0.355, with the published CNN at 0.326 marked in red. My reproduction gave 0.388. Using vibration only, as the paper describes, lifted it to 0.508: that is the bar. The leaky split, at 0.999, is not a valid comparison, which is the point. The hatched red bars are the deep methods I tried and dropped: DANN, SimCLR and an envelope CNN with test-time adaptation, all well below 0.508. Envelope features with trees were slightly below on this split, 0.490, but I kept them because they win over ten folds. Then training on real damage only, and despiking, reached 0.590. The last two steps are exploratory."

If asked: deep model bars are means of 3 seeds (DANN 0.348 ± 0.013, SimCLR 0.408 ± 0.034, env-CNN + AdaBN per bearing 0.393 ± 0.009). DANN against its own CNN control (0.404 ± 0.027, same trainer, 20 epochs) is −0.056. The vibration-only CNN (50 epochs) is 0.436 ± 0.023. LR and the tree models are deterministic single runs."""

N["old12"] = """[7:20–7:45] 25 s
"To conclude: leakage reverses which model looks best. DANN, the textbook answer to domain shift, did not help with 28 bearings as domains. What generalised was physics in the input and cleaner training data, from 0.326 to 0.590 on the benchmark split and 0.570 over ten folds. Next, I want to test the shifts the field usually studies, condition and machine, without leakage."

Limitations, if asked: one dataset and one bearing per class in each fold; folds share bearings, so p-values are indicative; Phase 4 choices were made after looking at the data; deep runs used 20 epochs and 3 seeds."""

N["waterfall10"] = """[5:05–5:35] 30 s
"Because one bearing per class decides fold 0, I repeat every step on ten held-out bearing sets. The chain is cleaner here. Swapping generic features for envelope-order features adds 0.093 with the same logistic regression. Switching to boosted trees adds another 0.080. Together that is 0.174 over the generic baseline, better on 9 of 10 folds, p = 0.004. The two data-cleaning steps add 0.035 and 0.007. The table shows fold 0 next to the 10-fold mean for each step, so you can see why I rely on the 10-fold numbers."

If asked: the 0.355 here is logistic regression on our 11 generic features in our pipeline, not the paper's 0.355 (18 features, 3 channels). Folds are drawn with seed 2026 in lsbfd.novel.folds; Wilcoxon signed-rank, two-sided, paired by fold. Real-damage-only alone: +0.035, 9 of 10 folds, p = 0.010. Despiking on top: +0.007."""

N["why"] = """[6:30–7:20] 50 s
"It can look counterintuitive that logistic regression beats DANN, so here is why I think it happens.

One: DANN needs many domains, and they should be unrelated to the label. Here there are 28 bearings, as few as 2 per class, so bearing identity nearly decides the class, and removing it also removes fault information. {why_fs}

Two: the CNN can memorise a bearing, as the 0.999 shows. A linear model on 11 statistics cannot.

Three: the adversary's strength has to be tuned without seeing the target, and with four validation bearings that tuning picked the worst setting.

Four: the invariance we need is already known from physics: fault orders are the same on every bearing of this type. Putting that in the input gives invariance by construction.

This is not unusual: a large study, DomainBed, found that plain training matches or beats DANN-style methods when tuned fairly."

If asked: the within-class adversary was meant to avoid point one, and it does better than the plain bearing adversary on validation, but on test it still lost 0.056 to its control. DomainBed: Gulrajani & Lopez-Paz, ICLR 2021."""

N["future"] = """[7:45–8:10] 25 s
"For future work, the shifts the field usually studies. For operating conditions, a leakage-safe split holds out a condition and different bearings together, in a two-by-two design, so the condition shift and the bearing shift can be measured separately. For machines, train on Paderborn and test on CWRU or another rig, with fault orders computed from each bearing's geometry. With a few clearly different machines as domains, DANN is closer to the setting it was designed for, so it deserves another test there."

If asked: CWRU's drive-end bearing is a 6205-2RS (fan end 6203-2RS), sampled at 12 or 48 kHz; PU is a 6203 at 64 kHz. Envelope features are in shaft orders, so a new speed should not move the fault peaks; only the resonance bands may change."""

N["fs_maps"] = fs.get("notes_maps", "")
N["fs_bars"] = fs.get("notes_bars", "")
for k in list(N):
    if "{why_fs}" in N[k]:
        N[k] = N[k].replace("{why_fs}", fs.get("why_fs", ""))
(FIG / "notes.json").write_text(json.dumps(N, indent=1))
print("notes", len(N))
