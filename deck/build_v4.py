"""Build LSBFD-final-presentation-v4.pptx from v3: new slides in the v3 style, notes, reorder."""
import copy
import json
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

SRC, DST, FIG = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NOTES = json.loads((FIG / "notes.json").read_text())
prs = Presentation(SRC)
old = list(prs.slides)  # v3 slides 1..20 -> old[0..19]
CONTENT = [l for l in prs.slide_layouts if l.name == "CONTENT"][0]
TITLE = next(x for x in old[7].shapes if x.name == "Text 0")
SLDNUM = next(s for s in old[7].shapes if s.name.startswith("Slide Number"))

L, W = 548640, 11064240  # content left edge and width (as in v3)
TX1, TX2, BG2, ACC6 = (MSO_THEME_COLOR.TEXT_1, MSO_THEME_COLOR.TEXT_2,
                       MSO_THEME_COLOR.BACKGROUND_2, MSO_THEME_COLOR.ACCENT_6)
RED, GREEN, BLUE, MUTED = (RGBColor(0xC0, 0x39, 0x2B), RGBColor(0x2F, 0x7D, 0x4F),
                           RGBColor(0x1F, 0x5F, 0xAD), RGBColor(0x52, 0x60, 0x6D))


def new_slide(title):
    s = prs.slides.add_slide(CONTENT)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    el = copy.deepcopy(TITLE._element)
    s.shapes._spTree.append(el)
    t = s.shapes[-1]
    t.text_frame.paragraphs[0].runs[0].text = title
    s.shapes._spTree.append(copy.deepcopy(SLDNUM._element))
    return s


def text(s, x, y, w, h, paras, size=16, color=TX1, bold=False, bullets=False, align=None,
         anchor=MSO_ANCHOR.TOP, space=6):
    """paras: list of str or list of [(text, {bold, color, italic}) runs]."""
    tb = s.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(space)
        if align:
            para.alignment = align
        if bullets:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", "203200")
            pPr.set("indent", "-203200")
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
        runs = p if isinstance(p, list) else [(p, {})]
        for t, st in runs:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.italic = st.get("italic", False)
            c = st.get("color", color)
            if isinstance(c, RGBColor):
                r.font.color.rgb = c
            else:
                r.font.color.theme_color = c
    return tb


def panel(s, x, y, w, h, fill=BG2):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    shp.fill.solid()
    shp.fill.fore_color.theme_color = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def card(s, x, y, w, h, head, body, fill=BG2, size=15, bullets=False):
    panel(s, x, y, w, h, fill)
    text(s, x + 182880, y + 137160, w - 365760, 380000, [head], size=17, color=TX2, bold=True)
    text(s, x + 182880, y + 560000, w - 365760, h - 640000, body, size=size, bullets=bullets)


def table(s, x, y, w, rows, widths, size=13, row_h=330000, bold_last=False, highlight=None):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Emu(x), Emu(y), Emu(w),
                             Emu(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        tblPr.remove(sid)
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Emu(int(w * cw / sum(widths)))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Emu(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Emu(64008)
            c.margin_top = c.margin_bottom = Emu(45720)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            tf.text = ""
            runs = val if isinstance(val, list) else [(val, {})]
            p = tf.paragraphs[0]
            for t, st in runs:
                r = p.add_run()
                r.text = t
                r.font.size = Pt(st.get("size", size))
                r.font.bold = (i == 0) or st.get("bold", False) or (highlight == i)
                if i == 0:
                    r.font.color.theme_color = MSO_THEME_COLOR.BACKGROUND_1
                elif "color" in st:
                    r.font.color.rgb = st["color"]
                else:
                    r.font.color.theme_color = TX1
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.theme_color = TX2
            elif highlight == i:
                c.fill.fore_color.theme_color = ACC6
            else:
                c.fill.fore_color.theme_color = MSO_THEME_COLOR.BACKGROUND_1
            tcPr = c._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(side), w="9525")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr"), val="D3D8DE")
            # borders must precede the fill in tcPr
            fill = tcPr.find(qn("a:solidFill"))
            tcPr.remove(fill)
            tcPr.append(fill)
    return tbl


def picture(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Emu(x), Emu(y), Emu(w) if w else None,
                                Emu(h) if h else None)


def notes(s, key):
    s.notes_slide.notes_text_frame.text = NOTES[key]


B = {"bold": True}
made = {}

# ------------------------------------------------------------ problem definition (benchmark)
s = new_slide("Problem definition: how the benchmark sets up domain shift")
panel(s, L, 1188720, W, 1005840, ACC6)
text(s, L + 182880, 1280160, W - 365760, 860000, [
    [("Source domain ", B), ("= labelled recordings used for training.   ", {}),
     ("Target domain ", B), ("= a shifted set of recordings used only for scoring.", {})],
    [("No target data, labelled or unlabelled, is used for training or model choice, so every "
      "scenario is ", {}), ("domain generalisation", B),
     (". Split by recording first, then cut windows; score = macro-F1 on the target.", {})],
], size=15, space=4)
rows = [["pdm-bench scenario (Knap et al., 2026)", "Source (train)", "Target (test)",
         "Shift tested", "Test bearings seen in training?"],
        ["CWRU Cross-Load", "loads 0–2 hp (1797–1750 rpm)", "3 hp (1730 rpm)", "load / speed",
         [("Yes, all 16", {"color": RED, "bold": True})]],
        ["CWRU Cross-FS", "12 kHz drive end", "48 kHz drive end", "sampling rate",
         [("Yes, 13", {"color": RED, "bold": True})]],
        ["CWRU Cross-Fault Instance", "0.007 and 0.014 in faults", "0.021 in faults",
         "fault size", "No"],
        ["PU Cross-Operating Condition", "N15_M01_F10, N15_M07_F04, N15_M07_F10",
         "N09_M07_F10 (900 rpm)", "speed", [("Yes, all 32", {"color": RED, "bold": True})]],
        ["PU Cross-Damage Provenance", "K002–K006 + artificial damage", "K001 + real damage",
         "artificial → real", "No"],
        ["PU Cross-Bearing Instance (mine)", "28 bearings, all 4 conditions",
         "K001, KA30, KI21, KB23", "new bearing", "No"]]
table(s, L, 2340000, W, rows, [3.3, 3.4, 2.6, 1.7, 2.2], size=13, row_h=455000, highlight=6)
text(s, L, 5650000, W, 500000, [
    "No scenario shares a recording between train and test. My audit (lsbfd.splits) found three "
    "that reuse the test bearings, so they measure a new load, sampling rate or speed on a "
    "bearing the model has already seen."], size=13, color=MUTED)
notes(s, "problem")
made["problem"] = s

# ------------------------------------------------------------ my task and domains
s = new_slide("My task: Cross-Bearing Instance, and what a domain is")
cw, gap, top, h = 3566160, 182880, 1234440, 3200400
card(s, L, top, cw, h, "Why this scenario", [
    "The leakage audit: Cross-Load, Cross-FS and Cross-Operating Condition test on bearings already "
    "in training, so a high score there can come from recognising the bearing.",
    "Cross-Bearing holds out whole bearings, and it is where LR (0.355) beat the best deep "
    "model (0.326)."], size=14, bullets=True)
card(s, L + cw + gap, top, cw, h, "Source and target", [
    [("Source: ", B), ("28 bearings, all 4 operating conditions, with labels.", {})],
    [("Target: ", B), ("K001, KA30, KI21, KB23 (one per class), never seen; not even their "
                       "unlabelled windows are used.", {})],
    "A target class is one bearing, so 10 rotated hold-outs back up the main claim."],
    size=14, bullets=True)
card(s, L + 2 * (cw + gap), top, cw, h, "Domains in my DANN", [
    [("Each of the 28 training bearings is a domain ", B),
     ("(the label is the bearing ID).", {})],
    "The adversary only separates bearings of the same class, so it removes identity, not the "
    "fault.",
    "Domain generalisation: λ chosen on 4 training bearings (K006, KA22, KB27, KI18)."],
    size=14, bullets=True)
panel(s, L, 4617720, W, 900000, ACC6)
text(s, L + 182880, 4709160, W - 365760, 800000, [
    [("What this task does not test. ", {"bold": True, "color": TX2}),
     ("Most domain-adaptation papers shift the operating condition or the machine. Here both "
      "sides contain all 4 conditions from one test rig, so only the bearing changes. A "
      "leakage-safe cross-condition and cross-machine split is future work (slide {N_future}).", {})]],
    size=15)
notes(s, "task")
made["task"] = s

# ------------------------------------------------------------ chronological waterfall (fold 0)
s = new_slide("What I tried, in order: effect of each step on the benchmark split")
picture(s, FIG / "waterfall_fold0.png", L, 1150000, w=W)
text(s, L, 5560000, W, 700000, [
    [("Fold 0 only: ", B), ("macro-F1 on the 4 benchmark test bearings. Deep models are means "
      "of 3 seeds; LR and trees are single deterministic runs. Dropped steps are compared with LR "
      "0.508. Envelope + trees was kept despite −0.018 here because it wins over 10 folds "
      "(slide {N_w10}).", {})]], size=13, color=MUTED)
notes(s, "waterfall0")
made["waterfall0"] = s

# ------------------------------------------------------------ 10-fold waterfall
s = new_slide("The same chain over 10 held-out bearing sets")
picture(s, FIG / "waterfall_10fold.png", L, 1150000, h=4300000)
x2 = L + 7050000
rows = [["Step", "Fold 0", "10 folds"],
        ["LR, 11 generic features", "0.376", "0.355 ± 0.101"],
        ["LR, envelope features", "0.360", "0.448 ± 0.098"],
        ["Trees, envelope features", "0.490", "0.528 ± 0.100"],
        ["+ real damage only", "0.560", "0.563 ± 0.093"],
        ["+ despiking", "0.590", "0.570 ± 0.091"]]
table(s, x2, 1250000, L + W - x2, rows, [2.6, 1.0, 1.6], size=13, row_h=380000, highlight=5)
text(s, x2, 3700000, L + W - x2, 2000000, [
    "Fold 0 alone is noisy: one target bearing per class, so one bearing flipping moves "
    "macro-F1 by about 0.1.",
    "Folds 1–9 hold out one random real-damage bearing per class plus one healthy bearing "
    "(fixed seed). Same folds for every method; paired Wilcoxon test.",
    "Steps after trees came from a data audit, so they are exploratory."],
    size=13, bullets=True, color=TX1)
text(s, L, 5600000, 6800000, 500000, [
    "LR on our 11 generic features (0.355) is not the paper's LR (0.355 by coincidence; 18 "
    "features, 3 channels)."], size=12, color=MUTED)
notes(s, "waterfall10")
made["waterfall10"] = s

# ------------------------------------------------------------ feature space slides
FS = json.loads((FIG / "fs_metrics.json").read_text()) if (FIG / "fs_metrics.json").exists() else None
if FS:
    s = new_slide("Feature space: how each representation arranges the windows")
    picture(s, FIG / "fs_maps.png", L, 1100000, w=W)
    text(s, L, 6020000, W, 400000, [FS["maps_caption"]], size=12, color=MUTED)
    notes(s, "fs_maps")
    made["fs_maps"] = s

    s = new_slide("Feature space, measured: compactness, bearing identity, transfer")
    picture(s, FIG / "fs_bars.png", L, 1100000, w=7200000)
    xr = L + 7400000
    card(s, xr, 1150000, L + W - xr, 4500000, "What the numbers say", FS["takeaways"], size=14,
         bullets=True)
    text(s, L, 5800000, 7200000, 600000, [FS["bars_caption"]], size=12, color=MUTED)
    notes(s, "fs_bars")
    made["fs_bars"] = s

# ------------------------------------------------------------ why LR / trees beat DANN
s = new_slide("Why do LR and trees beat DANN here?")
why = json.loads((FIG / "why.json").read_text())
cw2, ch = 5440680, 1965960
for k, (head, body) in enumerate(why["cards"]):
    x = L + (k % 2) * (cw2 + 182880)
    y = 1188720 + (k // 2) * (ch + 164592)
    card(s, x, y, cw2, ch, head, [body], size=14)
panel(s, L, 5420000, W, 820000, ACC6)
text(s, L + 182880, 5490000, W - 365760, 700000, [why["footer"]], size=14)
notes(s, "why")
made["why"] = s

# ------------------------------------------------------------ future work: leakage-safe shifts
s = new_slide("Future work: condition and machine shift, without leakage")
card(s, L, 1234440, 5440680, 3474720, "Leakage-safe cross-condition (PU)", [
    "Hold out a condition and bearings together: train on bearing set A at 3 conditions, "
    "test on a disjoint set B at the 4th (e.g. 900 rpm).",
    "A 2 × 2 design (seen / unseen bearing × seen / unseen condition) separates the two "
    "shifts, which today's Cross-Operating Condition split mixes up.",
    "Envelope features are in shaft orders, so they should carry over to a new speed; this "
    "tests that claim."], size=14, bullets=True)
card(s, L + 5440680 + 182880, 1234440, 5440680, 3474720, "Cross-machine", [
    "Train on PU (6203 bearing, 64 kHz), test on CWRU (6205 drive end, 12 kHz) or another rig: "
    "new machine, bearing type, sensor and sampling rate.",
    "Needs a shared label set, resampling, and fault orders computed from each bearing's "
    "geometry. Test bearings are disjoint by construction.",
    "With a few, clearly different domains, DANN or UDA is closer to the setting it was "
    "designed for."], size=14, bullets=True)
panel(s, L, 4892040, W, 1050000, ACC6)
text(s, L + 182880, 4983480, W - 365760, 900000, [
    [("Also: ", {"bold": True, "color": TX2}),
     ("retrain the CNN and DANN on the cleaned training data, and build the envelope-order front "
      "end into the network, then repeat the feature-space measurements above.", {})]], size=15)
notes(s, "future")
made["future"] = s

# ------------------------------------------------------------ edits to v3 slides
for sh in old[11].shapes:  # conclusion: point the next-steps bullet at the new future-work slide
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if r.text.startswith("Test on more datasets"):
                    r.text = ("Leakage-safe cross-condition and cross-machine splits (PU → CWRU), "
                              "where condition and machine change too.")
for sh in old[13].shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if r.text.startswith("Backup slides"):
                    r.text = ("Backup slides: GroupKFold, fault physics, windowing, SimCLR, "
                              "data audit")
for i, sh in [(12, None)]:  # references: add DomainBed
    ref = old[12]
    boxes = [x for x in ref.shapes if x.has_text_frame and "[8]" in x.text_frame.text]
    if boxes:
        tf = boxes[0].text_frame
        last = tf.paragraphs[-1]
        newp = copy.deepcopy(last._p)
        last._p.addnext(newp)
        from pptx.text.text import _Paragraph
        p = _Paragraph(newp, tf)
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
        p.runs[0].text = ("[9]  Gulrajani & Lopez-Paz, \"In Search of Lost Domain Generalization\", "
                          "ICLR, 2021.")
for k, key in [(3, "old4"), (11, "old12")]:
    if key in NOTES:
        old[k].notes_slide.notes_text_frame.text = NOTES[key]

# ------------------------------------------------------------ order (drop v3 backup 20)
order = [old[0], old[1], old[2], made["problem"], old[3], made["task"], made["waterfall0"],
         old[4], old[5], old[6], old[7], old[8], old[9], old[10], made["waterfall10"]]
order += [made[k] for k in ("fs_maps", "fs_bars") if k in made]
order += [made["why"], old[11], made["future"], old[12], old[13]] + old[14:19]
sldIdLst = prs.slides._sldIdLst
ids = {sl.slide_id: el for el in sldIdLst for sl in [None] if False}
by_id = {el.get("id"): el for el in sldIdLst}
drop = old[19]
rid_drop = [el for el in sldIdLst if int(el.get("id")) == drop.slide_id][0]
prs.part.drop_rel(rid_drop.get(qn("r:id")))
sldIdLst.remove(rid_drop)
els = [by_id[str(sl.slide_id)] for sl in order]
for el in list(sldIdLst):
    sldIdLst.remove(el)
for el in els:
    sldIdLst.append(el)
# ------------------------------------------------------------ slide refs, timings, notes
import re
pos = {id(sl): i + 1 for i, sl in enumerate(order)}
refs = {"{N_w10}": pos[id(made["waterfall10"])], "{N_future}": pos[id(made["future"])]}
for sl in order:
    for sh in sl.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    for k, v in refs.items():
                        if k in r.text:
                            r.text = r.text.replace(k, str(v))
DUR = {id(old[0]): 10, id(old[1]): 25, id(old[2]): 35, id(made["problem"]): 35, id(old[3]): 25,
       id(made["task"]): 30, id(made["waterfall0"]): 40, id(old[4]): 15, id(old[5]): 20,
       id(old[6]): 25, id(old[7]): 20, id(old[8]): 30, id(old[10]): 25,
       id(made["waterfall10"]): 25, id(made["why"]): 45, id(old[11]): 20, id(made["future"]): 25}
for k, d in (("fs_maps", 40), ("fs_bars", 30)):
    if k in made:
        DUR[id(made[k])] = d
t = 0
fmt = lambda x: f"{x // 60}:{x % 60:02d}"
TIMING = {}
for sl in order:
    body = sl.notes_slide.notes_text_frame.text
    body = re.sub(r"^\s*\[[^\]]*\][ \t]*(\d+ s)?[ \t]*\n?", "", body, count=1).lstrip("\n")
    if id(sl) in DUR:
        d = DUR[id(sl)]
        head = f"[{fmt(t)}–{fmt(t + d)}] {d} s"
        t += d
    elif sl is old[9]:
        head = "[optional, about 20 s]"
    else:
        head = None
    TIMING[pos[id(sl)]] = head
    sl.notes_slide.notes_text_frame.text = (head + "\n" + body) if head else body
(FIG / "timing.json").write_text(json.dumps(TIMING, indent=1))
print("talk length", fmt(t))
prs.save(DST)
print("slides:", len(prs.slides))
