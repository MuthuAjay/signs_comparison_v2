# Detector defects and fixes

Everything downstream inherits detector recall — the digital/scanned counts in
`detect_signs.py`, and later the duplicate and similarity stages. A signature
the detector misses is silently absent from every number the pipeline produces
and cannot be recovered by any later stage. So this is the right place to spend
effort before scaling up.

Companion to [PLAN.md](PLAN.md). Model: `detection/models/model_best.pth`
(epoch 8, val_loss 0.213). Dataset: `/home/eyadmin/Documents/Datasets/signs_dataset`
— 2,765 images, 8,022 boxes, `train.csv` / `test.csv` split already present.

---

## Summary

| # | Defect | Impact | Fix cost |
|---|---|---|---|
| 1 | Anchor aspect ratios don't match signature shape | **High** — recall + fragmented boxes | retrain |
| 2 | 512x512 squash destroys aspect ratio, crushes boxes to ~19px tall | **High** — small-object recall | retrain |
| 3 | `date` boxes silently trained as `redaction`; class 3 misnamed `stamp` | Medium — 7% of labels corrupt | retrain |
| 4 | No mAP / recall ever measured — val loss only | **Blocking** — no error rate to quote | 1h, no retrain |
| 5 | NMS cannot suppress nested boxes | Over-counting | worked around |

Do **#4 first**. It needs no retraining and no labelling, and its result decides
whether #1-#3 are worth doing.

---

## 1. Anchor aspect ratios don't match signatures

Measured over all 8,022 training boxes:

| | |
|---|---|
| box aspect ratio (w/h), median | **4.5 : 1** |
| p10 / p90 | 1.6 : 1 / **9.7 : 1** |
| wider than 4:1 | 58% |
| wider than 8:1 | 20% |

Torchvision's `aspect_ratios` are **h/w**. The default `(0.5, 1.0, 2.0)` spans
2:1-wide to 2:1-tall. The median signature needs h/w = **0.22**, p90 needs
**0.10**. Nothing in the default set is close, so every box is regressed from a
badly mismatched prior. This is the most likely cause of the fragmented,
nested detections that `merge_nested()` in `detect_signs.py` currently mops up.

Anchor *sizes* are fine and should not be changed: median box is 112x19px at
training resolution, geometric mean 46, sitting between the 32 and 64 anchors.
**The problem is specifically the aspect ratios.**

```python
from torchvision.models.detection.anchor_utils import AnchorGenerator

# FPN has 5 levels -- one sizes tuple and one aspect_ratios tuple per level.
anchor_gen = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.1, 0.2, 0.35, 0.5, 1.0),) * 5,
)
model = fasterrcnn_resnet50_fpn(weights=None, rpn_anchor_generator=anchor_gen)
```

Passing `rpn_anchor_generator` alone is enough — `FasterRCNN.__init__` builds the
RPN head from `num_anchors_per_location()`. Note this changes the RPN head shape,
so **existing checkpoints cannot be loaded**; retrain from the pretrained
backbone.

---

## 2. The 512x512 squash

`detection_training.py:102`

```python
image = image.resize(self.target_size[::-1])   # (512, 512) -- aspect destroyed
```

Source images are 2560x3300, 3400x4400, 1200x1575, 1728x2292 — all portrait,
~0.78 aspect. Squashing to square compresses vertically by ~1.29x, and shrinks
boxes to:

| box height at 512x512 | |
|---|---|
| median | **19 px** |
| p10 / p90 | 9 / 41 px |
| under 16px tall | 35.6% |
| under 32px tall | **81.3%** |

After Faster R-CNN's internal 512->800 upscale the median is 30px, but the
detail was already thrown away at 512.

**Fix:** drop the pre-resize entirely and let `GeneralizedRCNNTransform` size the
image, aspect preserved:

- remove the `image.resize(...)` line and scale boxes by the *actual* image size
  (the CSV stores normalized `[x, y, w, h]`, so this is a straight multiply)
- build the model with `min_size=1000, max_size=1400`
- `collate_fn` at `detection_training.py:178` already returns lists, so variable
  image sizes need no further change

At `min_size=1000` a 2560x3300 page becomes 1000x1289 and the median box is
~284x30px at true aspect — roughly 1.6x the vertical detail, with the shape
distortion gone. Batch 8 at that size is trivial on the 96 GB card.

---

## 3. Label corruption

`categories.csv` is:

```
1 signature   2 initials   3 redaction   4 date
```

Two bugs:

**a.** `detection_training.py:142`

```python
category_id = max(1, min(category_id, self.num_classes - 1))
```

With `num_classes=4` this clamps to 3, so all **570 `date` boxes were trained as
`redaction`** — 7% of the dataset, silently, behind a warning.

**b.** `inference.py:60` declares class 3 as `"stamp"`. **This dataset has no
stamp class.** Any output labelled "stamp" is actually redaction-or-date.

**Fix:** `num_classes=5`, remove the clamp (drop or fix out-of-range rows
instead), correct the names in `inference.py`. Counting only uses class 1 so the
current digital/scanned numbers are unaffected, but the corrupt signal
indirectly costs class-1 quality.

`ms_wi_comparisons/detect_signs.py` already carries the corrected map as
`redaction_or_date`.

---

## 4. No detection metrics — do this first

`detection_training.py:416` `evaluate()` returns **val loss only**. There is no
mAP, no precision, no recall, for any checkpoint. Nothing in the pipeline can
quote an error rate.

`test.csv` already exists, so this needs no labelling:

```bash
pip install torchmetrics          # not currently installed
```

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
for images, targets in test_loader:
    preds = model([i.to(device) for i in images])
    metric.update(preds, targets)
print(metric.compute())     # map, map_50, map_75, mar_100, per-class
```

The number that matters for counting is **recall on class 1 at IoU 0.5**
(`mar_100` / per-class recall), not mAP. Record it for the current checkpoint
before changing anything — that is the baseline every later change is judged
against.

### RESULT — baseline measured 2026-07-28

`eval_detector.py` on `model_best.pth`, held-out split (553 images / 1,640 boxes,
reproduced via `detection_training.load_annotations_and_split`, seed 42).
Full output in `eval_output/eval_report.json`.

| metric | |
|---|---|
| mAP@[.5:.95] | 0.500 |
| mAP@0.50 | 0.881 |
| mAP@0.75 | 0.507 |
| by size | small 0.324 · medium 0.492 · large 0.675 |
| signature mAP / mAR@100 | 0.558 / 0.636 |
| **signature recall @ IoU 0.50, score>=0** | **0.969** (ceiling) |

Signature class at IoU 0.50, by confidence threshold — the operating point the
counts actually run at:

| conf | recall | precision | TP | FP | FN | **pred/true** |
|---|---|---|---|---|---|---|
| 0.3 | 0.957 | 0.794 | 885 | 230 | 40 | 1.21 |
| **0.5** | **0.952** | **0.828** | 881 | 183 | 44 | **1.15** |
| 0.7 | 0.951 | 0.864 | 880 | 139 | 45 | 1.10 |
| **0.9** | **0.939** | **0.922** | 869 | 74 | 56 | **1.02** |

**Recall is not the problem — precision is.** At the current default of conf 0.5
the detector finds 95% of signatures but **over-counts by 15%**, and the excess
is false positives, not missed marks.

**Raising conf to 0.9 costs 1.3pp of recall and takes the count ratio from 1.15
to 1.02.** For a deliverable that is entirely counts, that is a far larger win
than retraining, and it is free. Recommended change to `detect_signs.py`
(currently `--conf` default 0.5).

Caveat: pred/true 1.02 is an *aggregate* ratio, where 74 FP and 56 FN partly
cancel. Per-document counts still carry both error types; roughly 14% of
signatures are involved in an error at conf 0.9.

**Verdict on steps 2-3 (retrain): not justified for the counting deliverable.**
The anchor and resize fixes would mainly improve *localization* — mAP@0.75 is
0.507 against mAP@0.50 of 0.881, so boxes are found but loosely fitted — and
loose boxes are adequate for intersecting with PDF object bboxes at the 0.50
coverage threshold `detect_signs.py` uses. Revisit only if the corpus check
below disagrees, or if a later stage needs tight crops.

### Second, free source of ground truth

`pdf_probe.py` already records the exact bbox of every placed image object in
the real corpus. Any `placed_image` the detector fails to fire on is a measured
recall failure on a real document, at **zero labelling cost**. Run this on the
corpus as a reality check — the test split is NIST forms, which may not resemble
the case documents.

Scanned wet ink has no free ground truth. Hand-label ~100 scanned pages if the
scanned count needs an error bar.

---

## 5. Nested-box over-counting (worked around)

Torchvision applies NMS internally, but NMS uses IoU, which **cannot** suppress a
small box nested inside a large one: a box fully contained in another twice its
area scores IoU exactly 0.5, at the default threshold. Observed live — three
boxes on one signature, one fully containing another at IoU 0.487.

Worked around in `detect_signs.py` `merge_nested()`, suppressing by
intersection-over-smaller-area at 0.70 (1.0 for any contained box, regardless of
size difference).

The 0.70 threshold is a guess. It is deliberately conservative: lowering it would
merge two people who signed side by side, and under-counting distinct signers is
the worse error here. **Re-tune after the retrain**, against measured data —
fixing the anchors (#1) should reduce fragmentation at source and may make this
workaround close to inert.

---

## Order of work

1. **Measure the current model** — add `MeanAveragePrecision` on `test.csv`,
   record class-1 recall @ IoU 0.5. *1h, no retrain.* Decides whether the rest
   is worth doing.
2. **Retrain** with fixed anchors (#1) + aspect-preserving resize (#2) +
   corrected labels (#3). *30-60 min per run on GPU 1.*
3. **Compare on the same split.** Keep whichever wins — do not assume the
   retrain is better, verify it.
4. **Re-tune the IoS threshold** (#5) against measured data.
5. **Re-run the corpus counts** with the winning model.

Run everything on `CUDA_VISIBLE_DEVICES=1` (RTX PRO 6000, 96 GB). GPU 0 drives
the display.

---

## Also outstanding, separate from the detector

Recorded in [PLAN.md](PLAN.md), not fixed by any of the above:

- The **floor calibration has never been run**, so no similarity threshold has a
  known error rate. Masked SSIM measured 0.330 same-writer vs 0.300
  different-writer on the inherited crops — close to uninformative.
- `calibrate_pipeline.py` still hardcodes the old matter's paths and pre-fix
  `OBSERVED` values.
