"""
Detector evaluation (DETECTOR_FIXES.md step 1).

Scores a checkpoint on the held-out test split with real detection metrics.
The training loop only ever recorded val loss, so no checkpoint has a known
recall -- and recall is the number the whole pipeline rests on, because a
signature the detector misses is silently absent from every downstream count.

The split is reproduced by importing detection_training's own
`load_annotations_and_split`, not reimplemented: it shuffles with
`np.random.default_rng(RANDOM_SEED=42)` and takes the first 80%, so calling it
with the checkpoint's stored config yields exactly the images this model was
NOT trained on. Reimplementing that would risk scoring on training data and
reporting inflated numbers.

Preprocessing matches training exactly (the 512x512 squash), so the result
reflects how the model is actually used.

Headline number is class-1 (signature) RECALL at IoU 0.5, not mAP.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_detector.py \
        --model ../detection/models/model_best.pth --out eval_output
"""

import argparse
import json
import os
import sys

import torch
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader

DETECTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "detection")
sys.path.insert(0, DETECTION_DIR)

from detection_training import (  # noqa: E402
    DEFAULT_CONFIG, build_datasets, collate_fn, get_model,
)

CLASS_NAMES = {1: "signature", 2: "initials", 3: "redaction_or_date"}


def load_checkpoint_config(model_path: str) -> tuple[dict, dict]:
    """Prefer the config stored in the checkpoint -- the module default may
    have drifted since this model was trained, and the split depends on it."""
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = dict(DEFAULT_CONFIG)
    stored = ck.get("config") or {}
    cfg.update({k: v for k, v in stored.items() if k in cfg})
    # Paths in the checkpoint may not exist on this machine; fall back.
    for key in ("data_csv_path", "image_dir", "image_ids"):
        if not os.path.exists(cfg[key]):
            cfg[key] = DEFAULT_CONFIG[key]
    return cfg, ck


@torch.no_grad()
def evaluate(model_path: str, out_dir: str, batch_size: int = 8,
             workers: int = 8, score_thresh: float = 0.0):
    os.makedirs(out_dir, exist_ok=True)
    cfg, ck = load_checkpoint_config(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Checkpoint : {model_path}")
    print(f"  epoch    : {ck.get('epoch')}   val_loss: {ck.get('val_loss')}")
    print(f"Device     : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Split      : train_split={cfg['train_split']}, seed=42 "
          f"(reproduced from detection_training)\n")

    _, test_ds, _ = build_datasets(cfg)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn)

    model = get_model(cfg["num_classes"])
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval().to(device)

    # extended_summary gives per-class precision/recall arrays alongside mAP.
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True,
                                  extended_summary=True)

    # Kept for the operating-point table below: COCO recall is measured at
    # score threshold 0, but detect_signs.py runs at conf 0.5, so the COCO
    # number is a ceiling rather than the recall the counts actually get.
    raw = []

    n_img = n_gt = n_pred = 0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        preds = model(images)
        cpu_preds, cpu_tgts = [], []
        for p, t in zip(preds, targets):
            keep = p["scores"] >= score_thresh
            cpu_preds.append({"boxes": p["boxes"][keep].cpu(),
                              "scores": p["scores"][keep].cpu(),
                              "labels": p["labels"][keep].cpu()})
            cpu_tgts.append({"boxes": t["boxes"].cpu(),
                             "labels": t["labels"].cpu()})
            n_gt += len(t["boxes"])
            n_pred += int(keep.sum())
            pm, tm = p["labels"].cpu() == 1, t["labels"].cpu() == 1
            raw.append((p["boxes"].cpu()[pm], p["scores"].cpu()[pm],
                        t["boxes"].cpu()[tm]))
        metric.update(cpu_preds, cpu_tgts)
        n_img += len(images)
        if n_img % (batch_size * 20) < batch_size:
            print(f"  {n_img}/{len(test_ds)} images")

    print(f"  {n_img}/{len(test_ds)} images\n")
    res = metric.compute()

    def f(x):
        return float(x) if x is not None and float(x) >= 0 else float("nan")

    classes = [int(c) for c in res.get("classes", torch.tensor([])).tolist()] \
        if "classes" in res else []

    report = {
        "checkpoint": os.path.abspath(model_path),
        "epoch": ck.get("epoch"),
        "val_loss": ck.get("val_loss"),
        "test_images": n_img,
        "gt_boxes": n_gt,
        "predicted_boxes": n_pred,
        "map": f(res["map"]), "map_50": f(res["map_50"]),
        "map_75": f(res["map_75"]),
        "mar_100": f(res["mar_100"]),
        "map_small": f(res["map_small"]), "map_medium": f(res["map_medium"]),
        "map_large": f(res["map_large"]),
        "per_class": {},
    }

    per_c_map = res.get("map_per_class")
    per_c_mar = res.get("mar_100_per_class")
    for i, c in enumerate(classes):
        report["per_class"][CLASS_NAMES.get(c, str(c))] = {
            "map": f(per_c_map[i]) if per_c_map is not None else None,
            "mar_100": f(per_c_mar[i]) if per_c_mar is not None else None,
        }

    # Recall at IoU 0.50 for class 1, read off the extended summary's
    # recall tensor [iou_thr, class, area_rng, max_det].
    recall_50_sig = None
    if "recall" in res:
        try:
            rec = res["recall"]
            ci = classes.index(1)
            recall_50_sig = float(rec[0, ci, 0, -1])
        except (ValueError, IndexError, TypeError):
            pass
    report["signature_recall_at_iou50"] = recall_50_sig

    with open(os.path.join(out_dir, "eval_report.json"), "w") as f_:
        json.dump(report, f_, indent=2)

    print("=" * 62)
    print("DETECTION METRICS -- held-out test split")
    print("=" * 62)
    print(f"images {n_img}   gt boxes {n_gt}   predicted boxes {n_pred}\n")
    print(f"  mAP@[.5:.95]      {report['map']:.3f}")
    print(f"  mAP@0.50          {report['map_50']:.3f}")
    print(f"  mAP@0.75          {report['map_75']:.3f}")
    print(f"  mAR@100           {report['mar_100']:.3f}")
    print(f"\n  by object size:   small {report['map_small']:.3f}   "
          f"medium {report['map_medium']:.3f}   large {report['map_large']:.3f}")
    print("\n  per class:")
    for name, m in report["per_class"].items():
        mp = m["map"] if m["map"] is not None else float("nan")
        mr = m["mar_100"] if m["mar_100"] is not None else float("nan")
        print(f"    {name:<20} mAP {mp:.3f}   mAR@100 {mr:.3f}")

    if recall_50_sig is not None:
        print(f"\n  >> signature recall @ IoU 0.50, score>=0 : {recall_50_sig:.3f} "
              f"(ceiling -- all detections kept)")

    # ---- operating points: what the counts actually get ----
    print("\n" + "=" * 62)
    print("SIGNATURE @ IoU 0.50, BY CONFIDENCE THRESHOLD")
    print("=" * 62)
    print(f"{'conf':>6}{'recall':>9}{'precision':>11}{'TP':>7}{'FP':>7}"
          f"{'FN':>7}{'pred/true':>11}")
    ops = {}
    for thr in (0.3, 0.5, 0.7, 0.9):
        tp = fp = fn = 0
        n_p = n_t = 0
        for pboxes, pscores, tboxes in raw:
            sel = pscores >= thr
            pb = pboxes[sel]
            n_p += len(pb)
            n_t += len(tboxes)
            if len(tboxes) == 0:
                fp += len(pb)
                continue
            if len(pb) == 0:
                fn += len(tboxes)
                continue
            # Greedy highest-score-first matching, one GT per prediction.
            order = torch.argsort(pscores[sel], descending=True)
            used = set()
            for i in order.tolist():
                b = pb[i]
                ix0 = torch.maximum(b[0], tboxes[:, 0])
                iy0 = torch.maximum(b[1], tboxes[:, 1])
                ix1 = torch.minimum(b[2], tboxes[:, 2])
                iy1 = torch.minimum(b[3], tboxes[:, 3])
                inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
                ab = (b[2] - b[0]) * (b[3] - b[1])
                at = (tboxes[:, 2] - tboxes[:, 0]) * (tboxes[:, 3] - tboxes[:, 1])
                iou = inter / (ab + at - inter).clamp(min=1e-9)
                iou = torch.tensor([v if j not in used else -1.0
                                    for j, v in enumerate(iou.tolist())])
                best = int(torch.argmax(iou))
                if float(iou[best]) >= 0.5:
                    used.add(best)
                    tp += 1
                else:
                    fp += 1
            fn += len(tboxes) - len(used)
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        ratio = n_p / max(n_t, 1)
        ops[thr] = {"recall": rec, "precision": prec, "tp": tp, "fp": fp,
                    "fn": fn, "pred_over_true": ratio}
        print(f"{thr:>6.1f}{rec:>9.3f}{prec:>11.3f}{tp:>7}{fp:>7}{fn:>7}"
              f"{ratio:>11.2f}")
    report["operating_points"] = ops
    print("\npred/true is the count ratio -- the number that matters when the "
          "deliverable is counts.")

    with open(os.path.join(out_dir, "eval_report.json"), "w") as f_:
        json.dump(report, f_, indent=2)
    print(f"\nReport: {os.path.join(out_dir, 'eval_report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../detection/models/model_best.pth")
    ap.add_argument("--out", default="eval_output")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    evaluate(args.model, args.out, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
