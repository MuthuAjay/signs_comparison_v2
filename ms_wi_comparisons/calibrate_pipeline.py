"""
Calibration for the signature comparison pipeline.

Measures, under the EXACT pipeline in signature_compare.py:
  Experiment 1 (ceiling): what identical signatures score when only
      capture conditions vary (rotation, scale, shift, stroke thickness).
  Experiment 2 (floor): what signatures known to be unrelated score.

Then reports where the observed pair (e.g. cosine 92 / patch 35 / ssim 82)
falls relative to those measured ranges, for all three metrics.

Usage:
    1. Put this file in the same directory as signature_compare.py.
    2. Edit the CONFIG block at the bottom:
         - REAL_CROPS: your actual signature crop paths (perturbation ceiling
           is computed per crop and pooled).
         - UNRELATED_DIR: a folder of genuine signature images from DIFFERENT
           writers (e.g. a handful of CEDAR / BHSig260 genuine samples,
           one or two images per writer, filenames starting with a writer id
           like "w01_...", "w02_..." so cross-writer pairs can be formed).
           Your Shafiul-vs-Shafqat crops can also be added as writer folders.
         - OBSERVED: the scores of the pair you are calibrating against.
    3. python calibrate_pipeline.py

Outputs:
    calibration_output/calibration_report.csv   - every scored pair
    calibration_output/calibration_summary.txt  - ranges + placement verdicts
    calibration_output/calibration_hist_<metric>.png - distributions with the
        observed value marked (drop these straight into the report)

Rule of use: decide to include the results BEFORE running. Whatever ranges
come out go into the report, including if they do not help the finding.
"""

import itertools
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from comapare_signs import (
    CANVAS, PATCH,
    preprocess, dinov2_features, global_cosine,
    mutual_nn_match, ink_patch_mask, align_to, ssim,
)

OUT_DIR = "calibration_output"
N_PERTURB = 25
SEED = 42


# ----------------------------------------------------------------------------
# Capture-condition perturbations (NOT writing variation)
# ----------------------------------------------------------------------------

def perturb_image(path: str, rng: random.Random) -> Image.Image:
    """
    Simulate a different scan/capture of the SAME physical signature:
    small rotation, scale, translation, and stroke-thickness change.
    Operates on the raw crop; the result then goes through the normal
    preprocess() so the perturbed copy experiences the full pipeline.
    """
    img = Image.open(path).convert("L")
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0)  # 1,H,W

    angle = rng.uniform(-3.0, 3.0)
    scale = rng.uniform(0.95, 1.05)
    tx = rng.randint(-4, 4)
    ty = rng.randint(-4, 4)
    x = TF.affine(x, angle=angle, translate=[tx, ty], scale=scale,
                  shear=[0.0], interpolation=TF.InterpolationMode.BILINEAR,
                  fill=255.0)

    # Stroke thickness jitter: ink is dark, so
    #   dilate ink  = min-filter = -maxpool(-x)
    #   erode ink   = max-filter =  maxpool(x)
    mode = rng.choice(["dilate", "erode", "none"])
    if mode == "dilate":
        x = -F.max_pool2d(-x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    elif mode == "erode":
        x = F.max_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

    arr = x.squeeze(0).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------------------------------------------------------
# Scoring one pair through the full pipeline (identical to production path)
# ----------------------------------------------------------------------------

def score_pair(path_a: str, path_b: str) -> dict:
    grid = CANVAS // PATCH
    pa, pb = preprocess(path_a), preprocess(path_b)
    cls_a, patches_a = dinov2_features(pa["tensor"])
    cls_b, patches_b = dinov2_features(pb["tensor"])
    cos = global_cosine(cls_a, cls_b)
    m = mutual_nn_match(patches_a, patches_b,
                        ink_patch_mask(pa["ink"], grid),
                        ink_patch_mask(pb["ink"], grid))
    aligned_b = align_to(pb["ink"], pa["ink"])
    s = ssim(pa["ink"], aligned_b)
    return {"cosine": cos, "patch": m["score"], "ssim": s}


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

def experiment_same_signature(real_crops: list[str]) -> list[dict]:
    """Ceiling: each real crop vs N perturbed copies of itself."""
    rng = random.Random(SEED)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for crop in real_crops:
            base = os.path.splitext(os.path.basename(crop))[0]
            for i in range(N_PERTURB):
                pert_path = os.path.join(tmp, f"{base}_pert{i}.png")
                perturb_image(crop, rng).save(pert_path)
                r = score_pair(crop, pert_path)
                r.update({"experiment": "same_signature",
                          "pair": f"{base} vs perturbed_{i}"})
                rows.append(r)
                print(f"[same] {r['pair']}: cos={r['cosine']:.3f} "
                      f"patch={r['patch']:.3f} ssim={r['ssim']:.3f}")
    return rows


def writer_id(filename: str) -> str:
    """Writer id = filename prefix before the first underscore."""
    return os.path.basename(filename).split("_")[0]


def experiment_unrelated(unrelated_dir: str, max_pairs: int = 30) -> list[dict]:
    """Floor: cross-writer pairs from a folder of genuine samples."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted(os.path.join(unrelated_dir, f)
                   for f in os.listdir(unrelated_dir)
                   if f.lower().endswith(exts))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 images in {unrelated_dir}")
    cross = [(a, b) for a, b in itertools.combinations(files, 2)
             if writer_id(a) != writer_id(b)]
    if not cross:
        raise ValueError("No cross-writer pairs found. Prefix filenames with "
                         "a writer id, e.g. w01_sig1.png, w02_sig1.png")
    rng = random.Random(SEED)
    rng.shuffle(cross)
    rows = []
    for a, b in cross[:max_pairs]:
        r = score_pair(a, b)
        r.update({"experiment": "unrelated",
                  "pair": f"{os.path.basename(a)} vs {os.path.basename(b)}"})
        rows.append(r)
        print(f"[unrel] {r['pair']}: cos={r['cosine']:.3f} "
              f"patch={r['patch']:.3f} ssim={r['ssim']:.3f}")
    return rows


# ----------------------------------------------------------------------------
# Summary, placement verdicts, figures
# ----------------------------------------------------------------------------

def stats(vals):
    v = np.array(vals)
    return {"mean": v.mean(), "min": v.min(), "max": v.max(), "n": len(v)}


def placement(observed: float, same: dict, unrel: dict) -> str:
    """Where does the observed value fall? Pre-committed decision logic."""
    in_unrel = observed <= unrel["max"]
    in_same = observed >= same["min"]
    if in_unrel and in_same:
        return ("Inside BOTH ranges (they overlap) - this metric does not "
                "separate same from unrelated under this pipeline; treat it "
                "as uninformative for this pair.")
    if in_unrel:
        return ("WITHIN the range produced by unrelated signatures - "
                "indistinguishable from unrelated under this pipeline.")
    if in_same:
        return ("WITHIN the range produced by identical signatures under "
                "varied capture conditions - consistent with same signature.")
    gap = same["min"] - unrel["max"]
    frac = (observed - unrel["max"]) / gap
    return (f"BETWEEN the two ranges, {frac:.0%} of the way from the "
            f"unrelated maximum toward the identical-signature minimum. "
            f"Closer to {'identical' if frac > 0.5 else 'unrelated'}.")


def run(real_crops, unrelated_dir, observed):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = experiment_same_signature(real_crops)
    rows += experiment_unrelated(unrelated_dir)

    # CSV of every scored pair
    csv_path = os.path.join(OUT_DIR, "calibration_report.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,pair,cosine,patch,ssim\n")
        for r in rows:
            f.write(f"{r['experiment']},{r['pair']},"
                    f"{r['cosine']:.4f},{r['patch']:.4f},{r['ssim']:.4f}\n")

    # Summary + placement
    lines = ["CALIBRATION SUMMARY (all values under the production pipeline)",
             "=" * 70]
    metric_stats = {}
    for metric in ("cosine", "patch", "ssim"):
        same = stats([r[metric] for r in rows if r["experiment"] == "same_signature"])
        unrel = stats([r[metric] for r in rows if r["experiment"] == "unrelated"])
        metric_stats[metric] = (same, unrel)
        lines += [
            f"\n{metric.upper()}",
            f"  identical signature, varied capture (n={same['n']}): "
            f"mean {same['mean']:.3f}, range [{same['min']:.3f}, {same['max']:.3f}]",
            f"  unrelated signatures (n={unrel['n']}): "
            f"mean {unrel['mean']:.3f}, range [{unrel['min']:.3f}, {unrel['max']:.3f}]",
            f"  observed pair: {observed[metric]:.3f}",
            f"  placement: {placement(observed[metric], same, unrel)}",
        ]
    lines += ["\nNote: the identical-signature range reflects capture-condition",
              "variation only. Natural writing variation between two genuine",
              "signatures of the same person widens the true same-writer range",
              "downward; the ceiling here is therefore conservative."]
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "calibration_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)

    # Histograms with the observed value marked
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("cosine", "patch", "ssim"):
        same_v = [r[metric] for r in rows if r["experiment"] == "same_signature"]
        unrel_v = [r[metric] for r in rows if r["experiment"] == "unrelated"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(unrel_v, bins=15, alpha=0.6, label="unrelated signatures")
        ax.hist(same_v, bins=15, alpha=0.6, label="identical, varied capture")
        ax.axvline(observed[metric], color="red", lw=2,
                   label=f"observed pair ({observed[metric]:.2f})")
        ax.set_title(f"{metric} - calibration under production pipeline")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend()
        fig.savefig(os.path.join(OUT_DIR, f"calibration_hist_{metric}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved: {csv_path}, calibration_summary.txt, "
          f"calibration_hist_*.png in {OUT_DIR}/")


if __name__ == "__main__":
    # ---------------- CONFIG: edit these ----------------
    REAL_CROPS = [
        "signatures/shafiul_cmm.png",
        "signatures/shafiul_hcd.png",
    ]
    UNRELATED_DIR = "unrelated_signatures"   # w01_*.png, w02_*.png, ...
    OBSERVED = {"cosine": 0.92, "patch": 0.35, "ssim": 0.82}
    # -----------------------------------------------------
    run(REAL_CROPS, UNRELATED_DIR, OBSERVED)