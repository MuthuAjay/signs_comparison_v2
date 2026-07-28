"""
Zero-shot signature comparison pipeline (Steps 2-5).

Step 2: Uniform preprocessing (background removal, ink crop, aspect-preserving resize)
Step 3: Three independent scorers
        A. DINOv2 global embedding cosine similarity
        B. DINOv2 patch-token mutual nearest-neighbor matching (+ visualization)
        C. Classical: rigid alignment (centroid + principal axis) -> SSIM
Step 4: Calibration against the reference-vs-reference baseline
Step 5: Pairwise report (CSV + console table + match visualizations)

Usage:
    Edit SIGNATURES config at the bottom, then:
        python signature_compare.py

Deps: torch, torchvision, pillow, numpy, matplotlib
    pip install torch torchvision pillow numpy matplotlib
"""

import itertools
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CANVAS = 224                      # DINOv2 needs multiples of 14
PATCH = 14                        # ViT-S/14 patch size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OUT_DIR = "comparison_output"


# ----------------------------------------------------------------------------
# Step 2: Preprocessing
# ----------------------------------------------------------------------------

def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's method on a uint8 grayscale array. Returns threshold in [0, 255]."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    best_t, best_var = 0, -1.0
    w0, sum0 = 0.0, 0.0
    sum_all = np.dot(np.arange(256), hist)
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0 = sum0 / w0
        m1 = (sum_all - sum0) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return float(best_t)


def preprocess(path: str, canvas: int = CANVAS, margin: int = 8) -> dict:
    """
    Load -> grayscale -> soft background removal (keep ink intensity) ->
    tight crop to ink bounding box -> aspect-preserving resize onto a white
    canvas -> return both the display image and the model-ready tensor.
    """
    img = Image.open(path).convert("L")
    gray = np.asarray(img, dtype=np.uint8)

    # Soft background removal: pixels brighter than Otsu -> pure white,
    # ink pixels keep their intensity (pressure signal preserved).
    t = otsu_threshold(gray)
    soft = gray.astype(np.float32).copy()
    soft[gray > t] = 255.0

    # Tight crop to ink bounding box (+margin)
    ink_mask = soft < 255.0
    if not ink_mask.any():
        raise ValueError(f"No ink found in {path} - check the crop/scan.")
    ys, xs = np.where(ink_mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, soft.shape[0] - 1)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, soft.shape[1] - 1)
    crop = soft[y0:y1 + 1, x0:x1 + 1]

    # Aspect-preserving resize onto white canvas
    h, w = crop.shape
    scale = (canvas - 2 * margin) / max(h, w)
    nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
    crop_t = torch.from_numpy(crop).unsqueeze(0)  # 1,H,W
    crop_t = TF.resize(crop_t, [nh, nw], antialias=True)
    canvas_t = torch.full((1, canvas, canvas), 255.0)
    top, left = (canvas - nh) // 2, (canvas - nw) // 2
    canvas_t[:, top:top + nh, left:left + nw] = crop_t

    display = canvas_t.squeeze(0).numpy().astype(np.uint8)      # for viz/SSIM
    ink = 1.0 - canvas_t / 255.0                                # ink=1, bg=0

    # Model tensor: replicate to 3 channels + ImageNet normalization
    rgb = (canvas_t / 255.0).repeat(3, 1, 1)
    model_in = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)

    return {"display": display, "ink": ink.squeeze(0), "tensor": model_in}


# ----------------------------------------------------------------------------
# Step 3A: DINOv2 global embedding similarity
# ----------------------------------------------------------------------------

_dinov2 = None

def get_dinov2():
    global _dinov2
    if _dinov2 is None:
        _dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _dinov2.eval().to(DEVICE)
    return _dinov2


@torch.no_grad()
def dinov2_features(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cls_embedding [D], patch_tokens [N, D]) for one image."""
    model = get_dinov2()
    out = model.forward_features(tensor.unsqueeze(0).to(DEVICE))
    cls = out["x_norm_clstoken"].squeeze(0)          # [D]
    patches = out["x_norm_patchtokens"].squeeze(0)   # [N, D], N=(224/14)^2=256
    return cls.cpu(), patches.cpu()


def global_cosine(cls_a: torch.Tensor, cls_b: torch.Tensor) -> float:
    return F.cosine_similarity(cls_a, cls_b, dim=0).item()


# ----------------------------------------------------------------------------
# Step 3B: Patch-token mutual nearest-neighbor matching
# ----------------------------------------------------------------------------

def ink_patch_mask(ink: torch.Tensor, grid: int, min_ink: float = 0.08) -> torch.Tensor:
    """Boolean mask [grid*grid] of patches that actually contain ink.

    min_ink is the mean ink fraction a 14x14 patch must carry to be scored.
    The inherited default (0.02) admitted patches holding only stroke
    anti-aliasing; DINOv2 tokens on near-blank patches are mutually highly
    correlated, so those patches produced spurious mutual-NN matches and
    inflated the score. A stroke genuinely crossing a patch covers well above
    0.08.
    """
    pooled = F.avg_pool2d(ink.unsqueeze(0).unsqueeze(0),
                          kernel_size=PATCH, stride=PATCH).flatten()
    return pooled > min_ink


def mutual_nn_match(pa: torch.Tensor, pb: torch.Tensor,
                    mask_a: torch.Tensor, mask_b: torch.Tensor,
                    sim_floor: float = 0.5) -> dict:
    """
    Mutual nearest neighbors between ink patches of the two signatures.
    Score = (fraction of ink patches with a mutual match above sim_floor,
             mean similarity of those matches).
    """
    ia, ib = torch.where(mask_a)[0], torch.where(mask_b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return {"score": 0.0, "mean_sim": 0.0, "matches": []}
    a = F.normalize(pa[ia], dim=1)
    b = F.normalize(pb[ib], dim=1)
    sim = a @ b.T                                    # [Na, Nb]
    nn_ab = sim.argmax(dim=1)                        # a -> b
    nn_ba = sim.argmax(dim=0)                        # b -> a
    mutual = nn_ba[nn_ab] == torch.arange(len(ia))
    matches = []
    for k in torch.where(mutual)[0]:
        s = sim[k, nn_ab[k]].item()
        if s >= sim_floor:
            matches.append((ia[k].item(), ib[nn_ab[k]].item(), s))
    # Symmetric normalization. The inherited version divided by
    # min(len(ia), len(ib)), which lets a small ink area match into a large one
    # and score near 1.0 -- biased upward whenever the two ink areas differ.
    denom = 0.5 * (len(ia) + len(ib))
    score = len(matches) / denom if denom else 0.0
    mean_sim = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return {"score": score, "mean_sim": mean_sim, "matches": matches}


def visualize_matches(disp_a, disp_b, matches, grid, out_path, max_lines=60):
    """Side-by-side images with lines connecting mutually matched patches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = CANVAS
    combo = np.full((h, 2 * h + 20), 255, dtype=np.uint8)
    combo[:, :h] = disp_a
    combo[:, h + 20:] = disp_b

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(combo, cmap="gray")
    matches_sorted = sorted(matches, key=lambda m: -m[2])[:max_lines]
    for pa_idx, pb_idx, s in matches_sorted:
        ya, xa = divmod(pa_idx, grid)
        yb, xb = divmod(pb_idx, grid)
        x1 = xa * PATCH + PATCH / 2
        y1 = ya * PATCH + PATCH / 2
        x2 = xb * PATCH + PATCH / 2 + h + 20
        y2 = yb * PATCH + PATCH / 2
        ax.plot([x1, x2], [y1, y2], lw=0.7, alpha=0.6,
                color=plt.cm.viridis((s - 0.5) / 0.5))
    ax.set_title(f"Mutual NN patch matches (n={len(matches)})")
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Step 3C: Rigid alignment + SSIM
# ----------------------------------------------------------------------------

def principal_angle(ink: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Ink-mass centroid and principal-axis angle (degrees) via PCA."""
    ys, xs = torch.where(ink > 0.05)
    pts = torch.stack([xs.float(), ys.float()], dim=1)
    c = pts.mean(dim=0)
    centered = pts - c
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    v = evecs[:, -1]                                 # dominant axis
    ang = math.degrees(math.atan2(v[1].item(), v[0].item()))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang, c


def align_to(ink_src: torch.Tensor, ink_ref: torch.Tensor) -> torch.Tensor:
    """Rotate + translate src ink map so its principal axis/centroid match ref."""
    ang_s, c_s = principal_angle(ink_src)
    ang_r, c_r = principal_angle(ink_ref)
    x = ink_src.unsqueeze(0)                         # 1,H,W
    x = TF.rotate(x, angle=(ang_s - ang_r),
                  center=[c_s[0].item(), c_s[1].item()],
                  interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    dx, dy = (c_r - c_s).tolist()
    x = TF.affine(x, angle=0.0, translate=[int(round(dx)), int(round(dy))],
                  scale=1.0, shear=[0.0],
                  interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    return x.squeeze(0)


def ink_region_mask(a: torch.Tensor, b: torch.Tensor,
                    thr: float = 0.05, dilate: int = 15) -> torch.Tensor:
    """Region where either signature has ink, dilated to include stroke context.

    Dilation is what keeps the mask fair: it retains the immediate neighbourhood
    of every stroke, so a stroke present in one image and absent in the other is
    still scored as a disagreement rather than being masked away.
    """
    union = ((a > thr) | (b > thr)).float().unsqueeze(0).unsqueeze(0)
    grown = F.max_pool2d(union, kernel_size=dilate, stride=1, padding=dilate // 2)
    return grown.squeeze() > 0


def ssim(a: torch.Tensor, b: torch.Tensor, win: int = 11, sigma: float = 1.5,
         masked: bool = True) -> float:
    """SSIM on ink maps in [0,1], Gaussian window, torch-only.

    Averaged over the dilated ink region, not the whole canvas. In a window that
    is pure background, mu = var = cov = 0 and the SSIM expression collapses to
    (c1*c2)/(c1*c2) = 1.0 exactly. Roughly 90% of a 224x224 signature canvas is
    background, so a whole-canvas mean is dominated by background agreement and
    reports a high score for signatures that share nothing but their margins.

    Pass masked=False to reproduce the inherited (inflated) behaviour.
    """
    coords = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    kernel = (g.T @ g).unsqueeze(0).unsqueeze(0)

    def filt(x):
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), kernel, padding=win // 2).squeeze()

    mu_a, mu_b = filt(a), filt(b)
    var_a = filt(a * a) - mu_a ** 2
    var_b = filt(b * b) - mu_b ** 2
    cov = filt(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))

    if not masked:
        return s.mean().item()
    m = ink_region_mask(a, b)
    if not m.any():
        return 0.0
    return s[m].mean().item()


# ----------------------------------------------------------------------------
# Steps 4-5: Pairwise scoring, baseline calibration, report
# ----------------------------------------------------------------------------

@dataclass
class Signature:
    name: str
    path: str
    role: str = "questioned"   # "reference" | "questioned"
    group: str = "."           # source sub-directory, from the corpus layout


def safe_name(rel_path: str) -> str:
    """Filesystem-safe, collision-free id derived from the corpus-relative path."""
    stem = os.path.splitext(rel_path)[0]
    return stem.replace(os.sep, "__").replace(" ", "_")


def signatures_from_folder(root: str, image_role: str = "crop") -> list[Signature]:
    """Build the signature list by walking a corpus folder recursively."""
    from ingest import discover

    records, _ = discover(root, image_role=image_role)
    crops = [r for r in records if r.kind == "image" and r.role == "crop"]
    if not crops:
        raise ValueError(
            f"No signature crops found under {root}. "
            f"PDFs need Stage 1 detection first (see PLAN.md build order).")
    return [Signature(name=safe_name(r.rel_path), path=r.abs_path, group=r.group)
            for r in crops]


def run(signatures: list[Signature], out_dir: str = OUT_DIR,
        viz_max: int = 50, save_preprocessed: bool = True):
    """
    Score every pair and report. Visualizations are capped: the inherited
    version wrote one PNG per pair, which at corpus scale is n(n-1)/2 files
    (5k crops -> 12.5M PNGs). Here all pairs are scored, but only the
    `viz_max` highest-cosine pairs get a match figure.
    """
    os.makedirs(out_dir, exist_ok=True)
    grid = CANVAS // PATCH
    n = len(signatures)
    n_pairs = n * (n - 1) // 2
    print(f"Signatures: {n}   pairs to score: {n_pairs:,}")
    if n_pairs > 5_000_000:
        print("  NOTE: this is the O(n^2) reference path. For a corpus this "
              "size use the batched GPU path (PLAN.md build step 6).")

    feats = {}
    for i, s in enumerate(signatures, 1):
        pre = preprocess(s.path)
        cls, patches = dinov2_features(pre["tensor"])
        feats[s.name] = {"pre": pre, "cls": cls, "patches": patches,
                         "mask": ink_patch_mask(pre["ink"], grid),
                         "role": s.role, "group": s.group}
        if save_preprocessed:
            Image.fromarray(pre["display"]).save(
                os.path.join(out_dir, f"preprocessed_{s.name}.png"))
        if i % 100 == 0 or i == n:
            print(f"  featurized {i}/{n}")

    rows = []
    for k, (a, b) in enumerate(itertools.combinations([s.name for s in signatures], 2), 1):
        fa, fb = feats[a], feats[b]
        g = global_cosine(fa["cls"], fb["cls"])
        m = mutual_nn_match(fa["patches"], fb["patches"], fa["mask"], fb["mask"])
        aligned_b = align_to(fb["pre"]["ink"], fa["pre"]["ink"])
        s_ssim = ssim(fa["pre"]["ink"], aligned_b)
        pair_type = ("ref-ref" if fa["role"] == fb["role"] == "reference"
                     else "ref-questioned" if "reference" in (fa["role"], fb["role"])
                     else "questioned-questioned")
        rows.append({"sig_a": a, "sig_b": b,
                     "group_a": fa["group"], "group_b": fb["group"],
                     "cross_group": fa["group"] != fb["group"],
                     "type": pair_type,
                     "global_cosine": g, "patch_match_score": m["score"],
                     "patch_mean_sim": m["mean_sim"], "ssim_aligned": s_ssim})
        if k % 1000 == 0:
            print(f"  scored {k:,}/{n_pairs:,}")

    # Match figures for the highest-cosine pairs only.
    top = sorted(rows, key=lambda r: -r["global_cosine"])[:viz_max]
    for r in top:
        fa, fb = feats[r["sig_a"]], feats[r["sig_b"]]
        m = mutual_nn_match(fa["patches"], fb["patches"], fa["mask"], fb["mask"])
        visualize_matches(fa["pre"]["display"], fb["pre"]["display"],
                          m["matches"], grid,
                          os.path.join(out_dir,
                                       f"matches_{r['sig_a']}_vs_{r['sig_b']}.png"))

    csv_path = os.path.join(out_dir, "comparison_report.csv")
    header = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                             for k in header) + "\n")

    print(f"\nMost similar pairs (top {min(25, len(top))} by DINOv2 cosine):")
    print(f"{'PAIR':<52}{'COSINE':>8}{'PATCH':>8}{'SSIM*':>8}  CROSS-GROUP")
    print("-" * 88)
    for r in top[:25]:
        pair = f"{r['sig_a']} vs {r['sig_b']}"
        if len(pair) > 50:
            pair = pair[:47] + "..."
        print(f"{pair:<52}{r['global_cosine']:>8.3f}"
              f"{r['patch_match_score']:>8.3f}{r['ssim_aligned']:>8.3f}"
              f"  {'yes' if r['cross_group'] else ''}")

    print("\n* SSIM is ink-masked. These scores have NO calibrated error rate "
          "yet -- the floor calibration (PLAN.md build step 2) has not been run, "
          "so they rank pairs but do not support any same-signer conclusion.")
    print(f"\nReport : {csv_path}")
    print(f"Visuals: {out_dir}/matches_*.png ({len(top)} figures), "
          f"{out_dir}/preprocessed_*.png")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Pairwise signature comparison over a corpus folder.")
    ap.add_argument("root", help="folder of signature crops (walked recursively)")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    ap.add_argument("--viz-max", type=int, default=50,
                    help="how many top pairs get a match figure (default 50)")
    ap.add_argument("--no-preprocessed", action="store_true",
                    help="skip writing preprocessed_*.png for every crop")
    args = ap.parse_args()

    sigs = signatures_from_folder(args.root)
    run(sigs, out_dir=args.out, viz_max=args.viz_max,
        save_preprocessed=not args.no_preprocessed)


if __name__ == "__main__":
    main()
