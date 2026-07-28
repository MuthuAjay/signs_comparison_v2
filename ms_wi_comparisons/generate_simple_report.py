"""
Minimal single-page PDF: every extracted signature shown side by side, plus
a pairwise similarity matrix computed from comparison_report.csv (no
explanatory text). Similarity % per pair = mean of the three scorers
(global_cosine, patch_match_score, ssim_aligned) already computed by
comapare_signs.py -- not an arbitrary number.

Usage:
    python generate_simple_report.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = "comparison_output"
REPORT_PATH = os.path.join(OUT_DIR, "signature_similarity_matrix.pdf")

LABELS = {
    "shafiul_cmm": "Shafiul\n(CMM)",
    "shafiul_hcd": "Shafiul\n(HCD)",
    "shafqat_hcd": "Shafqat\n(HCD)",
}


def load_pair_scores():
    with open(os.path.join(OUT_DIR, "comparison_report.csv")) as f:
        rows = list(csv.DictReader(f))
    scores = {}
    for r in rows:
        a, b = r["pair"].split(" vs ")
        cosine = float(r["global_cosine"])
        patch = float(r["patch_match_score"])
        ssim = float(r["ssim_aligned"])
        overall = (cosine + patch + ssim) / 3
        scores[(a, b)] = (overall, cosine, patch, ssim)
        scores[(b, a)] = (overall, cosine, patch, ssim)
    return scores


def main():
    keys = list(LABELS.keys())
    n = len(keys)
    pair_scores = load_pair_scores()

    matrix = np.ones((n, n))
    components = [[None] * n for _ in range(n)]
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i != j:
                overall, cosine, patch, ssim = pair_scores[(a, b)]
                matrix[i, j] = overall
                components[i][j] = (cosine, patch, ssim)

    fig = plt.figure(figsize=(8.5, 9.5))

    # Signature thumbnails across the top
    thumb_w = 0.9 / n
    for i, key in enumerate(keys):
        ax = fig.add_axes([0.05 + i * thumb_w, 0.68, thumb_w * 0.9, 0.27])
        ax.imshow(Image.open(os.path.join(OUT_DIR, f"preprocessed_{key}.png")), cmap="gray")
        ax.set_title(LABELS[key].replace("\n", " "), fontsize=10)
        ax.axis("off")

    # Similarity matrix
    ax_m = fig.add_axes([0.18, 0.08, 0.68, 0.52])
    im = ax_m.imshow(matrix, cmap="Greens", vmin=0, vmax=1)

    short_labels = [LABELS[k].replace("\n", " ") for k in keys]
    ax_m.set_xticks(range(n))
    ax_m.set_yticks(range(n))
    ax_m.set_xticklabels(short_labels, fontsize=10)
    ax_m.set_yticklabels(short_labels, fontsize=10)
    ax_m.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            if i == j:
                ax_m.text(j, i, "100%", ha="center", va="center",
                          fontsize=16, fontweight="bold", color=color)
            else:
                cosine, patch, ssim = components[i][j]
                ax_m.text(j, i - 0.12, f"{val*100:.0f}%", ha="center", va="center",
                          fontsize=16, fontweight="bold", color=color)
                ax_m.text(j, i + 0.22,
                          f"cos {cosine*100:.0f} · patch {patch*100:.0f} · ssim {ssim*100:.0f}",
                          ha="center", va="center", fontsize=6.8, color=color)

    for spine in ax_m.spines.values():
        spine.set_visible(False)
    ax_m.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_m.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_m.grid(which="minor", color="white", linewidth=3)

    fig.suptitle("Signature Similarity Matrix", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.035,
              "Headline % = average of cosine, patch-match and SSIM similarity (shown below each score)",
              ha="center", fontsize=8.5, color="#555555")

    fig.savefig(REPORT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
