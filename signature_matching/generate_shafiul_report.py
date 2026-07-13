"""
Single-page PDF: Shafiul Islam's signature from the CMM letter vs the HCD
letter only (no other signers). Headline % = average of the three scorers
from comparison_report.csv, with the component breakdown shown underneath.

Usage:
    python generate_shafiul_report.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = "comparison_output"
REPORT_PATH = os.path.join(OUT_DIR, "shafiul_islam_similarity.pdf")
PAIR = "shafiul_cmm vs shafiul_hcd"

LABELS = {
    "shafiul_cmm": "Shafiul Islam - CMM Letter",
    "shafiul_hcd": "Shafiul Islam - HCD Letter",
}


def main():
    with open(os.path.join(OUT_DIR, "comparison_report.csv")) as f:
        row = next(r for r in csv.DictReader(f) if r["pair"] == PAIR)

    cosine = float(row["global_cosine"])
    patch = float(row["patch_match_score"])
    ssim = float(row["ssim_aligned"])
    overall = (cosine + patch + ssim) / 3 * 100

    a_key, b_key = PAIR.split(" vs ")

    fig = plt.figure(figsize=(8.5, 7))
    fig.suptitle("Shafiul Islam - Signature Similarity", fontsize=17, fontweight="bold", y=0.97)

    ax_a = fig.add_axes([0.05, 0.42, 0.42, 0.45])
    ax_a.imshow(Image.open(os.path.join(OUT_DIR, f"preprocessed_{a_key}.png")), cmap="gray")
    ax_a.set_title(LABELS[a_key], fontsize=11)
    ax_a.axis("off")

    ax_b = fig.add_axes([0.53, 0.42, 0.42, 0.45])
    ax_b.imshow(Image.open(os.path.join(OUT_DIR, f"preprocessed_{b_key}.png")), cmap="gray")
    ax_b.set_title(LABELS[b_key], fontsize=11)
    ax_b.axis("off")

    ax_text = fig.add_axes([0, 0.02, 1, 0.35])
    ax_text.axis("off")
    ax_text.text(0.5, 0.72, f"{overall:.0f}% similar", ha="center", va="center",
                 fontsize=34, fontweight="bold")
    ax_text.text(0.5, 0.4,
                 f"cosine {cosine*100:.0f}%   ·   patch-match {patch*100:.0f}%   ·   ssim {ssim*100:.0f}%",
                 ha="center", va="center", fontsize=11, color="#444444")
    ax_text.text(0.5, 0.18, "(headline % = average of the three scores above)",
                 ha="center", va="center", fontsize=8.5, color="#777777")

    fig.savefig(REPORT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
