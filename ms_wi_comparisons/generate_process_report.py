"""
Focused process report: what happens when documents are submitted --
detection (what we detect, which model, why), preprocessing, and comparison.
Training is intentionally left out.

Usage:
    python generate_process_report.py
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

OUT_DIR = "comparison_output"
REPORT_PATH = os.path.join(OUT_DIR, "process_report.pdf")

PAGE_W, PAGE_H = 8.5, 11
MARGIN_L, MARGIN_R = 0.08, 0.95
LINE_H = 0.0195


class PageWriter:
    def __init__(self, pdf, title=None, subtitle=None):
        self.pdf = pdf
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 0.965
        if title:
            self.ax.text(MARGIN_L, self.y, title, fontsize=18, fontweight="bold", va="top")
            self.y -= 0.045
        if subtitle:
            self.ax.text(MARGIN_L, self.y, subtitle, fontsize=10, color="#555555", va="top")
            self.y -= 0.035
        if title:
            self.ax.axhline(self.y + 0.012, color="#cccccc", lw=1, xmin=MARGIN_L, xmax=MARGIN_R)
            self.y -= 0.02

    def heading(self, text):
        self.y -= 0.008
        self.ax.text(MARGIN_L, self.y, text, fontsize=13, fontweight="bold", color="#1a1a1a", va="top")
        self.y -= LINE_H * 1.3

    def para(self, text, width=100, fontsize=9.3, color="black", gap=0.014):
        for line in textwrap.wrap(text, width=width):
            self.ax.text(MARGIN_L, self.y, line, fontsize=fontsize, va="top", color=color)
            self.y -= LINE_H
        self.y -= gap

    def bullet(self, text, width=96, fontsize=9.3):
        wrapped = textwrap.wrap(text, width=width)
        for i, line in enumerate(wrapped):
            prefix = "•  " if i == 0 else "   "
            self.ax.text(MARGIN_L + 0.01, self.y, prefix + line, fontsize=fontsize, va="top")
            self.y -= LINE_H
        self.y -= 0.006

    def table(self, headers, rows, col_x, fontsize=8.7):
        for x, h in zip(col_x, headers):
            self.ax.text(x, self.y, h, fontsize=fontsize, fontweight="bold", va="top")
        self.y -= LINE_H
        self.ax.axhline(self.y + 0.008, color="#dddddd", lw=0.8, xmin=MARGIN_L, xmax=MARGIN_R)
        self.y -= 0.006
        for row in rows:
            for x, v in zip(col_x, row):
                self.ax.text(x, self.y, str(v), fontsize=fontsize, va="top")
            self.y -= LINE_H
        self.y -= 0.015

    def space(self, amount=0.015):
        self.y -= amount

    def image(self, path, x0, width, height):
        ax_img = self.fig.add_axes([x0, self.y - height, width, height])
        ax_img.imshow(Image.open(path), cmap="gray")
        ax_img.axis("off")
        self.y -= (height + 0.015)

    def footer(self, text):
        self.ax.text(0.5, 0.02, text, fontsize=7.5, color="#999999", ha="center")

    def close(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with PdfPages(REPORT_PATH) as pdf:

        # ---------------- Page 1 ----------------
        p = PageWriter(pdf, "Document Signature Verification - Process",
                        "What happens from document submission to a similarity score")
        p.heading("Flow")
        p.para(
            "Documents in -> Detection (find signatures) -> Preprocessing (clean up each crop) -> "
            "Comparison (score how similar two signatures are)."
        )

        p.heading("1. Detection - what we detect")
        p.para(
            "Every submitted page is scanned for three kinds of handwritten marks: signatures, initials, "
            "and stamps. Each detection comes with a confidence score and a bounding box, so low-confidence "
            "or clearly-wrong detections (letterhead logos, page-number scribbles, etc.) can be filtered out "
            "before they ever reach the comparison stage."
        )
        p.heading("Model used: Faster R-CNN (ResNet-50 + FPN backbone)")
        p.bullet("It's an object detector, not a classifier - we need both the location on the page and a "
                 "confidence score per mark, since a page can contain zero, one, or several signatures mixed "
                 "in with unrelated printed text and logos.")
        p.bullet("Two-stage detectors (propose regions, then classify them) are more accurate than one-stage "
                 "detectors for small, irregularly-shaped objects like handwritten ink - and this runs "
                 "offline on submitted documents, so accuracy is prioritized over real-time speed.")
        p.bullet("The ResNet-50 + FPN (Feature Pyramid Network) backbone detects objects at multiple scales "
                 "well, which matters because signatures vary a lot in size and aspect ratio from page to "
                 "page and document to document.")

        p.heading("Example: what got detected on the two submitted letters")
        p.table(
            ["Document", "Raw detections", "Kept as genuine signatures"],
            [
                ["CMM Letter", "1", "1 (Shafiul Islam)"],
                ["HCD Letter", "6", "2 (Shafiul Islam, Shafqat Ahmed)"],
            ],
            col_x=[0.09, 0.34, 0.58],
        )
        p.para(
            "The 4 discarded HCD detections were a letterhead logo, an exhibit heading, and two page/exhibit "
            "annotation marks - caught by low confidence score and confirmed by visual review, not signatures.",
            fontsize=8.6, color="#555555"
        )
        p.footer("Page 1")
        p.close()

        # ---------------- Page 2 ----------------
        p = PageWriter(pdf, "Preprocessing and Comparison")

        p.heading("2. Preprocessing - before any comparison")
        p.para(
            "Every detected signature crop goes through the same cleanup before scoring, so differences in "
            "scan quality or cropping don't get mistaken for differences between signatures:"
        )
        p.bullet("Convert to grayscale.")
        p.bullet("Soft background removal (Otsu threshold): background pixels are pushed to pure white, "
                 "ink pixels keep their intensity so pen-pressure detail survives.")
        p.bullet("Tight crop to the ink bounding box, with a small margin.")
        p.bullet("Aspect-preserving resize onto a standard 224x224 white canvas, so every signature is "
                 "compared at the same scale regardless of how large it was on the original page.")

        p.heading("3. Comparison - how similarity is scored")
        p.para(
            "Two preprocessed signatures are compared with three independent methods, because each one "
            "sees a different aspect of a signature and has different blind spots:"
        )
        p.bullet("Global style match (DINOv2 embedding, cosine similarity) - a broad 'does this look like "
                 "the same overall shape/ink style' check.")
        p.bullet("Local stroke match (DINOv2 patch-level mutual nearest-neighbor matching) - checks whether "
                 "specific stroke segments correspond between the two signatures; the most identity-sensitive "
                 "of the three checks.")
        p.bullet("Structural match (rigid alignment + SSIM) - a classical, non-learned pixel-level check "
                 "after rotating/translating one signature onto the other.")
        p.para(
            "A single headline similarity % (the average of the three) is reported alongside the individual "
            "scores, so it's always clear whether a result is backed by all three methods agreeing or driven "
            "by just one."
        )
        p.footer("Page 2")
        p.close()

    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
