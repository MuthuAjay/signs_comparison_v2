"""
Build a single-page PDF report from the comparison_output/ artifacts
produced by comapare_signs.py (preprocessed crops, match visualizations,
comparison_report.csv).

Usage:
    python generate_report.py
"""

import csv
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

OUT_DIR = "comparison_output"
REPORT_PATH = os.path.join(OUT_DIR, "signature_comparison_report.pdf")

CASE_META = {
    "shafiul_cmm": {
        "label": "Shafiul Islam - CMM Letter",
        "doc": "Letter of Authority - CR Case at CMM.pdf",
        "date": "16/06/2025",
        "role": "Chairman, Navana Limited",
    },
    "shafiul_hcd": {
        "label": "Shafiul Islam - HCD Letter",
        "doc": "Letter of Authority - Arbitration Case in HCD.pdf",
        "date": "02/09/2025",
        "role": "Chairman, Navana Limited",
    },
    "shafqat_hcd": {
        "label": "Shafqat Ahmed - HCD Letter (control)",
        "doc": "Letter of Authority - Arbitration Case in HCD.pdf",
        "date": "02/09/2025",
        "role": "Head of Strategic Planning & Insights (different signer)",
    },
}

PRIMARY_PAIR = "shafiul_cmm vs shafiul_hcd"


def load_rows():
    with open(os.path.join(OUT_DIR, "comparison_report.csv")) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    row_by_pair = {r["pair"]: r for r in rows}
    primary = row_by_pair[PRIMARY_PAIR]
    others = [r for r in rows if r["pair"] != PRIMARY_PAIR]

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(
        7, 1,
        figure=fig,
        height_ratios=[0.5, 0.5, 1.5, 0.3, 2.1, 1.9, 2.1],
        hspace=0.25,
        left=0.07, right=0.93, top=0.97, bottom=0.03,
    )

    # ---- Title ----
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(0, 0.75, "Signature Comparison Report", fontsize=20, fontweight="bold", va="top")
    ax_title.text(0, 0.05,
                  "Automated signature detection + DINOv2 / SSIM similarity analysis",
                  fontsize=10.5, color="#444444", va="top")

    # ---- Case metadata ----
    ax_meta = fig.add_subplot(gs[1])
    ax_meta.axis("off")
    meta_lines = (
        f"Document A:  {CASE_META['shafiul_cmm']['doc']}   (dated {CASE_META['shafiul_cmm']['date']})\n"
        f"Document B:  {CASE_META['shafiul_hcd']['doc']}   (dated {CASE_META['shafiul_hcd']['date']})\n"
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    ax_meta.text(0, 1.0, meta_lines, fontsize=9, va="top", family="monospace", linespacing=1.6)

    # ---- Signature crops ----
    gs_imgs = gs[2].subgridspec(1, 3, wspace=0.15)
    order = ["shafiul_cmm", "shafiul_hcd", "shafqat_hcd"]
    for i, key in enumerate(order):
        ax = fig.add_subplot(gs_imgs[i])
        img = Image.open(os.path.join(OUT_DIR, f"preprocessed_{key}.png"))
        ax.imshow(img, cmap="gray")
        ax.set_title(CASE_META[key]["label"], fontsize=8.3, pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)

    # ---- Section divider ----
    ax_div1 = fig.add_subplot(gs[3])
    ax_div1.axis("off")
    ax_div1.text(0, 0.5, "Similarity Metrics", fontsize=12, fontweight="bold", va="center")
    ax_div1.axhline(0.05, color="#cccccc", lw=1)

    # ---- Metrics table ----
    ax_table = fig.add_subplot(gs[4])
    ax_table.axis("off")
    col_labels = ["Pair", "Type", "Global\ncosine", "Patch match\nscore", "SSIM\n(aligned)"]
    table_rows = []
    row_colors = []
    for r in [primary] + others:
        is_primary = r is primary
        table_rows.append([
            r["pair"].replace(" vs ", "\nvs "),
            r["type"],
            f"{float(r['global_cosine']):.3f}",
            f"{float(r['patch_match_score']):.3f}",
            f"{float(r['ssim_aligned']):.3f}",
        ])
        row_colors.append("#e8f4ea" if is_primary else "#ffffff")

    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dddddd")
        else:
            cell.set_facecolor(row_colors[row - 1])
        cell.set_edgecolor("#bbbbbb")
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    ax_table.set_position([
        ax_table.get_position().x0, ax_table.get_position().y0 + 0.03,
        ax_table.get_position().width, ax_table.get_position().height - 0.03,
    ])
    ax_table.text(
        0, -0.1,
        "Highlighted row = same signer, different documents (Shafiul Islam). "
        "Other rows = different-signer control (Shafqat Ahmed).",
        fontsize=7.5, color="#555555", transform=ax_table.transAxes,
    )

    # ---- Match visualization ----
    ax_viz = fig.add_subplot(gs[5])
    ax_viz.axis("off")
    ax_viz.text(0, 1.05, "Patch-Level Match: Shafiul Islam (CMM vs HCD)",
                fontsize=12, fontweight="bold", va="bottom", transform=ax_viz.transAxes)
    viz_img = Image.open(os.path.join(OUT_DIR, "matches_shafiul_cmm_vs_shafiul_hcd.png"))
    ax_viz.imshow(viz_img)

    # ---- Assessment text ----
    ax_assess = fig.add_subplot(gs[6])
    ax_assess.axis("off")
    assessment = (
        "Assessment: The Shafiul Islam signature on the CMM letter and the Shafiul Islam signature on the\n"
        "HCD letter score highest on all three independent metrics (cosine, patch-match, SSIM), and are\n"
        "clearly separated from both comparisons against Shafqat Ahmed's signature (different signer,\n"
        "included as a control). This is consistent with the same individual having signed both documents.\n"
        "\n"
        "Caveats: (1) No second genuine exemplar of Shafiul Islam's signature was available to calibrate a\n"
        "formal accept/reject threshold; scores should be read as a clustering signal, not a certified\n"
        "match/no-match verdict. (2) The HCD letter is a lower-resolution scan than the CMM letter (visible\n"
        "stroke jaggedness persisted even at 600 DPI re-render), which likely suppresses the measured\n"
        "similarity somewhat -- true similarity is probably at least as high as reported."
    )
    ax_assess.text(0, 1.0, assessment, fontsize=8.3, va="top", linespacing=1.45)

    fig.savefig(REPORT_PATH, dpi=200)
    plt.close(fig)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
