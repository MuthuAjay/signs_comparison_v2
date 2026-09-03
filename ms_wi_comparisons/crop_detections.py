"""
Crop stage: turn detector boxes into signature images (bridges detect -> compare).

`detect_signs.py` reports *where* each signature is, in PDF point coordinates,
but writes no pixels. `comapare_signs.py` consumes a folder of crops. This is
the step between them.

Boxes come from the detector run at 512x512, so they are coarse. Rendering the
crop at a high DPI and padding generously matters more than it looks: the
comparison stage re-crops to the ink bounding box itself, so a padded box costs
nothing, while a tight box that clips a descender or a trailing flourish
destroys exactly the stroke evidence the comparison depends on.

Output layout mirrors the corpus groups, so `comapare_signs.py` picks the group
labels up from the directory names and its `cross_group` column becomes
"reference vs questioned".

Usage:
    python crop_detections.py --detections detect/detections.csv \
        --corpus corpus --out crops [--dpi 400] [--pad 0.12] [--min-score 0.3]
"""

import argparse
import csv
import os
import re

import fitz  # PyMuPDF
from PIL import Image


def safe_stem(rel_path: str) -> str:
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:100]


def crop_one(page, box, dpi, pad_frac):
    """Render just the padded box region at `dpi`. Returns a PIL image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # Padding is a fraction of the box's own size, with a point floor so a
    # small box still gets a usable margin.
    px, py = max(w * pad_frac, 6.0), max(h * pad_frac, 6.0)
    rect = fitz.Rect(x0 - px, y0 - py, x1 + px, y1 + py) & page.rect
    if rect.is_empty:
        return None
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True, help="detections.csv from detect_signs.py")
    ap.add_argument("--corpus", required=True, help="corpus root the rel_paths are relative to")
    ap.add_argument("--out", required=True, help="output crop directory")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--pad", type=float, default=0.12,
                    help="padding as a fraction of box size (default 0.12)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--classes", default="signature",
                    help="comma-separated detector_class values to keep")
    args = ap.parse_args()

    keep = {c.strip() for c in args.classes.split(",") if c.strip()}

    with open(args.detections) as f:
        rows = [r for r in csv.DictReader(f)
                if r["detector_class"] in keep
                and float(r["score"]) >= args.min_score]
    if not rows:
        raise SystemExit("No detections match the filters.")

    docs: dict[str, fitz.Document] = {}
    index = []
    for r in rows:
        rel = r["rel_path"]
        abs_path = os.path.join(args.corpus, rel)
        if rel not in docs:
            docs[rel] = fitz.open(abs_path)
        page = docs[rel][int(r["page"]) - 1]
        box = (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
        img = crop_one(page, box, args.dpi, args.pad)
        if img is None:
            print(f"  ! empty clip: {rel} p{r['page']}")
            continue

        # Group directory = the corpus sub-directory, so the comparison stage
        # inherits reference/questioned as its group labels.
        group = os.path.dirname(rel) or "."
        gdir = os.path.join(args.out, group)
        os.makedirs(gdir, exist_ok=True)
        name = (f"{safe_stem(rel)}__p{r['page']}"
                f"__x{int(float(r['x0']))}_y{int(float(r['y0']))}.png")
        path = os.path.join(gdir, name)
        img.save(path)
        index.append({**r, "crop_file": os.path.relpath(path, args.out),
                      "crop_px_w": img.width, "crop_px_h": img.height})

    for d in docs.values():
        d.close()

    idx_path = os.path.join(args.out, "crop_index.csv")
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)

    print(f"Crops written : {len(index)}  ({args.dpi} dpi, pad {args.pad})")
    for g in sorted({os.path.dirname(i['crop_file']) for i in index}):
        n = sum(1 for i in index if os.path.dirname(i['crop_file']) == g)
        print(f"  {g or '.':<24}{n}")
    print(f"Index         : {os.path.abspath(idx_path)}")


if __name__ == "__main__":
    main()
