"""
Extract placed images so `placed_image` candidates can be eyeballed.

`pdf_probe` can tell that an image was placed at some spot on a page, but not
what it depicts -- a pasted signature, a logo, a seal and a letterhead graphic
are indistinguishable in the object layer. This dumps them to disk so the
question can be settled by looking.

Input is a single PDF or a folder (walked recursively). Full-page scans are
skipped by default: they are pages, not placed artwork.

Images are composited onto white before saving. Pasted signatures usually carry
an alpha channel, and saving the raw stream renders the ink invisible or on
black, which defeats the point of looking at them.

Outputs:
    <out>/images/*.png   one file per placed image, named for its source
    <out>/index.csv      one row per image: source, page, bbox, dpi, sha256
    <out>/contact_sheet_*.png   optional grids for fast review

Usage:
    python extract_placed_images.py CORPUS_OR_PDF --out placed_images
    python extract_placed_images.py doc.pdf --out placed_images --contact-sheet
"""

import argparse
import csv
import hashlib
import os
import re

import fitz  # PyMuPDF
from PIL import Image

from pdf_probe import (FULLPAGE_AREA_RATIO, FULLPAGE_SPAN_RATIO,
                       FULLPAGE_SPAN_MIN_AREA)

MIN_PX = 24          # below this on either side it is an icon or a rule, not a mark
THUMB = 220          # contact-sheet cell size
COLS = 6


def safe_stem(rel_path: str) -> str:
    stem = os.path.splitext(rel_path)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem.replace(os.sep, "__"))[:120]


def is_full_page(bbox, prect) -> bool:
    """Same rule as pdf_probe, so the two agree on what counts as a scan."""
    parea = max(prect.width * prect.height, 1e-6)
    ratio = max(bbox.width * bbox.height, 0.0) / parea
    span_w = bbox.width / prect.width if prect.width else 0.0
    span_h = bbox.height / prect.height if prect.height else 0.0
    return (ratio >= FULLPAGE_AREA_RATIO
            or (max(span_w, span_h) >= FULLPAGE_SPAN_RATIO
                and ratio >= FULLPAGE_SPAN_MIN_AREA))


def load_image(doc, xref: int, smask: int) -> Image.Image | None:
    """Decode an image XObject to RGB, compositing any alpha onto white."""
    try:
        pix = fitz.Pixmap(doc, xref)
        if smask:
            try:
                pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
            except Exception:
                pass                       # keep the un-masked version
        if pix.colorspace is None:
            return None
        if pix.colorspace.n > 3:           # CMYK and friends
            pix = fitz.Pixmap(fitz.csRGB, pix)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img, mask=img.split()[3])
            img = bg
        return img
    except Exception:
        return None


def extract_from_pdf(abs_path, rel_path, img_dir, min_px, include_full_page):
    rows = []
    try:
        doc = fitz.open(abs_path)
    except Exception as e:
        print(f"  ! {rel_path}: {type(e).__name__}: {e}")
        return rows

    stem = safe_stem(rel_path)
    for page in doc:
        prect = page.rect
        for info in page.get_images(full=True):
            xref, smask = info[0], info[1]
            try:
                bbox = page.get_image_bbox(info)
            except Exception:
                continue
            full = is_full_page(bbox, prect)
            if full and not include_full_page:
                continue

            img = load_image(doc, xref, smask)
            if img is None or img.width < min_px or img.height < min_px:
                continue

            name = f"{stem}__p{page.number + 1}__x{xref}.png"
            out_path = os.path.join(img_dir, name)
            try:
                img.save(out_path)
            except Exception:
                continue

            raw = b""
            try:
                raw = doc.extract_image(xref).get("image", b"")
            except Exception:
                pass
            eff_dpi = round(img.width / (bbox.width / 72.0), 1) \
                if bbox.width > 1 else ""
            parea = max(prect.width * prect.height, 1e-6)

            rows.append({
                "image_file": name,
                "rel_path": rel_path,
                "file_name": os.path.basename(rel_path),
                "page": page.number + 1,
                "xref": xref,
                "role": "full_page_scan" if full else "placed_image",
                "px_w": img.width, "px_h": img.height,
                "x0": round(bbox.x0, 1), "y0": round(bbox.y0, 1),
                "x1": round(bbox.x1, 1), "y1": round(bbox.y1, 1),
                "page_area_ratio": round(
                    (bbox.width * bbox.height) / parea, 4),
                "effective_dpi": eff_dpi,
                "has_alpha": bool(smask),
                "stream_sha256": hashlib.sha256(raw).hexdigest() if raw else "",
                "full_path": abs_path,
            })
    doc.close()
    return rows


def contact_sheets(rows, img_dir, out_dir, per_sheet=36):
    """Grids of thumbnails -- the fastest way to sort signatures from logos."""
    made = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        cols = COLS
        n_rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * THUMB, n_rows * THUMB), "white")
        for i, r in enumerate(chunk):
            try:
                im = Image.open(os.path.join(img_dir, r["image_file"]))
            except Exception:
                continue
            im.thumbnail((THUMB - 10, THUMB - 10))
            x = (i % cols) * THUMB + (THUMB - im.width) // 2
            y = (i // cols) * THUMB + (THUMB - im.height) // 2
            sheet.paste(im, (x, y))
        path = os.path.join(out_dir, f"contact_sheet_{s // per_sheet + 1:03d}.png")
        sheet.save(path)
        made.append(path)
    return made


def resolve_inputs(target: str) -> list[tuple[str, str]]:
    """(abs_path, rel_path) for a single PDF or every PDF under a folder."""
    if os.path.isfile(target):
        return [(os.path.abspath(target), os.path.basename(target))]
    if os.path.isdir(target):
        from ingest import discover
        recs, _ = discover(target, image_role="page")
        return [(r.abs_path, r.rel_path) for r in recs if r.kind == "pdf"]
    raise SystemExit(f"Not a file or directory: {target}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a PDF, or a folder walked recursively")
    ap.add_argument("--out", default="placed_images")
    ap.add_argument("--min-px", type=int, default=MIN_PX,
                    help=f"skip images smaller than this (default {MIN_PX})")
    ap.add_argument("--include-full-page", action="store_true",
                    help="also extract full-page scans")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also write thumbnail grids for fast review")
    ap.add_argument("--limit", type=int, help="stop after N source PDFs")
    args = ap.parse_args()

    inputs = resolve_inputs(args.target)
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        raise SystemExit("No PDFs found.")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Source PDFs : {len(inputs)}")
    rows = []
    for i, (abs_path, rel_path) in enumerate(inputs, 1):
        rows += extract_from_pdf(abs_path, rel_path, img_dir,
                                 args.min_px, args.include_full_page)
        if i % 50 == 0 or i == len(inputs):
            print(f"  {i}/{len(inputs)} PDFs, {len(rows)} images")

    if not rows:
        print("\nNo placed images found "
              "(try --min-px 0, or --include-full-page).")
        return

    idx = os.path.join(args.out, "index.csv")
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dupes = {}
    for r in rows:
        if r["stream_sha256"]:
            dupes.setdefault(r["stream_sha256"], []).append(r["image_file"])
    repeated = {k: v for k, v in dupes.items() if len(v) > 1}

    print(f"\nImages extracted : {len(rows)}")
    print(f"  with alpha     : {sum(1 for r in rows if r['has_alpha'])}"
          f"   (pasted signatures usually do)")
    if repeated:
        print(f"  byte-identical : {len(repeated)} image(s) appear more than "
              f"once across the corpus")
    if args.contact_sheet:
        sheets = contact_sheets(rows, img_dir, args.out)
        print(f"  contact sheets : {len(sheets)}")
    print(f"\nImages : {os.path.abspath(img_dir)}")
    print(f"Index  : {os.path.abspath(idx)}")


if __name__ == "__main__":
    main()
