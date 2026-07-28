"""
Signature counting pass (stage B). Consumes pdf_probe output, runs the
detector, and reports how many signatures in each document are DIGITAL versus
SCANNED. No crops are written.

Why a detector is needed at all: the object layer knows exactly how many
crypto-signature fields and /Ink annotations a PDF has, but it cannot tell a
pasted signature image from a logo, and it cannot see wet ink inside a scan at
all. The detector supplies "where a signature is"; the probe supplies "what
that location is made of". Counts come from the intersection.

CLASSIFICATION of each detected box, by what it overlaps:
    pasted_image      a tightly placed image object      -> DIGITAL
    vector_ink        an /Ink annotation                 -> DIGITAL
    scanned_wet_ink   only a full-page image (a scan)    -> SCANNED
    unmatched         no object explains it              -> reported separately

Crypto-signature fields are added to the digital count directly from the probe:
they frequently have no visual appearance, so no detector will ever find them.

PREPROCESSING NOTE: training (detection_training.py:102) resized every image to
a hard 512x512, destroying aspect ratio, while inference.py did no resize at
all. That mismatch is reproduced here deliberately -- pages are squashed to
512x512 exactly as in training, then boxes are mapped back to page coordinates.

Usage:
    CUDA_VISIBLE_DEVICES=1 python detect_signs.py CORPUS \
        --probe probe_output --model ../detection/models/model_best.pth \
        --out detect_output
"""

import argparse
import csv
import os
from datetime import datetime, timezone
from collections import Counter, defaultdict

import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ingest import discover

TRAIN_SIZE = (512, 512)      # must match detection_training.py config input_size
RENDER_DPI = 200             # page render before the 512x512 squash
NUM_CLASSES = 4
# Per the dataset's categories.csv: 1=signature, 2=initials, 3=redaction,
# 4=date. inference.py:60 calls class 3 "stamp", which is wrong -- this dataset
# has no stamp class. Class 3 is also polluted: detection_training.py:142
# clamps category_id into [1, num_classes-1], so all 570 "date" boxes were
# silently trained as "redaction". Only class 1 is used for counting, so the
# counts are unaffected, but class 3 output is a redaction/date mixture.
CLASS_NAMES = {1: "signature", 2: "initials", 3: "redaction_or_date"}

# Fraction of a detection box that must fall inside an object's bbox for the
# object to be considered the explanation for that box.
OVERLAP_MIN = 0.50


# ----------------------------------------------------------------------------
# Probe output
# ----------------------------------------------------------------------------

def load_probe(probe_dir: str) -> dict:
    """Index the probe CSVs by (rel_path, page)."""
    def rows(name):
        path = os.path.join(probe_dir, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    images = defaultdict(list)
    for r in rows("images.csv"):
        bbox = r.get("bbox") or "{}"
        try:
            import json
            b = json.loads(bbox)
        except Exception:
            b = {}
        if not b:
            continue
        images[(r["rel_path"], int(r["page"]))].append({
            "x0": float(b["x0"]), "y0": float(b["y0"]),
            "x1": float(b["x1"]), "y1": float(b["y1"]),
            "role": r.get("role", ""),
        })

    annots = defaultdict(list)
    crypto = Counter()
    for r in rows("signature_candidates.csv"):
        key = (r["rel_path"], int(r["page"]))
        if r["kind"] == "crypto_signature":
            crypto[r["rel_path"]] += 1
        elif r["kind"] == "vector_ink":
            try:
                annots[key].append({"x0": float(r["x0"]), "y0": float(r["y0"]),
                                    "x1": float(r["x1"]), "y1": float(r["y1"])})
            except (KeyError, ValueError):
                pass

    pages = {}
    for r in rows("pages.csv"):
        pages[(r["rel_path"], int(r["page"]))] = r["composition"]

    return {"images": images, "ink": annots, "crypto": crypto, "pages": pages}


# ----------------------------------------------------------------------------
# Page rendering
# ----------------------------------------------------------------------------

_DOC_CACHE: dict[str, fitz.Document] = {}


def _get_doc(path: str) -> fitz.Document:
    """One open handle per path per worker process."""
    doc = _DOC_CACHE.get(path)
    if doc is None:
        doc = fitz.open(path)
        _DOC_CACHE[path] = doc
    return doc


class PageDataset(Dataset):
    """Renders a page and squashes it to 512x512, matching training."""

    def __init__(self, items: list[tuple]):
        self.items = items  # (abs_path, rel_path, page_no)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        abs_path, rel_path, pno = self.items[i]
        try:
            doc = _get_doc(abs_path)
            page = doc[pno - 1]
            prect = page.rect
            zoom = RENDER_DPI / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            # Same operation as detection_training.py:102 -- a hard squash to a
            # square, aspect ratio deliberately not preserved.
            img = img.resize(TRAIN_SIZE)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            return t, i, float(prect.width), float(prect.height), True
        except Exception:
            return (torch.zeros(3, *TRAIN_SIZE), i, 0.0, 0.0, False)


def collate(batch):
    return batch


# ----------------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------------

def covered_fraction(box: tuple, obj: dict) -> float:
    """Fraction of `box` area that falls inside `obj`'s bbox."""
    bx0, by0, bx1, by1 = box
    ix0, iy0 = max(bx0, obj["x0"]), max(by0, obj["y0"])
    ix1, iy1 = min(bx1, obj["x1"]), min(by1, obj["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    barea = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return inter / barea


def merge_nested(dets: list[dict], ios_thresh: float = 0.70) -> list[dict]:
    """
    Suppress duplicate detections of the same mark, keeping the highest score.

    Torchvision applies NMS internally, but NMS uses IoU, which cannot suppress
    a small box nested inside a large one: a box fully contained in another
    twice its area has IoU 0.5, right at the default threshold. On a single
    signature the detector routinely emits such a nest, and every extra box is
    a spurious +1 in a deliverable that is entirely counts.

    So suppression here is by intersection-over-smaller-area, which is 1.0 for
    any fully contained box regardless of the size difference. Two genuinely
    distinct signatures are rarely 70% contained in one another, so adjacent
    marks survive.
    """
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        a1 = max((d["x1"] - d["x0"]) * (d["y1"] - d["y0"]), 1e-6)
        dup = False
        for k in kept:
            ix0, iy0 = max(d["x0"], k["x0"]), max(d["y0"], k["y0"])
            ix1, iy1 = min(d["x1"], k["x1"]), min(d["y1"], k["y1"])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            a2 = max((k["x1"] - k["x0"]) * (k["y1"] - k["y0"]), 1e-6)
            if inter / min(a1, a2) >= ios_thresh:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def classify_box(box: tuple, rel_path: str, page: int, probe: dict) -> str:
    """
    Most specific explanation wins. A pasted signature sits on top of a scanned
    page in some documents, so placed images and ink annotations are tested
    before the full-page scan fallback.
    """
    key = (rel_path, page)
    for a in probe["ink"].get(key, []):
        if covered_fraction(box, a) >= OVERLAP_MIN:
            return "vector_ink"
    objs = probe["images"].get(key, [])
    for o in objs:
        if o["role"] == "placed_image" and covered_fraction(box, o) >= OVERLAP_MIN:
            return "pasted_image"
    for o in objs:
        if o["role"] == "full_page_scan" and covered_fraction(box, o) >= OVERLAP_MIN:
            return "scanned_wet_ink"
    return "unmatched"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _num(v):
    """Coerce probe CSV strings to numbers so Excel sorts them as numbers."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v if v not in (None, "") else ""


def write_xlsx(path: str, rows: list[dict], dets: list[dict],
               probe_dir: str, meta: dict) -> None:
    """
    Workbook joining the two stages on rel_path.

    The probe knows page composition and the exact digital counts; the detector
    supplies the scanned wet-ink count the probe could only mark
    `unknown_needs_detector`. One row per document, both halves side by side.
    """
    import pandas as pd

    docs: dict[str, dict] = {}
    dpath = os.path.join(probe_dir, "documents.csv")
    if os.path.exists(dpath):
        with open(dpath) as f:
            for r in csv.DictReader(f):
                docs[r["rel_path"]] = r

    main = []
    for r in rows:
        d = docs.get(r["rel_path"], {})
        main.append({
            "File name": r["file_name"],
            "Document": r["rel_path"],
            "Folder": d.get("group", ""),
            "Pages": _num(d.get("page_count")),
            "Native pages": _num(d.get("n_native_pages")),
            "Scanned pages": _num(d.get("n_scanned_pages")),
            "TOTAL signs": r["total_signs"],
            "TOTAL digital signs": r["total_digital_signs"],
            "TOTAL scanned signs": r["total_scanned_signs"],
            "  crypto sig fields": r["crypto_signature_fields"],
            "  pasted images": r["pasted_image"],
            "  vector ink": r["vector_ink"],
            "Unmatched": r["unmatched"],
            "Total detected": r["total_detected"],
            "Producer": d.get("producer", ""),
            "Creator": d.get("creator", ""),
            "Created": d.get("creation_date", ""),
            "Modified": d.get("mod_date", ""),
            "Encrypted": d.get("encrypted", ""),
            "SHA-256": d.get("sha256", ""),
            "Full path": r.get("full_path", ""),
        })
    main_df = pd.DataFrame(main)

    tot_digital = sum(r["total_digital_signs"] for r in rows)
    tot_scanned = sum(r["total_scanned_signs"] for r in rows)
    summary = pd.DataFrame([
        ("Corpus root", meta.get("root", "")),
        ("Run (UTC)", meta.get("run_utc", "")),
        ("Model", meta.get("model", "")),
        ("Confidence threshold", meta.get("conf", "")),
        ("Documents", len(rows)),
        ("TOTAL signatures", tot_digital + tot_scanned),
        ("DIGITAL signatures", tot_digital),
        ("  crypto signature fields", sum(r["crypto_signature_fields"] for r in rows)),
        ("  pasted images", sum(r["pasted_image"] for r in rows)),
        ("  vector ink annotations", sum(r["vector_ink"] for r in rows)),
        ("SCANNED signatures (wet ink)", tot_scanned),
        ("Unmatched detections", sum(r["unmatched"] for r in rows)),
        ("Documents with any signature",
         sum(1 for r in rows if r["total_digital_signs"] or r["total_scanned_signs"])),
        ("", ""),
        ("Note", "Digital counts are exact where they come from the PDF object "
                 "layer. Scanned counts come from the detector and inherit its "
                 "recall; see DETECTOR_FIXES.md for measured error rates."),
    ], columns=["Metric", "Value"])

    det_df = pd.DataFrame(dets) if dets else pd.DataFrame(
        columns=["rel_path", "file_name", "page", "detector_class", "score",
                 "x0", "y0", "x1", "y1", "classification", "page_composition"])

    sheets = [("Summary", summary), ("Per document", main_df),
              ("Detections", det_df)]
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, df in sheets:
            df.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                longest = max([len(str(col))] +
                              [len(str(v)) for v in df[col].head(500)])
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter
                ].width = min(longest + 2, 60)


def load_model(model_path: str, device: str):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    return model.eval().to(device)


def run(root, probe_dir, model_path, out_dir, conf=0.5, batch_size=16,
        workers=16, scanned_only=False, keep_classes=(1,),
        xlsx_path=None):
    os.makedirs(out_dir, exist_ok=True)
    probe = load_probe(probe_dir)
    if not probe["pages"]:
        raise SystemExit(f"No pages.csv content in {probe_dir}. Run pdf_probe.py first.")

    records, _ = discover(root, image_role="page")
    pdfs = {r.rel_path: r.abs_path for r in records if r.kind == "pdf"}

    items = []
    for (rel, pno), comp in sorted(probe["pages"].items()):
        if rel not in pdfs:
            continue
        if comp == "empty":
            continue
        if scanned_only and comp not in ("scanned", "scanned_with_text_layer"):
            continue
        items.append((pdfs[rel], rel, pno))

    if not items:
        raise SystemExit("No pages selected.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Pages  : {len(items)} from {len(pdfs)} PDFs")
    print(f"Mode   : {'scanned pages only' if scanned_only else 'all non-empty pages'}\n")

    model = load_model(model_path, device)
    loader = DataLoader(PageDataset(items), batch_size=batch_size,
                        num_workers=workers, collate_fn=collate, shuffle=False)

    dets_out = []
    per_doc = defaultdict(Counter)
    n_render_fail = 0
    n_suppressed = 0
    done = 0

    with torch.no_grad():
        for batch in loader:
            tensors = [b[0].to(device) for b in batch if b[4]]
            metas = [b for b in batch if b[4]]
            n_render_fail += sum(1 for b in batch if not b[4])
            if tensors:
                preds = model(tensors)
                for pred, (_, idx, pw, ph, _) in zip(preds, metas):
                    _, rel, pno = items[idx]
                    keep = pred["scores"] > conf
                    boxes = pred["boxes"][keep].cpu().numpy()
                    labels = pred["labels"][keep].cpu().numpy()
                    scores = pred["scores"][keep].cpu().numpy()
                    # Undo the 512x512 squash: independent x and y scaling.
                    sx, sy = pw / TRAIN_SIZE[0], ph / TRAIN_SIZE[1]
                    page_dets = []
                    for bx, lb, sc in zip(boxes, labels, scores):
                        if int(lb) not in keep_classes:
                            per_doc[rel][f"other_{CLASS_NAMES.get(int(lb), lb)}"] += 1
                            continue
                        page_dets.append({
                            "rel_path": rel,
                            "file_name": os.path.basename(rel),
                            "page": pno,
                            "detector_class": CLASS_NAMES.get(int(lb), str(lb)),
                            "score": round(float(sc), 4),
                            "x0": round(float(bx[0] * sx), 2),
                            "y0": round(float(bx[1] * sy), 2),
                            "x1": round(float(bx[2] * sx), 2),
                            "y1": round(float(bx[3] * sy), 2),
                        })
                    merged = merge_nested(page_dets)
                    n_suppressed += len(page_dets) - len(merged)
                    for d in merged:
                        cls = classify_box((d["x0"], d["y0"], d["x1"], d["y1"]),
                                           rel, pno, probe)
                        per_doc[rel][cls] += 1
                        d["classification"] = cls
                        d["page_composition"] = probe["pages"].get((rel, pno), "")
                        dets_out.append(d)
            done += len(batch)
            if done % (batch_size * 20) < batch_size or done >= len(items):
                print(f"  {min(done, len(items))}/{len(items)} pages")

    # ---- per-document counts ----
    rows = []
    for rel in sorted(pdfs):
        c = per_doc.get(rel, Counter())
        crypto = probe["crypto"].get(rel, 0)
        digital = c["pasted_image"] + c["vector_ink"] + crypto
        scanned = c["scanned_wet_ink"]
        rows.append({
            "rel_path": rel,
            "file_name": os.path.basename(rel),
            "full_path": pdfs[rel],
            # Headline figure: every signature found in the document, however
            # it got there. Distinct from total_detected below, which counts
            # detector output only -- it excludes crypto fields (invisible, so
            # never detected) and includes unmatched boxes (not confirmed
            # signatures).
            "total_signs": digital + scanned,
            "total_digital_signs": digital,
            "total_scanned_signs": scanned,
            "crypto_signature_fields": crypto,
            "pasted_image": c["pasted_image"],
            "vector_ink": c["vector_ink"],
            "scanned_wet_ink": scanned,
            "unmatched": c["unmatched"],
            "total_detected": c["pasted_image"] + c["vector_ink"] + scanned
                              + c["unmatched"],
        })

    counts_path = os.path.join(out_dir, "signature_counts.csv")
    with open(counts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    det_path = os.path.join(out_dir, "detections.csv")
    with open(det_path, "w", newline="") as f:
        if dets_out:
            w = csv.DictWriter(f, fieldnames=list(dets_out[0].keys()))
            w.writeheader()
            w.writerows(dets_out)

    tot = Counter()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tot[k] += v

    print(f"\n{'='*60}\nSIGNATURE COUNTS\n{'='*60}")
    print(f"Documents            : {len(rows)}")
    print(f"\n  TOTAL signatures   : {tot['total_signs']}")
    print(f"\n  DIGITAL            : {tot['total_digital_signs']}")
    print(f"    crypto fields    : {tot['crypto_signature_fields']}")
    print(f"    pasted images    : {tot['pasted_image']}")
    print(f"    vector ink       : {tot['vector_ink']}")
    print(f"\n  SCANNED (wet ink)  : {tot['total_scanned_signs']}")
    print(f"\n  unmatched          : {tot['unmatched']}"
          f"   (detected, no object explains it -- inspect these)")
    if n_suppressed:
        print(f"\n  nested duplicates suppressed: {n_suppressed} "
              f"(same mark detected more than once)")
    if n_render_fail:
        print(f"  render failures    : {n_render_fail} pages")
    if xlsx_path:
        write_xlsx(xlsx_path, rows, dets_out, probe_dir, {
            "root": os.path.abspath(root),
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": os.path.abspath(model_path),
            "conf": conf,
        })
        print(f"\nExcel        : {os.path.abspath(xlsx_path)}"
              f"   (sheets: Summary, Per document, Detections)")

    print(f"\nPer document : {counts_path}")
    print(f"Detections   : {det_path}")
    print("\nNOTE: these counts inherit the detector's recall, which has never "
          "been measured (the checkpoint stores val_loss only). A missed "
          "signature is silently absent from every number above.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="corpus root folder")
    ap.add_argument("--probe", default="probe_output", help="pdf_probe output dir")
    ap.add_argument("--model", default="detection/models/model_best.pth")
    ap.add_argument("--out", default="detect_output")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scanned-only", action="store_true",
                    help="only detect on scanned pages (forfeits the "
                         "placed-image filter on native pages)")
    ap.add_argument("--include-initials", action="store_true")
    ap.add_argument("--xlsx", dest="xlsx_path",
                    help="write a workbook joining probe metadata with the counts")
    args = ap.parse_args()

    keep = (1, 2) if args.include_initials else (1,)
    run(args.root, args.probe, args.model, args.out, args.conf,
        args.batch_size, args.workers, args.scanned_only, keep,
        args.xlsx_path)


if __name__ == "__main__":
    main()
