"""
Object-layer probe (build step 4, reconnaissance pass).

Walks a corpus folder recursively and reports, for every PDF, what the object
layer says about signatures and how each page was produced -- before any
detector runs.

WHAT THIS CAN COUNT EXACTLY (it is read from the PDF structure, not inferred):
    crypto_signature   AcroForm /Sig fields, with signer / time / reason
    vector_ink         /Ink annotations (stylus, Adobe Fill & Sign)
    stamp_annot        /Stamp annotations
    image objects      every embedded image, with page bbox and effective DPI

WHAT IT CANNOT COUNT: wet-ink signatures on a scanned page. A full-page scan is
one image object; whether it carries zero, one or five handwritten signatures is
a pixel question that needs the detector. Scanned pages are therefore reported
as *candidate pages* with an unknown signature count, never as a count.

So "digital signature count" here is exact, and "scanned signature count" is
deliberately absent -- what you get instead is the count of pages that could
carry one. Conflating those two is the main way this analysis goes wrong.

Outputs (all in --out):
    documents.csv            one row per PDF: metadata + per-kind counts
    pages.csv                one row per page: composition, what was found
    signature_candidates.csv one row per candidate, with page and object ref
    images.csv               every embedded image object, with forensic metrics
    metadata.json            full dump incl. XMP, fonts, subset tags, raw meta

Usage:
    python pdf_probe.py /path/to/corpus --out probe_output
"""

import argparse
import csv
import json
import os
import re
import traceback
from collections import Counter
from datetime import datetime, timezone

import fitz  # PyMuPDF

from ingest import discover, sha256_file

# Full-page-scan test. Area ratio alone is too brittle: a scan is placed with
# its aspect ratio preserved, so a Letter-sized scan on an A4 page letterboxes
# to ~0.3 of the page area while still plainly being the whole page. So an
# image also counts as full-page if it spans essentially the entire width or
# height and is substantial. A letterhead banner spans the full width but is
# thin, and is excluded by the area floor.
FULLPAGE_AREA_RATIO = 0.80
FULLPAGE_SPAN_RATIO = 0.95
FULLPAGE_SPAN_MIN_AREA = 0.30
# Below this many extracted characters a page is not meaningfully "native text".
NATIVE_TEXT_MIN_CHARS = 50

SIG_FIELD_KEYS = ("Name", "M", "Reason", "Location", "ContactInfo",
                  "SubFilter", "Filter", "ByteRange")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _rect_dict(r) -> dict:
    return {"x0": round(r.x0, 2), "y0": round(r.y0, 2),
            "x1": round(r.x1, 2), "y1": round(r.y1, 2),
            "w": round(r.width, 2), "h": round(r.height, 2)}


def _deref(value: str) -> int | None:
    """'123 0 R' -> 123. Returns None if the value is not an indirect ref."""
    m = re.match(r"^\s*(\d+)\s+\d+\s+R\s*$", value or "")
    return int(m.group(1)) if m else None


def _clean_pdf_string(v: str | None) -> str | None:
    """Strip PDF literal-string wrapping so values land readable in CSV."""
    if v is None:
        return None
    v = v.strip()
    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1]
    return v.replace("\n", " ").replace("\r", " ").strip() or None


def count_incremental_saves(path: str) -> int:
    """
    Number of '%%EOF' markers in the raw file, a proxy for how many times the
    document was saved. More than one means the PDF was updated after its
    original write -- interesting when a signature is the thing that changed.

    Approximate: a full rewrite or linearize collapses the chain back to one,
    so a value of 1 does not prove the file was never edited.
    """
    try:
        with open(path, "rb") as f:
            return f.read().count(b"%%EOF")
    except OSError:
        return -1


def signature_field_details(doc, widget) -> dict:
    """Pull /V of a signature field: signer, signing time, reason, SubFilter."""
    out = {"signed": False}
    try:
        xref = getattr(widget, "xref", 0)
        if not xref:
            return out
        _, v = doc.xref_get_key(xref, "V")
        vref = _deref(v)
        if v in (None, "null") or (vref is None and not v):
            return out
        out["signed"] = True
        if vref is None:
            return out
        for key in SIG_FIELD_KEYS:
            try:
                _, val = doc.xref_get_key(vref, key)
            except Exception:
                continue
            if val and val != "null":
                out[key.lower()] = _clean_pdf_string(val)[:300] \
                    if isinstance(val, str) else val
    except Exception as e:
        out["detail_error"] = str(e)[:200]
    return out


# ----------------------------------------------------------------------------
# Per-document probe
# ----------------------------------------------------------------------------

def probe_document(abs_path: str, rel_path: str, group: str,
                   sha256: str) -> dict:
    doc_rec = {
        "rel_path": rel_path, "group": group, "sha256": sha256,
        "status": "ok", "error": None,
    }
    pages_out, cands_out, images_out = [], [], []

    # MuPDF reports damage as warnings on stderr, not as exceptions: a content
    # stream that fails to decrypt ("partial block in aes filter") yields an
    # empty stream and the page silently looks blank. Without capturing these,
    # an undecryptable document is indistinguishable from one that genuinely
    # holds no signatures -- an undercount with nothing to flag it.
    fitz.TOOLS.reset_mupdf_warnings()

    doc = fitz.open(abs_path)

    if doc.is_encrypted:
        # Try the empty owner password; many "encrypted" PDFs are only
        # permission-restricted and open fine.
        doc_rec["encrypted"] = True
        doc_rec["decrypted_with_empty_password"] = bool(doc.authenticate(""))
        if doc.is_encrypted:
            doc_rec["status"] = "encrypted_unreadable"
            return doc_rec, pages_out, cands_out, images_out
    else:
        doc_rec["encrypted"] = False

    meta = doc.metadata or {}
    doc_rec.update({
        "page_count": doc.page_count,
        "pdf_format": meta.get("format"),
        "title": _clean_pdf_string(meta.get("title")),
        "author": _clean_pdf_string(meta.get("author")),
        "subject": _clean_pdf_string(meta.get("subject")),
        "keywords": _clean_pdf_string(meta.get("keywords")),
        "creator": _clean_pdf_string(meta.get("creator")),
        "producer": _clean_pdf_string(meta.get("producer")),
        "creation_date": meta.get("creationDate"),
        "mod_date": meta.get("modDate"),
        "trapped": meta.get("trapped"),
        "is_form_pdf": bool(doc.is_form_pdf),
        "sig_flags": doc.get_sigflags(),   # -1 none, 1 has sigs, 3 append-only
        "xref_length": doc.xref_length(),
        "incremental_saves": count_incremental_saves(abs_path),
        "is_repaired": bool(getattr(doc, "is_repaired", False)),
    })

    try:
        doc_rec["xmp_present"] = bool(doc.get_xml_metadata())
    except Exception:
        doc_rec["xmp_present"] = False

    fonts_all, subset_tags = set(), {}
    counts = Counter()

    for page in doc:
        pno = page.number + 1
        prect = page.rect
        parea = max(prect.width * prect.height, 1e-6)
        text = page.get_text().strip()
        n_chars = len(text)

        # ---- images on this page ----
        page_images, full_page_images = [], 0
        for img in page.get_images(full=True):
            xref, smask = img[0], img[1]
            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                bbox = fitz.Rect(0, 0, 0, 0)
            try:
                info = doc.extract_image(xref)
            except Exception:
                info = {}
            iw, ih = info.get("width", img[2]), info.get("height", img[3])
            barea = max(bbox.width * bbox.height, 0.0)
            ratio = barea / parea
            # Effective DPI: native pixels across the placed width in points.
            eff_dpi = round(iw / (bbox.width / 72.0), 1) if bbox.width > 1 else None
            span_w = bbox.width / prect.width if prect.width else 0.0
            span_h = bbox.height / prect.height if prect.height else 0.0
            is_full_page = (
                ratio >= FULLPAGE_AREA_RATIO
                or (max(span_w, span_h) >= FULLPAGE_SPAN_RATIO
                    and ratio >= FULLPAGE_SPAN_MIN_AREA)
            )
            full_page_images += int(is_full_page)

            rec = {
                "rel_path": rel_path, "page": pno, "xref": xref,
                "px_w": iw, "px_h": ih,
                "ext": info.get("ext"), "colorspace": info.get("colorspace"),
                "bpc": info.get("bpc"), "has_smask": bool(smask),
                "bbox": _rect_dict(bbox), "page_area_ratio": round(ratio, 4),
                "span_w": round(span_w, 4), "span_h": round(span_h, 4),
                "effective_dpi": eff_dpi,
                "role": "full_page_scan" if is_full_page else "placed_image",
                "stream_sha256": None,
            }
            if info.get("image"):
                import hashlib
                rec["stream_sha256"] = hashlib.sha256(info["image"]).hexdigest()
            images_out.append(rec)
            page_images.append(rec)

            # A tightly placed image on a page that also has native text is the
            # classic pasted-signature shape. Reported as a candidate, not a
            # conclusion -- it may equally be a logo, a seal or a letterhead.
            if not is_full_page and barea > 0:
                counts["placed_image"] += 1
                cands_out.append({
                    "rel_path": rel_path, "group": group, "page": pno,
                    "kind": "placed_image", "object_ref": f"xref:{xref}",
                    **_rect_dict(bbox),
                    "px_w": iw, "px_h": ih, "ext": info.get("ext"),
                    "alpha": bool(smask), "effective_dpi": eff_dpi,
                    "page_area_ratio": round(ratio, 4),
                    "stream_sha256": rec["stream_sha256"],
                    "signed": "", "signer": "", "signed_at": "",
                    "reason": "", "location": "", "subfilter": "",
                })

        # ---- signature form fields ----
        page_sigs = 0
        try:
            widgets = list(page.widgets())
        except Exception:
            widgets = []
        for w in widgets:
            if w.field_type != fitz.PDF_WIDGET_TYPE_SIGNATURE:
                continue
            page_sigs += 1
            counts["crypto_signature"] += 1
            d = signature_field_details(doc, w)
            counts["crypto_signature_signed"] += int(d.get("signed", False))
            cands_out.append({
                "rel_path": rel_path, "group": group, "page": pno,
                "kind": "crypto_signature",
                "object_ref": f"field:{w.field_name or ''}",
                **_rect_dict(w.rect),
                "signed": d.get("signed", False),
                "signer": d.get("name", ""),
                "signed_at": d.get("m", ""),
                "reason": d.get("reason", ""),
                "location": d.get("location", ""),
                "subfilter": d.get("subfilter", ""),
            })

        # ---- annotations ----
        annot_types = []
        try:
            for a in page.annots() or []:
                aname = a.type[1]
                annot_types.append(aname)
                if aname in ("Ink", "Stamp", "FreeText"):
                    kind = {"Ink": "vector_ink", "Stamp": "stamp_annot",
                            "FreeText": "freetext_annot"}[aname]
                    counts[kind] += 1
                    cands_out.append({
                        "rel_path": rel_path, "group": group, "page": pno,
                        "kind": kind, "object_ref": f"xref:{a.xref}",
                        **_rect_dict(a.rect),
                        "signed": "", "signer": "", "signed_at": "",
                        "reason": "", "location": "", "subfilter": "",
                    })
        except Exception:
            pass

        # ---- fonts (subset tags kept: they fingerprint the authoring session) ----
        page_fonts = []
        try:
            for f in page.get_fonts(full=True):
                basefont = f[3]
                page_fonts.append(basefont)
                fonts_all.add(basefont)
                if "+" in basefont:
                    tag, base = basefont.split("+", 1)
                    subset_tags.setdefault(base, set()).add(tag)
        except Exception:
            pass

        try:
            n_drawings = len(page.get_drawings())
        except Exception:
            n_drawings = -1

        # ---- page composition ----
        has_text = n_chars >= NATIVE_TEXT_MIN_CHARS
        if full_page_images and not has_text:
            composition = "scanned"
        elif full_page_images and has_text:
            composition = "scanned_with_text_layer"   # OCR'd, or scan + overlay
        elif has_text or n_drawings > 0:
            composition = "native"
        elif page_images:
            # Imagery but nothing full-page and no text. Not empty -- flagged
            # separately so it cannot be silently dropped from the counts.
            composition = "image_only"
        else:
            composition = "empty"

        if composition in ("scanned", "scanned_with_text_layer"):
            counts["scanned_page"] += 1
        elif composition == "native":
            counts["native_page"] += 1
        elif composition == "image_only":
            counts["image_only_page"] += 1
        else:
            counts["empty_page"] += 1

        pages_out.append({
            "rel_path": rel_path, "group": group, "page": pno,
            "composition": composition,
            "page_w": round(prect.width, 2), "page_h": round(prect.height, 2),
            "rotation": page.rotation,
            "n_chars": n_chars, "n_images": len(page_images),
            "n_full_page_images": full_page_images,
            "n_drawings": n_drawings,
            "n_crypto_sig_fields": page_sigs,
            "annots": ";".join(sorted(set(annot_types))),
            "fonts": ";".join(sorted(set(page_fonts)))[:500],
        })

    # Same base font carrying more than one subset tag: the pages were not all
    # produced in one authoring session. An indicator only -- re-saving and
    # merging in some tools renormalizes these tags.
    multi_tag = {b: sorted(t) for b, t in subset_tags.items() if len(t) > 1}

    doc_rec.update({
        "n_crypto_signatures": counts["crypto_signature"],
        "n_crypto_signatures_signed": counts["crypto_signature_signed"],
        "n_vector_ink": counts["vector_ink"],
        "n_stamp_annots": counts["stamp_annot"],
        "n_freetext_annots": counts["freetext_annot"],
        "n_placed_images": counts["placed_image"],
        "n_native_pages": counts["native_page"],
        "n_scanned_pages": counts["scanned_page"],
        "n_image_only_pages": counts["image_only_page"],
        "n_empty_pages": counts["empty_page"],
        "scanned_sig_count": "unknown_needs_detector" if counts["scanned_page"] else 0,
        "n_fonts": len(fonts_all),
        "n_multi_subset_fonts": len(multi_tag),
        "multi_subset_fonts": ";".join(sorted(multi_tag))[:300],
    })
    # rel_path alone is unreadable in a spreadsheet once paths get deep, so
    # every row also carries the bare filename and the absolute path.
    fname = os.path.basename(rel_path)
    doc_rec["file_name"] = fname
    doc_rec["full_path"] = abs_path
    for lst in (pages_out, cands_out, images_out):
        for row in lst:
            row["file_name"] = fname
            row["full_path"] = abs_path

    warn = (fitz.TOOLS.mupdf_warnings() or "").strip()
    doc_rec["mupdf_warnings"] = " | ".join(
        dict.fromkeys(w.strip() for w in warn.splitlines() if w.strip()))[:500]
    doc_rec["has_warnings"] = bool(warn)
    doc_rec["decrypt_failure"] = "aes filter" in warn or "decrypt" in warn.lower()
    if doc_rec["decrypt_failure"]:
        # Counts for this document cannot be trusted -- content did not decode.
        doc_rec["status"] = "content_undecodable"

    doc_rec["_fonts"] = sorted(fonts_all)
    doc_rec["_multi_subset_detail"] = multi_tag
    doc.close()
    return doc_rec, pages_out, cands_out, images_out


# ----------------------------------------------------------------------------
# Corpus driver
# ----------------------------------------------------------------------------

LEAD_COLS = ("rel_path", "file_name", "group", "page", "full_path")


def write_csv(path: str, rows: list[dict], fields: list[str] | None = None):
    if not rows:
        open(path, "w").close()
        return
    if fields is None:
        fields = [k for k in rows[0] if not k.startswith("_")]
        lead = [c for c in LEAD_COLS if c in fields]
        fields = lead + [c for c in fields if c not in lead]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items() if k in fields})


def run(root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    records, _ = discover(root, image_role="page")
    pdfs = [r for r in records if r.kind == "pdf"]
    if not pdfs:
        print(f"No PDFs found under {root}")
        return

    print(f"Probing {len(pdfs)} PDFs under {os.path.abspath(root)}\n")
    docs, pages, cands, images, failures = [], [], [], [], []

    for i, r in enumerate(pdfs, 1):
        try:
            d, p, c, im = probe_document(r.abs_path, r.rel_path, r.group, r.sha256)
            docs.append(d)
            pages.extend(p)
            cands.extend(c)
            images.extend(im)
        except Exception as e:
            failures.append({"rel_path": r.rel_path, "error": str(e)[:300],
                             "trace": traceback.format_exc()[-1500:]})
            docs.append({"rel_path": r.rel_path,
                         "file_name": os.path.basename(r.rel_path),
                         "full_path": r.abs_path, "group": r.group,
                         "sha256": r.sha256, "status": "error",
                         "error": str(e)[:300]})
        if i % 25 == 0 or i == len(pdfs):
            print(f"  {i}/{len(pdfs)}")

    doc_fields = [
        "rel_path", "file_name", "group", "sha256", "status", "error",
        "page_count",
        "n_crypto_signatures", "n_crypto_signatures_signed", "n_vector_ink",
        "n_stamp_annots", "n_freetext_annots", "n_placed_images",
        "n_native_pages", "n_scanned_pages", "n_image_only_pages",
    "n_empty_pages", "scanned_sig_count",
        "encrypted", "is_form_pdf", "sig_flags", "incremental_saves",
        "pdf_format", "producer", "creator", "author", "title", "subject",
        "keywords", "creation_date", "mod_date", "xmp_present", "is_repaired",
        "xref_length", "n_fonts", "n_multi_subset_fonts", "multi_subset_fonts",
        "has_warnings", "decrypt_failure", "mupdf_warnings", "full_path",
    ]
    write_csv(os.path.join(out_dir, "documents.csv"), docs, doc_fields)
    write_csv(os.path.join(out_dir, "pages.csv"), pages)
    cand_fields = ["rel_path", "file_name", "group", "page", "kind", "object_ref",
                   "x0", "y0", "x1", "y1", "w", "h",
                   "px_w", "px_h", "ext", "alpha", "effective_dpi",
                   "page_area_ratio", "stream_sha256",
                   "signed", "signer", "signed_at", "reason", "location",
                   "subfilter", "full_path"]
    write_csv(os.path.join(out_dir, "signature_candidates.csv"), cands,
              cand_fields)
    write_csv(os.path.join(out_dir, "images.csv"), images)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"corpus_root": os.path.abspath(root),
                   "probed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "n_pdfs": len(pdfs), "documents": docs,
                   "failures": failures}, f, indent=2, default=str)

    # ---- console summary ----
    ok = [d for d in docs if d.get("status") in ("ok", "content_undecodable")]
    undecodable = [d for d in docs if d.get("status") == "content_undecodable"]
    warned = [d for d in docs if d.get("has_warnings")]
    tot = Counter()
    for d in ok:
        for k in ("n_crypto_signatures", "n_crypto_signatures_signed",
                  "n_vector_ink", "n_stamp_annots", "n_placed_images",
                  "n_native_pages", "n_scanned_pages", "n_image_only_pages",
                  "n_empty_pages", "page_count"):
            tot[k] += d.get(k, 0) or 0

    print(f"\n{'='*66}\nCORPUS SUMMARY\n{'='*66}")
    print(f"PDFs probed          : {len(docs)}  ({len(failures)} failed)")
    print(f"Pages                : {tot['page_count']}")
    print(f"  native             : {tot['n_native_pages']}")
    print(f"  scanned            : {tot['n_scanned_pages']}")
    print(f"  image-only         : {tot['n_image_only_pages']}")
    print(f"  empty              : {tot['n_empty_pages']}")
    print("\nExact object-layer counts:")
    print(f"  crypto signatures  : {tot['n_crypto_signatures']} "
          f"({tot['n_crypto_signatures_signed']} actually signed)")
    print(f"  vector ink annots  : {tot['n_vector_ink']}")
    print(f"  stamp annots       : {tot['n_stamp_annots']}")
    print(f"  placed images      : {tot['n_placed_images']}  "
          f"(pasted-signature candidates -- also logos/seals/letterhead)")
    print(f"\nScanned wet-ink signatures: UNKNOWN -- {tot['n_scanned_pages']} "
          f"scanned pages need the detector to be counted.")

    prod = Counter(d.get("producer") or "(none)" for d in ok)
    print(f"\nProducers ({len(prod)} distinct):")
    for p, n in prod.most_common(10):
        print(f"  {n:>5}  {p[:60]}")

    if undecodable:
        print(f"\n!! {len(undecodable)} document(s) had content that would not "
              f"decode (encrypted streams failed).")
        print("   Their signature counts are NOT reliable -- the pages decoded "
              "as blank. Filter documents.csv on decrypt_failure=True.")
    if warned:
        print(f"\n{len(warned)} document(s) produced MuPDF warnings "
              f"(malformed structure). See mupdf_warnings in documents.csv.")

    multi = [d for d in ok if (d.get("n_multi_subset_fonts") or 0) > 0]
    if multi:
        print(f"\n{len(multi)} document(s) carry one base font under multiple "
              f"subset tags -- pages not produced in a single authoring "
              f"session. Indicator only; see documents.csv.")

    incr = [d for d in ok if (d.get("incremental_saves") or 0) > 1]
    if incr:
        print(f"{len(incr)} document(s) show more than one save generation.")

    print(f"\nWritten to {os.path.abspath(out_dir)}/:")
    for name in ("documents.csv", "pages.csv", "signature_candidates.csv",
                 "images.csv", "metadata.json"):
        print(f"  {name}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="corpus root folder (walked recursively)")
    ap.add_argument("--out", default="probe_output", help="output directory")
    args = ap.parse_args()
    run(args.root, args.out)


if __name__ == "__main__":
    main()
