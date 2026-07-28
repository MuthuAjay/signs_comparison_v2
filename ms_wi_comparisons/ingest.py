"""
Corpus ingest for the MS/WI signature comparison pipeline (build step 3).

Walks an input folder recursively. Sub-directory structure is meaningful and is
preserved as a `group` label on every record: the relative path is how a
signature is traced back to the document and the matter it came from.

Every record carries the SHA-256 of its source file. Nothing downstream refers
to a file by path alone -- a path can change, a hash cannot.

Two input shapes are handled:
    pdf    - routed through detection (Stage 1)
    image  - either a page render or an already-cropped signature; which one
             is not inferable from the bytes, so it is a flag, not a guess

Usage:
    python ingest.py /path/to/corpus --out ingest_output
    python ingest.py /path/to/corpus --out ingest_output --image-role page
"""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

PDF_EXTS = (".pdf",)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# Archives are containers, never documents. unzip_corpus.py leaves the original
# archive in place next to its extracted contents, so without this the archive
# would be analysed alongside the very files it was unpacked into.
ARCHIVE_EXTS = (".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".tbz",
                ".xz", ".txz", ".cab", ".arj", ".lzh", ".z")
READ_CHUNK = 1 << 20  # 1 MiB


@dataclass
class InputFile:
    rel_path: str        # path relative to the corpus root -- the stable id
    abs_path: str
    group: str           # relative directory; "." for files at the root
    kind: str            # "pdf" | "image"
    detected_by: str     # "extension" | "content" (extension was missing/wrong)
    role: str            # "document" | "crop"
    sha256: str
    size_bytes: int
    mtime_utc: str


def sha256_file(path: str) -> str:
    """SHA-256 of a file, streamed so corpus size does not bound memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_kind(path: str) -> str | None:
    """
    Identify a file by its content when the extension does not.

    Cloud exports routinely strip extensions -- a PDF delivered as bare
    `28191854` is still a PDF. Without this it would be skipped silently and
    would never appear in any count, probe or comparison, which is the worst
    kind of failure here: a document quietly absent from the analysis.

    The PDF header is searched in the first 1 KiB rather than required at
    offset 0, because the spec permits leading bytes before `%PDF-`.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
    except OSError:
        return None
    if not head:
        return None
    # Rule out container formats FIRST. An archive holding a stored
    # (uncompressed) PDF carries that PDF's `%PDF-` header a few dozen bytes
    # into the file, so a naive search would classify the archive itself as a
    # document -- and every row it produced would be attributed to the .zip
    # instead of to the file inside it.
    if (head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
            or head[:6] == b"7z\xbc\xaf\x27\x1c"
            or head[:4] == b"Rar!"
            or head[:2] == b"\x1f\x8b"          # gzip
            or head[:3] == b"BZh"               # bzip2
            or head[:6] == b"\xfd7zXZ\x00"      # xz
            or head[257:262] == b"ustar"):      # tar
        return None
    if b"%PDF-" in head:
        return "pdf"
    if (head.startswith(b"\x89PNG\r\n\x1a\n")
            or head.startswith(b"\xff\xd8\xff")                 # JPEG
            or head.startswith((b"II*\x00", b"MM\x00*"))        # TIFF
            or head.startswith(b"BM")                           # BMP
            or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")):
        return "image"
    return None


def classify(filename: str, path: str | None = None) -> tuple[str | None, str]:
    """Returns (kind, how) where `how` is 'extension' or 'content'."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ARCHIVE_EXTS:
        return None, ""          # a container, not a document
    if ext in PDF_EXTS:
        return "pdf", "extension"
    if ext in IMAGE_EXTS:
        return "image", "extension"
    if path is not None:
        kind = sniff_kind(path)
        if kind:
            return kind, "content"
    return None, ""


def discover(root: str, image_role: str = "crop",
             follow_symlinks: bool = False) -> tuple[list[InputFile], list[str]]:
    """
    Recursively walk `root`. Returns (records, skipped_paths).

    `image_role` decides whether loose image files are treated as page renders
    to be detected on ("page") or as signature crops to compare directly
    ("crop"). PDFs are always documents.
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Input root is not a directory: {root}")

    records: list[InputFile] = []
    skipped: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Deterministic traversal order -- reruns must produce identical manifests.
        dirnames.sort()
        # Skip our own outputs and common noise.
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d != "__pycache__"]

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            abs_path = os.path.join(dirpath, name)
            if os.path.islink(abs_path) and not follow_symlinks:
                skipped.append(abs_path)
                continue

            kind, how = classify(name, abs_path)
            if kind is None:
                skipped.append(abs_path)
                continue

            rel_path = os.path.relpath(abs_path, root)
            group = os.path.dirname(rel_path) or "."
            role = "document" if kind == "pdf" else (
                "document" if image_role == "page" else "crop")

            st = os.stat(abs_path)
            records.append(InputFile(
                rel_path=rel_path,
                abs_path=os.path.abspath(abs_path),
                group=group,
                kind=kind,
                detected_by=how,
                role=role,
                sha256=sha256_file(abs_path),
                size_bytes=st.st_size,
                mtime_utc=datetime.fromtimestamp(
                    st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            ))

    return records, skipped


def duplicate_sources(records: list[InputFile]) -> dict[str, list[str]]:
    """Source files that are byte-identical to each other, keyed by hash.

    This is a corpus-hygiene signal, not a Stage-3 finding: it means the same
    file was supplied more than once, which would otherwise inflate every
    duplicate and similarity count downstream.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_hash[r.sha256].append(r.rel_path)
    return {h: sorted(paths) for h, paths in by_hash.items() if len(paths) > 1}


def write_manifest(records: list[InputFile], skipped: list[str],
                   root: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "manifest.csv")
    fields = list(asdict(records[0]).keys()) if records else \
        [f.name for f in InputFile.__dataclass_fields__.values()]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))

    dupes = duplicate_sources(records)
    by_group: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    by_content = [r.rel_path for r in records if r.detected_by == "content"]
    for r in records:
        by_group[r.group] += 1
        by_kind[f"{r.kind}/{r.role}"] += 1

    summary = {
        "corpus_root": os.path.abspath(root),
        "ingested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(records),
        "n_groups": len(by_group),
        "n_skipped": len(skipped),
        "by_kind": dict(sorted(by_kind.items())),
        "by_group": dict(sorted(by_group.items())),
        "n_detected_by_content": len(by_content),
        "detected_by_content": sorted(by_content),
        "duplicate_source_files": dupes,
        "skipped": sorted(skipped),
    }
    json_path = os.path.join(out_dir, "manifest.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="corpus root folder (walked recursively)")
    ap.add_argument("--out", default="ingest_output", help="output directory")
    ap.add_argument("--image-role", choices=["crop", "page"], default="crop",
                    help="how to treat loose image files (default: crop)")
    ap.add_argument("--follow-symlinks", action="store_true")
    args = ap.parse_args()

    records, skipped = discover(args.root, args.image_role, args.follow_symlinks)
    if not records:
        print(f"No PDFs or images found under {args.root}")
        return

    csv_path = write_manifest(records, skipped, args.root, args.out)

    by_kind: dict[str, int] = defaultdict(int)
    for r in records:
        by_kind[f"{r.kind}/{r.role}"] += 1
    groups = sorted({r.group for r in records})

    print(f"Corpus root : {os.path.abspath(args.root)}")
    print(f"Files       : {len(records)}  ({len(skipped)} skipped)")
    print(f"Groups      : {len(groups)} sub-directories")
    for k, n in sorted(by_kind.items()):
        print(f"  {k:<18}{n}")

    by_content = [r for r in records if r.detected_by == "content"]
    if by_content:
        kinds = defaultdict(int)
        for r in by_content:
            kinds[r.kind] += 1
        detail = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        print(f"\nIdentified by content, not extension: {len(by_content)} "
              f"({detail}).")
        print("  An extension-only scan would have skipped these entirely. "
              "Listed in manifest.json.")

    dupes = duplicate_sources(records)
    if dupes:
        print(f"\nByte-identical source files: {len(dupes)} group(s) "
              f"-- the same file was supplied more than once.")
        for h, paths in list(dupes.items())[:5]:
            print(f"  {h[:16]}  {len(paths)} copies: {paths[0]} ...")
        print("  (full list in manifest.json; de-duplicate before Stage 3 or "
              "duplicate counts will be inflated)")

    print(f"\nManifest: {csv_path}")


if __name__ == "__main__":
    main()
