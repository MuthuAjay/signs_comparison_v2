"""
Recursively extract archives from a corpus folder into an output folder,
mirroring the input directory structure.

    INPUT/matterA/docs/bundle.zip   ->   OUTPUT/matterA/docs/bundle/...

Each archive gets its own folder named after the archive, so several archives
sitting in one directory cannot overwrite each other's contents.

Archives found *inside* extracted content are themselves extracted, up to
--max-depth. The originals are never modified or deleted.

SAFETY -- archives here are untrusted input:
  * path traversal ("zip slip"): every member is resolved and rejected if it
    would land outside its target directory. Python's ZipFile.extractall does
    not protect against this.
  * symlink and device entries are skipped rather than recreated.
  * absolute member paths are rejected.
  * a per-archive uncompressed-size cap guards against zip bombs.
  * encrypted archives are reported and skipped, not silently half-extracted.

Supports .zip, .rar, .tar, .tar.gz/.tgz, .tar.bz2, .tar.xz. Archives are identified by
magic bytes as well as by extension, so cloud exports that strip or mangle
extensions are still found. .7z is detected and reported but needs an
extra dependency. Continuation volumes of multi-volume sets are skipped: only
the first volume is opened, and the extractor pulls in the rest itself.

If no archives are found, the extension histogram of what IS there is printed,
so an already-extracted corpus is obvious rather than silent.

Usage:
    python unzip_corpus.py INPUT OUTPUT
    python unzip_corpus.py INPUT OUTPUT --max-depth 3 --max-size-gb 5
"""

import argparse
import csv
import os
import re
from collections import Counter
import shutil
import tarfile
import zipfile



ZIP_EXTS = (".zip",)
TAR_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz")
RAR_EXTS = (".rar",)
DEFAULT_MAX_SIZE_GB = 20.0

# Continuation volumes of a multi-volume set. Only the first volume is opened;
# the extractor pulls in the rest itself, and handing it volume 2 directly just
# produces a confusing failure.
_VOL_CONT = re.compile(r"\.part0*([2-9]|\d{2,})\.rar$|\.r\d{2}$|\.z\d{2}$",
                       re.IGNORECASE)


def is_continuation_volume(name: str) -> bool:
    return bool(_VOL_CONT.search(name))


# Formats that ARE zip containers but are documents, not archives. Magic-byte
# sniffing cannot tell them apart from a real zip -- every .docx starts with
# PK\x03\x04 -- so they are excluded by extension. Without this, every Office
# file in the corpus is exploded into a folder of XML parts and the actual
# document disappears from the extracted tree.
ZIP_BASED_DOCS = {
    # OOXML
    ".docx", ".docm", ".dotx", ".dotm",
    ".xlsx", ".xlsm", ".xltx", ".xltm", ".xlsb",
    ".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm",
    ".vsdx", ".vsdm",
    # OpenDocument
    ".odt", ".ods", ".odp", ".odg", ".odf", ".odb",
    # other zip-container formats
    ".jar", ".war", ".ear", ".apk", ".ipa", ".epub", ".xpi", ".crx",
    ".whl", ".egg", ".kmz", ".3mf", ".numbers", ".pages", ".key",
}


def is_zip_based_document(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in ZIP_BASED_DOCS


def archive_kind(name: str) -> str | None:
    low = name.lower()
    if low.endswith(ZIP_EXTS):
        return "zip"
    if low.endswith(TAR_EXTS):
        return "tar"
    if low.endswith(RAR_EXTS):
        return "rar"
    return None


def sniff_kind(path: str) -> str | None:
    """
    Identify an archive by its magic bytes.

    Extension alone is unreliable: cloud exports and mail gateways routinely
    strip or mangle extensions, and a file called `bundle` or `bundle.dat` can
    still be a zip. Content is authoritative, so it is checked whenever the
    extension does not already answer the question.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            f.seek(257)
            ustar = f.read(5)
    except OSError:
        return None
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    if ustar == b"ustar":
        return "tar"
    if head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00":
        return "tar"          # compressed stream; tarfile opens these directly
    if head[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if head[:6] in (b"Rar!\x1a\x07",) or head[:7] == b"Rar!\x1a\x07\x01\x00":
        return "rar"
    return None


try:
    import rarfile
    rarfile.tool_setup()
    _RAR_OK = True
except Exception:                       # missing package or missing unrar
    _RAR_OK = False

UNSUPPORTED_HINT = {
    "7z": "pip install py7zr",
}
if not _RAR_OK:
    UNSUPPORTED_HINT["rar"] = ("pip install rarfile  (also needs an `unrar`, "
                               "`unar` or `bsdtar` binary on PATH)")


def archive_stem(name: str) -> str:
    """bundle.tar.gz -> bundle ; bundle.zip -> bundle"""
    low = name.lower()
    for ext in sorted(TAR_EXTS + ZIP_EXTS + RAR_EXTS, key=len, reverse=True):
        if low.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def _safe_target(dest_root: str, member_name: str) -> str | None:
    """
    Resolve where a member would land, or None if it escapes dest_root.

    Rejects absolute paths and any '..' traversal. Compared against the real
    path of dest_root so a symlinked output directory cannot be used to break
    out either.
    """
    if not member_name or member_name.startswith("/") or os.path.isabs(member_name):
        return None
    if os.path.splitdrive(member_name)[0]:
        return None
    root = os.path.realpath(dest_root)
    target = os.path.realpath(os.path.join(root, member_name))
    if target != root and not target.startswith(root + os.sep):
        return None
    return target


def extract_zip(path: str, dest: str, max_bytes: int) -> dict:
    out = {"files": 0, "bytes": 0, "skipped": 0, "status": "ok", "error": "",
           "failed_members": 0}
    failed: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > max_bytes:
            out["status"] = "skipped_too_large"
            out["error"] = f"uncompressed {total/1e9:.1f} GB exceeds cap"
            return out
        for info in zf.infolist():
            # Unix mode lives in the top 16 bits of external_attr.
            mode = info.external_attr >> 16
            if mode and not (mode & 0o170000) in (0o100000, 0o040000, 0):
                out["skipped"] += 1          # symlink, fifo, device
                continue
            target = _safe_target(dest, info.filename)
            if target is None:
                out["skipped"] += 1
                continue
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if info.flag_bits & 0x1:       # whole archive is password protected
                out["status"] = "encrypted"
                out["error"] = "archive requires a password"
                return out
            try:
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as e:
                # Same rule as the RAR path: a corrupt member costs that member,
                # not every file after it.
                if os.path.exists(target) and os.path.getsize(target) == 0:
                    os.remove(target)
                failed.append((info.filename,
                               " ".join(f"{type(e).__name__}: {e}".split())[:200]))
                continue
            out["files"] += 1
            out["bytes"] += info.file_size
    if failed:
        out["status"] = "partial"
        out["failed_members"] = len(failed)
        out["error"] = (f"{len(failed)} member(s) unreadable: "
                        + "; ".join(n for n, _ in failed[:3])
                        + (" ..." if len(failed) > 3 else "")
                        + f" | first error: {failed[0][1]}")
    return out


def extract_tar(path: str, dest: str, max_bytes: int) -> dict:
    out = {"files": 0, "bytes": 0, "skipped": 0, "status": "ok", "error": "",
           "failed_members": 0}
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        total = sum(m.size for m in members if m.isfile())
        if total > max_bytes:
            out["status"] = "skipped_too_large"
            out["error"] = f"uncompressed {total/1e9:.1f} GB exceeds cap"
            return out
        keep = []
        for m in members:
            if not (m.isfile() or m.isdir()):
                out["skipped"] += 1          # symlink, hardlink, device
                continue
            if _safe_target(dest, m.name) is None:
                out["skipped"] += 1
                continue
            keep.append(m)
        # filter="data" is Python 3.12's hardened extraction path; the explicit
        # checks above stay because they also cover the size cap and reporting.
        tf.extractall(dest, members=keep, filter="data")
        out["files"] = sum(1 for m in keep if m.isfile())
        out["bytes"] = sum(m.size for m in keep if m.isfile())
    return out


def _write_member(rf, info, target: str, dest: str) -> str | None:
    """
    Write one RAR member. Returns None on success, or an error string.

    Two attempts: rarfile's streaming reader, then its direct-extract path.
    They invoke unrar differently -- streaming pipes through `unrar p`, direct
    extraction uses `unrar x` -- and one can succeed where the other fails, so
    the fallback is worth the second try before giving up on a document.
    """
    try:
        with rf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return None
    except Exception as first:
        try:
            if os.path.exists(target):
                os.remove(target)
            rf.extract(info, path=dest)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                return None
        except Exception:
            pass
        if os.path.exists(target) and os.path.getsize(target) == 0:
            os.remove(target)          # don't leave a truncated stub behind
        return " ".join(f"{type(first).__name__}: {first}".split())[:200]


def extract_rar(path: str, dest: str, max_bytes: int) -> dict:
    """Same member-level safety model as extract_zip -- no blanket extractall."""
    out = {"files": 0, "bytes": 0, "skipped": 0, "status": "ok", "error": "",
           "failed_members": 0}
    failed: list[tuple[str, str]] = []
    with rarfile.RarFile(path) as rf:
        if rf.needs_password():
            out["status"] = "encrypted"
            out["error"] = "archive requires a password"
            return out
        infos = rf.infolist()
        total = sum(i.file_size for i in infos if not i.is_dir())
        if total > max_bytes:
            out["status"] = "skipped_too_large"
            out["error"] = f"uncompressed {total/1e9:.1f} GB exceeds cap"
            return out
        for info in infos:
            if getattr(info, "is_symlink", lambda: False)():
                out["skipped"] += 1
                continue
            target = _safe_target(dest, info.filename)
            if target is None:
                out["skipped"] += 1
                continue
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            err = _write_member(rf, info, target, dest)
            if err:
                # One unreadable member must not cost the rest of the archive:
                # a CRC error part-way through would otherwise discard every
                # file after it.
                failed.append((info.filename, err))
                continue
            out["files"] += 1
            out["bytes"] += info.file_size
    if failed:
        out["status"] = "partial"
        out["failed_members"] = len(failed)
        out["error"] = (f"{len(failed)} member(s) unreadable: "
                        + "; ".join(n for n, _ in failed[:3])
                        + (" ..." if len(failed) > 3 else "")
                        + f" | first error: {failed[0][1]}")
    return out


EXTRACTORS = {"zip": extract_zip, "tar": extract_tar, "rar": extract_rar}


def scan(root: str) -> tuple[list[tuple[str, str]], Counter, int]:
    """
    Walk `root`. Returns (archives, extension_histogram, total_files).

    `archives` is [(path, kind)]. Extension decides first; anything it does not
    recognise is sniffed by magic bytes, so extension-less or mislabelled
    archives are still found.
    """
    found: list[tuple[str, str]] = []
    exts: Counter = Counter()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            total += 1
            exts[os.path.splitext(name)[1].lower() or "(none)"] += 1
            if is_continuation_volume(name) or is_zip_based_document(name):
                continue
            kind = archive_kind(name) or sniff_kind(path)
            if kind:
                found.append((path, kind))
    return found, exts, total


def find_archives(root: str) -> list[tuple[str, str]]:
    return scan(root)[0]


def run(src_root: str, out_root: str, max_depth: int = 3,
        max_size_gb: float = DEFAULT_MAX_SIZE_GB):
    if not os.path.isdir(src_root):
        raise SystemExit(f"Not a directory: {src_root}")
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)
    if out_root == src_root or out_root.startswith(src_root + os.sep):
        raise SystemExit("Output folder must be outside the input folder, "
                         "otherwise extracted archives get re-scanned.")
    os.makedirs(out_root, exist_ok=True)
    max_bytes = int(max_size_gb * 1e9)

    rows = []
    archives, exts, total_files = scan(src_root)

    print(f"Input  : {src_root}")
    print(f"Output : {out_root}")

    if not archives:
        # Report what IS there, by extension only -- enough to explain the
        # result without listing any filenames.
        print(f"\nNo archives found. Scanned {total_files:,} file(s); "
              f"extension counts:")
        for ext, n in exts.most_common(15):
            print(f"  {ext:<16}{n:>8,}")
        if total_files == 0:
            print("\n  The folder is empty, or everything in it is hidden.")
        elif exts.get(".pdf"):
            print(f"\n  {exts['.pdf']:,} PDFs are already present -- this corpus "
                  f"looks extracted already.\n  Skip this step and run "
                  f"count_files.py / pdf_probe.py on the input folder directly.")
        else:
            print("\n  Nothing matched by extension or by magic bytes "
                  "(zip/tar/gz/bz2/xz/7z/rar).")
        return

    unsupported = [(p, k) for p, k in archives if k in UNSUPPORTED_HINT]
    archives = [(p, k) for p, k in archives if k not in UNSUPPORTED_HINT]
    if unsupported:
        kinds = Counter(k for _, k in unsupported)
        print(f"\n{len(unsupported)} archive(s) in an unsupported format:")
        for k, n in kinds.items():
            print(f"  {k}: {n}   -> {UNSUPPORTED_HINT[k]}")

    if not archives:
        print("\nNothing extractable with the current dependencies.")
        return

    queue = [(p, k, os.path.dirname(os.path.relpath(p, src_root)), 0)
             for p, k in archives]
    print(f"Found  : {len(queue)} archive(s) at depth 0\n")

    seen: set[str] = set()
    while queue:
        path, kind, rel_dir, depth = queue.pop(0)
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)

        stem = archive_stem(os.path.basename(path))
        dest = os.path.join(out_root, rel_dir, stem) if rel_dir else \
            os.path.join(out_root, stem)
        # Collision between two archives with the same stem in one directory.
        base_dest, n = dest, 2
        while os.path.exists(dest) and os.listdir(dest):
            dest = f"{base_dest}__{n}"
            n += 1
        os.makedirs(dest, exist_ok=True)

        indent = "  " * depth
        try:
            res = EXTRACTORS[kind](path, dest, max_bytes)
        except Exception as e:
            res = {"files": 0, "bytes": 0, "skipped": 0, "failed_members": 0,
                   "status": "error",
                   "error": " ".join(f"{type(e).__name__}: {e}".split())[:400]}

        flag = "" if res["status"] == "ok" else f"  [{res['status']}]"
        print(f"{indent}{os.path.relpath(path, src_root) if depth == 0 else os.path.basename(path)}"
              f" -> {os.path.relpath(dest, out_root)}"
              f"  ({res['files']} files, {res['bytes']/1e6:.1f} MB)"
              f"{'  ' + str(res['skipped']) + ' unsafe skipped' if res['skipped'] else ''}"
              f"{flag}")
        if res["error"]:
            print(f"{indent}    {res['error']}")

        rows.append({
            "archive": os.path.relpath(path, src_root) if depth == 0 else path,
            "depth": depth,
            "output_dir": os.path.relpath(dest, out_root),
            "files": res["files"], "bytes": res["bytes"],
            "unsafe_entries_skipped": res["skipped"],
            "failed_members": res.get("failed_members", 0),
            "status": res["status"], "error": res["error"],
        })

        if depth < max_depth:
            for inner, ikind in find_archives(dest):
                if ikind in UNSUPPORTED_HINT:
                    continue
                queue.append((inner, ikind,
                              os.path.relpath(os.path.dirname(inner), out_root),
                              depth + 1))

    manifest = os.path.join(out_root, "_extraction_manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] in ("ok", "partial")]
    bad = [r for r in rows if r["status"] not in ("ok", "partial")]
    partial = [r for r in rows if r["status"] == "partial"]
    unsafe = sum(r["unsafe_entries_skipped"] for r in rows)

    print(f"\n{'='*58}")
    print(f"Archives extracted : {len(ok)}/{len(rows)}")
    print(f"Files written      : {sum(r['files'] for r in rows):,}")
    print(f"Total size         : {sum(r['bytes'] for r in rows)/1e9:.2f} GB")
    print(f"Max depth reached  : {max(r['depth'] for r in rows)}")
    if unsafe:
        print(f"\nUnsafe entries skipped: {unsafe} "
              f"(path traversal, symlink or device entries)")
    if partial:
        nm = sum(r["failed_members"] for r in partial)
        print(f"\nPartial ({len(partial)} archive(s), {nm} unreadable member(s) "
              f"-- everything else in them was extracted):")
        for r in partial:
            print(f"  {r['archive']}")
            print(f"      {r['error']}")
    if bad:
        print(f"\nFailed / skipped ({len(bad)}):")
        for r in bad:
            print(f"  {r['status']:<20} {r['archive']}")
            if r["error"]:
                print(f"      {r['error']}")
    print(f"\nManifest: {manifest}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder containing archives (walked recursively)")
    ap.add_argument("output", help="folder to extract into (must be outside input)")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="how many levels of nested archives to follow (default 3)")
    ap.add_argument("--max-size-gb", type=float, default=DEFAULT_MAX_SIZE_GB,
                    help=f"per-archive uncompressed cap (default {DEFAULT_MAX_SIZE_GB})")
    args = ap.parse_args()
    run(args.input, args.output, args.max_depth, args.max_size_gb)


if __name__ == "__main__":
    main()
