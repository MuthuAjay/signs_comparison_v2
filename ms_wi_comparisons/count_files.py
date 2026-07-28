"""
Corpus file count. Walks a folder recursively and reports what is in it.

Deliberately does no hashing, so it stays fast on a large tree -- run this to
size a corpus before committing to `ingest.py` (which hashes every file) or
`pdf_probe.py` (which opens every PDF).

Usage:
    python count_files.py /path/to/corpus
    python count_files.py /path/to/corpus --by-dir        # per-directory table
    python count_files.py /path/to/corpus --csv out.csv
    python count_files.py /path/to/corpus --xlsx counts.xlsx
"""

import argparse
import csv
import os
from datetime import datetime, timezone
from collections import Counter, defaultdict

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def walk(root: str, follow_symlinks: bool = False):
    """Yield (rel_dir, filename, size_bytes, ext) for every file under root."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d != "__pycache__"]
        rel_dir = os.path.relpath(dirpath, root)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path) and not follow_symlinks:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            yield rel_dir, name, size, os.path.splitext(name)[1].lower()


def write_xlsx(path: str, root: str, total: int, total_size: int,
               by_ext: Counter, size_by_ext: Counter,
               per_dir: dict, per_dir_size: Counter) -> None:
    """Three sheets: Summary, By extension, By directory."""
    import pandas as pd

    n_pdf = sum(v for k, v in by_ext.items() if k in PDF_EXTS)
    n_img = sum(v for k, v in by_ext.items() if k in IMAGE_EXTS)

    summary = pd.DataFrame([
        ("Corpus root", os.path.abspath(root)),
        ("Counted (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Files", total),
        ("Directories with files", len(per_dir)),
        ("Total size (bytes)", total_size),
        ("Total size", human(total_size)),
        ("PDFs", n_pdf),
        ("Images", n_img),
        ("Other", total - n_pdf - n_img),
        ("Distinct extensions", len(by_ext)),
    ], columns=["Metric", "Value"])

    ext_rows = [{"Extension": e, "Files": n,
                 "Size (bytes)": size_by_ext[e], "Size": human(size_by_ext[e])}
                for e, n in by_ext.most_common()]
    ext_df = pd.DataFrame(ext_rows)

    dir_rows = []
    for d in sorted(per_dir):
        c = per_dir[d]
        dir_rows.append({
            "Directory": d,
            "Files": sum(c.values()),
            "PDFs": sum(v for k, v in c.items() if k in PDF_EXTS),
            "Images": sum(v for k, v in c.items() if k in IMAGE_EXTS),
            "Size (bytes)": per_dir_size[d],
            "Size": human(per_dir_size[d]),
        })
    dir_df = pd.DataFrame(dir_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        ext_df.to_excel(xl, sheet_name="By extension", index=False)
        dir_df.to_excel(xl, sheet_name="By directory", index=False)
        for sheet, df in (("Summary", summary), ("By extension", ext_df),
                          ("By directory", dir_df)):
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                longest = max([len(str(col))] +
                              [len(str(v)) for v in df[col].head(500)])
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter
                ].width = min(longest + 2, 80)


def run(root: str, by_dir: bool = False, csv_path: str | None = None,
        follow_symlinks: bool = False, xlsx_path: str | None = None):
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    by_ext = Counter()
    size_by_ext = Counter()
    per_dir = defaultdict(Counter)
    per_dir_size = Counter()
    total = total_size = 0
    dirs = set()

    for rel_dir, _, size, ext in walk(root, follow_symlinks):
        ext = ext or "(no extension)"
        by_ext[ext] += 1
        size_by_ext[ext] += size
        per_dir[rel_dir][ext] += 1
        per_dir_size[rel_dir] += size
        dirs.add(rel_dir)
        total += 1
        total_size += size

    n_pdf = sum(v for k, v in by_ext.items() if k in PDF_EXTS)
    n_img = sum(v for k, v in by_ext.items() if k in IMAGE_EXTS)
    n_other = total - n_pdf - n_img

    print(f"Root         : {os.path.abspath(root)}")
    print(f"Files        : {total:,}")
    print(f"Directories  : {len(dirs):,} (containing files)")
    print(f"Total size   : {human(total_size)}")
    print(f"\n  PDFs       : {n_pdf:,}")
    print(f"  images     : {n_img:,}")
    print(f"  other      : {n_other:,}")

    if by_ext:
        print(f"\nBy extension ({len(by_ext)} distinct):")
        print(f"  {'ext':<18}{'count':>9}{'size':>12}")
        for ext, n in by_ext.most_common():
            print(f"  {ext:<18}{n:>9,}{human(size_by_ext[ext]):>12}")

    if by_dir:
        print(f"\nBy directory ({len(per_dir)}):")
        print(f"  {'directory':<50}{'files':>8}{'pdfs':>7}{'size':>12}")
        for d in sorted(per_dir):
            c = per_dir[d]
            n = sum(c.values())
            npdf = sum(v for k, v in c.items() if k in PDF_EXTS)
            label = d if len(d) <= 48 else "..." + d[-45:]
            print(f"  {label:<50}{n:>8,}{npdf:>7,}{human(per_dir_size[d]):>12}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["directory", "extension", "count", "size_bytes"])
            for d in sorted(per_dir):
                for ext, n in sorted(per_dir[d].items()):
                    w.writerow([d, ext, n, ""])
        print(f"\nWritten: {csv_path}")

    if xlsx_path:
        write_xlsx(xlsx_path, root, total, total_size, by_ext, size_by_ext,
                   per_dir, per_dir_size)
        print(f"\nExcel: {os.path.abspath(xlsx_path)}"
              f"   (sheets: Summary, By extension, By directory)")

    if total == 0:
        print("\nNothing found. Check the path, or pass --follow-symlinks if "
              "the corpus is linked in.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder to count (walked recursively)")
    ap.add_argument("--by-dir", action="store_true",
                    help="per-directory breakdown")
    ap.add_argument("--csv", dest="csv_path", help="write breakdown to CSV")
    ap.add_argument("--xlsx", dest="xlsx_path",
                    help="write the counts to an Excel workbook")
    ap.add_argument("--follow-symlinks", action="store_true")
    args = ap.parse_args()
    run(args.root, args.by_dir, args.csv_path, args.follow_symlinks,
        args.xlsx_path)


if __name__ == "__main__":
    main()
