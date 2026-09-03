"""
Measure the "M-like" feature: tall, steep, closely-spaced vertical strokes.

The observation under test is structural rather than holistic -- the reference
signatures carry two or three tall near-vertical staffs standing close together,
like the legs of an M, and the questioned signatures are said not to. None of
the three similarity scorers addresses that directly: they compare whole marks,
so a specific construction feature can be present or absent without moving them
much.

METHOD. A staff is a run of ink that is tall and steep. For each column of the
binarised crop the longest unbroken vertical run of ink is measured, and a
column counts as staff-bearing when that run reaches MIN_H of the signature's
own ink height. Columns are then grouped into staffs, and the gaps between
adjacent staffs recorded.

Two details decide whether this measures the feature or an artefact:

  Lean tolerance. These signatures slant. Following a leaning stroke straight
  down one pixel column breaks the run long before the stroke ends, so a strict
  column test finds no staffs in any signature, genuine or not. The mask is
  therefore dilated horizontally by LEAN before run lengths are measured, and
  intersected back with the original ink -- the dilation decides which pixels
  are eligible, never adds ink.

  Height normalisation. Every measurement is a fraction of that signature's own
  ink height, never an absolute pixel count. A wide mark and a tall mark are
  therefore assessed on the same footing, and the width:height difference
  between the two sets cannot by itself produce a difference in staff count.

Usage:
    python staff_analysis.py [--sig-dir signatures]
"""

import argparse
import os

import numpy as np
from PIL import Image

MIN_H = 0.45         # a staff must reach this fraction of the signature's height
LEAN = 0.030         # horizontal slack, as a fraction of ink width
MIN_STAFF_W = 0.004  # ignore staff groups narrower than this fraction of width
CLOSE = 0.35         # staffs closer than this fraction of ink height are "close"

ORDER = [
    ("ref_shafiul_invitation", "R1  Invitation letter", "reference"),
    ("ref_shafiul_navana_sa", "R2  Service agreement", "reference"),
    ("ref_shafiul_rjsc", "R3  RJSC Form XVIII", "reference"),
    ("q_shafiul_hcd", "Q1  HCD letter", "questioned"),
    ("q_shafiul_cmm", "Q2  CMM letter", "questioned"),
]


def load_ink(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    ink = g <= 128
    ys, xs = np.where(ink)
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def longest_vertical_run(mask):
    """Per column, the longest unbroken vertical run of True."""
    h, w = mask.shape
    best = np.zeros(w, dtype=np.int32)
    cur = np.zeros(w, dtype=np.int32)
    for y in range(h):
        cur = np.where(mask[y], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def find_staffs(ink):
    H, W = ink.shape
    lean = max(int(W * LEAN), 1)
    wide = np.zeros_like(ink)
    for dx in range(-lean, lean + 1):
        wide |= np.roll(ink, dx, axis=1)
    runs = longest_vertical_run(wide)

    staff_cols = runs >= MIN_H * H
    staffs, start = [], None
    for x in range(W + 1):
        on = staff_cols[x] if x < W else False
        if on and start is None:
            start = x
        elif not on and start is not None:
            if (x - start) >= max(int(W * MIN_STAFF_W), 1):
                seg = runs[start:x]
                staffs.append({"cx": (start + x - 1) / 2,
                               "h_frac": float(seg.max()) / H})
            start = None
    return staffs, runs, H, W


def analyse(path):
    ink = load_ink(path)
    staffs, runs, H, W = find_staffs(ink)
    gaps = [round((b["cx"] - a["cx"]) / H, 2) for a, b in zip(staffs, staffs[1:])]
    return {"n_staffs": len(staffs),
            "h_fracs": [round(s["h_frac"], 2) for s in staffs],
            "tallest": round(max((s["h_frac"] for s in staffs), default=0.0), 2),
            "gaps": gaps,
            "n_close_pairs": len([g for g in gaps if g <= CLOSE]),
            "max_vert_frac": round(float(runs.max()) / H, 2),
            "aspect": round(W / H, 2)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sig-dir", default="signatures")
    args = ap.parse_args()

    print(f"Staff = unbroken vertical ink run reaching {MIN_H:.0%} of the "
          f"signature's own ink height")
    print(f"(lean tolerance {LEAN:.1%} of width; gaps in units of ink height)\n")
    print(f"{'signature':<24}{'staffs':>7}{'tallest':>9}{'close':>7}"
          f"{'staff heights':>24}{'gaps':>16}")
    print("-" * 88)

    res = {}
    for key, label, role in ORDER:
        p = os.path.join(args.sig_dir, f"{key}.png")
        if not os.path.exists(p):
            print(f"{label:<24}  missing: {p}")
            continue
        r = analyse(p)
        res[key] = (role, r)
        print(f"{label:<24}{r['n_staffs']:>7}{r['tallest']:>9.2f}"
              f"{r['n_close_pairs']:>7}{str(r['h_fracs']):>24}{str(r['gaps']):>16}")
        if key == "ref_shafiul_rjsc":
            print("-" * 88)

    ref = [r for role, r in res.values() if role == "reference"]
    que = [r for role, r in res.values() if role == "questioned"]
    if ref and que:
        print(f"\n{'':<32}{'references':>12}{'questioned':>14}")
        for label, k, f in (
                ("staffs found", "n_staffs", lambda v: f"{min(v)}-{max(v)}"),
                ("tallest staff (of own height)", "tallest",
                 lambda v: f"{min(v):.2f}-{max(v):.2f}"),
                ("close staff pairs", "n_close_pairs",
                 lambda v: f"{min(v)}-{max(v)}")):
            print(f"  {label:<30}{f([x[k] for x in ref]):>12}"
                  f"{f([x[k] for x in que]):>14}")


if __name__ == "__main__":
    main()
