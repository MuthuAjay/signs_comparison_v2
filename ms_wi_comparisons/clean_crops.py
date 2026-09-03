"""
Isolate the signature stroke inside a detector crop (bridges crop -> compare).

A detector box is a rectangle, and on a real signature block a rectangle cannot
avoid enclosing the ruled signature line, the printed name caption under it and
whatever letterhead text the flourish happens to cross. `comapare_signs.py`
re-crops to the ink bounding box and scores every ink pixel, so all of that
enters the comparison as if it were stroke -- and printed text is identical
across documents from the same template, which biases every score upward for
exactly the pairs that matter.

Rectangles cannot separate them, because the signature routinely overlaps the
caption. Connected-component geometry can:

    ruled line    a long horizontal run of ink. It usually TOUCHES the
                  signature (the flourish crosses it), so it is not a separate
                  component and no per-component test can see it. It is removed
                  at pixel level first, by a horizontal opening: only ink that
                  survives erosion by a long 1xL horizontal kernel is rule, and
                  a signature stroke is never L pixels of unbroken horizontal
                  run. Cutting it leaves a small gap where it crossed the
                  stroke, which costs far less than leaving a full-width bar in
                  the ink map.
    printed text  many small components of similar height, sitting on a shared
                  baseline, each far smaller than the signature
    signature     one component (or a few, where a pen lifted) whose bounding
                  box is dramatically larger than any glyph

So the ruled line is removed by a thinness test, and what survives is filtered
to components near the size of the largest one. Scale is set by the biggest
component rather than by absolute pixel sizes, so the same thresholds hold for
a 400 dpi crop and a degraded fax.

Usage:
    python clean_crops.py --in crops --out crops_clean [--debug]
"""

import argparse
import os

import numpy as np
from PIL import Image

# A component is kept if its bounding-box diagonal is at least this fraction of
# the largest component's diagonal. Pen lifts split a signature into a few big
# pieces; a printed glyph is an order of magnitude smaller.
KEEP_FRAC = 0.30
# Ruled-line test: wider than this fraction of the image, and thinner than this
# fraction of its own width. Applies to a rule that stands as its own component.
LINE_MIN_WIDTH_FRAC = 0.55
LINE_MAX_THINNESS = 0.06
# Horizontal opening for a rule fused to the signature: the kernel length as a
# fraction of image width, and how tall a detected run may be to count as rule
# rather than as a genuine near-horizontal stroke.
RULE_KERNEL_FRAC = 0.35
RULE_MAX_ROW_HEIGHT = 0.045


# Ink darker than this is stroke for the "grey" erase mode. Printed captions
# rendered in grey measure around 120-170; pen stroke measures around 26.
GREY_CUTOFF = 80
# "vkeep" mode: ink is kept only if it sits on a vertical run at least this
# fraction of the image height. A signature staff runs several times longer
# than the stem of a printed capital letter.
VKEEP_FRAC = 0.15
# Horizontal slack, as a fraction of image width, allowed when following a
# leaning staff down the page.
VKEEP_SLANT_FRAC = 0.012

# Hand-audited erasures, keyed by crop filename. Only for printed matter that
# TOUCHES the signature and so survives both the rule strip and the component
# filter -- geometry cannot separate those, so they are named explicitly rather
# than guessed at. Each box is (x0, y0, x1, y1) as a fraction of the crop, and
# each was set by eye against a grid overlay of that crop.
#
#   box    erase every ink pixel in the box. Only where the box is clear of
#          the signature entirely.
#   vkeep  erase ink in the box EXCEPT pixels on a long vertical run. Removes
#          bold letterhead text that the signature's vertical staffs pass
#          through.
#   grey   erase ink in the box lighter than GREY_CUTOFF. Removes a grey
#          printed caption while keeping black stroke crossing it.
ERASE: dict[str, list[tuple]] = {
    # Shafqat Ahmed (HCD): ruled line remnant and the printed name below it.
    "Letter_of_Authority_-_Arbitration_Case_in_HCD__p1__x317_y547.png": [
        ("box", 0.00, 0.74, 1.00, 1.00),
    ],
    # Shafiul Islam (CMM): ruled line remnant and "Shafiul Islam" below it.
    "Letter_of_Authority_-_CR_Case_at_CMM__p1__x71_y408.png": [
        ("box", 0.00, 0.80, 1.00, 1.00),
    ],
    # Shafiul Islam (RJSC, 1970s photocopy with a purple stamp overprinted).
    # The stamp is too faded to separate by colour, so only the parts of it
    # clear of the signature are removed; the letters crossing the strokes are
    # left in place rather than risk cutting stroke. Noted in the report.
    "Docs_collected_from_RJSC_-_signature_sample__p1__x923_y790.png": [
        ("box", 0.00, 0.00, 0.34, 0.32),   # "sh) Limited" typescript
        ("box", 0.00, 0.40, 0.29, 0.58),   # "For NAV..." left of the signature
        ("box", 0.64, 0.40, 1.00, 0.58),   # "...ED" right of the signature
        ("box", 0.45, 0.66, 1.00, 0.82),   # "Chairman" stamp
        ("box", 0.00, 0.26, 0.32, 0.36),   # ruled-line stub left of the staffs
        ("box", 0.50, 0.26, 1.00, 0.36),   # ruled-line stub right of the staffs
    ],
    # Shafiul Islam (Navana service agreement).
    "Navana_Service_agreement_signature_page__p1__x313_y80.png": [
        # "NAVANA LIMITED" letterhead. The two staffs pass through the text
        # band; their columns were read off the strip just above it (they sit
        # at x 0.484-0.499 and 0.561-0.581 there), so the band is erased
        # everywhere except those columns plus slack for the lean.
        ("box",   0.000, 0.245, 0.474, 0.340),
        ("box",   0.510, 0.245, 0.548, 0.340),
        ("box",   0.594, 0.245, 1.000, 0.340),
        ("box",   0.00, 0.70, 0.18, 1.00),  # "By:" / "Title:" labels
        ("grey",  0.18, 0.72, 1.00, 1.00),  # grey "Shafiul Islam / Chairman"
        ("box",   0.82, 0.50, 1.00, 0.72),  # ruled-line stub right of the sweep
        ("box",   0.00, 0.50, 0.14, 0.72),  # ruled-line stub left of the curl
    ],
}


def tall_mask(ink: np.ndarray) -> np.ndarray:
    """Ink belonging to a tall, near-vertical stroke.

    A signature staff is rarely plumb -- it leans, so following it down a single
    pixel column breaks the run long before the stroke actually ends, and a
    strict per-column test amputates the staff instead of protecting it. So the
    mask is dilated horizontally first: the lean is absorbed into a band wide
    enough that some column does run the full height. The result is intersected
    back with the original ink, so the dilation only decides which pixels are
    protected, never adds any.
    """
    h, w = ink.shape
    L = max(int(h * VKEEP_FRAC), 12)
    if L >= h:
        return ink.copy()

    k = max(int(w * VKEEP_SLANT_FRAC), 3)
    wide = np.zeros_like(ink)
    for dx in range(-k, k + 1):
        wide |= np.roll(ink, dx, axis=1)

    cs = np.cumsum(np.pad(wide.astype(np.int32), ((1, 0), (0, 0))), axis=0)
    eroded = np.zeros_like(ink)
    eroded[L - 1:] = (cs[L:] - cs[:-L]) == L
    if not eroded.any():
        return np.zeros_like(ink)

    tall = np.zeros_like(ink)
    for x in np.flatnonzero(eroded.any(axis=0)):
        ys = np.flatnonzero(eroded[:, x])
        for y in ys:
            tall[max(y - L + 1, 0):y + 1, x] = True
    # Give the protected band the same horizontal slack it was measured with.
    grown = np.zeros_like(ink)
    for dx in range(-k, k + 1):
        grown |= np.roll(tall, dx, axis=1)
    return grown & ink


def apply_erase(ink, gray, boxes):
    """Apply the hand-audited erase boxes. Returns (ink, pixels_removed)."""
    h, w = ink.shape
    before = int(ink.sum())
    for mode, x0f, y0f, x1f, y1f in boxes:
        y0, y1 = int(y0f * h), int(y1f * h)
        x0, x1 = int(x0f * w), int(x1f * w)
        sub = ink[y0:y1, x0:x1]
        if mode == "box":
            sub[:] = False
        elif mode == "grey":
            sub &= gray[y0:y1, x0:x1] <= GREY_CUTOFF
        elif mode == "vkeep":
            sub &= tall_mask(ink)[y0:y1, x0:x1]
        else:
            raise ValueError(f"unknown erase mode: {mode}")
    return ink, before - int(ink.sum())


def otsu(gray: np.ndarray) -> float:
    hist = np.histogram(gray.ravel(), bins=256, range=(0, 256))[0].astype(np.float64)
    total, sum_all = hist.sum(), np.dot(np.arange(256), hist)
    w0 = sum0 = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        var = w0 * w1 * (sum0 / w0 - (sum_all - sum0) / w1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return float(best_t)


def strip_rule(ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Erase long horizontal runs (the ruled signature line) from an ink mask.

    Horizontal erosion followed by dilation keeps only ink lying on an unbroken
    horizontal run at least L pixels long. A pen stroke, even a flat terminal
    sweep, rises or falls across that distance and so is erased by the erosion;
    a printed rule is not. The height guard then rejects a "rule" thicker than a
    real one, so a genuinely horizontal chunk of signature is not sacrificed.
    """
    h, w = ink.shape
    L = max(int(w * RULE_KERNEL_FRAC), 30)
    if L >= w:
        return ink, 0
    # Erosion by a 1xL kernel: a pixel survives iff its whole L-window is ink.
    cs = np.cumsum(np.pad(ink.astype(np.int32), ((0, 0), (1, 0))), axis=1)
    runs = cs[:, L:] - cs[:, :-L]
    eroded = np.zeros_like(ink)
    eroded[:, L - 1:] = runs == L
    if not eroded.any():
        return ink, 0
    # Dilation by the same kernel restores the full run, per row.
    dil = np.zeros_like(ink)
    for y in np.flatnonzero(eroded.any(axis=1)):
        for x in np.flatnonzero(eroded[y]):
            dil[y, max(x - L + 1, 0):x + 1] = True
    rule = dil & ink

    # Reject bands too tall to be a printed rule.
    rows = np.flatnonzero(rule.any(axis=1))
    keep = np.zeros_like(rule)
    max_h = max(RULE_MAX_ROW_HEIGHT * h, 4)
    start = rows[0]
    prev = rows[0]
    for r in list(rows[1:]) + [rows[-1] + 99]:
        if r - prev > 1:
            if (prev - start + 1) <= max_h:
                keep[start:prev + 1] = rule[start:prev + 1]
            start = r
        prev = r
    return ink & ~keep, int(keep.sum())


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labelling. Union-find over a two-pass raster scan."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        row, prev = mask[y], mask[y - 1] if y else None
        for x in np.flatnonzero(row):
            neigh = []
            if x and lab[y, x - 1]:
                neigh.append(lab[y, x - 1])
            if y:
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if 0 <= xx < w and lab[y - 1, xx]:
                        neigh.append(lab[y - 1, xx])
            if neigh:
                m = min(neigh)
                lab[y, x] = m
                for n in neigh:
                    union(m, n)
            else:
                lab[y, x] = nxt
                parent.append(nxt)
                nxt += 1

    remap = np.zeros(nxt, dtype=np.int32)
    seen: dict[int, int] = {}
    for i in range(1, nxt):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen) + 1
        remap[i] = seen[r]
    return remap[lab], len(seen)


def clean(path: str, debug_dir: str | None = None) -> Image.Image | None:
    img = Image.open(path).convert("L")
    gray = np.asarray(img, dtype=np.uint8)
    ink = gray <= otsu(gray)
    if not ink.any():
        return None

    ink, n_rule_px = strip_rule(ink)
    if not ink.any():
        return None

    boxes = ERASE.get(os.path.basename(path), [])
    ink, n_erased = apply_erase(ink, gray, boxes) if boxes else (ink, 0)
    if not ink.any():
        return None

    lab, n = label(ink)
    if n == 0:
        return None

    h, w = ink.shape
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < 8:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        comps.append({"id": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "w": bw, "h": bh, "px": len(ys),
                      "diag": float(np.hypot(bw, bh))})
    if not comps:
        return None

    # Drop the ruled signature line: wide, and thin relative to its own width.
    def is_rule(c):
        return (c["w"] >= LINE_MIN_WIDTH_FRAC * w
                and c["h"] <= LINE_MAX_THINNESS * c["w"])

    kept = [c for c in comps if not is_rule(c)]
    n_rules = len(comps) - len(kept)
    if not kept:
        kept, n_rules = comps, 0

    # Keep everything close in size to the largest survivor.
    biggest = max(c["diag"] for c in kept)
    sig = [c for c in kept if c["diag"] >= KEEP_FRAC * biggest]

    out = np.full((h, w), 255, dtype=np.uint8)
    for c in sig:
        m = lab == c["id"]
        out[m] = gray[m]

    ys, xs = np.where(out < 255)
    if len(ys) == 0:
        return None
    pad = 10
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w - 1)
    res = Image.fromarray(out[y0:y1 + 1, x0:x1 + 1])

    if debug_dir:
        print(f"  {os.path.basename(path):<58} comps {len(comps):>4} -> "
              f"kept {len(sig):>2}   rule px {n_rule_px:>6}   "
              f"erased px {n_erased:>6}   rule comps {n_rules}")
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    n_ok = n_fail = 0
    for dirpath, _, names in os.walk(args.src):
        for name in sorted(names):
            if not name.lower().endswith(".png"):
                continue
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, args.src)
            out = os.path.join(args.dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            img = clean(src, args.dst if args.debug else None)
            if img is None:
                print(f"  ! no ink retained: {rel}")
                n_fail += 1
                continue
            img.save(out)
            n_ok += 1
    print(f"\nCleaned {n_ok} crops -> {os.path.abspath(args.dst)}"
          + (f"   ({n_fail} failed)" if n_fail else ""))


if __name__ == "__main__":
    main()
