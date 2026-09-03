"""
Letter of Authority case: are the two questioned Shafiul Islam signatures
consistent with his genuine signature?

QUESTIONED (the two Letters of Authority)
    q_shafiul_hcd     Letter of Authority - Arbitration Case in HCD
    q_shafiul_cmm     Letter of Authority - CR Case at CMM

REFERENCE (genuine Shafiul Islam signatures from three other documents)
    ref_shafiul_invitation   Toyota/Navana 50-year invitation letter, 2014
    ref_shafiul_navana_sa    Navana / Toyota Tsusho service agreement
    ref_shafiul_rjsc         RJSC Form XVIII, "For NAVANA LIMITED / Chairman"

CONTROL (different writers, taken from the same five documents)
    ctl_shafqat_hcd          Shafqat Ahmed, second signatory on the HCD letter
    ctl_muto_navana_sa       Kazuyuki Muto, Toyota Tsusho, same agreement page
    ctl_second_signer_rjsc   second signatory on the RJSC form

WHY THE CONTROLS MATTER. The earlier run in this folder scored the two
questioned signatures against each other and had nothing else: its own
calibration note says "No floor/unrelated-writer experiment was run ... Do not
cite this alone as evidence the signatures match." A similarity number is
meaningless without knowing what this pipeline scores for two signatures known
to be by the same hand, and what it scores for two known to be by different
hands. The three reference documents supply the first; the three other
signatories on those same pages supply the second, captured through the same
scanners and the same degradation. So every pair falls into one of:

    ref-ref        same writer (Shafiul), different documents  -> CEILING
    ref-control    different writers                           -> FLOOR
    ref-questioned the actual question
    q-q            the two questioned signatures to each other

A questioned pair scoring in the ceiling band is consistent with his genuine
signature; one scoring in the floor band is not. A score between the two bands
supports neither conclusion, and is reported as inconclusive rather than
rounded toward whichever answer is wanted.

Usage:
    python run_loa_case.py [--out comparison_output_loa]
"""

import argparse
import csv
import itertools
import os
from dataclasses import dataclass

import numpy as np

from comapare_signs import (align_to, dinov2_features, global_cosine,
                            ink_patch_mask, mutual_nn_match, preprocess, ssim,
                            visualize_matches, CANVAS, PATCH)

SIG_DIR = "signatures"

# role: reference (genuine Shafiul) | questioned | control (different writer)
CASE = [
    ("ref_shafiul_invitation", "reference",
     "Shafiul Islam - invitation letter (2014)"),
    ("ref_shafiul_navana_sa", "reference",
     "Shafiul Islam - Navana service agreement"),
    ("ref_shafiul_rjsc", "reference",
     "Shafiul Islam - RJSC Form XVIII"),
    ("q_shafiul_hcd", "questioned",
     "QUESTIONED - Letter of Authority, HCD"),
    ("q_shafiul_cmm", "questioned",
     "QUESTIONED - Letter of Authority, CMM"),
    ("ctl_shafqat_hcd", "control",
     "Shafqat Ahmed - HCD letter"),
    ("ctl_muto_navana_sa", "control",
     "Kazuyuki Muto - Navana service agreement"),
    ("ctl_second_signer_rjsc", "control",
     "second signatory - RJSC Form XVIII"),
]

SCORERS = ("global_cosine", "patch_match_score", "ssim_aligned")


@dataclass
class Sig:
    name: str
    role: str
    label: str
    path: str


def pair_type(ra: str, rb: str) -> str:
    """Label a pair by what it can be used for."""
    roles = {ra, rb}
    if roles == {"reference"}:
        return "ref-ref (same writer)"
    if roles == {"reference", "control"}:
        return "ref-control (diff writer)"
    if roles == {"reference", "questioned"}:
        return "ref-questioned (THE QUESTION)"
    if roles == {"questioned"}:
        return "q-q"
    if roles == {"questioned", "control"}:
        return "q-control (diff writer)"
    return "control-control (diff writer)"


def band(v: float, b: dict) -> str:
    """Where a score sits relative to the two calibration bands.

    A band that was never measured (no controls in the run, or fewer than two
    references) is reported as no verdict rather than as a pass: an absent
    floor is an absence of evidence, not evidence of a high score.

    If the bands overlap, the scorer has been shown -- on this very corpus --
    to give same-writer and different-writer pairs indistinguishable scores.
    No threshold can then separate them, so no per-pair verdict is available at
    any score. Emitting one anyway is the single most misleading thing this
    script could do, so the overlap check comes first.
    """
    if not b["have_floor"] or not b["have_ceiling"]:
        return "NO VERDICT (uncalibrated)"
    if not b["separated"]:
        return "NOT DIAGNOSTIC (bands overlap)"
    if v >= b["ceil_lo"]:
        return "in same-writer band"
    if v <= b["floor_hi"]:
        return "in diff-writer band"
    return "between bands"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="comparison_output_loa")
    ap.add_argument("--sig-dir", default=SIG_DIR)
    ap.add_argument("--exclude", default="",
                    help="comma-separated signature names to drop (sensitivity runs)")
    args = ap.parse_args()

    drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
    sigs = [Sig(n, r, l, os.path.join(args.sig_dir, f"{n}.png"))
            for n, r, l in CASE if n not in drop]
    missing = [s.path for s in sigs if not os.path.exists(s.path)]
    if missing:
        raise SystemExit("Missing crops:\n  " + "\n  ".join(missing))

    os.makedirs(args.out, exist_ok=True)
    grid = CANVAS // PATCH

    print(f"Signatures: {len(sigs)}"
          + (f"   (excluded: {', '.join(sorted(drop))})" if drop else ""))
    feats = {}
    for s in sigs:
        pre = preprocess(s.path)
        cls, patches = dinov2_features(pre["tensor"])
        feats[s.name] = {"pre": pre, "cls": cls, "patches": patches,
                         "mask": ink_patch_mask(pre["ink"], grid), "sig": s}
        from PIL import Image
        Image.fromarray(pre["display"]).save(
            os.path.join(args.out, f"preprocessed_{s.name}.png"))

    rows = []
    for a, b in itertools.combinations([s.name for s in sigs], 2):
        fa, fb = feats[a], feats[b]
        m = mutual_nn_match(fa["patches"], fb["patches"], fa["mask"], fb["mask"])
        aligned_b = align_to(fb["pre"]["ink"], fa["pre"]["ink"])
        rows.append({
            "sig_a": a, "sig_b": b,
            "type": pair_type(fa["sig"].role, fb["sig"].role),
            "global_cosine": global_cosine(fa["cls"], fb["cls"]),
            "patch_match_score": m["score"],
            "patch_mean_sim": m["mean_sim"],
            "ssim_aligned": ssim(fa["pre"]["ink"], aligned_b),
        })
        if pair_type(fa["sig"].role, fb["sig"].role).startswith(
                ("ref-ref", "ref-questioned", "ref-control")):
            visualize_matches(fa["pre"]["display"], fb["pre"]["display"],
                              m["matches"], grid,
                              os.path.join(args.out, f"matches_{a}_vs_{b}.png"))

    for r in rows:
        r["mean_score"] = float(np.mean([r[k] for k in SCORERS]))

    def group(t):
        return [r for r in rows if r["type"].startswith(t)]

    ceiling = group("ref-ref")
    # The floor is reference-vs-control ONLY: a genuine Shafiul signature scored
    # against another writer's. That is the same comparison being made for a
    # questioned mark, differing in exactly one respect -- whether the second
    # signature is his -- so it is the only group that calibrates it.
    #
    # Pooling control-vs-control and questioned-vs-control into the floor, as an
    # earlier version did, destroys the calibration: those pairs answer a
    # different question, they run high because the marks involved happen to
    # share a wide flat proportion, and their presence made a cleanly separated
    # measure look non-diagnostic.
    floor = group("ref-control")
    context = group("control-control") + group("q-control")
    questioned = group("ref-questioned")

    # ---- calibration bands ----
    # Either band can be empty: drop every control and there is no
    # different-writer floor, drop all but one reference and there is no
    # same-writer ceiling. A missing band is not a passing score -- it means
    # the run is uncalibrated, and `separated` stays False so no verdict is
    # ever emitted off a band that was never measured.
    bands = {}
    for k in SCORERS + ("mean_score",):
        cv = [r[k] for r in ceiling]
        fv = [r[k] for r in floor]
        bands[k] = {
            "ceil_lo": min(cv) if cv else None,
            "ceil_mean": float(np.mean(cv)) if cv else None,
            "ceil_hi": max(cv) if cv else None,
            "floor_lo": min(fv) if fv else None,
            "floor_mean": float(np.mean(fv)) if fv else None,
            "floor_hi": max(fv) if fv else None,
            "separated": bool(cv and fv and min(cv) > max(fv)),
            "have_floor": bool(fv), "have_ceiling": bool(cv),
        }

    csv_path = os.path.join(args.out, "comparison_report.csv")
    hdr = ["sig_a", "sig_b", "type", "global_cosine", "patch_match_score",
           "patch_mean_sim", "ssim_aligned", "mean_score"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in sorted(rows, key=lambda r: (r["type"], -r["mean_score"])):
            w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.4f}"
                        for k in hdr])

    # ---- console report ----
    W = 92
    print("\n" + "=" * W)
    print("CALIBRATION BANDS  (built from this corpus, same scanners)")
    print("=" * W)
    print(f"  {'scorer':<20}{'DIFF-WRITER (floor)':>28}"
          f"{'SAME-WRITER (ceiling)':>28}   separated")
    print(f"  {'':<20}{f'n={len(floor)} pairs':>28}{f'n={len(ceiling)} pairs':>28}")
    print("-" * W)
    for k in SCORERS + ("mean_score",):
        b = bands[k]
        fl = (f"{b['floor_lo']:.3f}-{b['floor_hi']:.3f}"
              if b["have_floor"] else "none in run")
        ce = (f"{b['ceil_lo']:.3f}-{b['ceil_hi']:.3f}"
              if b["have_ceiling"] else "none in run")
        fm = f" (mean {b['floor_mean']:.3f})" if b["have_floor"] else " " * 13
        cm = f" (mean {b['ceil_mean']:.3f})" if b["have_ceiling"] else " " * 13
        verdict = ("YES" if b["separated"]
                   else "NO -- overlapping" if b["have_floor"] and b["have_ceiling"]
                   else "UNCALIBRATED")
        print(f"  {k:<20}{fl:>18}{fm}{ce:>18}{cm}   {verdict}")

    mb = bands["mean_score"]
    if not mb["have_floor"] or not mb["have_ceiling"]:
        print("\n" + "!" * W)
        print("UNCALIBRATED RUN -- NO VERDICT IS AVAILABLE AT ANY SCORE.")
        print("!" * W)
        if not mb["have_floor"]:
            print("  No different-writer pairs in this run. Nothing establishes what")
            print("  this pipeline scores for two signatures by different hands, so a")
            print("  score cannot be read as high or low -- there is nothing to read")
            print("  it against. Add control signatures to calibrate.")
        if not mb["have_ceiling"]:
            print("  No same-writer pairs in this run: fewer than two genuine")
            print("  reference signatures, so there is no within-writer ceiling.")
    elif not mb["separated"]:
        print("\n" + "!" * W)
        print("BANDS OVERLAP -- THIS PIPELINE CANNOT ANSWER THE QUESTION ON THIS DATA.")
        print("!" * W)
        worst = max(floor, key=lambda r: r["mean_score"])
        best = max(ceiling, key=lambda r: r["mean_score"])
        print(f"  Highest different-writer pair : {worst['mean_score']:.3f}  "
              f"({worst['sig_a']} vs {worst['sig_b']})")
        print(f"  Best-matching same-writer pair: {best['mean_score']:.3f}  "
              f"({best['sig_a']} vs {best['sig_b']})")
        print("  Two signatures known to be by DIFFERENT hands score higher than")
        print("  two known to be by the SAME hand, so no threshold separates them.")
        print("  The per-pair scores below rank pairs; they support no conclusion")
        print("  about who signed the Letters of Authority, in either direction.")

    print("\n" + "=" * W)
    print("THE QUESTION: each questioned signature vs each genuine reference")
    print("=" * W)
    print(f"{'pair':<52}{'COS':>7}{'PATCH':>7}{'SSIM':>7}{'MEAN':>7}   verdict")
    print("-" * W)
    for r in sorted(questioned, key=lambda r: (r["sig_b"], r["sig_a"])):
        v = band(r["mean_score"], bands["mean_score"])
        pair = f"{r['sig_a']} vs {r['sig_b']}"
        print(f"{pair:<52}{r['global_cosine']:>7.3f}{r['patch_match_score']:>7.3f}"
              f"{r['ssim_aligned']:>7.3f}{r['mean_score']:>7.3f}   {v}")

    print("\n" + "-" * W)
    print("Reference pairs (same writer, for comparison):")
    for r in sorted(ceiling, key=lambda r: -r["mean_score"]):
        print(f"  {r['sig_a']} vs {r['sig_b']:<32}{r['mean_score']:.3f}")
    if floor:
        print("Different-writer pairs:")
        for r in sorted(floor, key=lambda r: -r["mean_score"])[:8]:
            print(f"  {r['sig_a']} vs {r['sig_b']:<32}{r['mean_score']:.3f}")
    else:
        print("Different-writer pairs: none in this run (no controls included).")

    if context:
        cv = [r["mean_score"] for r in context]
        print(f"\nNot used for calibration ({len(context)} pairs, "
              f"{min(cv):.3f}-{max(cv):.3f}): control-vs-control and "
              f"questioned-vs-control.\n  These compare two marks neither of "
              f"which is a genuine Shafiul signature, so they calibrate "
              f"nothing here.")

    qq = group("q-q")
    if qq:
        print("\nThe two questioned signatures to each other:")
        for r in qq:
            print(f"  {r['sig_a']} vs {r['sig_b']:<32}{r['mean_score']:.3f}")

    print(f"\nReport : {os.path.abspath(csv_path)}")
    print(f"Visuals: {args.out}/matches_*.png, {args.out}/preprocessed_*.png")
    print("\nThese three scorers measure visual/structural similarity of the ink. "
          "They are\nnot a forensic document examination: pen lifts, line quality, "
          "pressure and\nwriting order are not modelled. Treat this as triage "
          "evidence, not a finding.")


if __name__ == "__main__":
    main()
