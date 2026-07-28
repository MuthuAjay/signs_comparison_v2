# MS/WI Signature Comparison — Plan

Scope: **signature-level analysis only.** For every signature found in a corpus of
up to ~500 PDFs, answer four questions:

1. **Where is it?** — detect and extract every signature mark
2. **Is it digital or scanned?** — classify how the mark got onto the page
3. **Is it duplicated?** — does this exact mark appear in any other document
4. **How similar is it?** — to every other signature in the corpus

Out of scope (dropped deliberately): page-level splicing detection, document
structural forensics, text-layer analysis.

---

## Hardware

| Resource | Assignment |
|---|---|
| GPU 1 — RTX PRO 6000 Blackwell, 96 GB | all pipeline work (`CUDA_VISIBLE_DEVICES=1`) |
| GPU 0 — RTX 5000 Ada, 32 GB | **leave alone** — drives the display (Xorg/gnome-shell) |
| 64 CPU threads | PDF rasterization (the real bottleneck — PyMuPDF is CPU-bound) |
| 251 GB RAM | whole corpus intermediate state stays in memory; no DB, no chunking |
| 850 GB free disk | retain every render, crop and intermediate for reproducibility |

**Single process, not DDP.** The two GPUs are different architectures with a 3x
memory gap and GPU 0 is contended by the display server — DDP would sync to the
slower, busier device. Determinism and auditability are worth more here than the
~10 minutes DDP would save.

Because 96 GB of VRAM holds the entire patch-token cache (~2 GB at 5k crops),
Stage 4 runs **exhaustive pairwise** rather than top-k retrieval. There is no
retrieval cutoff to defend.

---

## Input

A folder, traversed recursively. Sub-directory structure is meaningful and is
preserved as a `group` label on every record — the relative path is how a
signature is traced back to the document and matter it came from.

Both input shapes are supported and auto-detected:

- **PDFs / page images** → routed through detection (Stage 1)
- **Pre-cropped signature images** → used directly as signature records

---

## Pipeline

### Stage 1 — Detect & extract

- Faster R-CNN ResNet50-FPN (`detection/inference.py`), classes:
  `signature / initials / stamp`
- Pages rendered at **200 DPI to locate**
- Each detection **re-cropped from a 600 DPI render** of its region — stroke-level
  forensics in Stage 2 needs the resolution; 200 DPI is fine for finding boxes,
  not for judging them
- Each bbox linked to the PDF objects it overlaps: image XObject, annotation,
  vector path, font glyph run

### Stage 2 — Digital vs scanned

| Class | Determined by |
|---|---|
| `crypto_signature` | AcroForm `/Sig` field — not a drawn mark at all |
| `typed_font` | script-font glyph run |
| `vector_ink` | `/Ink` annotation or path ops (stylus) |
| `pasted_image` | image XObject whose bbox ~= **signature** bbox |
| `scanned_wet_ink` | region sits inside an image XObject whose bbox ~= **page** bbox |
| `unknown_raster` | no object layer available -> pixel forensics |

The load-bearing discriminator is the last two rows: a *tightly cropped image
object* versus a *full-page scan*. **"Embedded image" does not mean "digital"** —
a scanned wet signature is also an image. On mostly-editable PDFs this call is
deterministic: no model, no threshold.

Pixel forensics runs only for `unknown_raster`, and as corroboration on scanned
pages: paste-boundary noise discontinuity, JPEG block-grid misalignment,
effective DPI of region vs page, stroke-width uniformity, curvature regularity
(font-generated marks are geometrically perfect), alpha binarity,
anti-aliasing/halo profile, ink colour variance.

### Stage 3 — Duplicates

| Level | Test |
|---|---|
| L0 | SHA-256 of raw embedded image stream |
| L1 | SHA-256 of decoded pixel array |
| L2 | pHash Hamming <= 2, then masked NCC >= 0.99 after affine fit |
| L3 | vector path set identical after affine normalization |

All four are hash-bucket or shortlist operations — trivial at this corpus size,
and the strongest findings the system produces.

**Interpretation depends on the Stage-2 class of both endpoints:**

| A x B | An L0/L1 hit means |
|---|---|
| `scanned_wet_ink` x `scanned_wet_ink` | **Anomalous.** Two independent scans of genuine wet ink are never pixel-identical. Escalate. |
| `scanned_wet_ink` x `pasted_image` | A scan was lifted and pasted. Escalate. |
| `pasted_image` x `pasted_image` | Same image file reused — expected for a same-signer template, notable otherwise |
| `crypto_signature` x any | Benign — e-sign platforms reuse the adopted signature image by design |

Every duplicate pair also carries `same_page_context`: whether the surrounding
page content matches too. Without it, a byte-identical pair cannot be
distinguished between *a pasted signature* and *a transplanted page*, which
support very different conclusions. One hash per page; the only thing retained
from the dropped page-level track.

### Stage 4 — Similarity

- DINOv2 embeddings over all crops, **exhaustive** pairwise on GPU 1
- Re-rank above a low cosine floor with ink-masked SSIM + patch mutual-NN
- Output a **calibrated position** against measured same-writer / different-writer
  distributions, with n and confidence interval

DINOv2 is a generic visual encoder, not a signature verifier. It is used for
screening and ranking. Any number it produces is quoted only alongside its
measured error rate on held-out signature data.

---

## Known defects in the inherited code (fix before calibrating)

1. **SSIM is inflated by background.** `comapare_signs.py:243` averages over the
   whole 224x224 canvas. In a pure-background window `mu = var = cov = 0`, so the
   SSIM expression collapses to `(c1*c2)/(c1*c2) = 1.0`. With ~90% of the canvas
   white, the score is dominated by background agreement.
   **Fix:** average only over a dilated ink region. **Done** (`ssim(..., masked=True)`).

   Measured on the three inherited crops, same alignment, old vs new:

   | pair | | SSIM old | SSIM masked |
   |---|---|---|---|
   | shafiul_cmm vs shafiul_hcd | same writer | 0.815 | 0.330 |
   | shafiul_cmm vs shafqat_hcd | different writer | 0.738 | 0.300 |
   | shafiul_hcd vs shafqat_hcd | different writer | 0.730 | 0.300 |

   Note what this does and does not fix. It removes the inflation — 0.82 reads
   as a strong match and is not one. It does **not** improve discrimination:
   the different-writer score was 90% of the same-writer score before the fix
   and 91% after. **SSIM is close to uninformative on this data either way.**
   The old value simply hid that behind a large-looking number. This is the
   concrete argument for build step 2: without a measured floor, no score in
   this pipeline can be read at all. (n=3 crops — the mechanism is proven and
   deterministic, the separation estimate is not.)

2. **Patch matching is inflated by near-blank patches.** `ink_patch_mask` admits
   patches at `min_ink > 0.02`; DINOv2 tokens on near-blank patches are mutually
   highly correlated and produce spurious mutual-NN matches.
   **Fix:** raise the ink floor.

3. **Patch score normalization is asymmetric.** `comapare_signs.py:172` divides by
   `min(len(ia), len(ib))`, which biases upward when the two ink areas differ.
   **Fix:** symmetric normalization, `2*matches / (na + nb)`.

4. **The baseline is not defensible.** `comapare_signs.py:305` averages over
   reference-reference pairs; the old matter had exactly one reference.
   **Fix:** replace with the calibrated distributions from step 2 of the build.

5. **The floor calibration never ran.** Only `calibration_ceiling_only.*` exists
   in the old output. There is no measured different-writer distribution, so no
   threshold in the inherited code has a known error rate.

6. **`calibrate_pipeline.py` config is stale.** It imports the scorers, so it
   picks up fixes 1-3 automatically, but its `__main__` block still hardcodes
   the old matter's crop paths and `OBSERVED = {cosine 0.92, patch 0.35,
   ssim 0.82}`. Those numbers were produced by the **pre-fix** metrics and are
   not comparable to anything this pipeline now emits. Must be re-pointed at
   the new corpus before build step 2.

---

## Record schema

Per signature:

```
sig_id, source_path, group, file_sha256, page, bbox, dpi,
class, class_evidence, class_confidence,
crop_path, page_context_hash,
model_version, git_commit, seed, timestamp
```

Per pairwise finding:

```
sig_a, sig_b, dup_level,
scores { cosine, patch, ssim_masked },
calibrated_position, same_page_context, interpretation
```

Append-only. Nothing overwrites; every claim traces to a retained crop and a
source page.

---

## Build order

| # | Step | Why here |
|---|---|---|
| 1 | Scorer fixes (defects 1-3) | every threshold downstream calibrates against these |
| 2 | Floor calibration + encoder selection (vits14/vitl14/vitg14 on measured FAR/FRR) | fixes the model before anything depends on it |
| 3 | Folder ingest — recursive, SHA-256, group labels | input layer |
| 4 | Object-layer parser -> Stage 1 linkage + Stage 2 classification | highest evidential value per hour |
| 4a | **`pdf_probe.py` — reconnaissance pass, done** | exact counts of what the object layer knows, before any detector runs |
| 4b | **`detect_signs.py` — digital vs scanned counts, done** | detector supplies *where*, probe supplies *what*; counts come from the intersection. No crops written. |
| 5 | Duplicate index L0-L3 + interpretation table | cheap, exact, strongest output |
| 6 | Exhaustive similarity + calibrated reporting | |
| 7 | Contact-sheet review UI | 100% human verification is feasible at ~5k crops |

Steps 1-2 come first because the encoder choice and every threshold depend on
them. Building the pipeline first means recalibrating all of it afterwards.

---

## Reporting posture

The system flags; a human examiner concludes. Output language is capped at
`consistent with` / `inconsistent with` / `indeterminate`. The word "forged"
never appears in generated output.

Structural indicators (font subset tags, object numbering, and similar) direct
attention but are not proof — re-saving or merging in some tools normalizes them.
They are reported with the specific artifact quoted and never as standalone
conclusions.

---

## Open item

The corpus has not been inspected yet. The one number that could still move this
plan is the fraction of signatures landing in `unknown_raster` — that determines
how much of Stage 2 is deterministic versus statistical. Resolved by running the
object-layer probe over the real files.
