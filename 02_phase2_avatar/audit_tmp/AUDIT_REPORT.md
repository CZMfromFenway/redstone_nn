# Audit: `02_phase2_avatar/archive_phase1_attempts` data pipeline

Auditor: delegated audit subagent. Date: session-time. Workspace: `E:\redstone_nn` (Windows).
Interpreter: bundled `C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`
(numpy 2.3.5, Pillow 12.3.0).

All scripts live in `02_phase2_avatar\audit_tmp\` and were run as
`& "<bundled python>" "E:\redstone_nn\02_phase2_avatar\audit_tmp\<script>.py"`.
Nothing outside `audit_tmp\` was created, modified or deleted.
Note: the DSH sandbox denied *all* writes from `pwsh`/Python inside the workspace
(`New-Item`/`Set-Content`/`open(...,'w')` -> `Permission denied`); only the harness
`write` tool could create files, so every result below is stdout evidence, not a saved artifact.

---

## 0. Verdict

The dataset is **real, correctly identified, and the face extraction is correct**. The
palette-index/quantization invariant holds exactly, and all three claimed statistics
reproduce. The 2.6 GB zip is **bit-identical to the upstream HuggingFace artifact**
(SHA-256 match). The npy rows **can** be mapped back to individual skin files, and I did so.
There is **no wrong crop, no BGR swap, no transpose and no out-of-range index**.

Real defects found (details in §6):
1. The printed "RMS 11.9/255" is dimensionally wrong (should be 20.62/255).
2. Alpha is silently dropped: 4.76 % of the whole source archive has a *fully transparent*
   head-front that the pipeline turns into a black avatar.
3. Face-level redundancy: only 932 636 distinct face crops among 1 107 411 entries; one
   all-black crop occurs 51 156 times.
4. `faces8.npy`/`faces8_rgb.npy` were overwritten in place, so the 2026-10-04 results are
   no longer reproducible from the artifacts that remain.
5. "Quantization error" means two different things in two files (3x apart).
6. Two of the 48 palette colours are 10.4/255 apart (effectively identical).

---

## 1. Inventory of `archive_phase1_attempts\data\`

| file | bytes | exists | verified |
|---|---:|---|---|
| `minecraft_skins_64x64.zip` | 2 714 488 499 | yes | 1 107 411 entries, all PNG, **all decode as 64×64**, 0 corrupt; SHA-256 `b184c113…f8ed64c4` = HF LFS oid (exact match) |
| `cifar-10-binary.tar.gz` | 170 052 171 | yes | valid gzip/tar, 9 members, 6 × 30 730 000-byte batches (magic `1f 8b 08 00`) |
| `zip_index.json` | 25 842 610 | yes | JSON list, **200 000** entries (18.1 % of the zip), `name/csz/lho/crc` match the real central directory 200 000/200 000 |
| `faces8_rgb.npy` | 9 600 128 | yes | (50000, 8, 8, 3) uint8 |
| `faces8.npy` | 3 200 128 | yes | (50000, 64) uint8 |
| `mc8.npy` | 82 709 888 | yes | (430780, 8, 8, 3) uint8 — MC block-texture patches, different provenance, not faces |
| `distill_d15_data.npz` | 3 887 370 | yes | not audited |
| `palettes.json` | 6 742 | yes | pal16 (16) + pal48 (48) + by_family, reproduced exactly |
| `mc16_palette.json` | 1 858 | yes | `[k][0]` == `palettes.json.pal16` (True) |
| `palette_avatar16.json` | 1 324 | yes | 16 distinct colours + ordered-path permutation |
| `palette_hypercube.json` / `..._cmp.json` | 2 438 / 6 342 | yes | 16-vertex assignment, colours all from avatar16 palette |
| `skins\` | 13 516 868 | yes | **exactly 5820** PNGs, all 64×64, 0 unreadable (5816 RGBA + 4 palette-mode) |

All claimed assets exist; nothing is missing. Every `skins\` filename also exists as a zip entry name.

Upstream cross-check (`https://hf-mirror.com/api/datasets/MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64/tree/main`
and its README): the dataset card states 2.6 GB, BLAKE3 pixel-level dedup, "Result is
1107411 64x64 skins" — matching our 1 107 411 entries and the zip size exactly. The card
also lists "Filtering troll skins (skins that are formed of just a single color)" as a
*future* improvement, i.e. blank/uniform skins are a known upstream property.

## 2. `faces8_rgb.npy` / `faces8.npy`

```
faces8.npy     shape=(50000, 64)     dtype=uint8  min=0  max=15   nbytes=3 200 000
faces8_rgb.npy shape=(50000, 8, 8, 3) dtype=uint8 min=0  max=255  nbytes=9 600 000
```
* Same 50 000 samples in both files; no truncation, no NaN (uint8), labels 0..15 only —
  **0 labels ≥ 16** (palette length 16), 0 labels < 0.
* `faces8.npy` is **flat (N,64)**, row-major over `(r,c)` → `p = r*8+c`. Consistent with
  `faces8_rgb.npy.reshape(-1,3)` (verified: 0/3 200 000 mismatches, §3).
* A transposed reading of the flat array is decisively worse (vertical-mirror symmetry
  0.189 vs 0.701 raw RGB; label pair-equality 0.322 vs 0.860), so the axis order is right.
* `extract_faces.py` (the older producer) saved `(N,8,8)` while `extract_from_zip.py` saved
  `(N,64)`; the current file is `(N,64)`. Every consumer script indexes it as `(N,64)`.

## 3. Palette derivation and the index/quantization invariant

**How the palettes were derived** (read from `src\build_palettes.py`, `src\custom_palette.py`):
* `palettes.json.pal16` (16) = mode colour of `<colour>_concrete.png` for the 16 dye names in
  `ORDER16` (colour-wheel order).
* `palettes.json.pal48` (48) = the same hue order with concrete/wool/terracotta interleaved
  (`concrete, wool, terracotta` per hue).
* `palette_avatar16.json` (16, the one actually used for `faces8.npy`) = greedy subset
  selection from a candidate pool of MC block-texture mode colours, initialised from a free
  k-means on the **old** 5820-skin `faces8_rgb.npy`, then ordered into a path by
  nearest-neighbour + 2-opt.

**Independent re-derivation** (`07_palette_derivation.py`, walks the local
`D:\czm我的世界\.minecraft`, 10 339 PNGs): concrete 16/16, wool 16/16, terracotta 16/16 found;
re-derived `pal16` == `palettes.json.pal16` (**True**), re-derived `pal48` == `pal48` (**True**),
all three `by_family` dicts equal. Example paths:
`...\resourcepacks\oCD_Pack_1.20\assets\minecraft\textures\block\white_concrete.png` → (207,213,214).

**Invariant check** (`01_npy_basic.py`, `11_palette_crosschecks.py`): recomputing
`argmin_k ||pixel - palette_k||²` for all 3 200 000 pixels of `faces8_rgb.npy` against
`palette_avatar16.json` reproduces `faces8.npy` **exactly: 0 mismatches (0.0000 %)**.
(1 149 of the 3.2 M pixels have an exact tie between best and 2nd-best entry; `np.argmin`
breaks it by lowest index in both the extractor and my re-derivation, so this is harmless.)
Re-deriving with the MC16 concrete palette gives 88.34 % mismatches → confirms
`faces8.npy` carries **custom-palette** labels, not the old MC16 labels.

**Quantization error (current 50 000-sample data):**
```
mean per-pixel squared error (sum over 3 channels) = 1275.8506
per-channel MSE (mean over all 3.2M values)        =  425.2835   <-- the claimed 425.3, exact
per-channel RMS  sqrt(425.2835)                    =   20.6224/255
per-pixel RMS    sqrt(1275.8506)                   =   35.7191/255
max per-channel error                              =  167
MC16 concrete reference (same data): per-channel MSE 914.0990
```
So **425.3 is correct** (it is a per-channel MSE) but **RMS 11.9 is wrong** (§6.1).

## 4. Face crop re-derivation and row→file mapping

**The crop is the conventional face** — `im.crop((8,8,16,16))` = rows 8..15, cols 8..15 of the
64×64 layout. Evidence:
* Re-running `extract_from_zip.py`'s exact logic (zip central-directory order, `convert('RGBA')`,
  crop (8,8,16,16), dedup on the 192-byte crop) reproduces **all 50 000 rows byte-identically**
  (`03_reproduce_zip_extract.py`: `RGB mismatches vs npy: 0`; 54 442 entries scanned to accept
  50 000; first row ← `skin_00930587.png`, row 10000 ← `skin_00269730.png`,
  row 49999 ← `skin_00374656.png`). Labels also reproduce: **0 rows differ**.
* Direct disk check (`05_alpha_axis_cifar.py`): of the 5820 `skins\` PNGs, 288 fall inside the
  scanned window and **all 288 crops are byte-identical to their npy rows** (0 mismatches);
  the other 23 in-window files were skipped by the byte-dedup, and all 23 are byte-duplicates
  of an accepted face — the dedup accounting closes exactly.
  Expected overlap if `skins\` is a uniform sample of the 1 107 411 entries and the npy covers
  the first 54 442: 0.0492; observed 288/5820 = **0.0495**.
* Region comparison over the 5820 skins (`04_rowmap_alpha_crops.py`):

  | region | mean opaque | fully opaque | fully clear | mirror sym | uniq RGB | pixel std |
  |---|---:|---:|---:|---:|---:|---:|
  | **head front (8,8)-(16,16) [used]** | 0.951 | **0.946** | 0.046 | 0.710 | **14.25** | **61.6** |
  | head right (0,8)-(8,16) | 0.945 | 0.941 | 0.053 | 0.533 | 11.46 | 34.3 |
  | head left (16,8)-(24,16) | 0.945 | 0.942 | 0.052 | 0.533 | 11.55 | 34.2 |
  | head top (8,0)-(16,8) | 0.940 | 0.938 | 0.059 | 0.694 | 8.01 | 15.3 |
  | head bottom (16,0)-(24,8) | 0.936 | 0.931 | 0.060 | 0.809 | 7.96 | 34.4 |
  | hat front (40,8)-(48,16) | 0.407 | 0.070 | 0.173 | 0.798 | 7.09 | 46.0 |
  | **shifted (9,9)-(17,17)** | 0.879 | **0.052** | 0.043 | 0.345 | 15.61 | 71.3 |

  The used region is (a) the most detailed opaque region (14.25 unique RGB, std 61.6) and
  (b) perfectly block-aligned: shifting by one pixel drops "fully opaque" from 94.6 % to 5.2 %,
  i.e. the off-by-one crop is misaligned and lands in transparent padding. The 40,8 hat layer
  is a *separate* overlay (only 7 % fully opaque), as expected.
* Visual confirmation: `results\mix50k_real.png` (real avatars rendered with the palette) shows
  unmistakable 8×8 faces; a raw `data\skins\skin_00000157.png` shows the standard layout with
  the face block at (8,8)-(16,16) and its hat duplicate at (40,8)-(48,16).

**Mapping recoverability — YES.** Row `k` of `faces8_rgb.npy`/`faces8.npy` corresponds to the
`k`-th accepted entry when iterating `zipfile.ZipFile(...).namelist()` in order, skipping
non-`.png`, images < 16 px, and crop-byte duplicates. I recovered this mapping and verified it
byte-exactly. Example pairs: row 0 ← `skin_00930587.png`, row 1 ← `skin_00669406.png`,
row 2 ← `skin_01040837.png`, row 100 ← `skin_00569038.png`, row 1000 ← `skin_00915929.png`,
row 49999 ← `skin_00374656.png`. Caveat: this is a *derived* mapping (the npy stores no ids);
it is exact only because the extraction is deterministic and the zip is unchanged.

## 5. Claimed statistics, re-measured with the project's own definitions

| claim | project definition (from source) | measured | verdict |
|---|---|---|---|
| 5.31 distinct colours / avatar | `mean(len(unique(labels)))` over the 16-colour set | **5.3132** over all 50 000 | matches |
| 5.29 "real" | same over `pool[:3000]` (rng(0) sample of 20 000) in `mixture_50k.py` | **5.2883** | matches exactly |
| diversity 57.8 | mean pairwise Hamming distance over 64 positions, `pool[:120]` | **57.7899** exact over all 50 000; 58.0676 for their `pool[:120]` | matches |
| symmetry 0.859 | fraction of the 32 mirror pairs `(r*8+c, r*8+(7-c))` with equal labels | **0.8598** (also recorded as `cover` 0.859799375 in `results\antisym_analysis.json`) | matches |
| MSE 425.3 | mean squared error over all channel values | **425.2835** | matches |
| RMS 11.9/255 | `sqrt(MSE/3)` | wrong; see §6.1 | **does not match** |

Extras: fully mirror-symmetric avatars 0.3362; label-usage entropy 3.771/4 bits; every one of
the 16 palette indices is used; no position is dominated by a single index.

## 6. Concrete problems found

**6.1 RMS understated by √3 (reporting bug).**
`extract_from_zip.py:98` prints `RMS {np.sqrt(qerr/3)}` where `qerr` is already a *per-channel*
mean (`((pal[idx]-F)**2).mean()` averages over N·8·8·**3**). The correct per-channel RMS is
`sqrt(425.28) = 20.62/255` (8.1 % of full scale), not 11.9/255. `extract_faces_all.py:46` has
the same bug; `extract_faces.py:58` computes the same quantity and prints `sqrt(qerr)` = 20.62,
so two of the project's own scripts contradict each other. The MSE 425.3 itself is right.

**6.2 Alpha is dropped without masking (`extract_from_zip.py:72`, `extract_faces.py:42`).**
`np.array(face)[:, :, :3]` takes RGB under alpha=0. Full-archive measurement
(`10_transparency_full.py`, all 1 107 411 entries):
* **52 690 entries (4.76 %) have a fully transparent head-front** (all 64 px alpha = 0) →
  the pipeline yields an all-black (or hidden-colour) 8×8 avatar. 51 156 of them collapse to
  one identical all-black crop, plus 638 distinct hidden-RGB variants.
* 61 636 entries (5.57 %) contain ≥ 1 transparent pixel in the crop, i.e. undefined/authoring-tool
  RGB leaks into the dataset.
* In the surviving 50 000-row sample: 686 rows (1.37 %) have non-opaque pixels, 484 rows have
  alpha-0 pixels, **40 rows (0.08 %) are fully transparent** (39 of the 40 store non-zero
  hidden RGB, e.g. whole-image values the artist never saw), and the transparent pixels are
  93 % exactly (0,0,0), which also inflates palette index 8 `(4,4,4)` to 12.6 % of all pixels.
* This is not catastrophic at 50 000 samples, but any future run over the full 1.1 M inherits
  ~53 k identical black avatars (4.8 %). Fix: drop or mask samples with low alpha coverage
  (the project's own `build_dataset.py` already did this for texture patches, `(a[:,:,3] > 8).mean() < 0.5`).

**6.3 Face-level redundancy in the source (the "1.1 M" is not 1.1 M distinct faces).**
Whole-file duplicates: **0 / 1 107 411** — the "deduped" claim holds at file level (and the
upstream card confirms BLAKE3 pixel dedup). But at the *face-crop* level there are only
**932 636 distinct crops** (15.78 % are repeats), of which ~931 998 are distinct *visible*
faces. The biggest cluster is the transparent-black crop above (51 156) plus 4 870 genuinely
opaque all-black faces = 56 026 entries with an all-black face (5.06 % of the archive).
Largest genuine face cluster: 2 333. Consequence for the extraction: it stops after the first
50 000 *distinct* crops in zip order, so the sample is the first 8.2 %-duplicate-heavy slice,
not a verified random sample; within the first 54 442 entries one all-black crop appeared
2 766 times (5.1 %). The 50 000 rows themselves are all distinct crops (verified), so training
data is not corrupted — but "50 000 samples" is 50 000 distinct crops of 1.1 M heavily
template-clustered skins.

**6.4 In-place overwrite destroyed provenance.**
`faces8.npy` / `faces8_rgb.npy` (and `palette_hypercube_cmp.json`, `results\extract_log.txt`)
were rewritten 2026-10-05 11:48 by `extract_from_zip.py`. The previous dataset (5820 skins from
`skins\`, MC16 labels, `(N,8,8)`) is gone. Therefore every result dated 2026-10-04
(`gan_*`, `capacity*`, `mixture_model`, `sym_*`) was fitted on a dataset that no longer exists
and cannot be reproduced from the archive. Corroborating evidence: `gan_train.py:72` still
comments "faces8.npy is the OLD MC16 labelling, unusable" while the current file is
custom-palette labelled (88.34 % of pixels differ under MC16); `palette_avatar16.json` records
`quant_err 1183.2 / mc16_err 2628.5` from the old sample, whereas on the current sample the same
palette gives 1275.85 and MC16 gives 2742.30. Recommend freezing versioned filenames
(e.g. `faces8_rgb_v2.npy` + a manifest with args/hashes) rather than overwriting.

**6.5 "Quantization error" is 3x ambiguous across files.**
`custom_palette.py:quant_err` and `palette_avatar16.json` use the **per-pixel sum over 3
channels** (1183.2 old / 1275.85 now); `extract_from_zip.py` prints the **per-channel mean**
(425.3 now). They differ by exactly 3x, so the stored 1183.2 must not be compared with the
logged 425.3 (doing so suggests a bogus 2.8x improvement). Also note `pal48` (48 colours) is no
better than the custom 16 on this data (1288.57 vs 1275.85 per-pixel SSE).

**6.6 `pal48` contains a near-collision.** Minimum pairwise RGB distance in `pal48` is **10.44**
(green concrete (73,91,36) vs green terracotta (76,83,42)) — visually the same colour. `pal16`
min distance 39.72, custom avatar16 min distance 49.32 (both fine). Only matters if the
48-colour set drives a redstone LUT where two states must be distinguishable.

**6.7 Smaller notes.**
* Duplicate *label* maps: 47 670 distinct 16-colour maps among 50 000 rows (2 330 rows share a
  label map with another row). Inherent to 16-colour quantization; no defect, but per-sample
  statistics over label maps are effectively over 47.7 k distinct images.
* Blank/uniform avatars are included with no filter (376 flat-RGB rows, 761 single-colour
  label rows, 1 all-black from transparency, 1 all-white). The upstream dataset card lists
  filtering these as a *future* improvement.
* `zip_index.json` is only a **partial** index (first 200 000 of 1 107 411 entries, 18.1 %);
  its content is byte-exact for the entries it covers, but the name suggests completeness.
* The size guard `if im.size[0] < 16 or im.size[1] < 16` (`extract_from_zip.py:70`) is far too
  weak for a crop at (8,8)-(16,16) — a 16×16 image would be accepted and cropped into garbage.
  Moot for this zip (all 1 107 411 entries are exactly 64×64, verified by decoding every one),
  but it is a latent bug.
* 637 `P`, 21 `RGB`, 6 `LA` PNGs exist in the zip; `convert('RGBA')` handles them (the 4 `P`
  files in `skins\` are covered by the 288-row disk verification), so no mode bug.
* Dedup is on the crop only, so distinct skins sharing a face collapse; combined with the
  "stop at N" loop this makes the sample order-dependent rather than random.

## 7. What is genuinely fine (do not "fix" these)

* Zip integrity and identity: SHA-256 `b184c1131fb6037e6920a9011f4482d749b99f185797db0d3f44a171f8ed64c4`
  equals the HuggingFace LFS object id for `minecraft_skins_64x64.zip`; size 2 714 488 499 B
  equals the API's reported size. Count 1 107 411 matches the dataset card.
* All 1 107 411 PNGs decode; all are 64×64; 0 corrupt. `skins\` 5820/5820 readable.
* Crop (8,8)-(16,16) is right, block-aligned, and byte-consistent with the source files.
* No RGB/BGR swap (byte-exact match to PIL's RGB crop), no channel permutation, no transposed axes.
* Labels are exactly `argmin` over the stated 16-colour palette; 0 out-of-range; 0 unused palette
  entries; palette files are mutually consistent and reproducible from the MC install.
* MSE 425.3 and the statistics 5.31 / 5.29 / 57.8 / 0.859 all reproduce; the symmetry rate is
  independently recorded as 0.859799375 in the project's own `antisym_analysis.json`.
* No duplicate rows in `faces8_rgb.npy` (the byte-dedup worked), so there is no train/test
  leakage from repeated crops *within* the 50 000-row file.

## 8. Scripts (all under `02_phase2_avatar\audit_tmp\`)

| script | purpose |
|---|---|
| `01_npy_basic.py` | npy headers/shapes/dtypes/ranges, index invariant, MSE variants, stats, symmetry, duplication |
| `02_zip_and_skins.py` | zip central directory, test decompression, `zip_index.json`, `skins\` inventory |
| `03_reproduce_zip_extract.py` | exact re-extraction from the zip; row→filename mapping; per-row byte check |
| `04_rowmap_alpha_crops.py` | alpha per row, alternative-crop comparison, ASCII face rendering, `skins\`↔row matching |
| `05_alpha_axis_cifar.py` | transpose test, transparency accounting, disk verification, CIFAR tar, `mc8.npy` sanity |
| `06_stats_and_consistency.py` | project-identical reference stats, exact whole-set diversity, dedup accounting, CIFAR batches |
| `07_palette_derivation.py` | independent palette re-derivation from the MC textures; error recomputation |
| `08_dedup_and_index.py` | face-crop multiplicity in the scanned window, `zip_index.json` vs real headers |
| `09_full_zip_scan.py` | all 1 107 411 entries decoded: sizes/modes/corruption, face-crop redundancy |
| `10_transparency_full.py` | full-archive transparency split, hidden RGB, whole-file duplicate check |
| `11_palette_crosschecks.py` | palette file cross-checks, MC16-vs-custom labelling proof, entropy |
| `00_probe.txt` | sandbox write probe (6 bytes) |
