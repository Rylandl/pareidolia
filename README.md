# Acus / Rectifier Lab

The current pilot is Acus: an interactive local fiber-volume study. From a
selected scan point it renders an adjustable N³ raw voxel cube, extracts short
unsigned needle-like ridge primitives, and estimates the direction most nearly
orthogonal to their orientations. It does not assign a papyrus sheet or claim
that each ridge primitive is an individual physical fiber.

The browser provides three linked orthogonal scan views and an orbitable GPU
volume rendering. Acus needles and the recovered shared normal are projected
into that same 3D view. The fitted result also includes an orientation-density
profile over signed position along the shared normal, using the circular
unsigned 0–180° orientation domain.

The optional 3×3 Acus field places neighboring cube centers in the anchor
tangent plane. It independently refits every center, parallel-transports local
orientation frames to the anchor, and reports normal deviation, best profile
correlation, and the signed depth lag at maximum profile agreement.

Acus fits an inner N³ analysis region from a larger real-data context cube.
Needle length is an explicit finite support constraint: every accepted ridge
must maintain evidence along its full axis, while candidate centers remain in
the inner cube. By default the real-data halo equals the requested needle
length; it remains independently adjustable for stability audits. Fits are
rejected when that context would extend beyond the loaded cuboid rather than
silently using synthetic padding.

## Current pilot

- The tailnet launcher uses a full-resolution 256³ cuboid from PHerc. 358 when
  `data/pherc0358-z7168-y5888-x4608.npy` is present, and otherwise falls back to
  the deterministic synthetic scroll.
- A different local 3D NumPy volume can be supplied as a ZYX `.npy` array.
- `POST /api/needles` computes a polarity-agnostic 3D Hessian ridge response,
  refines local candidates with weighted PCA, and robustly solves the shared
  unsigned normal in orientation space.
- When `cupy-cuda12x` is available, Acus automatically computes the dense
  Hessian line field on CUDA. Its analytic symmetric 3×3 eigensolver avoids the
  large cuSOLVER workspace; the normal system Python remains a complete CPU
  fallback. Set `ACUS_COMPUTE=cpu` to force the reference path.
- Every needle is assigned a signed normal coordinate. A small circular kernel
  density model then reports the strongest one or two orientation modes through
  depth and their exploratory two-mode coverage; these modes are not treated as
  sheet identities.
- The current result is deliberately not sheeted. There is no predicted
  surface, layer identity, winding, recto, verso, or fiber-direction sign.
- The earlier phase-neutral local chart endpoint remains available at
  `POST /api/fit` for comparison, but it is not the active UI workflow.
- `POST /api/field` runs the 3×3 neighborhood comparison. Because nearby N³
  cubes overlap, its spacing control must be swept before high coherence is
  interpreted as independent evidence. The anchor is solved once and the eight
  neighbor contexts are submitted as a bounded GPU batch.
- `POST /api/audit` sweeps tangent-field spacing, reports exact axis-aligned
  cube overlap, block-bootstraps spatial needle groups for normal uncertainty,
  and compares transported profile agreement against a shuffled-depth null.
- `POST /api/padding-audit` compares halos below, at, and above the current
  needle length, measuring boundary-face tangency, axial support, normal drift,
  and profile stability against the largest available real-data halo.
- `POST /api/region` analyzes a reusable finite-needle catalog across the loaded
  volume in haloed GPU tiles, then summarizes local normals and depth-pattern
  evidence on a regular grid. Candidate selection uses a globally calibrated
  strength scale and globally anchored bins so tile boundaries do not define
  the result. Completed analyses are cached on disk under `work/region-cache`.
- The volume-scale evidence view colors the local normal glyphs by adjacent-normal
  stability, orientation-pattern agreement, confidence, or depth coverage.
  Clicking a glyph moves all linked views to that seed for the existing local
  fit, field, and audit tools; it still does not connect cells into sheets.
- `GET /api/slab/flakes` derives up to three bounded depth–orientation modes per
  valid slab cell from the retained finite-needle catalog. Each exploratory
  flake records a fixed Acus normal, unsigned fiber axis, center, finite
  footprint, thickness, support, and quality. Adjacent cells are mutually
  matched by position, transported normal, and transported fiber direction;
  link scores explicitly discount reused needles.
- `GET /api/slab/flake-holdout` deterministically splits the raw needle catalog
  into two disjoint halves, fits each half independently, and measures mutual
  rediscovery against a fully rematched depth/fiber permutation null. Across
  the six current planes, 39–40% of full-data flakes replicate, with about
  0.9-voxel median depth disagreement, 2.2° median fiber disagreement, and
  7.7–8.1× as many validated pairs as the null.
- `GET /api/slab/sheetlets` links only those held-out-replicated flakes at
  non-overlapping 64-voxel X/Y/Z spacing. The first graph contains 122,618
  validated nodes and 30,678 mutual links (7,951 across Z), producing 17,173
  multi-flake components. Link density is 45.9× the whole-cell spatial null
  overall and 36.9× the null along Z; these remain sheetlet hypotheses rather
  than physical page identities.
- `python3 scripts/analyze-sheetlets.py` builds the denser exploratory successor
  offline, without adding more UI controls. It matches adjacent flakes by
  transported fiber direction, finite-footprint reach, and the residual where
  their endpoint tangent planes extrapolate to meet. Normal change is recorded
  as curvature rather than penalized. Strongest-first component assembly also
  forbids two competing flakes from the same Acus cell. On the current slab,
  the selected construction links 273,770 of 304,348 usable flakes into 30,642
  components, with 1,146 components crossing all six axial planes and 3,135
  substantial candidates of at least 20 flakes. Its largest candidate contains
  363 unique cells; retained edges have 0.71-voxel median meeting residual and
  1.07° median transported-fiber disagreement.
- `python3 scripts/screen-sheetlet-carriers.py` performs a resumable coarse
  carrier-and-texture pass over all 3,135 substantial components without
  writing per-candidate imagery. The current whole-catalog screen completes in
  37 seconds. `python3 scripts/build-sheetlet-carriers.py --screened-top 64`
  then turns only its winners into exact continuous carriers. It blends each
  component's local tangent-plane predictions into a smooth supported height
  field, carries the varying normals through the raster, and samples the native
  CT volume from -12 to +12 voxels along those normals. A depth-resolved
  structure-tensor measurement ranks construction yield from supported physical
  area, carrier residual, normal residual, and directional texture. Coarse and
  exact yield scores correlate at 0.992 across the selected 64; 18 of the exact
  top 20 are already in the coarse top 20 and all are in the coarse top 32.
  The exact pool includes coherent carriers originally ranked as low as 260th
  by component geometry, and 61 of 64 have both a median height residual below
  three voxels and a median normal residual below six degrees. Geometry, depth
  stacks, best-texture previews, montages, and both rankings are stored under
  `work/cross-scroll-analysis` for later visualization.
- `python3 scripts/assemble-sheetlet-carriers.py` extracts 108,508 compact 3D
  boundary samples from the 1,855 carriers with construction fit at least 0.7.
  Nearby outward-facing edges are linked only when their tangent planes,
  transported fiber directions, and local normals continue across the gap;
  strongest-first assembly still forbids any repeated Acus cell. At the
  selected 0.45 edge score, 75 carriers form 37 multi-carrier hypotheses. The
  largest contains three original carriers, so this stage does not collapse
  into a giant transitive component. `python3
  scripts/preview-sheetlet-assemblies.py --top 12` rebuilds the leading joins
  as single exact 25-plane carriers. All 12 remain below three voxels median
  surface residual and six degrees median normal residual; joining increases
  the median surface residual by only 0.105 voxels and leaves median normal
  residual effectively unchanged. The merged stacks and previews are stored
  under `work/cross-scroll-analysis/sheetlet-assemblies-v1`.
- `python3 scripts/grow-sheetlet-carriers.py` iteratively extrapolates the 12
  leading merged carriers into all 26 neighboring Acus cells. A candidate flake
  must agree with the refit carrier's predicted position, local normal, and
  transported fiber direction, win its cell by at least 0.04 score, and remain
  globally unclaimed. Growth converges naturally after 11 rounds, adding 468
  unique flakes to 2,181 seed flakes (+21.5%) with zero repeated-flake or
  same-seed cell collisions. The resulting supported flattened area grows
  21.9%. All 12 exact grown carriers remain below three voxels median surface
  residual and six degrees median normal residual; the median height and normal
  residuals both improve slightly after growth. Exact grown geometry and depth
  stacks are stored under `work/cross-scroll-analysis/sheetlet-growth-v1`.
- `python3 scripts/iterate-sheetlet-carriers.py` runs growth and boundary
  rematching to a fixed point from all 37 merged hypotheses. The first cycle
  grows 4,039 seed flakes to 4,963 unique flakes (+22.9%); the second cycle
  adds nothing, so the local growth stage has converged. No new carrier pairs
  meet the existing 40-voxel boundary rule after growth, and 11 spatially close
  pairs are correctly excluded because merging would repeat an Acus cell. The
  final catalog therefore remains 37 distinct sheets with zero repeated-flake
  assignments or within-sheet cell collisions. Exact previews of the leading
  final sheets remain below three voxels median surface residual and six
  degrees median normal residual. Results are stored under
  `work/cross-scroll-analysis/sheetlet-iteration-v1`; further consolidation now
  requires an explicit longer-range gap-bridging model rather than more local
  growth cycles.
- `python3 scripts/bridge-sheetlet-carriers.py` tests that longer-range model
  against outward-facing carrier boundaries 40–128 voxels apart. It scores
  endpoint tangent-plane, transported-fiber, local-normal, and facing
  agreement, then samples the native CT volume along the interpolated gap and
  looks independently for compatible unclaimed Acus flakes. Only one of the
  37 fixed-point sheets has a supported continuation at every tested score
  threshold from 0.28 through 0.45: a 78.4-voxel join with 2.19-voxel endpoint
  plane residual, 1.79° fiber disagreement, 4.07° normal bend, and 0.99 facing
  cosine. All interior CT samples contain material and 44.4% show a local
  normal-direction ridge, producing a 0.86 CT score, but no intermediate Acus
  flake supports the interval. The collision-safe merge reduces the catalog
  from 37 to 36 hypotheses with no repeated flakes or cells. Its exact
  965-flake reconstruction remains geometrically stable at 2.07 voxels median
  surface residual and 4.45° median normal residual. We therefore retain this
  as a strong CT-supported continuation hypothesis, not as a recovered
  continuous sheet. Results and exact previews are stored in
  `work/cross-scroll-analysis/sheetlet-carrier-bridges-v1.json` and
  `work/cross-scroll-analysis/sheetlet-bridges-v1`.
- The doubled-depth run in `work/cross-scroll-analysis-z512` repeats the same
  pipeline over 512 source slices and 14 Acus grid planes. It retains 704,145
  usable flakes and 790,050 collision-safe direction/edge links. At the same
  0.60 construction threshold, 241 components span all 14 planes and 538 span
  at least 11; the largest raw component grows from 363 to 586 cells while the
  median edge residual (0.708 voxels), transported-fiber disagreement (1.068°),
  and normal bend (5.668°) remain essentially unchanged. Its exact carrier has
  1.307-voxel median height residual and 3.043° median normal residual.
  Screening all 7,133 substantial components takes 85 seconds, exact-building
  the top 64 takes 21 seconds, and boundary assembly forms 128 collision-free
  multi-carrier hypotheses in 145 seconds. Fixed-point growth converges in four
  cycles, adding 6,434 flakes (16,351 to 22,785) with no repeated assignments
  or same-sheet cell collisions. Seven long-range bridges then reduce the 128
  grown states to 121; three have intermediate flake support and four are kept
  as provisional CT-only continuations. The largest exact joined carrier has
  965 flakes with 2.069-voxel median height residual and 4.446° median normal
  residual.
- `python3 scripts/fill-sheetlet-gaps.py --root
  work/cross-scroll-analysis-z512 --top 24` audits only fully enclosed holes in
  the final carriers. It projects every flake hypothesis into each flattened
  hole, applies the existing height, normal, fiber, score-margin, ownership,
  and one-flake-per-cell rules, and separately samples CT texture along an
  expanded carrier without treating that texture as permission to fill. The
  first run accepts only three unclaimed flakes, all in rank 3's 9,860-square-
  voxel enclosed gap. They make 665 of 1,812 gap pixels newly supported while
  changing exact median carrier residual only from 1.622 to 1.635 voxels and
  normal residual from 3.784° to 3.791°. Rank 2 contains four compatible gap
  flakes, but all are owned by other carriers and remain untouched. Other
  holes frequently peak 5–6 voxels off the predicted surface or carry a nearly
  orthogonal fiber family. Ranks 11 and 12 instead have near-depth,
  direction-matched CT texture but no compatible flake hypothesis, making them
  focused candidates for denser local Acus re-analysis rather than permissive
  CT-only filling. Gap maps and exact previews are stored under
  `work/cross-scroll-analysis-z512/sheetlet-gaps-v1`.
- `python3 scripts/reanalyze-sheetlet-gaps.py --root
  work/cross-scroll-analysis-z512 --ranks 11,12` re-extracts Acus needles only
  around CT-positive enclosed gaps. It uses an 8-voxel cell covering,
  2-voxel candidate spacing, up to 640 needles per cell, and the GPU Hessian
  path, then requires a 0.55 carrier score, a 0.50 depth-aligned CT texture
  score, best-vs-second mode separation, and best ownership across all 121
  carriers. The two-rank run takes 9.5 seconds and checks 39 new modes against
  every carrier. Rank 11 remains empty for a useful reason: its two
  near-surface modes are about 88.3 degrees from the carrier fiber, while the
  matching fiber family is at least 13.5 voxels away. Rank 12 gains one
  independently fitted flake with 56 needles, 3.744-voxel height residual,
  5.491-degree normal residual, and 1.980-degree fiber residual. That single
  flake supports all 133 pixels of the enclosed hole while changing median
  carrier height residual from 1.408 to 1.431 voxels and median normal residual
  from 3.014 to 3.017 degrees. Outputs are stored under
  `work/cross-scroll-analysis-z512/sheetlet-gap-reanalysis-v1`.

The included bounded Zarr importer reads raw, uncompressed Zarr v2 chunks into
a local `.npy` cuboid. Whole-scroll multiscale navigation and demand-loaded
full-resolution fitting remain the next data-adapter step.

## Native cross-scroll slab

The first volume-scale experiment uses a 256 × 7,783 × 7,783 native-resolution
slab at source Z 7168–7423. It covers approximately 2.4 × 72.9 × 72.9 mm. The
nominal uint8 array is 15.5 GB, but the masked source is downloaded into a
sparse NumPy memmap and occupies about 10 GB on the current filesystem.

The current extension is 512 × 7,783 × 7,783 at source Z 7168–7679, or about
4.8 × 72.9 × 72.9 mm. `scripts/extend-zarr-slab.py` seeds any larger nested
fetch from a completed slab without redownloading material or fill chunks. The
second half transferred 4.92 GB in 695 seconds; the resulting array is 31.0 GB
logical and about 20 GB allocated. Both sources live in
`/mnt/t5/acus-cross-scroll`.

The fetcher records every completed or fill chunk in an atomic manifest and can
be rerun unchanged after interruption:

```bash
python3 scripts/fetch-zarr-slab.py \
  --url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr \
  --level 0 --origin-zyx 7168 0 0 --shape-zyx 256 7783 7783 \
  --output /mnt/t5/acus-cross-scroll/pherc0358-z7168-d256-yfull-xfull.npy \
  --name 'PHerc. 358 · full cross-scroll slab · Z 7168–7423' \
  --voxel-size-microns 9.362 --workers 24
```

The slab analyzer is separately resumable. It uses a globally calibrated ridge
strength, globally anchored candidate bins, haloed GPU tiles, a fixed spatial
needle catalog, and memory-mapped evidence arrays. The first macro pass uses an
N=64 context every 32 voxels, yielding 351,384 possible local summaries:

```bash
CUDA_PATH=/usr ACUS_GPU_BATCH_VOXELS=22000000 \
  .venv/bin/python scripts/analyze-acus-slab.py \
  --source /mnt/t5/acus-cross-scroll/pherc0358-z7168-d256-yfull-xfull.npy \
  --output work/cross-scroll-analysis \
  --grid-stride 32 --tile-core 128 --calibration-tiles 96
```

The 512-depth run reuses the first slab's fixed strength calibration and fails
closed if CUDA is unavailable:

```bash
.venv/bin/python scripts/analyze-acus-slab.py \
  --source /mnt/t5/acus-cross-scroll/pherc0358-z7168-d512-yfull-xfull.npy \
  --output work/cross-scroll-analysis-z512 \
  --grid-stride 32 --tile-core 128 \
  --strength-scale 0.049903104081749916 --compute gpu
```

On the GTX 1080 this processes 14,884 tiles and 4,244,755 finite needles in
31.7 minutes, then summarizes 819,896 grid cells (251,742 valid). This is about
the same wall time as the original 256-depth run despite doubling the source
depth. The manifest records the resolved backend and device so a silent CPU
fallback cannot be mistaken for a GPU run.

`GET /api/slab/status` exposes fetch/extraction/summary progress to the webpage.
Once complete, `GET /api/slab/overview` serves a bounded level-of-detail normal
field instead of serializing all cells or loading the complete catalog into the
browser.

The completed pilot contains 1,988,000 finite needles and 108,915 valid local
summaries. Median adjacent-normal disagreement is 6.851 degrees and median
neighbor orientation-pattern agreement is 0.8662. A sign-invariant transverse
radial fit lands at local XY `(3981.36, 3970.74)` with a 32.568-degree median
residual, versus 49.349 degrees when normals are shuffled among locations. The
six depth planes also recover a monotonic fitted-center drift of about 114
voxels. These are macro-scale evidence diagnostics, not sheet assignments.

The first cached flake pass contains 309,123 local hypotheses across the six
planes. At the default overlap-discounted link threshold, median matched-fiber
disagreement is about 1.2 degrees versus about 42.3 degrees after shuffling.
Matched neighbors still share roughly 59 percent of their supporting needles,
so the webpage reports that dependence and treats linked tracks as hypotheses,
not independently verified physical sheets.

The independence audit reruns the complete mutual matcher after fiber, depth,
and spatial shuffles rather than scoring already-selected links. It compares
32-voxel overlapping neighbors with 64-voxel non-overlapping windows and
96-voxel windows separated by a 32-voxel gap. Across all six planes, the
64-voxel links retain a median 56.8 percent of the adjacent link density with
zero shared needles. Their median fiber disagreement is 2.37 degrees versus
7.31 degrees after fiber shuffling and rematching, and their link density is
about 46 times the spatially shuffled control. The 96-voxel links retain a
median 23.9 percent of adjacent density. These controls strengthen the local
continuity result but still do not assign physical sheets.

## PHerc. 358 real-data cuboid

The checked-in JSON sidecar records the source Zarr, global XYZ origin, source
shape, scale level, voxel size, and suggested local seed. The 16 MB volume is a
local ignored data artifact and can be reproduced with:

```bash
python3 scripts/fetch-zarr-cuboid.py \
  --url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr \
  --level 0 --origin-zyx 7168 5888 4608 --shape-zyx 256 256 256 \
  --output data/pherc0358-z7168-y5888-x4608.npy \
  --name 'PHerc. 358 · 9.362 µm real cuboid' --voxel-size-microns 9.362 \
  --suggested-seed-xyz 64 64 64
```

## Run locally

Requirements are Python 3.12 with NumPy and Node.js 22.13 or newer.

```bash
npm ci
python3 backend/server.py --host 127.0.0.1 --port 8000
npm run dev
```

Open `http://127.0.0.1:3000/`.

To use a local volume:

```bash
python3 backend/server.py --volume /data/crop.npy --host 127.0.0.1 --port 8000
```

For the tailnet launcher, set `RECTIFIER_VOLUME=/data/crop.npy` to override the
PHerc. 358 default.

The volume path is fixed when the backend starts. Browser requests cannot open
arbitrary server paths or remote URLs.

## Run over Tailscale

When Tailscale is running on the host:

```bash
scripts/start-tailnet.sh
```

The launcher builds and starts the production UI, then binds one same-origin
proxy specifically to the machine's Tailscale IPv4 address. Both the vinext
process and data API remain on loopback. This avoids cross-port browser
restrictions, omits Vite's WebSocket-dependent hot-reload client, and does not
expose the pilot on every LAN interface.

## Coordinate contract

- Source array and source bounds: ZYX.
- User-visible points and vectors: XYZ.
- Acus cube bytes: ZYX, normalized unsigned 8-bit intensity.
- Needle centers, endpoints, directions, and shared normal: XYZ.
- Needle direction sign and fiber or sheet identity: intentionally absent.

## Validation

```bash
npm run build
node --test tests/*.test.mjs
python3 -m unittest backend.test_rectify -v
```

The optional local CUDA environment is described by `requirements-gpu.txt`.
`ACUS_GPU_BATCH_VOXELS` bounds each GPU launch (eight million voxels by default),
so larger contexts are split without changing the fit contract.

The backend tests cover the analytic rolled volume, exact seed anchoring,
finite chart geometry, cube padding, Acus normal recovery on a crossed-needle
phantom, the cached cuboid-wide neighbor field, air-seed rejection, and PNG
output.
