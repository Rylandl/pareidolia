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
- `python3 scripts/build-normal-families.py --root
  work/cross-scroll-analysis-z512` adds a conservative multi-normal census
  without changing the Acus needle bake. It records 21,621 standalone
  secondary candidates separately from the 9,560 cells (3.80% of valid cells)
  admitted by a three-cell spatial-support rule, so neighbor agreement is an
  inclusion filter rather than circular evidence. The included cells have
  19.66% median exclusive needle coverage, 0.573 median refit confidence,
  5.65% genuinely margin-ambiguous weight, and 22.79% broader plane overlap.
  The largest secondary-normal region contains 397 cells and spans all 14
  planes. Primary flakes retain their original normal, confidence, inputs, and
  membership exactly; the z512 audit finds 714,987 unchanged primary flakes,
  9,685 additive secondary flakes, and zero shared needle IDs.
- Normal families are kept as separate surface hypotheses in sheetlet
  construction. An early cross-family join trial failed the declared carrier
  gates because a few alternate-family flakes were absorbed as extreme
  curvature; the final graph therefore permits no cross-family links. This
  preserves all baseline macro counts exactly (538 components spanning at
  least 11 planes, 241 spanning all 14, and 3,237 spanning at least six) while
  4,298 secondary nodes form 2,699 independent links with zero cell
  collisions. Sixty-three pure secondary fragments of 5–13 flakes are retained
  as a separate small-seed class instead of being attached to legacy carriers.
  Their median carrier residual is 0.217 voxels / 2.09°, their median fit factor
  is 0.939, and their CT screen covers 114,736 gross square voxels with a 0.332
  median best-plane texture score. Gross support is not unique recovered area
  and is not assumed to lie in an existing carrier hole. Secondary seeds are
  deliberately excluded from legacy boundary assembly and growth until a
  family-constrained seed-growth stage is evaluated.
  `python3 scripts/evaluate-normal-families.py --root
  work/cross-scroll-analysis-z512` independently checks persisted primary
  values and needle memberships, per-family carrier residuals, graph
  preservation, CT screen results, and the predeclared construction gates. As
  throughout this pipeline, these are surface hypotheses rather than claimed
  physical papyrus identities.
- `python3 scripts/science-ci.py --root
  work/cross-scroll-analysis-z512 --verify-artifacts` freezes and checks the
  active multi-normal science state before architectural experiments. The
  committed benchmark content-hashes 37 active artifacts (781.6 MB), preserves
  all primary values and memberships, requires disjoint family ownership and
  zero cell collisions, and guards the established long-span and secondary-fit
  results. It intentionally excludes stale single-normal carrier descendants.
- `python3 scripts/build-material-intervals.py --root
  work/cross-scroll-analysis-z512` samples native CT from -32 through +32
  voxels along 261,302 local normal-family hypotheses. Material is thresholded
  before consulting any flake; a separate overlay records air, unassigned,
  singly claimed, and contested material without assigning sheet IDs. The full
  census takes 26 seconds. It finds 96.32% material samples, 243,069 fully dense
  windows, 234,496 contested profiles, and only 72 non-boundary singly claimed
  intervals eligible for an apparent-thickness statistic. A deterministic
  threshold sweep shows that apparent separators at higher cutoffs come with
  substantial loss of supported flake evidence, so CT air gaps are retained as
  accountability constraints rather than promoted to layers.
- `python3 scripts/prototype-monotone-layers.py --root
  work/cross-scroll-analysis-z512` applies reversal-invariant partial sequence
  alignment in a 32 x 32 x 14-cell primary-family window. Both relative depth
  orientations are solved for every adjacent cell; exact-score ties retain
  only orientation-invariant links. The chosen link carries relative parity,
  never an absolute normal side. Collision-safe links are then processed in
  descending score and the weakest edge in every parity-inconsistent cycle is
  explicitly deferred. A representative dense window retains 45,297 links,
  rejects 238 parity-cycle edges, and has zero pairwise order crossings.
- `python3 scripts/associate-monotone-branches.py --root
  work/cross-scroll-analysis-z512` scores compatible facing endpoints of those
  local branches. Independent branch gauges are aligned only through parity
  votes in cells the branches actually share; tied or frustrated order
  observations are omitted from the hard order graph. Material support,
  collision safety, overlapping subwindows, and the active MLS reconstruction
  remain construction gates. In the representative dense window, 54 stable
  candidates become 32 exact-coherent joins in 28 associations; 22 are
  explicitly deferred by the 3-voxel / 6-degree median carrier gate.
- `python3 scripts/audit-branch-association-integrity.py --root
  work/cross-scroll-analysis-z512` reconstructs and triangulates every accepted
  merged association and reports support-skirt and evidence-core intersections
  separately. The representative dense solve has zero intersections and one
  consistently ordered near-contact. Integrity remains a veto rather than a
  score: tiled associations involved in any mesh intersection are retained in
  a quarantine catalog and excluded from consensus joins.
- `python3 scripts/run-window-schedule.py --root
  work/cross-scroll-analysis-z512 --maximum-workers 4` runs the same bounded
  solve over every occupied tile. The 242 x 242 x 14 grid requires 51 occupied
  windows and 86 face-overlap reconciliations; 49 empty windows are skipped
  without losing a claim. A fresh four-worker pass takes 310 seconds and covers
  all 704,145 primary flakes. All 818,414 raw matches and all relative parities
  agree in every observing window. Later collision/parity pruning leaves 2,148
  context-dependent retained edges. Of 521 local accepted join pairs, the
  integrity veto quarantines 12 unique pairs from 14 local association
  occurrences; 125 remaining joins are both overlap-observed and unanimous.
  The resumable manifest and consensus arrays are stored in
  `tiled-window-schedule-v4`.
- `python3 scripts/build-global-monotone-graph.py --root
  work/cross-scroll-analysis-z512` consumes only unanimous raw matches and
  parities. The 12.1-second whole-volume sparse solve retains 783,846 edges
  after 29,151 cell-collision and 5,417 parity-cycle rejections. It produces
  63,783 linked branches and 636,717 linked flakes with zero cell collisions;
  the largest branch has 586 flakes, 531 branches span at least 11 axial
  planes, and 235 span all 14. Every repeated edge-score observation is exactly
  equal. These remain sparse local surface branches, not pages or sheets.
  Detailed contracts and findings are in `docs/design-revision.md`.
- `python3 scripts/associate-global-branches.py --root
  work/cross-scroll-analysis-z512` preserves three evidence tiers rather than
  hiding crop disagreement: 125 overlap-validated joins, 214 unanimous
  single-window joins, and 172 context-disputed endpoint observations. Of the
  latter, two remain excluded by the local mesh quarantine and 13 are already
  linked inside one global branch, leaving 157 novel disputed candidates. Every
  novel candidate is reconstructed from its complete global branches. The solve
  retains all 125 overlap joins, 213 single-window joins, and 141 disputed joins;
  13 fail because an input carrier remains incoherent, two cause cell collisions,
  and the two weakest disputed edges in intersecting carrier pairs are removed.
  The resulting 479 joins form 458 exact-coherent associations over 937 branches
  and 9,415 flakes: 439 pairs, 17 triples, and two four-branch groups. Final
  carrier medians are 1.25 voxels / 3.88 degrees, with zero exact failures and
  zero intersections after 101,818 broad-phase and 2,486 narrow-phase triangle
  checks. The 25-second result reduces 63,783 linked branches to 63,304 groups.
  In the full catalog it adds 17 fragments with at least 25 flakes, four with at
  least 100 flakes, eight spanning at least 11 planes, and six spanning all 14
  relative to the unassociated graph. `--clean-only` exactly reproduces all 17
  compared v2 geometric and decision arrays. Evidence, provenance, residuals,
  and decisions are retained in `global-branch-association-v3`.
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
- `python3 scripts/census-sheetlet-gaps.py --root
  work/cross-scroll-analysis-z512` scans enclosed gaps in every final carrier.
  CT sampling is cropped to block-aligned gap bounds, preserving the texture
  estimator's original grid phase while avoiding full-carrier resampling. The
  census takes 17.3 seconds: 44 of 121 carriers contain 65 enclosed gaps with
  78,083 square voxels of total area, but only ranks 11, 12, and 24 pass the
  depth-aligned texture, material, depth, and fiber gates. The nearest rejected
  gap has a 0.4121 texture score, leaving a clear separation from the weakest
  queued score of 0.5521. Upstream artifacts are content-hashed so a stale
  census cannot survive regenerated carrier inputs. Results are stored in
  `work/cross-scroll-analysis-z512/sheetlet-gap-census-v1.json`.
- `python3 scripts/reanalyze-sheetlet-gaps.py --root
  work/cross-scroll-analysis-z512 --ranks auto` re-extracts Acus needles only
  for the census queue. It uses an 8-voxel cell covering, 2-voxel candidate
  spacing, up to 640 needles per cell, and the GPU Hessian path, then requires
  a 0.55 carrier score, a 0.50 depth-aligned CT texture score,
  best-vs-second mode separation, and best ownership across all 121 carriers.
  The three-rank run takes 10.4 seconds and checks 55 new modes through 6,655
  carrier comparisons. Rank 11 is classified as orthogonal near-surface
  evidence: its two near-surface modes are about 88.3 degrees from the carrier
  fiber, while the matching family is at least 13.5 voxels away. Rank 12 gains
  one 56-needle flake that supports all 133 gap pixels; its acceptance score is
  0.5606 with 0.0106 threshold slack and zero post-fit score drift. Rank 24 has
  strong near-surface evidence, but carrier 3 fits it better (0.7419 versus
  0.7111), so ownership rejects the fill. Per-threshold slack, global
  ownership, post-fit rescoring, and gap classifications are recorded in
  `work/cross-scroll-analysis-z512/sheetlet-gap-reanalysis-v2.json`.

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
