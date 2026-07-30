# Raw Acus cubical pipeline

This is the production-oriented path from native CT voxels to cubical papyrus
surface hypotheses. It does not read the legacy Acus needle catalog, the
three-mode flake files, a sheetlet graph, or any prior component identity.
Those artifacts remain useful historical experiments, but they are not a
dependency of this pipeline.

The core command is:

```bash
python3 -m backend.cubical full-acus \
  --source /mnt/t5/acus-cross-scroll/pherc0358-z7168-d512-yfull-xfull.npy \
  --voxel-origin 3520 2784 160 \
  --shape 8 8 6 \
  --shard-shape 4 4 3 \
  --compute gpu \
  --output /mnt/t5/pareidolia/raw-acus-8x8x6-v1
```

`--voxel-origin` is XYZ in the source `.npy`, whose array order is ZYX.
Scanner-space `originXYZ` and `voxelSizeMicrons` are read from the `.json`
sidecar. The output grid therefore preserves the original scan coordinates.

An optional JSON object can override `RawAcusSettings` fields. The exact file,
source stat, metadata hash, settings, shard geometry, artifact schema, and
implementation hashes are folded into one pipeline identity. A root with a
different identity is rejected rather than silently mixing results.

Adjacent blocks from the same native source may explicitly share one
source-level calibration with `--calibration /path/to/calibration-v1.json`.
Only the calibrated scalar values and sampled source bounds are reused; the
reference path and content hash become part of the new block identity, and a
new identity-bound local calibration artifact is written. Needle extraction,
evidence, configurations, and patch selection are still rerun independently.

## Cell and shard ownership

At the default settings, one cubical cell owns a disjoint 32³-voxel region.
Its Acus evidence is estimated from a 64³ context centered on that cell. A
further 16-voxel halo supplies a full needle length to candidate centers near
the evidence boundary. A 4 × 4 × 3-cell shard consequently reads one
192 × 192 × 160 raw block.

Neighboring shards deliberately overlap in raw evidence but never in cell
ownership. This removes voxel-boundary orientation bias without creating
duplicate surface patches. Every shard can be deleted and regenerated from the
native CT volume independently; completed matching artifacts are verified by
SHA-256 before they are reused.

Hessian extraction is partitioned separately from inference. Fixed 128³
candidate cores are anchored at voxel 16 of the native source and padded by a
16-voxel raw halo, so their usual GPU input is 160³. Evidence shards gather
the immutable candidates whose centers fall in their 64³ cell contexts and
deduplicate only inside each half-open cell context. Changing the evidence
shard shape therefore changes scheduling and duplicated shard-local storage,
not the candidates seen by a cell. A real 192-cell audit produced byte-exact
normal, depth-orientation, CT-profile, covariance, fiber, and top-M
configuration arrays at both 8 × 8 × 3 and 4 × 4 × 3 partitioning.

## Stages and artifacts

1. `calibration-v1.json` records the intensity, air-threshold, and
   Hessian-strength scales. By default they are sampled from the native CT
   window. An explicitly supplied source-level calibration can instead keep
   adjacent independent blocks on one scale without reusing inferred Acus
   evidence.
2. `extraction-tiles/*/needles-v1.npz` contains the canonical, source-anchored
   refined finite-support axial ridge primitives. The GPU path keeps the
   Hessian field, candidates, and needle refinement resident on CUDA and
   transfers only retained needles. Analytic symmetric 3 × 3 eigensolvers
   avoid cuSOLVER and its large/version-sensitive dependency stack. The same
   code has a CPU reference path. `shards/*/needles-v1.npz` is the deterministic
   spatial gather needed by that evidence shard, not another Hessian solve.
3. `shards/*/evidence-v1.npz` retains the complete local
   `normal hypothesis × signed depth × unsigned orientation` likelihood, its
   absolute density scale, depth support, and independent native-CT material
   profile. The default depth axis has 65 one-voxel samples and the orientation
   axis has 36 five-degree bins over `[0, 180)`.
4. `shards/*/modes-v1.npz` stores every independently fitted local Acus mode
   before configuration pruning. A mode keeps its source depth/orientation,
   fitted plane covariance, unsigned fiber axis, material probability, and
   effective support. This is the reusable evidence bank for contextual
   inference; top-M compression is no longer the only surviving copy.
5. `shards/*/stratigraphies-v1.npz` stores up to eight competing layer-count
   configurations per cell. Each layer mode is refitted as a sloped plane from
   its own needles, with a covariance over two normal tilts and height plus an
   unsigned fiber axis and angular uncertainty.
6. `selection-v1.npz` chooses one configuration per cell by combining its
   local log posterior with the ordered shared-face trace likelihoods of its
   six neighbors. The face term is measured relative to leaving all traces
   unmatched, so a supported continuation is rewarded while a real open edge
   is not automatically penalized.
7. `selected-patches-v1.npz` is the structure-of-arrays cubical geometry.
   Hierarchical block assembly then performs global same-cell collision and
   crossing-topology vetoes, writes `surface.obj`, and produces
   `projections.png` plus a readable `largest-component.png` check-in view.

The root `pipeline.json` is checkpointed after each stage and shard. Re-running
the same command verifies and skips this pipeline's completed artifacts.
`--force` rebakes them from the native CT data. It never makes legacy artifacts
eligible inputs.

For a volume-scale local bake, stop before the in-memory window finalizer:

```bash
python3 -m backend.cubical full-acus \
  --compute gpu \
  --voxel-origin 32 32 32 \
  --shape 241 241 14 \
  --shard-shape 8 8 3 \
  --local-only \
  --output /mnt/t5/pareidolia/pherc0358-raw-acus-local-v1
```

The clean regular grid excludes the unsupported outer 32 voxels rather than
adding the old analysis grid's irregular terminal centers. `--limit-shards N`
performs a bounded resumable batch. `--only-shard x0000-y0000-z0000` targets a
specific planned shard and is repeatable. These modes write `local-summary.json`
and never enter global selection.

The clean full slab contains 813,134 cells, 4,805 evidence shards at
8 × 8 × 3, and 14,884 canonical extraction tiles (61 × 61 × 4). At the
current measured single-process rates, the rich local bake remains roughly an
11-hour job on this machine; resumable bounded batches make that an operational
estimate rather than one uninterrupted requirement. Extraction tiles are
baked once even when several evidence shards require them.

## Physical priors and ambiguity

All normal and fiber vectors are axial. Their signs are gauge choices and are
never treated as evidence.

The local configuration search uses physical units from the scan metadata. Its
default 35 µm minimum layer spacing is a broad duplicate-mode exclusion, not a
claim about this scroll. An 80–400 µm sheet-thickness interval supplies only a
soft reward when successive modes also have approximately orthogonal fibers.
Parallel successive modes remain legal: folds in contact, a missing or weak
ply, and two neighboring sheets can all produce them. An empty configuration is
always retained. These settings are explicit and can be swept without changing
the geometry or artifact schema.

At 9.362 µm resolution, a 32-voxel owned cell is about 300 µm wide and a
64-voxel Acus context is about 599 µm wide. The physical minimum spacing, not a
hardcoded mode count, bounds how many distinct layers can occupy one cell.

## Measured pilot

The final fresh 8 × 8 × 6 pilot used eight independent 4 × 4 × 3 evidence
shards and 18 canonical extraction tiles on an 8 GB GTX 1080. It completed in
30.5 seconds and produced:

- 35,191 candidates in the complete canonical tile catalogs and 24,552
  shard-context candidate occurrences (the latter intentionally counts
  overlaps between evidence shards);
- 764 normal hypotheses over 384 valid cells;
- 3,072 competing stratigraphies containing 7,091 non-crossing layer
  alternatives;
- 1,247 selected patches after five configuration sweeps;
- 1,284 retained topology-safe joins and 177 components; and
- a largest current fragment of 71 patches.

A hot 160³ canonical GPU tile spends about 0.255 seconds on the Hessian field
and 0.0065 seconds on resident candidate/refinement work. The complete pilot
occupies 6.5 MB, of which 3.39 MB is compressed full evidence. Small windows
pay disproportionate fixed-tile overhead; scaling the measured evidence,
stratigraphy, shard-gather, and canonical-catalog artifacts gives roughly
11 GB for a full 241 × 241 × 14 local bake before previews and meshes. Store
full-volume bakes under `/mnt/t5`, not the repository work tree.

## Gap diagnosis and full-mode continuation

An unresolved trace is not automatically missing papyrus evidence. The
diagnostic path separates five cases without weakening the assembler:

```bash
python3 -m backend.cubical gap-census \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-v1
```

The 16 × 16 × 14 full-depth slab's original 195-cell leading component had 248
unresolved interior traces: 103 were same-component cell-collision vetoes, 17
were crossing-topology vetoes, 84 were ordered face-assignment decisions, ten
could be recovered by an already retained configuration, and only 34 lacked a
compatible trace in every retained top-M configuration.

Older completed bakes can recover the new first-class mode artifact from their
own immutable needle and evidence shards; this does not rerun Hessian
extraction or read a legacy cache:

```bash
python3 -m backend.cubical mode-bank \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-v1 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-mode-bank-v2
```

That slab contains 88,084 fitted modes across 3,584 cells (24.58 per cell) and
the 20-shard backfill takes 42.3 seconds. The full bank contains an accepted
shared-face continuation for 27 of the 34 apparent mode gaps, grouped into 21
distinct target-cell modes. The dominant failure was therefore top-M
stratigraphy pruning, not absent raw Acus signal.

Continuation search constructs a complete same-normal, non-crossing physical
stratigraphy containing the required mode. It replaces the target cell as a
unit, then performs full global assembly for every trial. A candidate is
recommended only if it closes a recorded gap without deleting layers,
shrinking the source component, losing retained joins, increasing unresolved
traces, or adding collision/topology debt:

```bash
python3 -m backend.cubical mode-continuation-search \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-v1 \
  --mode-bank /mnt/t5/pareidolia/raw-acus-16x16x14-mode-bank-v2 \
  --maximum-modes-per-gap 1 \
  --maximum-configurations-per-candidate 3 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-v1/mode-continuation-search-config3-final-v1.json

python3 -m backend.cubical apply-mode-continuations \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-v1 \
  --mode-bank /mnt/t5/pareidolia/raw-acus-16x16x14-mode-bank-v2 \
  --search /mnt/t5/pareidolia/raw-acus-16x16x14-v1/mode-continuation-search-config3-final-v1.json \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1
```

The bounded three-configuration search evaluates 55 full-slab assemblies. It
closes at least one seam in 42 trials, with 13 passing every conservative gate.
Choosing the best independently safe configuration in each of six distinct
target cells verifies all eight intended joins together, adds three supported
patches, raises retained joins from 12,380 to 12,395, reduces unresolved
interior traces from 19,546 to 19,528, and grows the leading component from 195
to 201 cells. Collision deferrals fall by five and crossing-topology deferrals
by two. This is contextual recovery from independently fitted local evidence,
not extrapolated geometry.

`selection-variant` remains available for explicit prior sweeps, but the
unmatched-trace prior defaults to zero. A measured 0.1 global penalty reduced
unmatched traces partly by deleting 198 selected layers and emptying nine
cells, so it was rejected as the recovery mechanism.

## Native-CT fragment flattening

Connectivity statistics can improve while a component drifts between physical
plies. `flatten-components` therefore provides a qualitative stop/go artifact
before another growth pass:

```bash
python3 -m backend.cubical flatten-components \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1 \
  --component-ranks 1 2 3 7 \
  --depth-min -12 --depth-max 12 --depth-step 1 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-flattened-sanity-v1
```

The exporter triangulates the exact welded cubical polygons without smoothing
their positions. It cuts only the UV topology at nonmanifold or contradictory
edges, redundant cycles, and tangent-chart boundaries whose signed normal
would exceed the declared 40-degree cone. Stronger accepted joins are retained
when a redundant cycle needs a chart seam. Each remaining chart is projected
in physical voxel units and packed into an atlas; the reconstruction itself is
not modified.

Native CT is sampled from -12 through +12 voxels along the original patch
normal. One depth offset applies to the complete component—there is no per-cell
or per-tile best-depth alignment that could conceal layer hopping. Every rank
writes the raw compressed stack, a center image, a cyan cell-boundary overlay,
the complete fixed-depth montage, and orthogonal depth crossings. Red pixels
mark nonadjacent UV overlap and remain a failure diagnostic rather than being
blended away. The manifest records every seam, triangle flip, projection
distortion, overlap fraction, scanner-space source bound, and source identity.

The first four-component checkpoint is deliberately mixed rather than a
success-only gallery. Ranks 1, 2, 3, and 7 contain 201, 194, 188, and 165
selected patches. Their median linear atlas distortion stays within 2%, and
their nonadjacent UV-overlap fractions are 0.83%, 1.85%, 0.10%, and 0.88%.
Ranks 2 and 3 show broad native-fiber continuity across many cells. The initial
rank-7 image raised a mixed-ply concern, but the local tests below do not
separate it from the other large components; it remains unverified rather than
being labeled a failure. Rank 1 exposed nine orientation-cycle conflicts,
which identified a missing assembly invariant.

## Orientability and native-CT join refinement

Every accepted face join now contributes a binary polygon-orientation parity
constraint. The hierarchical assembler carries those constraints through its
disjoint set and defers a join as `orientation-parity-cycle` when it would
close a contradictory loop. On the complete slab this rejects 11 redundant
cycle edges, removes all orientation conflicts from the twelve largest
components, and does not remove a patch or split a component.

The original rank-1 `nonmanifoldEdges: 1` report was a diagnostic error rather
than a branching surface. Several crossing identities had snapped to exactly
the same cube corner, creating zero-length polygon edges with three apparent
incidences. These are now excluded from manifold edge counts and recorded as
`coincidentZeroLengthEdges`; rank 1 has five such degeneracies and zero
physical nonmanifold edges.

Native CT provides a second, independent refinement stage:

```bash
python3 -m backend.cubical refine-join-continuity \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-join-continuity-final-v2
```

For every retained join, seven shared-trace locations are sampled at fixed
normal depths. Points 1.5 voxels inside the seam are compared across cells;
points another three voxels inside each patch provide equal-span controls.
Mismatch ratios are calibrated independently for each face axis with a robust
median/MAD scale. Only ratios in the four-standard-deviation outer tail and at
least 1.5 times their within-patch controls change connectivity. Tiled source
reads keep this stage schedulable for larger volumes.

The pass scores all 12,384 parity-safe joins in 25 seconds. It rejects 14,
retains 12,370, and splits eight small components. Removing a bad redundant
edge does not split a component when a clean alternate route remains, so none
of the twenty largest component sizes changes. The complete table records
surface-texture angles and best normal-profile shifts as diagnostics, but they
do not gate connectivity: raw texture angles correlate only 0.07--0.09 with
independent Acus fiber disagreement, and 3,333 joins prefer a profile shift of
at least four voxels without forming a robust outlier tail. Promoting either
would therefore be slice-specific tuning.

The refinement table can be applied without altering selected patch evidence:

```bash
python3 -m backend.cubical flatten-components \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1 \
  --join-refinement /mnt/t5/pareidolia/raw-acus-16x16x14-join-continuity-final-v2 \
  --component-ranks 1 2 3 7 \
  --depth-min -12 --depth-max 12 --depth-step 1 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-flattened-refined-final-v2
```

For rank 1, orientability plus corrected corner accounting reduces conflict
seams from 10 to zero, p90 projection distortion from 1.174 to 1.106, and UV
overlap from 0.83% to 0.31%, while preserving all 201 patches. This is a real
topological improvement. The unchanged large-component sizes are also an
important result: local CT mismatch safely removes obvious discontinuities,
but does not by itself establish ply identity.

## Full-mode stratigraphic continuity

A locally plausible face match can still jump to a nearby ply. The next stage
therefore uses the complete pre-pruning Acus mode bank as context rather than
loosening or retuning the geometric face matcher:

```bash
python3 -m backend.cubical refine-stratigraphic-continuity \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1 \
  --mode-bank /mnt/t5/pareidolia/raw-acus-16x16x14-mode-bank-v2 \
  --join-refinement /mnt/t5/pareidolia/raw-acus-16x16x14-join-continuity-final-v2 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-stratigraphic-continuity-final-v1
```

Every selected patch is first matched to its exact same-normal-family bank
mode. The match is a provenance check, not another fit. On this slab all
11,622 patches anchor successfully: height residual is exactly zero, maximum
normal residual is 0.0231 degrees, and maximum unsigned-fiber residual is
0.000006 degrees. Removing the anchor itself leaves 135,177 contextual modes,
with a median of twelve per patch.

Each contextual mode contributes a confidence-weighted kernel at its signed
depth relative to the anchor plane. A second channel records whether its
unsigned fiber is parallel or transverse to the anchor fiber. Normal signs are
still not observations: the two axial normals are put into one pairwise gauge,
which determines whether the depth sequence must reverse. Flipping both gauges
leaves the comparison unchanged.

The local comparison is repeated over graph-connected three-hop neighborhoods.
For a tested face, the two neighborhoods are constrained to their respective
spatial half-spaces and the tested join is removed, so evidence does not leak
directly across the seam under test. At least three valid patches are required
on each side. Density overlap uses the common physically observable depth
interval; missing terminal modes outside that interval are not treated as
disagreements.

Scores are calibrated independently for the three face axes. A join is removed
only when both its single-cell signature and its multi-cell signature exceed
the four-robust-standard-deviation tail. Of 12,370 joins surviving the native
CT pass, 12,287 have usable local signatures, 7,953 have sufficient multi-cell
context, and 7,906 enter joint calibration. Twenty-four joins fail both gates.
They split twelve components while retaining 12,346 joins. The largest
component remains 201 patches. The second component loses one independently
inconsistent three-patch appendage, changing from 194 to 191; the other large
checkpoint components retain their patch counts. The complete fingerprint and
join pass takes 25.2 seconds on the pilot slab.

The stage is deliberately selective rather than a visual cleanup heuristic.
The still-questionable rank-7 geometry is unchanged, so this result does not
claim that every remaining component follows one physical ply. It says that
the 24 removed joins have both an anomalous local layer neighborhood and an
anomalous larger surrounding neighborhood under the same full-Acus evidence.
The native-CT stage remains the independent appearance check.

The refined connectivity composes with the existing visual checkpoint:

```bash
python3 -m backend.cubical flatten-components \
  --root /mnt/t5/pareidolia/raw-acus-16x16x14-mode-continuation-config3-final-v1 \
  --join-refinement /mnt/t5/pareidolia/raw-acus-16x16x14-join-continuity-final-v2 \
  --stratigraphic-refinement /mnt/t5/pareidolia/raw-acus-16x16x14-stratigraphic-continuity-final-v1 \
  --component-ranks 1 2 3 7 \
  --depth-min -12 --depth-max 12 --depth-step 1 \
  --output /mnt/t5/pareidolia/raw-acus-16x16x14-flattened-stratigraphic-final-v1
```

The fingerprint artifact is a fixed-width structure of arrays and the
neighborhood radius is bounded. Construction is linear in selected patches,
retained local modes, and joins, so this contributes directly to the scalable
pipeline. A volume scheduler can shard fingerprints by owned cells and score
faces after loading only the bounded graph halo; no native-CT or Acus rerun is
required.

The raw and local-inference stages are already independently sharded. The
selected-patch assembler is hierarchical. Configuration selection currently
runs one window at a time; full-slab operation should schedule overlapping
macro windows and reconcile their boundary bands rather than materializing
every alternative plane as a Python object in one process. The artifact
contracts are designed for that scheduler, and no evidence-stage rewrite is
required.

## Structural saturation and dual-axis sheet packets

The selected surfaces now have a calibrated saturation audit. It uses only the
canonical, source-owned, finite-length Acus needles and weights each by its
score times the square root of axial coverage times support. Normal and fiber
directions remain axial. The raw intensity admission gate is not reused as a
voxel-occupancy label: 99.85% of this slab lies above it, so it cannot separate
papyrus from the broad attenuation field.

```bash
python3 -m backend.cubical audit-sheet-saturation \
  --root /path/to/selected-block \
  --output /path/to/saturation-audit
```

For each needle, the audit records distance to the selected plane, transported
unsigned-fiber residual, a two-dimensional standardized joint residual, and
assignment share among competing layers. The primary contour is 2.5 in joint
depth/fiber residual and the confident share is 0.8. The complete per-needle
classification is persisted, including failure decomposition and score-decile
calibration; summary percentages are not the only surviving result.

Full-bank saturation reselection enumerates every physically ordered stack in
every cell before retaining a bounded, diverse candidate set. One normal
family, minimum spacing, and within-cell non-crossing remain hard constraints.
The best-coverage physical stack is always retained alongside the current and
empty configurations. A confidence-normalized Gaussian mixture scores Acus
evidence without rewarding duplicate modes, and the ordinary shared-face ICM
then selects one stack per cell:

```bash
python3 -m backend.cubical saturation-reselect \
  --root /path/to/selected-block \
  --mode-bank /path/to/full-mode-bank \
  --pairwise-scale 0.2 \
  --output /path/to/saturation-reselection
```

The 16 × 16 × 14 slab contains 3,584 cells. Reselection enumerates 1,219,069
physical paths and retains 40,271 configurations with 87,813 layer
alternatives. Its structural-evidence ceiling ladder is:

| Constraint | supported evidence mass |
| --- | ---: |
| current contextual-growth stack | 60.09% |
| local unary winner | 62.63% |
| selected global stack | 64.79% |
| complete physical per-cell oracle | 72.70% |
| any modes from the best single normal family | 75.85% |
| any fitted mode | 79.09% |

The selected result contains 11,748 patches, 13,100 retained joins, 1,127
components, and 18,883 unresolved interior traces. Its independent audit finds
58.55% confidently assigned mass, 6.24% ambiguous mass, and 35.21%
unexplained mass. A 0.25 explicit utilization reward raises direct support by
0.62 percentage points but produces 36 more components and 233 more unresolved
traces, so it is not the default. This is an example of the audit preventing a
nominal coverage gain from silently replacing cohesion.

The candidate artifact is immutable and reusable:

```bash
python3 -m backend.cubical select-saturation-candidates \
  --candidates /path/to/saturation-reselection \
  --pairwise-scale 0.2 --no-visuals \
  --output /path/to/selection-variant
```

Geometry-level trace memoization reduces global selection from 136.3 to 78.9
seconds with exactly identical cells, energies, support, and topology. Reusing
the candidate bank and verified baseline statistics reduces a no-preview
selection iteration to 99.9 seconds, including selected-graph assembly.

The failure decomposition provides the next physical distinction. Only 30.8%
of unexplained mass lacks a selected plane within the 6.25-voxel depth gate;
69.2% is near selected geometry but incompatible with its one stored fiber
axis. Testing the actual transverse in-plane cross-product axis as a diagnostic
explains another 5.65% of total evidence. Direct or transverse support is
therefore 70.44% overall and rises monotonically with needle score to 90.83% in
the strongest
decile. This diagnostic does not relabel or join anything.

`dual-axis-packets` materializes that hypothesis as a separate sheet-level
connectivity graph. It never modifies the strict single-ply graph. Existing
strict joins are fixed, and only new quarter-turn candidates are considered.
They must satisfy explicit absolute normal and fiber-frame residual gates, then
pass the unchanged endpoint likelihood, ordered trace alignment, same-cell
collision, crossing-topology, and orientability selector:

```bash
python3 -m backend.cubical dual-axis-packets \
  --root /path/to/saturation-reselection \
  --maximum-normal-angle 15 \
  --maximum-fiber-residual 15 \
  --output /path/to/dual-axis-packets
```

The 15° run discovers 2,157 quarter-turn possibilities, admits 1,149 through
the absolute gates, and retains 187 after global topology. Retained joins have
median/p90 normal residuals of 8.85°/13.86°, fiber-frame residuals of
3.98°/10.41°, and endpoint residuals of 0.90/2.36 standard deviations. They
reduce components from 1,127 to 977, grow the largest fragment from 191 to 278
cells, and remove 374 unresolved traces. The graph NPZ stores retained joins,
quarter-turn provenance, residuals, and patch-to-component membership.

The absolute-gate sensitivity is smooth rather than singular:

| normal/fiber cap | retained quarter-turn joins | components | largest | unresolved traces |
| ---: | ---: | ---: | ---: | ---: |
| 10° | 99 | 1,037 | 270 | 18,685 |
| 15° | 187 | 977 | 278 | 18,509 |
| 20° | 243 | 935 | 278 | 18,397 |

The largest fragment saturates by 15°, while 20° mainly merges smaller
components with weaker angular agreement. Fifteen degrees is therefore the
current conservative default, not a claim about a scroll-specific fiber angle.
The packet graph remains a downstream interpretation layer; block merging can
exchange its retained boundary traces without changing Acus extraction,
physical stack selection, or the strict ply graph.

## Boundary-band handoff and split/recompose audit

A completed selected block can now emit the bounded state required by a
neighbor. The physical candidate bank is optional for a connectivity-only
probe but should be included when the next stage will jointly reselect the
meeting bands:

```bash
python3 -m backend.cubical export-boundary-band \
  --root /path/to/selected-block \
  --packet-root /path/to/dual-axis-packets \
  --candidate-root /path/to/saturation-reselection \
  --depth-cells 2 \
  --output /path/to/boundary-band

python3 -m backend.cubical merge-boundary-bands \
  --first /path/to/first-boundary-band \
  --second /path/to/second-boundary-band \
  --output /path/to/boundary-merge

python3 -m backend.cubical reselect-boundary-bands \
  --first /path/to/first-boundary-band \
  --second /path/to/second-boundary-band \
  --output /path/to/boundary-reselection
```

The exporter does not copy the block interior. It writes the shell's selected
patches, every shell-cell physical alternative, exterior traces in world
coordinates, packet ownership, full occupied-cell certificates for components
that reach the inner cut, orientation parity among immutable anchor patches,
and existing welded edge/vertex identities including their deep owners. Both
composition paths handle independently local patch IDs, validate exact
world-grid adjacency, and enforce same-cell collision, crossing-feature,
ordered-trace, unsigned-direction, and orientability invariants without loading
native CT.

`merge-boundary-bands` leaves both child selections unchanged and emits a
conservative component forest. `reselect-boundary-bands` instead builds a
`2d + 2`-cell-thick slab: `d` mutable layers from each child and one immutable
anchor layer at each outer edge. Warm-started conditional ICM reconsiders only
the mutable cells. The topology solve admits ordinary strict joins first, then
fixes that graph while considering separately gated quarter-turn packet joins.
The private interiors are represented only by compact occupancy, crossing, and
orientation certificates.

On the full 16 x 16 x 14 result, the two-cell shell contains 2,144 of 3,584
cells, 7,006 selected patches, 24,131 physical configurations with 52,556
layers, 2,945 cut-anchor patches, 11,252 component-cell certificates, and 4,434
crossing groups. The complete compressed handoff is 2.8 MB, including a 124 KB
frozen-topology certificate; 1,440 interior cells remain private. The shell
fraction is high only because this pilot is small. For an `N^3` block at fixed
depth, clipped geometry and physical candidates become `O(N^2)`. Complete
component-occupancy certificates can still be `O(N^3)` in the worst case;
they are compact integer state rather than CT or patch geometry, but bitmap or
run-length compression is a remaining volume-scale optimization.

The rebase and comparison utilities make the complete contract reproducible on
one known block without presenting that as independent raw-CT evidence. When
the source contains a saturation candidate bank and selection artifact,
`extract-selected-subblock` partitions and rebases those physical candidates
as well as its selected polygons:

```bash
python3 -m backend.cubical extract-selected-subblock \
  --root /path/to/selected-block --start 0 0 0 --stop 8 16 14 \
  --output /tmp/left-selected

# Repeat for X=8..16, rebuild dual-axis-packets and export-boundary-band for
# both children, then jointly reselect them as above.

python3 -m backend.cubical audit-boundary-reselection \
  --full-packet-root /path/to/full-dual-axis-packets \
  --reselection-root /tmp/recomposed-boundary \
  --output /tmp/boundary-reselection-audit.json
```

At the X=8 split, the joint solve reads 896 mutable cells, 665 immutable anchor
patches, and no private interior geometry. It finishes in 26.5 seconds and
exactly recovers all 13,287 retained joins and all 977 components of the
unsplit packet graph. This closes the conservative selected-only merge's
eight-component difference and is now the deterministic regression target.

The stronger test performs two genuinely independent `full-acus` runs over
the adjacent 8 x 16 x 14 CT halves. They share only an explicitly hashed
source-level calibration; both rerun all 60 GPU extraction tiles, ten evidence
shards, mode fitting, physical saturation selection, and packet assembly. The
selected-only merger gives 987 components. Joint boundary reselection changes
37 of 896 mutable cells and gives 978 components, one above the 977-component
full-context consistency reference. Its changes improve exact full-context
configuration agreement from 293 to 308 cells and layer-count agreement from
866 to 884. Of the 37 changed cells, 15 move to the exact reference
configuration and none move away. This reference is not asserted to be
physical truth; the result demonstrates that the bounded solve repairs real
independent-boundary effects in a direction consistent with extra context.

That comparison is reproducible as an artifact rather than an ad hoc notebook
calculation:

```bash
python3 -m backend.cubical audit-independent-boundary \
  --full-packet-root /path/to/full-context-packets \
  --selected-merge-root /path/to/selected-only-merge \
  --reselection-root /path/to/joint-boundary-reselection \
  --output /path/to/independent-boundary-audit.json
```

## Tests

```bash
python3 -m unittest \
  backend.test_rectify \
  backend.test_cubical \
  backend.test_raw_acus_pipeline -v
```

The tests cover disjoint shard ownership, source-anchored extraction tiling,
raw source and shared-calibration identities, content hash checks, unsigned
normal recovery, physical two-ply/empty alternatives, exact mode-bank
persistence, gap classification, configuration-aware face selection, cubical
clipping, trace alignment, frozen topology round trips, conditional boundary
reselection, topology vetoes, and hierarchical assembly.
