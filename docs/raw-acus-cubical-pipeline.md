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

That orientation channel is the second harmonic `cos(2 theta)`, expressed in
the anchor patch's local fiber frame. A classified quarter-turn continuation
therefore negates the moment when it transports the neighboring frame; treating
the two arrays as if they shared a frame would systematically penalize valid
orthogonal-ply edges. The same binary gauge is propagated through every
multi-hop neighborhood. The global topology selector independently forbids an
odd quarter-turn cycle as `fiber-frame-parity-cycle`, just as polygon parity
forbids a non-orientable surface cycle.

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

The current command also accepts composed block evidence through
`--sheet-evidence` and a complete restitch candidate catalog through
`--candidate-restitch`. Calibration still uses only the retained graph, but
the frozen robust test is then applied to every geometric alternative. The
output is no longer only a modifier table: it copies the immutable selected
patch artifact and writes the exact filtered `surface-graph-v1`, with the input
graph hashes in its identity. A subsequent round can therefore use that output
directly and stop when `rejectedJoins` reaches zero. Fixed-point rounds normally
use the latest gate; taking the intersection of several gates is available as
an explicitly more conservative restitch policy, not the default iteration.

On the 4,784-sheetlet owned 12 x 12 x 10 checkpoint, complete-candidate gating
followed by global restitching produced 5,016 joins and a 166-cell leading
component. The next materialized rescore removed two retained bridges with
jointly extreme local and neighborhood disagreement. A following round was a
fixed point: 5,014 joins, 929 components, 56.312% retained interior traces,
leading sizes 166, 149, and 143, and zero rejected retained joins while 106 of
11,953 alternatives remain inadmissible. All 929 reconstructed component meshes
have zero physical nonmanifold edges and zero orientation conflicts; the graph
also has no same-cell returns, foldbacks, or quarter-turn parity contradictions.

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

python3 -m backend.cubical reselect-boundary-cluster \
  --boundary /path/to/x0-y0-boundary \
  --boundary /path/to/x1-y0-boundary \
  --boundary /path/to/x0-y1-boundary \
  --boundary /path/to/x1-y1-boundary \
  --output /path/to/cluster-reselection

python3 -m backend.cubical materialize-boundary-cluster \
  --cluster /path/to/cluster-reselection \
  --boundary /path/to/x0-y0-boundary \
  --boundary /path/to/x1-y0-boundary \
  --boundary /path/to/x0-y1-boundary \
  --boundary /path/to/x1-y1-boundary \
  --output /path/to/materialized-cluster

python3 -m backend.cubical flatten-components \
  --root /path/to/materialized-cluster \
  --component-ranks 1 2 3 4 5 6 7 8 \
  --output /path/to/materialized-cluster-flattened
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

The same export also includes 26 frozen-region certificates: every nonempty
combination in which each axis contributes no face, its low face, or its high
face. This is the bounded state needed to remove two or three mutually
orthogonal bands simultaneously at a child corner. The compressed addition is
roughly 158--194 KB per 8 x 8 x 14 pilot child.

`merge-boundary-bands` leaves both child selections unchanged and emits a
conservative component forest. `reselect-boundary-bands` instead builds a
`2d + 2`-cell-thick slab: `d` mutable layers from each child and one immutable
anchor layer at each outer edge. Warm-started conditional ICM reconsiders only
the mutable cells. The topology solve admits ordinary strict joins first, then
fixes that graph while considering separately gated quarter-turn packet joins.
The private interiors are represented only by compact occupancy, crossing, and
orientation certificates.

`reselect-boundary-cluster` accepts the complete Cartesian set of two to eight
equal child blocks. It takes the union of their participating face bands,
adds one immutable cut shell, and runs one sparse configuration and topology
solve. Every corner or edge cell is represented once, so no later union of
incompatible pairwise decisions is needed.

The materializer does not perform another configuration or topology search.
It replaces each child's internal mutable bands with the cluster result and
copies only complementary immutable child geometry and joins. It writes a
complete retained graph and verifies that direct reconstruction has the
cluster-certified component count. `flatten-components` detects that graph
automatically instead of assembling a new strict graph from the patch table.
Each flattened component includes `center-raw.png`, an overlap-diagnostic
`center.png`, a cell-boundary overlay, depth products, and clean and diagnostic
ranked overview PNGs.

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

The corresponding 2 x 2 experiment uses four independently inferred
8 x 8 x 14 children. Pairwise X/Y solves produce 23 contradictory choices in
224 overlapping corner cells and different component partitions for all four
crossing seam pairs. The joint cluster solve covers 1,568 mutable cells and
616 immutable shell cells, changes 133 configurations, retains 6,401 joins,
and completes in 48.9 seconds. All 23 conflicts resolve to one of the locally
supported alternatives.

Against the unsplit 16 x 16 x 14 consistency reference, layer-count agreement
rises from 1,456 to 1,527 cells. Thirty changed cells become exact reference
configurations and three cease to be exact; in the 224-cell corner subset the
balance is six toward and zero away. The 2,426 common mappable joins give
98.38% Jaccard, and both graphs have 977 components. Their mapped connectivity
is not identical: co-component precision is 90.74%, recall is 90.88%, and
Jaccard is 83.17%. These are explicit remaining block-level targets rather
than evidence for declaring the fragments solved.

Complete materialization makes the fragment growth directly measurable. The
four independent children contain 1,185 components, with a largest fragment of
161 occupied cells and only one fragment at or above 128 cells. The joint graph
contains 977 components, a 271-cell largest fragment, and 17 fragments at or
above 128 cells. It has 153 multi-child components: 124 span two children, 18
span three, and 11 span all four. The unsplit consistency reference has a
278-cell largest fragment and the same count of 17 at or above 128 cells. This
close size distribution does not imply identical membership, but confirms that
the bounded solve produces genuinely larger physical fragments rather than only
changing component labels.

```bash
python3 -m backend.cubical audit-multiseam \
  --reselection /path/to/x-seam-0 \
  --reselection /path/to/x-seam-1 \
  --reselection /path/to/y-seam-0 \
  --reselection /path/to/y-seam-1 \
  --cluster-root /path/to/cluster-reselection \
  --output /path/to/multiseam-audit.json

python3 -m backend.cubical audit-boundary-cluster-reference \
  --full-packet-root /path/to/full-context-packets \
  --full-selected-root /path/to/full-context-selection \
  --cluster-root /path/to/cluster-reselection \
  --output /path/to/cluster-reference-audit.json
```

## Block-global needle inference

Cells are an ownership, indexing, and eventual meshing device; they do not
have to be independent inference domains.  The block needle-field stage unions
the source-anchored extraction tiles from one or more raw-Acus roots,
deduplicates overlap by canonical tile identity and content hash, crops needle
centers to an explicit world-space cuboid, and builds one spatial graph over
every retained needle.

Each unsigned fiber needle admits a family of perpendicular page normals, so a
single normal per cell and a single global orientation cluster are both too
restrictive.  The block solve instead keeps several local unsigned normal
hypotheses per needle.  A GPU mean-field solve couples their probabilities
using local evidence, axial normal agreement, and a curvature-aware Hermite
chord.  After aligning the unsigned endpoint normals, a smooth bend has
equal-and-opposite signed tangent-plane offsets.  Their symmetric part is a
layer-shift residual, while their antisymmetric part is checked against the sag
implied by the endpoint normal angle.  This replaces the incorrect flat-patch
assumption that each endpoint must lie in the other endpoint's tangent plane.
A parallel layer jump is penalized while a resolved bend is not penalized
merely for being curved.

The resulting high-affinity connected components are conservative diagnostic
normal carriers, not final papyrus sheet identities.  Large normal change
across a carrier is permitted when it is accumulated through locally
compatible edges, which is necessary for real hairpins.

```bash
python3 -m backend.cubical solve-block-needle-field \
  --raw-root /path/to/raw-acus-child-0 \
  --raw-root /path/to/raw-acus-child-1 \
  --world-start 3456 2720 7264 \
  --world-stop 3840 3104 7584 \
  --compute gpu \
  --output /path/to/block-needle-field
```

All solver values have declared dataclass defaults and are recorded in the
artifact identity.  A `BlockNeedleFieldSettings` keyword object may be passed
with `--settings-json`; explicit command-line options override matching JSON
values.  On the current 12 x 12 x 10, 32-source-voxel-cell core, canonical tile
ownership gives 43,950 unique needles and 1,048,874 directed neighborhood
edges.  The curvature-aware GTX 1080 reference solve takes 6.77 seconds end to
end and writes a 15.16 MB immutable artifact.  It forms 42 conservative normal
carriers with at least 128 needles, 12 with at least 256, and five with at least
512; the largest has 974 of 43,950 needles, so the added curvature does not
produce one transitive giant component.  These measurements establish
feasibility, not a claim that its normal carriers are finished sheets.

The block needle-topology stage then distinguishes fiber-coherent physical
plies.  It builds a normal-depth density and `cos(2 fiber-angle)` stack
fingerprint around every needle directly from the dense graph.  Robust caps are
calibrated from the strongest real edges, while the minimum layer spacing,
curvature radius, fiber scale, and depth kernel come from the immutable
raw-Acus physical settings.  Conservative seed components grow to a fixed point
only through bridge packets with independent endpoints and nonzero spatial
span.  A singleton may use one endpoint on its own side, but still requires two
independent target needles.  Normal-separated branches are not globally
repulsive—a real hairpin can return beside itself—so the physical invariant is
zero selected direct layer-shift edges rather than global spatial injectivity.

```bash
python3 -m backend.cubical solve-block-needle-topology \
  --field /path/to/block-needle-field \
  --output /path/to/block-needle-topology
```

On the same core this stage evaluates 612,222 unique needle pairs and selects
91,618 ply edges in 1.95 seconds including compressed output and two projection
previews.  It selects zero direct edges at or beyond the physical 3.74-voxel
layer-spacing floor.  The largest fiber-coherent carrier has 245 needles;
50.53% of total needle-quality mass belongs to carriers of at least eight
needles and 78.84% belongs to a carrier of at least two.  The leading carrier's
local selected-edge normal angle is 7.38° median / 17.24° p90 and its global
fiber projector has 95.95% of its mass on one axis.  Orthogonal plies remain
separate by design and are the next level of papyrus-sheet association.

The surface stage converts those graph carriers into explicit ordered geometry
without treating a graph component as a finished mesh.  Unsigned normal and
fiber axes receive deterministic transport signs solely to define local
coordinates.  A branch-free predecessor/successor solve first exposes ordered
fiber traces and monotone cross-trace strip evidence.  Meshing then freezes the
topology carrier identities and re-evaluates every within-radius pair inside
each carrier through the same curvature, layer-shift, fiber, and stack-
fingerprint gates.  This removes the fixed maximum-neighbor count as a meshing
artifact without making any new cross-carrier merge.

Each accepted local chord contributes a signed `(fiber, cross-fiber)` increment.
A matrix-free, Jacobi-preconditioned conjugate-gradient solve per carrier
integrates those weighted increments into an intrinsic chart, so a 3D hairpin
may unroll into an ordinary 2D strip.  The chart solve stores only graph edges
and vectors rather than a dense node-by-node Laplacian, keeping its memory
linear in the carrier graph.
Delaunay faces are mapped back to the exact 3D needle centers and retained only
when all three edges independently pass the physical continuation gates.  Mesh
components are connected through shared edges, not merely a shared vertex;
this prevents a bow-tie contact from masquerading as one surface.

```bash
python3 -m backend.cubical solve-block-needle-surfaces \
  --topology /path/to/block-needle-topology \
  --output /path/to/block-needle-surfaces

python3 -m backend.cubical flatten-block-needle-surfaces \
  --surfaces /path/to/block-needle-surfaces \
  --grouping topology-carrier \
  --pixel-step 0.5 --maximum-pixels 768 \
  --depth-min -12 --depth-max 12 --depth-step 1 \
  --output /path/to/block-needle-flattening

python3 -m backend.cubical associate-block-needle-surfaces \
  --surfaces /path/to/block-needle-surfaces \
  --output /path/to/block-needle-bundles
```

On the current core, complete within-carrier evaluation considers 325,665
pairs, keeps 119,824, and recovers 18,056 valid pairs omitted by the original
fixed-degree neighbor graph.  All 1,096 carriers of at least eight needles
converge, requiring 9 iterations at the median and 46 at worst.  Their
chord-integration residual is 0.44 voxels median / 1.60 p90 / 3.66 p99.  The
final physically gated atlas has 12,828 triangles, no edge with more than two
incident faces, and a leading edge-connected patch of 72 needles / 93
triangles.  The complete stage takes 6.89 seconds on the reference block.  The
original 245-needle graph carrier splits where a chart edge fails the physical
gates; that is an explicit remaining continuity question rather than a hidden
triangulation shortcut.

`flatten-block-needle-surfaces` rasterizes those intrinsic charts and samples
the immutable native CT at fixed normal offsets.  On the twelve leading
patches, eleven select the exact zero-offset needle plane as the strongest
texture slice at half-voxel raster spacing; their center-plane texture score is
typically 1.4--2.7 times the stronger of the two ±12-voxel endpoint scores.
Eleven have zero nonadjacent chart-overlap pixels and one has a single pixel.
This is a qualitative physical check, not independent ground truth, but it
shows that the recovered surfaces follow CT-dense papyrus rather than only a
visually plausible graph projection.

The flattening command can rank either individual edge-connected mesh islands
or every island already belonging to one frozen topology carrier.  Carrier
grouping does not fill a hole: it uses the common intrinsic chart to expose the
hole.  The largest current carrier has 217 meshed needles / 233 triangles in
20 islands, versus 72 / 93 for its largest island.  It has zero nonadjacent
raster overlaps, and eleven of the twelve leading carrier views prefer native
CT depth zero; one prefers +10 voxels and remains an explicit nearby-ply
ambiguity.

`associate-block-needle-surfaces` then builds a separate evidence artifact for
that ambiguity.  A cross-ply packet must have aligned normals, nearly
orthogonal unsigned fibers, physical layer separation, at least three
independent endpoints on each side, half a needle length of spatial span, a
consistent normal side, and low separation dispersion.  Its geometric gates
are dimensionless multiples of the immutable Acus needle length, depth kernel,
orthogonal-ply spread, layer-spacing floor, and plausible sheet-thickness
range; endpoint counts and side consistency are declared general settings.
None are fitted to this block.  The current solve retains 179 of 978
component-pair hypotheses.  They have 21 edges and 12.83 voxels of span at the
median, with 9.27-voxel median ply separation and only 0.52-voxel median
separation MAD.  Their median normal disagreement is 9.69°, and median error
from orthogonal fibers is 8.40°.

An evidence-only shadow bridge is emitted when one intact crossed-fiber island,
or two islands in one already-frozen crossed-fiber carrier, support two
disconnected islands of the same ply carrier.  Ninety such bridges remain
within one physical needle length in the current block: 80 have one intact
partner island and ten use the broader frozen partner carrier.  The resulting
largest support group has 41 needles / 39 triangles in five still-disconnected
islands.  They never add graph edges or mesh triangles, so this stage does not
yet decide whether an orthogonal neighbor is the other ply of the same papyrus
sheet or the closest ply of an adjacent sheet.  The complete association pass
takes 0.36 seconds and leaves that page-pairing decision available to a later
stack-order optimization.

## Dense isolated-slab seeds

The needle surface experiment exposes a representation limit as well as useful
fiber evidence: a missing Acus sample cannot be recovered by repeatedly
relaxing graph topology without risking an invented layer jump.  The
complementary isolated-slab stage therefore starts again from native CT and is
deliberately Acus-independent.  It detects the simplest trustworthy geometry
first: one material interval with clear air beyond both faces.

The stage block-averages CT onto a source-aligned two-voxel sampling lattice,
smooths by a declared native-voxel scale, and calibrates air/material classes
with an Otsu split over the owned block.  Every sufficiently strong
air-to-material interface is followed along its inward gradient.  A retained
profile must have a second material-to-air transition, physical thickness in
the declared 80--400 micrometre range, at least 50 micrometres of independently
clear air on both sides, material-bearing interior samples, and an opposing
exit gradient.  Its two subvoxel boundaries, midpoint, unsigned normal,
thickness, and all confidence terms are persisted.  The processing region has
a thickness-plus-clearance halo, but only midpoint-owned pairs are emitted, so
the stage composes over adjacent blocks.

```bash
python3 -m backend.cubical detect-isolated-slabs \
  --source /path/to/native-ct.npy \
  --metadata /path/to/native-ct.json \
  --world-start 3456 2720 7264 \
  --world-stop 3840 3104 7584 \
  --output /path/to/isolated-slabs
```

Low-confidence physical pairs remain in the immutable array.  The default 0.5
threshold only controls a descriptive component graph whose edges require
local coplanarity, aligned unsigned normals, and compatible thickness.  It
never fills a missing profile or alters either interface.  This lets later
growth lower a confidence threshold under independent sheet context without
rebaking CT.

On the current 384 x 384 x 320 source-voxel core, the CPU reference run takes
5.19 seconds.  It tests 1,494,785 candidate boundary samples, retains 175,073
opposing profiles before reciprocal/spatial de-duplication, and writes 62,661
unique midpoint-owned pairs.  Of those, 47,384 pass the conservative seed
threshold.  They form 280 components with at least 32 samples; the largest has
1,073.  Retained thickness is 15.81 voxels / 148.04 micrometres at the median
and 33.00 voxels / 308.90 micrometres at p90.  The artifact includes native CT
cross sections, orthographic component projections, and a binary PLY point
cloud, making the accepted and deliberately unresolved regions directly
inspectable.

The evidence-only comparison with the existing block needle field uses the
actual finite 16-voxel Acus segments rather than pretending their centers are
points of surface coverage:

```bash
python3 -m backend.cubical audit-isolated-slabs-with-acus \
  --slabs /path/to/isolated-slabs \
  --field /path/to/block-needle-field \
  --output /path/to/isolated-slab-acus-audit
```

Only 37.38% of conservative slab seeds lie within four voxels of any finite
needle segment; 65.36% lie within six and 84.58% within eight.  At the old
center-distance interpretation, 59.72% lie within half a needle length.  Among
the 280 substantial slab components, nominal four-voxel segment coverage is
37.16% at the median and seven components have zero coverage.  At the closest
finite segment the Acus fiber is 13.78 degrees out of the CT slab plane at the
median, while the inferred Acus page normal differs by 28.72 degrees.  A
nearest segment can belong to an adjacent ply in dense regions, so these angle
figures are diagnostics rather than ground truth.  The coherent uncovered
stretches nevertheless establish that sparse needle sampling cannot be the
sole carrier of surface continuity.  Acus remains valuable as directional
fiber evidence to attach after dense CT sheets are seeded.

### Paired-interface bank and contextual growth

The conservative detector is a seed generator, not the full clear-slab
representation.  `build-paired-surface-bank` repeats the same source-aligned
CT preparation but retains every profile with exactly two threshold crossings
and physically plausible papyrus thickness.  Air clearance, material
interior, and opposing-gradient tests become persisted unary evidence rather
than irreversible rejection gates.  Reciprocal boundary detections are
suppressed geometrically, while up to four genuinely distinct paired-interface
hypotheses may remain at one sampling-lattice key.  Existing conservative
samples are immutable anchors and always win duplicate suppression.

```bash
python3 -m backend.cubical build-paired-surface-bank \
  --slabs /path/to/isolated-slabs \
  --output /path/to/paired-surface-bank

python3 -m backend.cubical grow-paired-surfaces \
  --bank /path/to/paired-surface-bank \
  --output /path/to/paired-surface-growth
```

The growth graph compares the lower and upper CT interfaces separately after
unsigned-normal alignment.  A candidate edge must satisfy midpoint distance,
normal angle, midpoint height, both boundary-height residuals, and thickness
difference before it receives an affinity.  Thus a midpoint-plausible jump
whose physical faces do not continue is rejected.  Growth is a
maximum-bottleneck forest with one selected hypothesis per source-lattice key;
it never moves a boundary or invents a missing CT profile.

Clear seed fragmentation is handled before that final ownership solve.  A
two-source pass stores the two strongest seed-component explanations for each
candidate.  Foreign support may reach an immutable seed profile, but cannot
cross it.  Two seed components share an identity only when a spatially broad
set of candidates supports both and each component reaches multiple seed
samples of the other in both directions.  Consequently one plausible shear
edge cannot create a transitive union.  The emitted audit image shows the four
largest multi-seed assemblies in all three projections, coloring their
original seed patches separately over the selected contextual surface.

On the current core, bank construction takes about 8.0 seconds on CPU.  It reduces
353,450 owned physical profiles to 309,672 candidates at 268,523 spatial keys;
35,542 keys retain multiple hypotheses, with a maximum of four.  All 62,661
original isolated samples are recovered exactly (maximum normalized matching
cost 0.002795), and the compressed bank is 22 MB.

The contextual solve evaluates 1,922,114 candidate pairs, retains 1,244,658
strict two-boundary edges, and completes graph construction, reciprocal seed
association, growth, compressed output, and previews in about 4.8 seconds.  Of
309,672 bank candidates, 95,041 are selected: all 38,309 eligible immutable
seeds plus 56,732 contextual additions.  Fifty-nine reciprocally supported
seed-patch pairs reduce 937 seed components to 878 identities; no current
identity contains more than three seed patches.  The largest selected surface
has 1,859 profiles, versus 1,605 before association, without changing total
coverage.  Added profiles are not predominantly single chains: their median
seed-parent depth is four, median growth-eligible same-label graph degree is
six, and 92.97% have at least two such continuity neighbors.  These are
deterministic in-slab measurements, not proof of page identity.  Profiles in
dense material without a physical air--papyrus--air interval remain
deliberately unresolved
for a later boundary-evidence stage.

### Direct paired-profile reconstruction

The direct reconstruction path deliberately stops before the one-sided face
graph. It treats every retained air--papyrus--air profile as one immutable
physical observation, chooses at most one hypothesis at each source sampling
key, and reconnects those observations from their two measured boundaries.
The generic macro orientation tensor is an independent hard gate: a locally
coherent profile family that is transverse to the visible laminar structure
cannot become a sheet merely by supporting itself.

```bash
python3 -m backend.cubical build-direct-paired-profile-surfaces \
  --bank /path/to/paired-surface-bank \
  --growth /path/to/paired-surface-growth \
  --macro-orientation /path/to/macro-sheet-orientation \
  --output /path/to/direct-paired-profile-surfaces
```

Profile normals are axial. Before comparing the two physical faces, the
second normal is aligned to the first and its lower/upper boundaries are
swapped whenever that alignment changes sign. Thus an arbitrary vector-sign
flip cannot break a real bend or silently compare opposite faces. Short gaps
may be closed only when midpoint, both boundary heights, thickness, unsigned
normal, and macro orientation agree. A local tangent-column guard rejects a
transitive union that revisits a separated normal depth.

On the current 384 x 384 x 320 PHerc. 358 core, the CPU stage selects 144,898
profiles from 309,672 candidates, retains 1,280,635 within-fragment edges, and
produces 11,335 components in about 26 seconds. On an independently unrolled
PHerc. 1667 control, the same defaults cover 88.56% of the known surface. The
largest recovered component covers 18.08% of that surface with 87.39% matched
node purity and 100% closest-boundary-side consistency; the covered surface is
split across 49 components. These figures diagnose candidate recall,
fragmentation, and layer mixing independently and are not training labels.

The optional control audit reads an official TIFXYZ surface without importing
it into reconstruction:

```bash
python3 -m pip install -r requirements-truth.txt
python3 -m backend.cubical audit-tifxyz-surface-control \
  --tifxyz /path/to/known-surface.tifxyz \
  --source-metadata /path/to/source-crop.json \
  --mid-surfaces /path/to/direct-paired-profile-surfaces \
  --output /path/to/control-audit
```

The alternative `frontier-bundles` component solver is persisted as an
explicit experiment, not a production default. Its first formulation raised
fragmentation on the independent control, so ordinary sign-correct connected
geometry remains the selected construction until a frontier objective passes
that control.

### Signed one-sided boundary patches

The paired bank cannot represent a clear material boundary when the opposite
face is occluded, touches another ply, or leaves the block.  The next stage
therefore reuses the identical source-aligned CT preparation to extract every
strong **signed air-to-material interface** without requiring an exit
crossing.  It anchors both exact physical faces of selected paired profiles;
the two faces have opposite signed normals but retain one sheet identity.
Ambiguous anchors from different identities are marked as conflicts and are
never assigned.

```bash
python3 -m backend.cubical build-one-sided-interface-bank \
  --growth /path/to/paired-surface-growth \
  --output /path/to/one-sided-interface-bank

python3 -m backend.cubical grow-one-sided-interfaces \
  --bank /path/to/one-sided-interface-bank \
  --output /path/to/one-sided-interface-growth
```

This field is dense, unlike the paired profiles.  Its default topology uses
only processing-lattice sites that share a face.  Reusing the paired graph's
sqrt(5)-step sparse links connected distinct nearby boundaries into a 12,559
interface component touching 59 trusted identities.  Shared-face topology
reduces the largest component to 377 while still exposing substantial new
boundary support.  Signed normal comparison rejects opposite sheet faces;
normal sign is never canonicalized away in this stage.

Connected patches retain ambiguity explicitly.  A patch with no seed is an
unowned boundary hypothesis, one touching a single identity may grow that
identity, and one touching multiple identities is deferred rather than split
by queue order.  Fragment identities may associate only when at least two
independent boundary components on **each** corresponding physical face show
consistent side parity and balanced seed support.  A parity-aware union keeps
canonical-side flips explicit and rejects contradictory association cycles.

On the current core, extraction takes about 3.0 seconds and writes 360,545
owned strong interfaces.  The strict 0.75-step / 15-degree anchor matches
88,147 of 190,082 paired endpoints to 74,063 unique interface seeds while
excluding 828 conflicting seeds.  Shared-face graph construction, bilateral
association, conservative growth, compressed output, and previews complete in
about 1.7 seconds.  Eight bilateral joins reduce 878 input identities to 870
assemblies, with no assembly larger than three identities.  The solve selects
139,089 interfaces: 74,063 immutable anchors plus 65,026 unambiguous
extensions.  It leaves 28,710 non-seed interfaces in contested components and
163,831 in wholly unseeded components for later sheet-pair inference.  These
deferred populations are first-class output, not failures hidden by a winner.

### Exact two-face ribbon bank

The strongest subset of the boundary field is represented explicitly as a
**ribbon**: one physically bounded paired profile whose lower and upper faces
both match strong signed interface samples.  Reciprocal or near-identical
profiles that land on the same exact interface pair collapse to one ribbon,
while their alternative count and original candidate mapping remain
persisted.  The strict paired-profile continuity graph is then projected onto
these ribbons without recomputing or weakening its two-boundary geometry.

```bash
python3 -m backend.cubical build-clear-ribbon-bank \
  --growth /path/to/one-sided-interface-growth \
  --output /path/to/clear-ribbon-bank

python3 -m backend.cubical select-clear-ribbons \
  --bank /path/to/clear-ribbon-bank \
  --output /path/to/clear-ribbon-selection

python3 -m backend.cubical grow-clear-ribbon-interfaces \
  --selection /path/to/clear-ribbon-selection \
  --output /path/to/clear-ribbon-interface-feedback

python3 -m backend.cubical grow-clear-ribbon-paired-profiles \
  --feedback /path/to/clear-ribbon-interface-feedback \
  --output /path/to/clear-ribbon-paired-feedback

python3 -m backend.cubical grow-paired-feedback-interfaces \
  --paired-feedback /path/to/clear-ribbon-paired-feedback \
  --output /path/to/clear-core-interface-refinement
```

This stage is an evidence bank and component census, not a selector.  In
particular it records competing ribbons at the same source-lattice key so a
later solve cannot accidentally select an entire connected component and
violate local mutual exclusion.

On the current core, 43,297 of 309,672 physically bounded candidates match
both signed faces within one sampling step and 20 degrees.  They reduce to
41,451 unique ribbons, including 1,846 duplicate face-pair profiles.
Projecting the persisted paired graph produces 134,855 unique strict
continuity edges and 7,137 components.  The largest components contain 764
and 557 ribbons.  Of all components, 2,651 touch exactly one trusted assembly,
4,480 are unseeded, and six touch multiple assemblies; the contested
population is 633 ribbons and is deferred in full.  Ninety-two unseeded
components contain at least eight ribbons and the largest contains 34.  There
are also 3,762 same-key alternatives across 967 components, which is why the
bank deliberately stops before selection.  Construction and both PNG audits
take about 1.5 seconds.

The following selection stage treats every previously selected paired surface
as an immutable anchor.  Within a component that touches exactly one trusted
assembly it grows a maximum-bottleneck forest over strict two-face continuity
edges.  Components touching multiple assemblies retain their anchors but
defer their interiors.  Entirely unseeded components receive a new identity
only when both of every candidate ribbon's signed-interface components are
also unseeded and at least eight ribbons remain after global source-lattice
mutual exclusion.  This cross-representation check matters: absence from the
paired-ribbon selection alone does not prove that a physical boundary is
unowned.  The hard constraint remains one selected ribbon per source-lattice
key, including across otherwise disconnected components.

On the current core, selection retains all 28,774 anchors, adds 824
collision-safe ribbons to anchored assemblies, and admits 395 ribbons as 37
entirely new clear cores.  Of 8,232 ribbons in components without a
paired-surface assembly, 4,521 also have two unseeded signed components and
3,711 are excluded as already claimed or contested boundary evidence.  The
37 surviving cores contain 8--26 ribbons, with a median of nine.  Eleven
candidate cores fall below the size minimum after mutual exclusion.  The
median path bottleneck is 0.766 for anchored growth and 0.718 for new cores.
Including compressed output and both PNG audits, the stage takes about 0.5
seconds and writes a 104 KB selection table.  This is intentionally a
high-confidence scaffold rather than an attempt to claim utilization in
unresolved dense material.

The feedback stage then adds only the 37 genuinely new identities to the
signed-interface seeds.  It rebuilds topology because an exact ribbon face
may have fallen below the original interface evidence gate and therefore had
no graph edges.  Existing interface assignments are a hard invariant: any
loss or relabel aborts the stage, and every component touched by more than one
identity remains deferred.

The 395 ribbons contribute 790 endpoint observations at 587 unique
interfaces, with no identity conflicts.  Rebuilding raises eligible
interfaces from 331,630 to 331,752 and continuity edges from 349,114 to
349,215.  Conservative growth assigns 2,508 interfaces to the 37 new cores
while preserving all 139,089 baseline assignments exactly.  The contested
population remains 28,710 and the wholly unseeded population falls from
163,831 to 161,445.  The complete feedback solve, 7.6 MB compressed table, and
two visual audits take about 1.4 seconds on CPU.

The endpoint caps were selected by a monotone safety sweep, not yield alone.
The original 0.75-step / 15-degree rung recovers five cores and 323
interfaces; 0.75 / 20 recovers 12 and 1,059; and 1.0 / 20 recovers 37 and
2,508 while preserving the baseline exactly.  The next 1.25 / 25 rung creates
eight seed conflicts, makes four previously owned components contested, and
would drop 66 prior assignments, so the feedback invariant rejects it.  The
1.0 / 20 setting is therefore the widest demonstrated safe rung on this
block, while the preservation check remains mandatory on every future block.

The final feedback pass returns those interface-validated identities to the
complete physical paired-profile graph.  This is not a second CT detector and
does not invent geometry: all candidates and continuity edges come from the
persisted paired bank.  Every one of the 95,041 baseline selections and its
occupied sampling-lattice key is immutable.  Baseline-occupied keys are
removed before free components are labeled, a free component reached by more
than one new identity is deferred in full, and source-key mutual exclusion is
enforced globally during maximum-bottleneck growth.

Paired connectivity alone is not accepted as a safety certificate.  Before
growth, both physical faces of every candidate in a singly seeded free
component are matched back to the signed-interface bank.  A candidate is
vetoed if either matched face is already owned by another identity; the free
graph is then relabeled because removing one contradictory profile can split
off an otherwise plausible branch.  Unmatched faces and matched-but-unowned
faces remain usable evidence, while a contradiction can never be hidden by a
strong paired-graph path.

On the current core, the 395 clear-ribbon seeds initially occupy 37 distinct
free graph components containing 2,664 candidates.  Their 5,328 physical
endpoints yield 2,571 signed-interface matches: 1,532 agree with the intended
identity, 750 are unowned, and 289 contradict an existing owner.  All 289
contradictory candidates are removed; none is a clear-core seed.  Safe
relabeling still leaves 37 singly seeded components, now containing 10--201
candidates with a median of 59.

Conservative growth then adds 1,503 already-measured paired profiles, for
1,898 profiles across the 37 new identities, while preserving all 95,041
baseline profiles exactly.  The selected result has zero foreign-owned
endpoint matches and exposes 519 unowned endpoint observations at 397 unique
interfaces without a cross-identity conflict.  Added profiles have a median
path bottleneck of 0.589 and median same-label graph degree of six; 92.35% have
at least two same-label neighbors.  The complete solve and visual audits take
0.92 seconds on CPU.  These larger patches are still evidence-bounded clear
cores, not a claim that the unresolved dense block has been partitioned into
sheets.

The paired-to-interface refinement closes the first monotone feedback loop.
It freezes every selected interface from the preceding state, groups all
unowned endpoint observations by exact interface and intended identity, and
adds only conflict-free groups as seeds.  Multiple labels for one interface,
an original seed conflict, or an unexpectedly occupied interface are recorded
and deferred.  Rebuilding the signed graph is necessary because a newly seeded
interface may previously have been below the ordinary evidence threshold.

On the current core, 519 endpoint observations reduce to 397 unique new
interface seeds across 35 of the 37 clear identities, with no label conflicts;
106 of those interfaces were previously below the evidence gate.  They unlock
1,000 additional interfaces, so the round adds 1,397 assignments while
preserving all 141,597 prior assignments and labels exactly.  Median new-growth
bottleneck is 0.699, median same-label degree is two, and 88.20% of additions
have at least two same-label neighbors.  Unseeded eligible interfaces fall
from 161,445 to 160,149, while five additional interfaces become contested and
remain deferred.  The complete CPU round and visual audits take 1.46 seconds.

The complementary resolution audit asks whether the existing planar cubical
representation is locally too coarse.  It preserves shared-face endpoint,
corner, and ordering constraints while relaxing only normal and fiber gates;
coherent high-bend correspondences are emitted solely as refinement evidence
and can never be inserted as graph joins.

```bash
python3 -m backend.cubical audit-sheet-resolution \
  --graph /path/to/surface-graph-v1.json \
  --voxel-size-microns 7.91 \
  --output /path/to/sheet-resolution-audit
```

## Tests

```bash
python3 -m unittest \
  backend.test_rectify \
  backend.test_cubical \
  backend.test_isolated_slab \
  backend.test_paired_surface_bank \
  backend.test_paired_surface_growth \
  backend.test_one_sided_interface \
  backend.test_one_sided_growth \
  backend.test_clear_ribbon \
  backend.test_clear_ribbon_selection \
  backend.test_clear_ribbon_feedback \
  backend.test_clear_ribbon_paired_feedback \
  backend.test_clear_core_interface_refinement \
  backend.test_raw_acus_pipeline -v
```

The tests cover disjoint shard ownership, source-anchored extraction tiling,
raw source and shared-calibration identities, content hash checks, unsigned
normal recovery, physical two-ply/empty alternatives, exact mode-bank
persistence, gap classification, configuration-aware face selection, cubical
clipping, trace alignment, frozen topology round trips, conditional boundary
reselection, topology vetoes, signed interface anchoring,
ambiguity-preserving boundary growth, bilateral seed association, and
exact two-face ribbon de-duplication, collision-aware component census, and
collision-safe ribbon selection, minimum-size rollback, contested-component
deferral, cross-representation seed validation, baseline-preserving ribbon
feedback, immutable-baseline paired-profile feedback, free-component ambiguity
deferral, spatial-key mutual exclusion, cross-representation ownership vetoes,
monotone paired-endpoint interface refinement, and hierarchical assembly.
