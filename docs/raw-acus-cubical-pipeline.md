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

1. `calibration-v1.json` samples the native CT window to determine intensity,
   air-threshold, and Hessian-strength scales. It does not import the scale from
   an older Acus run.
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

The raw and local-inference stages are already independently sharded. The
selected-patch assembler is hierarchical. Configuration selection currently
runs one window at a time; full-slab operation should schedule overlapping
macro windows and reconcile their boundary bands rather than materializing
every alternative plane as a Python object in one process. The artifact
contracts are designed for that scheduler, and no evidence-stage rewrite is
required.

## Tests

```bash
python3 -m unittest \
  backend.test_rectify \
  backend.test_cubical \
  backend.test_raw_acus_pipeline -v
```

The tests cover disjoint shard ownership, source-anchored extraction tiling,
raw source identities, content hash checks, unsigned normal recovery, physical
two-ply/empty alternatives, exact mode-bank persistence, gap classification,
configuration-aware face selection, cubical clipping, trace alignment,
topology vetoes, and hierarchical assembly.
