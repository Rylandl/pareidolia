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

## Cubical surface experiment

The active architectural experiment represents locally planar interfaces as a
cell complex rather than extending the legacy fragment-growth graph. Its core
is dataset-independent and lives in `backend/cubical`:

- every uncertain cell-centered plane is clipped into a deterministic three-
  through six-vertex polygon on canonical global grid edges;
- adjacent cells align their complete ordered trace sequences on the exact
  shared face using uncertainty-normalized position, unsigned-normal, and
  transported unsigned-fiber evidence;
- statistically supported near-corner topology changes weld to the canonical
  grid vertex, while inconsistent transitive corner cycles are deferred;
- a retained join may not place two locally planar patches from one cell in a
  surface component;
- regular leaf blocks cache exterior traces and interior component incidence,
  and hierarchical composition reproduces direct assembly on the analytic
  tests; and
- patch artifacts use versioned NumPy structures of arrays so later inference
  can be sharded and vectorized without changing the geometry contract.

The analytic smoke test is reproducible with:

```bash
python3 -m backend.cubical synthetic --verify-direct
```

The native-CT path now runs Acus itself and does not consume the persisted
needle, flake, graph, or component caches:

```bash
python3 -m backend.cubical full-acus \
  --compute gpu \
  --voxel-origin 3520 2784 160 \
  --shape 8 8 6 \
  --shard-shape 4 4 3 \
  --output /mnt/t5/pareidolia/raw-acus-8x8x6-v1
```

It retains full normal-by-depth-by-unsigned-orientation evidence and native CT
profiles, persists every fitted mode before top-M compression, constructs
physically constrained layer-count alternatives, selects configurations using
shared-face agreement, and only then performs topology-safe hierarchical
surface assembly. The first fresh 384-cell GPU
pilot completes in 30.5 seconds, selects 1,247 patches, retains 1,284 joins,
and has a largest current component of 71 patches. Its complete contract,
artifacts, exact shard-invariance result, measured storage, and scaling
boundary are documented in
[`docs/raw-acus-cubical-pipeline.md`](docs/raw-acus-cubical-pipeline.md).

Gap recovery now works from that full mode bank instead of weakening match or
topology gates. On the 16 × 16 × 14 full-depth slab, 27 of the leading
component's 34 apparent mode gaps already had an independently fitted matching
mode that top-M compression had hidden. The bounded conservative continuation
search evaluates up to three conditioned stratigraphies per candidate. Its
combined result uses six safe target cells, verifies eight closed seams, grows
the leading component from 195 to 201 cells, gains 15 retained joins, removes
18 unresolved traces, and reduces both collision and topology deferrals without
deleting a layer. Reproducible `gap-census`, `mode-bank`,
`mode-continuation-search`, and `apply-mode-continuations` commands and their
artifact contracts are documented in the same pipeline note.

Block saturation now has a calibrated structural audit and a full-bank
reselection path. On the 16 × 16 × 14 slab, exhaustive physical enumeration
tests 1,219,069 stacks and raises jointly supported Acus evidence from 60.09%
to 64.79% while reducing assembled components from 1,378 to 1,127. The audit
records the complete ceiling ladder: 79.09% has some fitted mode, 75.85% is
covered by one normal family, and 72.70% is covered by one physically ordered
stack. Candidate enumeration is immutable and can be reused for fast selection
sweeps with `select-saturation-candidates`.

A separate `dual-axis-packets` graph then gives sheet-level semantics to the
two orthogonal papyrus fiber axes without weakening the single-ply graph. The
strict graph is fixed; quarter-turn joins must pass absolute 15° normal and
fiber-frame gates plus the existing endpoint, ordered-trace, same-cell
collision, crossing-topology, and orientability checks. It adds 187 safe joins,
reduces components from 1,127 to 977, grows the largest fragment from 191 to
278 cells, and removes 374 unresolved traces. The evidence audit independently
finds 70.44% direct-or-transverse support overall and 90.83% in the strongest
needle-score decile. Details, commands, threshold sensitivity, and artifact
contracts are in
[`docs/raw-acus-cubical-pipeline.md`](docs/raw-acus-cubical-pipeline.md).

Block handoff is now concrete rather than a full-volume reload. A default
two-cell `export-boundary-band` shell carries selected polygons, retained
physical alternatives, packet ownership, occupied-cell collision certificates,
welded edge/vertex identities, and one immutable inner anchor layer.
`reselect-boundary-bands` jointly optimizes only the two meeting shells, then
rebuilds their strict and dual-axis topology against serialized frozen-interior
certificates. On a deterministic X=8 split of the current 16 × 16 × 14 result,
it exactly recovers all 13,287 retained full-block joins and all 977 components.
On two halves independently rerun from native CT, it changes 37 of 896 mutable
cells, improves layer-count agreement with the full-context reference from 866
to 884, and reduces the selected-only composition from 987 components to 978,
one above the 977-component reference. The full block is a consistency
reference rather than ground truth; importantly, all 15 changes that reach an
exact reference configuration move toward it and none move away.

Multi-seam composition now solves a complete regular 2 x 2 x 2 child cluster
once instead of unioning incompatible pairwise seams. Each boundary export
contains the 26 valid one-, two-, and three-face frozen-region certificates;
`reselect-boundary-cluster` removes the participating face bands as a union,
runs one sparse conditional configuration solve, and reconstructs one global
strict/dual-axis topology. On four independently inferred 8 x 8 x 14 blocks,
the 16 x 16 x 14 solve handles 1,568 mutable cells in 48.9 seconds and resolves
all 23 pairwise corner conflicts. Against the unsplit consistency reference,
layer-count agreement improves from 1,456 to 1,527 cells, comparable retained
joins agree at 98.38% Jaccard, and both graphs contain 977 components. Exact
component membership is not identical: connectivity precision/recall on the
4,069 geometrically mappable patches is about 90.8%, which remains a useful
target for the next block-level refinement.

`materialize-boundary-cluster` now expands that bounded solve into one complete,
hashed retained graph without rerunning inference: joint mutable bands replace
the corresponding child bands, while the certified child interiors contribute
their original geometry and joins. The real four-child result contains 11,723
patches and 13,230 joins in 977 fragments. Its largest fragment is 271 cells,
up from 161 in any independent child; 17 fragments contain at least 128 cells
versus one before composition, and 153 fragments cross a child boundary. The
size distribution is also close to the unsplit consistency reference (largest
278 cells, with the same 17 fragments at or above 128). Component flattening
automatically consumes this retained graph and emits both diagnostic grid
overlays and clean native-CT PNGs.

`diagnose-cell-refinement` reopens that materialized result at a single cell
without rerunning Acus. It maps the cell back to its immutable full physical
candidate bank, reports evidence, face, and retained-topology utilization as
separate quantities, and ranks all local stacks with a trace-count-normalized
continuity objective. Refinement is conservative and staged: single-cell and
adjacent-pair proposals must remain inside an evidence-coverage envelope; each
replacement is then replayed against a graph with every exterior join frozen;
topology-positive support cells are accepted first; and the focal cell is
retried only while every frozen exterior component remains connected, open
endpoints and component count do not increase, and collision safety plus
orientability hold. If individually safe supports conflict as a batch, a
deterministic exact-replay pass grows a maximal collision-safe subset and
revisits blocked additions after the context changes. A final exact replay also
admits the coordinated net-change set when individually unsafe replacements
become safe together and improve more evidence than the staged result.
Retained-trace fraction remains
diagnostic rather than a gate because legitimate removal of unsupported layers
changes its denominator.
Repeated annealing trials reuse an exact compressed certificate of the immutable
exterior graph: component occupancy, orientation parity, crossing ownership,
and detached components are frozen once for each active-cell cut. Only the
candidate patches and joins touching that cut are replayed. This is not a
weaker acceptance path—the final materializer independently reconstructs the
complete graph and requires its audit summary to match byte-for-byte before it
writes a new variant.
The command writes a hashed diagnostic/proposal artifact but does not mutate
the selected surface graph:

```bash
python3 -m backend.cubical diagnose-cell-refinement \
  --cluster /path/to/cluster-reselection-v2 \
  --materialized /path/to/cluster-materialized-v2 \
  --cell 5 4 4 --component-id 10 \
  --output /path/to/cell-refinement-c5-4-4-v1
```

An accepted annealing round can then be promoted to a complete graph variant.
`materialize-cell-refinement` preserves every exterior patch and retained join,
replays only the accepted cells, and writes a full per-cell configuration
ledger bound by hash to the new patch and surface-graph artifacts. The output
is therefore a valid `--materialized` input for another diagnostic round:

```bash
python3 -m backend.cubical materialize-cell-refinement \
  --cluster /path/to/cluster-reselection-v2 \
  --materialized /path/to/cluster-materialized-v2 \
  --diagnostic /path/to/cell-refinement-c5-4-4-v1 \
  --output /path/to/cell-refinement-variant-r1
```

`rank-cell-refinement-targets` provides the next-pass work queue. It ranks
cells by the geometric mean of two empirical percentiles—candidate-bank Acus
mass still recoverable and unresolved trace endpoints touching the cell—and
also emits a spatially separated ranking whose radius-one neighborhoods do not
overlap:

```bash
python3 -m backend.cubical rank-cell-refinement-targets \
  --cluster /path/to/cluster-reselection-v2 \
  --materialized /path/to/cell-refinement-variant-r1 \
  --output /path/to/refinement-targets-r2
```

The replacement for repeated cell-by-cell repair is an immutable block sheet
contract. `compile-sheet-evidence` deduplicates every retained full-Acus mode,
clips its cubical polygon once, and records every physical within-cell stack as
a hyperedge over stable mode IDs. Acus does not run during later sheet solves:

```bash
python3 -m backend.cubical compile-sheet-evidence \
  --input /path/to/block-x0-y0-saturation-v1 0 0 0 \
  --input /path/to/block-x1-y0-saturation-v1 8 0 0 \
  --output /path/to/block-sheet-evidence-v1
```

For an already selected geometry, `restitch-block-sheets` keeps every
pair-gated face alternative, solves exact order-preserving face assignments,
segments the dense face graph with minimum-cost same-cell collision cuts, and
performs topology-safe whole-sheet neighborhood exchanges. It writes a standard
selected-patch and retained-surface-graph root that can be flattened or merged
by the existing commands:

```bash
python3 -m backend.cubical restitch-block-sheets \
  --cluster /path/to/cluster-reselection-v2 \
  --materialized /path/to/materialized-block \
  --output /path/to/restitch-result \
  --curvature-refinement \
  --layer-partition
```

The collision cut runs to physical completion by default; an operational cap
can be supplied with `--collision-cut-limit`, and `--collision-cut-order both`
persists a deterministic forward/reverse sensitivity comparison. On the owned
12 x 12 x 10 audit core, whole-block belief propagation, coverage-preserving
configuration scoring, completed collision cuts, and four sheet-exchange rounds
raise structural-evidence coverage from 65.42% to 66.86%. The default pure
likelihood solve retains 5,692 rather than 5,633 joins, reduces open endpoints
from 6,590 to 6,424, lowers the local-to-global topology tax from 967 to 932,
and raises raw correspondence benefit. An explicit two-unit open-endpoint prior
gives the Pareto variant with 5,709 joins, 6,390 open endpoints, 404 rather than
418 components, and a 915-join topology tax for a 0.044% raw-benefit trade.
The optional curvature refinement is a geometry-correction stage, not a fixed
normal-angle gate. It measures unsigned one-cell bend and the contrast between
axial mean normals in neighborhoods on opposite sides of every retained join.
Within-side dispersion is subtracted so gradual macro-curvature remains valid.
Robust limits are calibrated from the block's own solved graph and then frozen;
minimum-cost cuts remove abrupt hinges and the exact topology validator rematches
their newly open traces. On the owned audit core this reduced 389 detected abrupt
hinges to zero in two rounds, recovered 110 valid joins after cutting, and retained
5,336 joins total. The three visibly folded 155-, 138-, and 134-cell components
resolved into smooth fragments, while the two convincing large components remained
169 and 161 cells. This tradeoff is explicit: retained endpoint utilization falls
from 63.93% to 59.93% until later configuration-level gap recovery can add genuinely
supported geometry.

Face matching now also freezes absolute angular validity before graph solving.
Plane normals use the axial angle `acos(abs(n1 dot n2))`; fiber axes are first
transported by the minimal normal-frame rotation and then compared axially.
The orthogonal-fiber family is separate: its score is the residual to 90
degrees and it can never turn a perpendicular fiber into a strict continuation.
Zero-valued `--strict-normal-angle-cap-degrees` and
`--strict-fiber-angle-cap-degrees` derive robust median-plus-MAD limits from
the retained population, with the declared policy and physical uncertainty
floors still acting as bounds. This keeps the policy block-adaptive rather than
tuned to one scroll crop.

Unsigned normals alone cannot distinguish a smooth continuation from a facet
folded back over the same shared trace. Each candidate therefore also projects
both polygon interiors onto the trace conormal in their common tangent frame.
A valid continuation must leave the trace on opposite sides. Same-sided pairs
are persisted as hard foldback exclusions, so an indirect graph path cannot
reconnect them. On the owned 12 x 12 x 10 core the current solve calibrates a
26.258-degree normal cap and a 15-degree strict-fiber cap. It retains 5,160
joins with 57.952% endpoint utilization; the maximum accepted axial hinge is
26.213 degrees. All 206 detected foldback pairs remain in different final
components, including those previously reconnected by short transitive loops.
The promoted curvature-refined successor removes all 81 remaining multiscale
hinge flags in three cut-and-refill rounds while retaining 5,127 joins and
57.581% endpoint utilization. Its largest component grows from 158 to 159
patches, the 150-patch runner-up is unchanged, and the maximum retained axial
hinge falls to 24.512 degrees. The pre-curvature graph remains available as the
v11 checkpoint so this geometry/coverage tradeoff stays inspectable.

The current owned-core successor also assigns cross-edge corner observations
to their nearest half-edge. A high-uncertainty trace may no longer snap past a
cube-edge midpoint merely because its covariance makes the distant corner
statistically admissible. The corner-safe graph has no physical nonmanifold
edges across all 922 components; its most distant retained corner transition
is 0.494786 edge lengths from the shared vertex.

The optional layer partition compiles fixed active sheetlets into a persisted
typed graph. Continuations retain raw likelihood plus exact whole-face matching
marginals. High-confidence local collisions remain hard constraints, while a
denser lifted graph supplies soft distinct-layer costs to a signed
component-level partition: a shear bridge must outweigh every repulsive
relation it would pull into one sheet, not merely look plausible at one face.
Same-cell returns and the narrower high-confidence collision set remain hard:
no indirect path or loop can reconnect their endpoints. The resulting cluster
is replayed through the exact topology validator. The signed partition is also
reversible: deterministic attractive min-cuts seeded by internal repulsive
pairs split a completed component whenever separated layer evidence outweighs
the continuity edges being cut. On the current 4,784-node owned core no such
post-solve split has positive gain, which is recorded rather than silently
forcing fragmentation. An experimental `--stack-transport` audit enforces an
independent integer cell-stack gauge in each connected sheet component, but is
deliberately off by default because incomplete stacks require partial monotone
correspondences. This fixed-node stage can reorganize or split selected
sheetlets but cannot activate unused Acus candidates to fill holes.

`audit-sheet-core` writes the complete reusable
evidence/configuration/topology failure decomposition.

The architecture and joint configuration/sheet solver are detailed
in [`docs/design/block-sheet-reconstruction.md`](docs/design/block-sheet-reconstruction.md).
The implemented continuation compiles the complete all-mode edge and stack-pair
factor banks, optimizes a reversible configuration initialization, and then
requires global topology replay:

```bash
python3 -m backend.cubical catalog-sheet-correspondences \
  --evidence /path/to/block-sheet-evidence-v1 \
  --cluster /path/to/cluster-reselection-v2 \
  --output /path/to/mode-correspondences-v1
python3 -m backend.cubical compile-sheet-factors \
  --evidence /path/to/block-sheet-evidence-v1 \
  --correspondences /path/to/mode-correspondences-v1 \
  --cluster /path/to/cluster-reselection-v2 \
  --output /path/to/configuration-factors-v1
python3 -m backend.cubical initialize-sheet-configurations \
  --evidence /path/to/block-sheet-evidence-v1 \
  --factors /path/to/configuration-factors-v1 \
  --initial /path/to/current-materialized-graph \
  --output /path/to/configuration-initialization-v1
python3 -m backend.cubical replay-joint-sheet-graph \
  --evidence /path/to/block-sheet-evidence-v1 \
  --correspondences /path/to/mode-correspondences-v1 \
  --configurations /path/to/configuration-initialization-v1 \
  --cluster /path/to/cluster-reselection-v2 \
  --output /path/to/joint-sheet-graph-v1
```

`refine-sheet-topology` is the nonlocal outer loop. It ranks configurations by
recoverable incompatible gaps and topology pressure, exchanges complete Acus
stacks, reopens every current sheet component touched by those stacks, and
freezes the untouched exterior. Every proposal is accepted only after exact
whole-block collision, face-order, crossing, and orientability replay:

```bash
python3 -m backend.cubical refine-sheet-topology \
  --evidence /path/to/block-sheet-evidence-v1 \
  --correspondences /path/to/mode-correspondences-v1 \
  --factors /path/to/configuration-factors-v1 \
  --configurations /path/to/configuration-initialization-v1 \
  --graph /path/to/joint-sheet-graph-v1 \
  --cluster /path/to/cluster-reselection-v2 \
  --output /path/to/sheet-topology-refinement-v1
```

The refinement root is both an auditable configuration selection and a normal
`selected-patches-v1`/`surface-graph-v1` root. Supplying it to
`replay-joint-sheet-graph` seeds the full multi-restart solve with the already
validated graph, so later whole-sheet exchange can improve its likelihood but
cannot discard it merely because another deterministic restart was weaker.

Assembly now carries two explicit binary gauge states. Polygon-orientation
parity prevents unsigned local normals from closing a globally contradictory
surface loop. Fiber-frame parity separately requires every cycle to contain an
even number of quarter-turn transitions; a contradictory edge is deferred as
`fiber-frame-parity-cycle`. Unknown fiber relations remain unconstrained.
`refine-join-continuity` then scores every retained face using fixed-depth
native CT against equal-span within-patch controls. Only robust per-axis
intensity-mismatch outliers split connectivity; noisier texture-angle and
normal-profile-shift measurements remain auditable diagnostics. Its compact
table can be supplied to `flatten-components` with `--join-refinement`.

`refine-stratigraphic-continuity` adds the larger-context layer test. It
anchors every selected patch back to its exact same-family mode in the complete
Acus bank, compares the surrounding depth/fiber distribution across a join,
and repeats that comparison after averaging graph-connected three-hop
neighborhoods on the two spatial sides of the face. Connectivity changes only
when both scales are robust per-axis outer-tail outliers. With
`--candidate-restitch`, the retained graph calibrates the robust scale while
the same frozen test is applied to every geometric alternative before another
global solve. Normal sign transports depth order. A quarter-turn edge
transports the sign of the axial `cos(2 theta)` orientation moment; this is a
fiber-frame gauge operation, not a signed-vector choice.

Each refinement now writes both its score tables and a complete
`selected-patches-v1`/`surface-graph-v1` checkpoint. Its identity includes the
input graph hashes, so fixed-point rounds cannot accidentally reuse scores from
a different join graph. On the current 4,784-sheetlet owned core, the
candidate-gated solve reached 5,016 joins and a 166-cell largest component. A
following rescore removed two complete-mode contradictions, then converged
with zero rejected retained joins: 5,014 joins, 929 components, 56.312%
interior-trace utilization, and leading component sizes 166, 149, and 143.
The converged candidate table rejects 106 of 11,953 alternatives. Across the
final graph, the maximum direct axial-normal hinge is 23.802 degrees, the
strict-fiber residual is at most 14.712 degrees, and the explicit quarter-turn
residual to 90 degrees is at most 14.796 degrees. Every component is free of
same-cell returns, foldbacks, polygon-orientation conflicts, fiber-frame parity
contradictions, and physical nonmanifold edges.

`flatten-components` is the corresponding visual checkpoint. It unfolds a
mixed set of reconstructed components into bounded-normal atlases and samples
the native CT at one component-wide sequence of fixed depth offsets. Cyan marks
cell boundaries and red marks UV overlap, so neither atlas seams nor local
layer switching can be hidden by a per-cell best-depth choice. The raw stacks,
montages, depth crossings, topology diagnostics, and reproducible command are
documented in the same pipeline note.

The earlier real-data plumbing check remains available as a historical geometry
comparison. It adapts persisted Acus modes only as plane evidence proxies and
does **not** call them physical layers:

```bash
python3 -m backend.cubical acus-window \
  --root work/cross-scroll-analysis-z512 \
  --origin 109 86 0 --shape 16 16 14
```

That 16 × 16 × 14-cell run currently converts 11,065 inherited modes into
8,173 core-owned polygons and completes adaptation, hierarchical assembly, and
mesh/projection export in 17 seconds. It retains 7,727 joins, defers 520 joins
that would cause a same-cell layer collision and 98 with an inconsistent
crossing topology, and leaves 1,827 components. This is deliberately a baseline
showing why the inherited three-mode local representation is insufficient. The
full geometry and artifact contract is documented in
[`docs/cubical-surfaces.md`](docs/cubical-surfaces.md).

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
- `python3 scripts/build-global-branch-candidates.py --root
  work/cross-scroll-analysis-z512` inventories locally withheld evidence before
  accepting any new join. From 47,981 candidate occurrences it finds 230 global
  branch pairs that passed the main window's score, material, order, and
  collision solve but lacked unanimous subwindow support, plus 429 pairs that
  passed subwindow stability but failed the small-window MLS fit. The 659-pair
  catalog retains all 1,045 supporting observations and their original
  decisions; 229 of the first tier are under-observed and one has an actual
  subwindow disagreement. Accepted, quarantined, already-linked, and duplicate
  global branch pairs cannot re-enter through `global-branch-candidates-v1`.
- `python3 scripts/associate-global-branches.py --root
  work/cross-scroll-analysis-z512 --local-evidence-only` rebuilds the accepted
  and rescue tiers from complete global branches. Global context passes 111 of
  230 subwindow-
  unresolved pairs and five of 429 locally exact-deferred pairs. The final solve
  retains 587 joins: 125 overlap-validated, 213 single-window, 140 context-
  disputed, 104 subwindow-unresolved, and five local-exact rescues. Six pairwise
  passes cause cell collisions and six weakest edges are removed to resolve eight
  initial carrier-pair intersections. The resulting 556 exact-coherent
  associations cover 1,143 branches and 12,042 flakes: 528 pairs, 25 triples,
  and three four-branch groups. Final carrier medians remain 1.25 voxels / 3.90
  degrees, with zero exact failures and zero intersections after 127,776 broad-
  phase and 3,802 narrow-phase triangle checks. Candidate discovery takes 11.2
  seconds and the solve 34.5 seconds. It reduces 63,783 linked branches to
  63,196 groups. Relative to the unassociated graph, the full catalog gains 25
  fragments with at least 25 flakes, 11 with at least 50, four with at least
  100, nine spanning at least 11 planes, and six spanning all 14. The
  `--accepted-only` mode exactly reproduces all 23 compared v3 geometric and
  decision arrays; `--clean-only` retains the earlier two-tier scope. The v4
  artifact preserves every candidate, provenance tier, residual, and final
  decision.
- `python3 scripts/build-global-boundary-candidates.py --root
  work/cross-scroll-analysis-z512` broadens “endpoint” to a directionally
  exposed fragment edge. For global graph nodes of degree one through six, it
  estimates the open direction opposite the resultant of retained tangent
  neighbors, then requires that no retained neighbor already occupies that
  cone. A streamed vectorized search evaluates 106,943,713 nearby node pairs in
  30.0 seconds without materializing them, leaving 276 novel branch pairs after
  the existing score, material, whole-volume order, and collision rules. It also
  audits the 5,425 otherwise scored pairs trapped in the giant cyclic order
  component inside every tiled window that observes both endpoints. Requiring
  at least two observations and unanimous acyclic, unblocked local order admits
  307 more candidates; 441 clean single-window cases remain deferred. The full
  v4 boundary artifact takes 37.7 seconds and retains every local observation.
- The globally order-clean directional subset is added only after all local-
  evidence joins and a directional-only edge is always pruned before a local
  edge. It therefore preserves all 60 common v4 artifact arrays exactly.
  Complete-branch reconstruction retains 124 of 276 candidates:
  139 have an incoherent input carrier, six fail pair geometry, one collides,
  one fails transitive reconstruction, and five are removed by mesh integrity.
  The resulting v5 scope has 711 joins in 672 associations over 1,383 branches
  and 14,509 flakes with 1.24-voxel / 3.90-degree median fits and zero
  intersections.
  Relative to v4, 120 directional-boundary associations extend axial span in
  74 cases; the whole catalog gains 12 fragments with at least 25 flakes, eight
  with at least 50, six spanning at least 11 planes, and three spanning all 14.
  The largest fragment remains 586 flakes and the count at least 100 is
  unchanged, so this is useful gap closure rather than a giant collapse.
- The default v6 solve constructs the 307 overlap-resolved local-order
  candidates after even the globally order-clean directional tier and prunes
  them first. Complete-branch reconstruction retains 153: 122 have an
  incoherent input carrier, 15 fail pair geometry, five collide, one fails
  transitive reconstruction, and 11 are removed by mesh integrity. All 1,431
  earlier candidate decisions and exact diagnostics remain identical after
  stable endpoint alignment. The final 864 joins form 812 associations over
  1,676 branches and 17,812 flakes with 1.24-voxel / 3.92-degree median fits and
  zero intersections. The 150 affected associations extend axial span in 82
  cases. Relative to v5, the full catalog gains eight fragments with at least
  25 flakes, five with at least 50, one with at least 100, two spanning at least
  11 planes, and one spanning all 14; the 586-flake maximum is unchanged.
  Because 152 of the 153 retained joins have both endpoints in contested
  material, this remains geometry-conditioned gap closure rather than
  independent physical-sheet evidence.
- `python3 scripts/census-fragment-terminations.py --root
  work/cross-scroll-analysis-z512` turns the final v6 catalog into a bounded
  follow-up queue instead of adding another speculative association tier. It
  reconciles every local-window, global-boundary, and complete-branch decision
  at the 32,112 definite degree-one ends of associations with at least 25
  flakes. Of 31,793 unresolved ends, 31,374 have a usable outward tangent and
  form 26,078 same-association, direction-consistent termination regions; 419
  remain explicitly unclustered. The regions split into 24,250 that never
  reach an accepted geometric continuation, 1,602 order failures, 108
  downstream geometry/collision/integrity failures, and 118 material deferrals.
  Sampling the 512 highest-priority evidence-poor targets finds 507 in dense,
  nontruncated CT. The capped queue selects 128 weak-geometry targets across 68
  associations, at most two per association; order and downstream failures are
  kept in separate review queues. The initial cold run takes 32.2 seconds and a
  forced warm-cache rerun 6.9 seconds; both write
  `fragment-termination-census-v1`. Absence of a candidate is not interpreted
  as a physical sheet edge.
- `python3 scripts/reanalyze-fragment-terminations.py --root
  work/cross-scroll-analysis-z512` performs the queued experiment rather than
  changing the graph. Resolving each cluster through its actual endpoint shows
  that 40 of the 128 nominal targets point into a cell already occupied by the
  same final association; these internal branch ends are skipped explicitly.
  The remaining 88 targets are packed into 81 bounded, globally phase-aligned
  GPU crops. At each identical 64-cube location, the existing whole-volume
  catalog and a fresh 2-voxel dense extraction receive the same 12-mode fit and
  local ownership test. Median usable needles rise from 136 to 676. Coarse Acus
  passes 26 targets and dense Acus passes 35: 24 are corroborated by both, two
  regress, and dense adds 11 passes. Six are genuinely absent from the stored
  target-cell modes; five strengthen a matching mode already owned by a small
  separate fragment, identifying association rather than extraction work.
  Forty-four remain below geometry threshold, six are ownership-ambiguous, and
  one is mode-ambiguous. All 81 crops use the GTX 1080 and finish in 46.9
  seconds. `fragment-termination-reanalysis-v1.json` preserves every crop,
  coarse/dense mode, residual, competitor margin, and classification; none of
  the recovered modes are inserted automatically.
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

The solved owned-core block is available at `/block-volume`. It registers the
current collision-safe physical boundary tracks against a stride-2 texture of
the corresponding 384 × 384 × 320 native CT block. Boundary observations are
colored by physical-face identity rather than by their paired-profile
midpoint, so a clear air--papyrus interface remains continuous when the
opposite exit crossing is ambiguous. The viewer can orbit and trackpad-zoom
the combined scene, select or isolate a boundary track, filter by component
size, and cut the volume and observations with the same X, Y, or Z plane. It
also reports the number of transitive layer-crossing joins rejected by the
macro-tangent depth guard. The default artifacts can be overridden without
changing the webpage:

```bash
PAREIDOLIA_BLOCK_SHEET_ROOT=/path/to/retained-graph \
PAREIDOLIA_BLOCK_VOLUME=/path/to/source-slab.npy \
python3 backend/server.py --host 127.0.0.1 --port 8000
```

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

The bounded whole-sheet assignment solver is installed separately:

```bash
python3 -m pip install -r requirements-optimization.txt
```

It uses HiGHS mixed-integer optimization only for small residual component
scopes; the declared node and time limits keep this diagnostic from changing
the block-scale complexity of the ordinary pipeline.

The backend tests cover the analytic rolled volume, exact seed anchoring,
finite chart geometry, cube padding, Acus normal recovery on a crossed-needle
phantom, the cached cuboid-wide neighbor field, air-seed rejection, and PNG
output.
