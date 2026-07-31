# Block sheet reconstruction

## Decision

Acus is an expensive, immutable evidence bake. Sheet inference is a separate,
repeatable graph optimization over that bake.

The pipeline must not commit to one local stack, one face alignment, or one
component identity before sheet inference has access to the complete block.
Those early choices caused the previous repair loop: a plausible local edge
occupied a trace, alternatives disappeared, and later code could only add to a
graph whose wrong joins were effectively permanent.

The block contract therefore has three levels:

1. **Mode nodes** are unique fitted Acus planes in one owned cubical cell. A
   node retains its source mode identity, plane covariance, unsigned normal and
   fiber axes, confidence, material probability, evidence score, and effective
   support. Its plane is clipped to the cell once to produce a reusable polygon
   loop.
2. **Configuration hyperedges** are physically admissible within-cell stacks.
   A configuration references mode IDs rather than copying plane geometry and
   retains its evidence, physical-prior, coverage, and posterior scores. The
   empty configuration remains legal. Exactly one configuration is eventually
   active per cell.
3. **Sheet edges** are alternative shared-face correspondences between mode
   nodes in adjacent cells. They are selected only after a configuration makes
   both endpoint modes active.

All direction comparisons are axial/unsigned. Sign is a coordinate gauge, not
an observation.

## Bake phase

The raw CT phase owns all high-cost work:

- source-anchored Hessian extraction with a full needle-length halo;
- dense cell evidence over normal, signed depth, and unsigned orientation;
- independent fitted modes with covariance and CT material evidence; and
- physical stack enumeration with spacing, support, and ply-orientation priors.

The existing `full-acus`, mode-bank, and saturation stages provide these
quantities. A production block may use larger needle budgets, more retained
modes, and a wider physical-configuration beam than the pilot. Those are bake
settings, recorded in the source identity. They are not solver knobs.

`compile-sheet-evidence` converts one or more disjoint Acus candidate banks into
`sheet-evidence-v1` and `mode-patches-v1`:

```bash
python3 -m backend.cubical compile-sheet-evidence \
  --input /data/block-x0-y0-saturation 0 0 0 \
  --input /data/block-x1-y0-saturation 8 0 0 \
  --input /data/block-x0-y1-saturation 0 8 0 \
  --input /data/block-x1-y1-saturation 8 8 0 \
  --output /data/block-sheet-evidence-v1
```

Mode IDs are stable hashes of the mode-bank identity, source shard, and source
mode index. The compiler rejects overlapping cell ownership, inconsistent
world-space offsets, duplicated source modes with different geometry, invalid
configuration membership, and ID collisions. The reader verifies all hashes,
offset tables, cell/configuration ownership, normal-family membership, current
stack uniqueness, and the complete clipped patch shard.

The current 16 × 16 × 14 slab compiles in about 16 seconds to 32,888 unique
modes, 40,253 physical configurations, and 87,724 mode memberships. The two
compressed data artifacts occupy about 6.3 MB. This is small enough to be the
normal downstream unit; CT and Acus arrays do not need to remain resident.

The remaining immutable sheet inputs are compiled independently:

```bash
python3 -m backend.cubical catalog-sheet-correspondences \
  --evidence /data/block-sheet-evidence-v1 \
  --cluster /data/matching-policy-root \
  --output /data/block-sheet-mode-correspondences-v1

python3 -m backend.cubical compile-sheet-factors \
  --evidence /data/block-sheet-evidence-v1 \
  --correspondences /data/block-sheet-mode-correspondences-v1 \
  --cluster /data/matching-policy-root \
  --output /data/block-sheet-configuration-factors-v1
```

The pilot contains 235,364 unselected mode correspondences and 1,268,860 exact
neighboring stack-pair factors. They take about 73 and 19 seconds to compile and
occupy 9.1 and 1.7 MB compressed. Matching policy can therefore be changed and
these relatively cheap artifacts regenerated without touching Acus.

## Sheet solve

For a fixed configuration assignment, all accepted face alternatives are kept.
The solver optimizes a retained edge set under exact constraints:

- one join per patch trace;
- order-preserving correspondences on a shared face;
- at most one patch from a cell in a sheet component;
- one consistent grid-edge/grid-vertex crossing feature;
- orientable polygon parity; and
- path-independent quarter-turn fiber-frame parity.

The current fixed-geometry solver uses four non-monotone levels:

1. exact weighted, order-preserving alignment on every complete shared face;
2. dense face-graph segmentation, which starts from every independently
   optimal face edge and repeatedly makes a minimum-cost cut between two
   patches from the same cell until every provisional sheet is physically
   collision-free;
3. deterministic whole-block proposals under the global constraints; and
4. component-neighborhood exchange, which removes every internal join from the
   one or two current sheets touched by a focal alternative, reconstructs the
   induced graph against a fixed exterior, and accepts only an objective
   improvement.

The dense segmentation reverses the strongest-first commitment that formerly
made sheet identity depend on edge arrival order. Each cut is an undirected
minimum cut weighted by correspondence evidence plus the configured open-trace
cost. The cut sequence terminates without a dataset-specific iteration count:
each cut permanently separates at least one previously connected same-cell
pair. Crossing-feature, face-order, polygon-orientation parity, quarter-turn
fiber-frame parity, and component/cell constraints are still replayed by the
ordinary exact selector, and all safe alternatives are offered again after
segmentation.

Graph selection is separate from mesh welding. Thousands of candidate edge
sets can therefore be evaluated without rebuilding vertices, boundary traces,
or flattening charts. Only the accepted state is materialized as a normal
`selected-patches-v1` plus `surface-graph-v1` root.

On the current selected geometry, complete face matching plus bounded sheet
exchange moved from 13,200 to 13,994 retained joins, from 968 to 819 components,
and from 58.81% to 62.35% retained interior traces while improving the explicit
join-likelihood objective. This demonstrates real stitching headroom, but it
also identifies the fixed-geometry floor: thousands of remaining endpoints
have no compatible selected plane on the adjacent cell.

### Owned-core utilization audit

`audit-sheet-core` separates evidence-bank, configuration, local-face, and
global-topology losses for one owned rectangular core. It uses stable mode and
configuration IDs to compare an owned graph with its complete immutable Acus
bank and records results by shell depth, cell, component, and hole class. An
unresolved endpoint is distinguished as an occupied compatible alternative,
an open compatible bridge, a same-component continuation, an inactive
configuration alternative, a bank-incompatible plane, or a complete face miss.

On the controlled 12 x 12 x 10 core, 80.15% of the 6,590 open endpoints already
had a compatible continuation in the immutable bank; only 133 missed the
target face entirely. The selected configurations retained 65.42% of
structural evidence against a 72.98% per-cell physical-stack oracle. They
permitted 6,600 local joins, but only 5,633 survived global topology—a 967-join
topology tax. Thus the boundary halo was not the mechanism needed to repair
core holes.

Whole-block max-sum configuration messages plus a coverage reward raised owned
evidence utilization to 66.86% and the local match count to 6,624. Under the
default pure correspondence-likelihood objective, fully converged dense
collision-cut segmentation followed by four whole-sheet exchange rounds
retained 5,692 joins, reduced open endpoints to 6,424, and lowered the topology
tax from 967 to 932. Raw retained correspondence benefit rose by 187. An
explicit continuity-Pareto run charging two additional benefit units per open
endpoint retained 5,709 joins, reduced open endpoints to 6,390, reduced
components from 418 to 404, and lowered the topology tax to 915 while giving up
only 27 raw benefit units (0.044%) relative to the pure optimum. The pure and
continuity solutions' largest components were 170 and 191 cells, respectively,
versus 212 before this refocus; component size remains an audit rather than an
acceptance reward. Forward and reverse deterministic cut orders differed by
only four retained joins before exchange; the higher-objective forward order
remains the default, while both remain available as an explicit sensitivity
audit.

### Axial curvature refinement

The retained graph now has an optional post-stitch geometry stage for the abrupt
fold-backs visible in otherwise large components. It never assigns signs to normal
vectors. For every retained join it measures:

1. direct axial normal bend across the shared face; and
2. multiscale axial-mean separation between graph-local half-neighborhoods on the
   two sides of that face, after subtracting the larger within-side normal
   dispersion.

The second signal distinguishes a coherent hinge between two locally smooth pieces
from a sheet whose normal rotates gradually over its span. Whole-component normal
spread remains diagnostic only because a valid scroll sheet can rotate substantially
at larger scales. Direct-bend and positive neighborhood-contrast limits are robust
median-plus-MAD estimates from the current solved block. They are frozen before
refinement so repeated rounds cannot tighten their own calibration by peeling off
the tail.

Each detected hinge seeds an undirected minimum cut. Edge capacity combines retained
correspondence benefit with curvature pressure, preserving strong smooth joins while
cutting the least-supported boundary. Removing joins is replayed through the exact
surface constructor; then only data-derived direct-bend-safe alternatives are offered
to the ordinary topology validator, with the untouched graph fixed. A following round
can detect neighborhood inconsistency introduced by that refill.

On the 12 x 12 x 10 owned core, the frozen model measured a 26.90-degree direct-bend
limit and a 15.53-degree positive neighborhood-contrast limit. Two rounds reduced
389 flagged joins to zero, forbade 468 cut edges, and recovered 110 topology-safe
alternatives for a final 5,336 joins. The largest fragments are 169, 161, and 137
cells; their direct-bend p90 values are 13.26, 16.40, and 16.07 degrees. The folded
138- and 134-cell identities became 116- and 88-cell smooth fragments. This is a
deliberate geometry/coverage tradeoff and identifies the now-clean open boundaries
for configuration-level recovery rather than hiding incorrect layer crossings inside
large component counts.

### Typed sheetlet graph and lifted layer exclusions

Smooth curvature is not sufficient to establish surface identity. Two nearly
parallel papyrus layers can be connected through a sequence of individually
plausible joins while every retained join remains below the abrupt-bend limit.
The resulting component is smooth but locally multi-valued along its own
normal. This is a transitive partition failure rather than a bad-edge failure.

`restitch-block-sheets --layer-partition` therefore persists and solves a typed
`sheetlet-graph-v1` with:

- one fixed active Acus sheetlet node per selected clipped mode;
- attractive shared-face continuation edges carrying both raw pair likelihood
  and their exact marginal probability over every order-preserving whole-face
  stack alignment;
- hard lifted exclusions for the original high-confidence local collision
  definition; and
- a denser soft repulsion graph between nearby, near-parallel sheetlets whose
  tangent projections overlap while their normal heights remain distinct.

An exclusion is independent of the current component assignment. Its two
endpoints may not enter the same local surface chart even through a long chain
of attractive edges. Dense face alignment is segmented by minimum-cost cuts
until both same-cell collisions and lifted exclusions are separated. Every
proposal, neighborhood exchange, and curvature refill then replays exclusions
alongside trace occupancy, face order, crossing consistency, cell uniqueness,
polygon orientability, and fiber-frame parity. Normal and fiber comparisons
remain axial/unsigned; the binary fiber gauge records whether a continuation
preserves the local axis or exchanges it with its orthogonal papyrus axis.

Hard exclusions alone do not identify a layer substitution in which a track
leaves one physical layer and continues on the next without retaining both
layers in the same local neighborhood. Conversely, requiring one constant
integer rank offset between every pair of adjacent cell stacks is too strict:
missing Acus modes and patches that do not cross a particular cube face are
insertions and deletions in a partial ordered correspondence. The implemented
integer-potential union-find is therefore retained as an explicit
`--stack-transport` audit, not enabled by default.

The production fixed-node proposal instead performs signed correlation
clustering. For two current components, a merge realizes the sum of every
attractive continuation edge between them and the sum of every soft lifted
repulsion between them. It is accepted only when the complete component-pair
gain is positive, while same-cell pairs and the narrower high-confidence
lifted exclusions remain hard-separated. Thus neither a direct edge nor an
arbitrarily long loop may reconnect a physically excluded layer pair, and one
locally plausible shear bridge must still pay for all weaker downstream
evidence that the two tracks occupy distinct layers. The clustering result is
only a proposed partition: retained joins are reconstructed afterward by the
unchanged exact trace-occupancy, face-order, crossing, cell-uniqueness, and
orientability validator.

The first hard-exclusion fixed-node pilot contains 4,784 nodes, 16,398 continuation edges, and
30,108 geometric exclusion edges. Only 101 exclusions were connected through a
normal-coherent path in the pre-curvature graph. The combined partition and
curvature solve reduced those 101 local multi-depth conflicts and 391 abrupt
hinges to zero under the frozen declared models. It retained 5,287 joins, 49
fewer than the curvature-only graph, with 98 additional open endpoints and 26
additional components. The former 161-cell suspect component was reorganized
primarily into 121- and 100-cell charts, while the prior 169-cell component
remained largely intact at 164 cells. This is the intended behavior: component
count and size remain audits, while supported continuation is maximized subject
to explicit physical incompatibility.

The following hybrid signed solve keeps those 30,108 exclusions hard and adds
57,660 lower-threshold lifted relations as soft costs. It eliminates all 101
strong conflicts while retaining 5,577 joins in 600 components, including four
components of at least 128 cells and a largest component of 172 cells. Relative
to the first hard-exclusion result, that is 290 more retained joins and 580
fewer open trace endpoints. Relative to a soft-only signed solve, its residual
weak repulsion cost falls from 359.94 to 181.01. The exact component audit finds
no duplicate occupied cell in any reconstructed sheet.

This pilot freezes node activation to isolate the partition problem. The next
stage applies the same typed-edge contract to the complete Acus mode bank, with
one physical configuration hyperedge active per cell. That joint state can
recover inactive modes at holes; a fixed-node partition can only reorganize or
split the sheetlets it was given.

## Joint configuration and sheet inference

The joint solver consumes `sheet-evidence-v1`, not only the current selected
patches. Its state is:

- one active configuration hyperedge per cell;
- the corresponding active mode nodes; and
- a topology-safe subset of face correspondence edges.

The objective separates evidence sources:

```text
cell Acus likelihood
+ physical stack prior
+ shared-face correspondence likelihood relative to open traces
+ measured sheet-scale coherence
```

Sheet coherence must be based on residual evidence—normal transport, fiber
frame, local curvature, and CT profile agreement—not an unconditional reward
for making a giant component. Component size and trace utilization are audit
metrics, not sufficient evidence by themselves.

Optimization alternates large neighborhoods rather than greedy cell repairs:

1. solve exact face alignments for the active configuration state;
2. perform whole-sheet edge exchanges;
3. identify a sheet boundary, collision, or incompatible-gap neighborhood;
4. reopen all configurations in that neighborhood together;
5. select the best joint configuration/edge state against the frozen exterior;
6. replay the full topology validator before committing; and
7. iterate to a fixed point.

This allows a local plane to disappear, appear, or change identity when doing
so repairs a coherent sheet neighborhood. Acus measurements remain unchanged.

The implemented initialization, topology-refinement, and replay boundary is:

```bash
python3 -m backend.cubical initialize-sheet-configurations \
  --evidence /data/block-sheet-evidence-v1 \
  --factors /data/block-sheet-configuration-factors-v1 \
  --initial /data/current-materialized-selection \
  --output /data/block-sheet-configuration-init-v1

python3 -m backend.cubical replay-joint-sheet-graph \
  --evidence /data/block-sheet-evidence-v1 \
  --correspondences /data/block-sheet-mode-correspondences-v1 \
  --configurations /data/block-sheet-configuration-init-v1 \
  --cluster /data/matching-policy-root \
  --output /data/joint-sheet-graph-v1

python3 -m backend.cubical refine-sheet-topology \
  --evidence /data/block-sheet-evidence-v1 \
  --correspondences /data/block-sheet-mode-correspondences-v1 \
  --factors /data/block-sheet-configuration-factors-v1 \
  --configurations /data/block-sheet-configuration-init-v1 \
  --graph /data/joint-sheet-graph-v1 \
  --cluster /data/matching-policy-root \
  --output /data/block-sheet-topology-refinement-v1
```

The initializer optimizes the complete unary-plus-face factor graph from the
declared state, the unary optimum, and an optional synchronous max-sum loopy
belief-propagation seed. Every seed receives the same deterministic ICM polish
and is compared under the exact local factor objective. Its output is explicitly
provisional: global topology replay is mandatory. On the pilot it changed 504/3,584 stacks,
raised Acus coverage from 64.83% to 65.14%, and added 387 locally matchable face
joins. After exact topology replay, the improvement remained real: 14,160 joins
versus 13,994 for the best fixed-geometry graph, 808 versus 819 components, a
374-cell versus 286-cell largest fragment, and five versus two fragments of at
least 256 cells. The selected state added 161 supported mode patches.

The first replay measured the optimization target precisely. Of 16,713 locally
matchable joins, 14,160 survived global topology (84.72%), leaving a 2,553-join
topology tax. `refine-sheet-topology` now targets that tax directly. A seed is
a complete physical stack substitution, never one patch or dangling endpoint.
The mutable region is the union of every current sheet component containing an
old stack mode or touching a newly proposed correspondence. All joins outside
that region are fixed, while the mutable region is rebuilt from the complete
active candidate catalog under the normal hard topology constraints.

Proposal ordering uses recoverable incompatible gaps, topology-conflicted
alternatives, and the compiled local factors. Acceptance does not: it uses the
exact combined Acus/stack unary plus globally retained correspondence
objective. Component size is only a tie-break and audit statistic. A lazy match
cache reconstructs the active 38,000-correspondence graph once; subsequent
whole-block proposals on the pilot take roughly two to three seconds each.

Two alternating refinement/replay cycles on the pilot changed only 10 of 3,584
configurations relative to the initial joint state. The final exact graph has
14,193 joins rather than 14,160, 801 components rather than 808, and 17,139 open
interior endpoints rather than 17,201. Total retained correspondence benefit
rose from 156,969.08 to 157,368.22. Including the changed stack unary, the
global objective rose by 75.11. The topology tax fell from 2,553 to 2,517 and
survival rose from 84.72% to 84.94%. The largest component remains 374 cells and
the count above 256 cells remains five; this iteration improves supported
continuity without using fragment size as evidence.

Every refinement output contains the selected configuration ledger, selected
patch shard, and exact surface graph. A subsequent `replay-joint-sheet-graph`
automatically uses that graph as a declared topology-safe proposal. The
multi-restart replayer therefore cannot regress the accepted likelihood merely
because a fresh deterministic proposal is weaker, though a higher-likelihood
sheet exchange may trade several low-benefit joins for fewer stronger ones.

## Sheet-inference halo and ownership

There are two distinct halos. The raw Acus **measurement halo** is measured in
voxels, is at least the fitted needle length, and exists so every owned fit has
directionally unbiased CT support. The downstream **sheet-inference halo** is
measured in cubical cells and provides neighboring stack context around a
disjoint owned core. It does not rerun Acus or change the immutable evidence.

An expanded solve cannot simply be clipped and published. Global topology in
the halo may consume a trace or force an edge choice inside the future core.
Removing the exterior then frees those constraints without reconsidering the
remaining graph, leaving artificial fragments. The block procedure is:

1. extract an exact evidence subblock for the owned core plus its inference
   halo, preserving stable global mode and configuration IDs;
2. optimize physical stack configurations and replay topology in that expanded
   context;
3. crop patches and the selected configuration ledger to the owned core while
   retaining parent-component lineage; and
4. re-stitch topology inside the crop with those halo-selected configurations
   fixed, producing the graph that the block actually owns.

The controlled 12 × 12 × 10-core audit can be reproduced with:

```bash
python3 -m backend.cubical audit-sheet-halos \
  --evidence /data/block-sheet-evidence-v1 \
  --cluster /data/matching-policy-root \
  --core-start 2 2 2 --core-stop 14 14 12 \
  --halos 0 1 2 \
  --output /data/sheet-halo-audit-v1
```

One and two halo cells converged: only 7/1,440 owned configurations differed,
all on the outermost cell shell. After mandatory owned-core re-stitching, their
patch Jaccard was 99.60% and retained-join Jaccard was 99.61%. The zero-halo
selection appeared 81.95 objective units better when scored only inside its
artificial cut, but lost 205.57 units on the omitted cross-boundary factors and
was 123.63 units worse in the common two-cell context. One cell was only 6.80
units below the two-cell reference.

This does not establish that a halo makes the isolated core less fragmented;
the local component counts were slightly worse. It establishes that the
zero-halo solution overfits the block cut and that one cell captures nearly all
of the missing configuration context. One sheet-inference cell is therefore
the provisional default. Owned topology is rebuilt after cropping, parent
lineage is retained for later reconciliation, and topology on the outer cell
band remains provisional until adjacent blocks are merged.

## Block scaling and merge

Cells have disjoint ownership. Raw evidence contexts overlap by the configured
halo, and canonical source identities make independent blocks comparable.

Merging two solved blocks does not reopen both interiors. The merge stacks the
block contracts in global grid coordinates, enumerates cross-boundary mode
correspondences, and reopens:

- configurations in a shallow boundary band;
- sheets touching that band; and
- crossing/orientation state anchored at the cut.

The rest of each interior is summarized as immutable component occupancy,
orientation parity, and crossing ownership. If a seam proposal propagates past
the current band, the band expands by whole affected sheet components. This is
the same exact frozen-topology boundary already used by local replay, applied
to block faces rather than individual cells.

## Required audits

Every solve records at least:

- Acus evidence mass covered by the active configurations;
- non-air cells and valid modes used;
- retained versus unresolved interior trace endpoints;
- incompatible and plane-misses-face gaps separately;
- candidate alternatives blocked by occupancy, order, collision, crossing, or
  parity;
- component-size distribution and sheet collision counts; and
- flattened native-CT previews of large components.

Deep holdout validation belongs after the joint pipeline produces stable
fragments. During algorithm development, exact artifact identities, hard
topology invariants, objective deltas, and visual CT sanity checks are the
fast feedback loop.
