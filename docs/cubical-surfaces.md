# Cubical surface reconstruction

This pipeline reconstructs locally planar interfaces in an axis-aligned cubical
cell complex. It is independent of a particular scroll, voxel resolution, Acus
parameterization, or slab. Papyrus-specific evidence supplies local hypotheses;
the geometry and composition layers operate only on their declared contracts.

## Invariants

- A grid has an XYZ origin, anisotropic XYZ cell size, and finite XYZ cell
  shape. Coordinates in artifacts are expressed in a declared world unit.
- A cell may contain zero or more mutually exclusive configurations. A
  configuration contains an ordered collection of locally planar patches.
- Plane normals and fiber directions are axial: negating either direction does
  not change the observation. A plane is stored in a deterministic gauge.
- A plane is expressed by its canonical normal and signed height from the cell
  center. This avoids large global offsets and makes a patch invariant to a
  translation of the dataset.
- Local uncertainty is a 3 x 3 covariance over two tangent-space normal tilts
  and plane height. Edge-crossing distributions are correlated consequences of
  this plane posterior; they are not independently fitted points.
- A generic plane/cube intersection is a cyclic convex polygon with three to
  six vertices. Every vertex lies on a canonical global grid edge, and every
  polygon side is a trace on a canonical global grid face.
- Adjacent cells can join patches only through the exact shared grid face.
  Matching preserves the ordered trace sequence and never assigns an absolute
  normal side. Endpoints normally share a global edge. Two adjacent edges may
  instead meet through their shared grid vertex when both posterior crossing
  distributions support that corner and each observation lies in the half of
  its edge owned by that nearest endpoint. Covariance may soften the residual;
  it may not move a crossing past the edge midpoint to a distant corner.
- Up to four incident cells can observe one physical crossing of a global grid
  edge. Accepted face joins weld those observations into one latent crossing.
- Pairwise-compatible corner transitions are processed in descending evidence
  order. A join is deferred if its transitive crossing group has no common edge
  or vertex, or if it would put two locally planar patches from one cell in one
  surface component.
- Retained cycles must satisfy two independent binary gauges: polygon loops
  remain orientable, and explicit strict/quarter-turn fiber relations are
  path-independent. Fiber direction itself remains unsigned.
- A macro block retains both its exterior trace graph and the mapping from each
  trace to an interior surface component. Boundary geometry alone is not a
  sufficient hierarchical summary.

Degenerate cases in which a fitted plane exactly contains a cube vertex or edge
are explicit, rather than resolved by a hidden floating-point tie break. A real
posterior can be sampled or perturbed into generic alternatives. A surface that
cannot be represented by one disk-like plane patch in a cell must use multiple
patches or trigger adaptive subdivision.

## Artifact layers

The scalable representation is a structure of arrays, sharded by regular macro
blocks. Human-readable objects are reference views used by tests and debugging;
they are not the volume-scale storage model.

Each patch shard contains these logical arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `cell_xyz` | `P x 3` | Owning cell |
| `configuration_id` | `P` | Mutually exclusive local configuration |
| `local_order` | `P` | Gauge-relative layer order in that configuration |
| `normal_xyz` | `P x 3` | Canonical unsigned plane normal |
| `height` | `P` | Height from cell center in world units |
| `plane_covariance` | `P x 6` | Packed symmetric tilt/tilt/height covariance |
| `fiber_xyz` | `P x 3` | Canonical unsigned fiber direction, or NaNs |
| `confidence` | `P` | Evidence confidence, not sheet probability |
| `vertex_offset` | `P + 1` | Ragged clipped-polygon offsets |
| `vertex_edge_axis` | `V` | Global edge axis |
| `vertex_edge_anchor` | `V x 3` | Global edge integer anchor |
| `vertex_t` | `V` | Crossing coordinate on that edge |
| `vertex_variance` | `V` | Marginal variance induced by the plane posterior |

Derived face traces and block joins are separately versioned caches. Native
evidence provenance and pipeline settings live in the shard manifest. Changing
a local evidence model therefore invalidates only the affected shards and their
ancestor block summaries.

## Processing stages

1. The raw Acus adapter derives finite-support needles directly from native CT,
   retains normal-by-depth-by-unsigned-orientation likelihoods and CT material
   profiles, and proposes several physical stratigraphic configurations per
   cell. Synthetic generators exercise the same downstream interface.
2. Plane hypotheses are clipped into cell patches and invalid geometry is
   rejected deterministically.
3. Shared-face trace sequences are aligned with uncertainty-normalized costs.
4. Accepted joins weld global-edge crossings and form polygonal components.
5. Regular blocks cache internal components and exterior trace graphs.
6. A completed block serializes a narrow outer band, immutable interior
   component occupancy, crossing-feature certificates, exterior traces, and
   retained physical alternatives. World-adjacent blocks can either establish
   a conservative shared-face component forest or jointly reselect their
   meeting bands without reopening either private interior.
7. Papyrus-specific sheet packets add thickness, paired-ply, fiber, and air-gap
   factors without changing the geometric representation.

The first packet layer is now implemented as a separate dual-axis connectivity
artifact. It fixes every retained single-ply join, admits only well-supported
parallel-to-orthogonal fiber-frame continuations, and reruns the same global
collision, crossing, and orientability checks for additions. The underlying
patch shard and strict graph remain immutable. Thickness and air-gap factors
remain later packet refinements rather than implicit properties of this graph.

## Boundary-band composition

`export-boundary-band` is the block-local handoff. The default two-cell shell
contains selected clipped patches, packet-component ownership, joins incident
to the shell, exact exterior trace endpoints, and the physical configuration
bank for every shell cell. It also stores one layer of clipped anchor patches
at each inner cut and compact face-specific certificates derived from the
immutable interior:

- every boundary-touching component's complete occupied-cell set, so a
  transitive merge cannot silently put two layers from one cell into one
  component;
- the orientation parity between its cut-anchor patches, so axial local normals
  cannot close an inconsistent global polygon loop; and
- the existing welded edge-or-vertex class and deep patch owner of every
  cut-anchor endpoint, so a seam cannot introduce an impossible crossing
  cycle after the private geometry has been discarded.

Component and patch IDs are namespaced by input during composition. Blocks are
located by their world-space grid origins, not by assuming their local integer
coordinates or IDs are globally unique. Parallel-fiber seam matching uses the
ordinary strict policy. Orthogonal-fiber candidates are a separate packet
addition and alone receive the configured 15-degree normal and fiber-frame
caps.

`merge-boundary-bands` aligns complete trace sequences on each shared unit
face. It serializes all supported alternatives and retains one representative
per component pair as a cell-collision-safe, crossing-feature-safe forest.
The forest is automatically orientable; redundant matches remain evidence
rather than being promoted to unconstrained topology cycles. Neither input's
interior geometry or configuration selection is changed.

`reselect-boundary-bands` is the refinement path. It forms a slab containing
the `d` mutable shell layers from each input plus one immutable anchor layer on
each outside edge. A warm-started conditional configuration solve can change
only the `2d` shell layers. Topology is then reconstructed in two stages: all
ordinary parallel-fiber joins are selected first, and that strict graph is
fixed while separately gated quarter-turn packet joins are considered. The
frozen occupancy, crossing, and orientation certificates participate in every
veto, so the result has the same global invariants as direct assembly without
reading native CT or either block's private interior.

Pairwise slabs are not independently composable at a corner: the X solve and
Y solve can legitimately choose different configurations for the same cell,
and their retained joins can induce different transitive partitions. The
exporter therefore also stores frozen-region certificates for every compatible
face mask. A regular 2 x 2 x 2 child has at most one low or high internal face
per axis, giving exactly `3^3 - 1 = 26` nonempty masks.

`reselect-boundary-cluster` is the hierarchical composition primitive. It
validates a complete Cartesian child layout, removes the union of every
participating face band from each child, and loads the certificate built for
that exact multi-face region. The active configuration graph is sparse: it
contains the mutable band union and one immutable cut shell, not the full
cluster cuboid. One conditional solve chooses every physical cell once; one
strict-then-quarter-turn topology solve then sees all X, Y, and Z seams and
their intersections together. A two-block cluster reduces to the same bounded
problem as pairwise reselection, while four- and eight-block clusters avoid
pairwise corner or edge contradictions by construction.

`audit-multiseam` measures the failure of a pairwise seam network and can
compare it with a joint cluster result. `audit-boundary-cluster-reference`
compares independent children and their cluster solve with an unsplit
full-context reconstruction by physical geometry, retained joins, and induced
component partitions. The unsplit result is explicitly a consistency
reference rather than ground truth.

The cluster solve remains compact by design and therefore is not itself a full
surface artifact. `materialize-boundary-cluster` is the lossless expansion
boundary. It takes the selected mutable patches and joins from the cluster,
then restores every child patch and retained join whose endpoints lie outside
all participating internal bands. Replaying the complete declared graph through
the ordinary collision, crossing, and orientation selector must reproduce the
cluster's certified component count exactly; any rejection is an artifact error,
not permission to silently alter connectivity. The output owns a standard
`selected-patches-v1` shard, a hashed `surface-graph-v1` artifact, and per-patch
source provenance. That complete graph is suitable for measurement, flattening,
and export into the next composition level.

For a rectangular `X x Y x Z` block and shell depth `d`, selected shell state
scales with
`XYZ - (X - 2d)(Y - 2d)(Z - 2d)`, or `O(dN^2)` for an `N^3` block. Component
occupancy certificates may reference deeper cells, but they contain integer
cell identities rather than voxels or patch geometry.

The deterministic real-block split audit provides an exact regression target
for the contract. Splitting the current 16 x 16 x 14 selected result at X=8,
partitioning its complete physical candidate bank, and jointly reselecting the
two-cell bands recovers every one of the full graph's 13,287 retained joins and
all 977 components. The solve reads 896 mutable shell cells, 665 immutable
anchor patches, and compact frozen topology certificates; it does not read the
private child interiors.

A separate test reruns each 8 x 16 x 14 half independently from native CT,
sharing only an explicitly hashed source calibration. Selected-only seam
bridging gives 987 components. Joint band reselection changes 37 of 896 cells
and gives 978 components, versus 977 in the full-context consistency reference.
Layer-count agreement rises from 866 to 884 cells. Fifteen changed cells become
exactly consistent with the full-context configuration and none become less
consistent. This does not prove the reference is physically correct, but it
shows that bounded neighbor context resolves real independent-boundary effects
in the expected direction rather than merely replaying a deterministic split.

The four-child test independently reruns four 8 x 8 x 14 blocks. Four pairwise
seam solves disagree on 23 of their 224 shared corner cells and on every
overlapping component partition, demonstrating why pairwise union is invalid.
The joint cluster covers 1,568 mutable cells plus 616 immutable shell cells,
changes 133 configurations, and finishes in 48.9 seconds. It resolves every
conflict to one locally supported alternative. Relative to the unsplit block,
30 changes move to an exact reference configuration and three move away; at
the 224 corner cells the split is six toward and zero away. Of the retained
joins whose endpoints map geometrically to the reference, 2,426 agree, 19 are
cluster-only, and 21 are reference-only (98.38% Jaccard). Both graphs have 977
components, while mapped co-component precision and recall are 90.74% and
90.88%, so equal component count is not overstated as identical fragmentation.

The geometry stages are validated independently on analytic surfaces. The
native-CT implementation and its measured pilot are documented in
[`raw-acus-cubical-pipeline.md`](raw-acus-cubical-pipeline.md).

## Label-free physical ribbons

The current dense-CT path does not use a propagated component identity to
discover or connect papyrus. It begins with signed air-to-material interfaces
and treats a local papyrus observation as a material ribbon bounded by two
opposing faces. Sheet identity is an output of the geometric configuration
solve, not an input constraint.

`build-physical-ribbon-bank` casts inward from every dense signed interface
over the configured physical thickness interval. It retains every opposing,
inward-facing boundary pair and records mutual first hits separately from later
ray alternatives. On the current 384 x 384 x 320 native-voxel pilot, 360,545
interfaces produce 1,702,134 explicit ribbon alternatives in 15.1 seconds;
282,276 interfaces participate in at least one alternative and 10,764 pairs
are mutual first hits. No candidate is owned at this stage.

`solve-physical-ribbon-continuity` connects candidates only when both boundary
faces translate tangentially together, thickness and unsigned normal remain
continuous, and neighbor directions span a local two-dimensional tangent
plane. One observed interface can bound at most one ribbon in an explicit
solution. The broad pilot evaluates 134,646 bidirectional candidates and
1,658,589 compatible continuation edges in 5.5 seconds.

`optimize-physical-ribbon-configuration` adds exact physical conflicts. It
rasterizes candidate thickness profiles, verifies their closest interior
approach geometrically, and forbids intersecting profiles. A local factor solve
then trades alternatives using boundary evidence, inward-ray rank, and
simultaneous two-face continuation. Rejected alternatives remain in the bank.
Support and topology can be separate immutable continuity artifacts: the wider
graph contributes votes to the configuration objective, while only the strict
graph defines component identity. This prevents one permissive support edge
from silently fusing otherwise distinct sheets. The current two-scale pilot
selects 37,889 ribbons and 75,778 distinct faces with zero crossings or reused
interfaces. The strict graph forms 4,307 components.

`analyze-physical-ribbon-patch-holes` replaces candidate-at-a-time gap growth
with a surface-level diagnostic. It integrates every eligible strict component
into an intrinsic chart, triangulates only supported edges, and extracts closed
interior boundary loops. A loop is the decision unit. Affine and regularized
quadratic patches are fitted from its complete boundary plus a two-hop context,
then sampled directly from native CT. The predicted layer must reproduce the
context's air-material-air profile and beat copies translated along the page
normal by half and one local ribbon thickness.

Candidate repair is an alternating interface re-pairing, not an additive cell
vote. Every geometrically compatible ribbon and every incumbent that owns one
of its interfaces enter one local factor graph. Shared interfaces and exact
profile crossings are hard exclusions; strict continuation and support from
fixed neighbors are pair factors. The command rebuilds components, charts, and
triangles counterfactually but never mutates its source configuration. On the
current slab, 34 closed holes contain three multi-ribbon gaps. The two gaps
with 0.97--0.99 context-profile correlation close after joint re-pairing. The
weaker 0.62-correlation proposal remains open and is therefore not mistaken for
a successful repair. The replay adds 14 supported triangles while preserving
all 4,307 component identities, with no interface collision, crossing, or
cross-component fusion.

`analyze-physical-ribbon-patch-corridors` addresses open surface islands and
missing bands, where no closed hole exists yet. It does not grow from one cell.
For every surface boundary edge it retains the three nearest reciprocal facing
alternatives, then resolves those ambiguous pairs as order-preserving boundary
arc alignments. Each corridor has at least three anchors, no crossing pairing,
and at least 50% anchor density along both arcs. Ruled and cubic-Hermite strips
compete so a tight physical bend can satisfy both endpoint tangent planes.

The complete strip is sampled from native CT. Its air-material-air profile must
agree on both sides, beat copies translated along the page normal, and preserve
high-pass boundary texture across the gap. Passing strips drive complete
alternating interface re-pairings under the same interface-exclusivity,
profile-crossing, and strict-continuation factors as the main configuration.
The global result is rebuilt counterfactually. A repair survives only when at
least half of each original boundary arc enters the same edge-connected
triangle region; sharing a vertex is explicitly insufficient. Competing repairs
within one sheet are then enumerated locally, keeping only a subset that does
not increase its triangle-region count or discard more than 2% of supported
surface area.

On the current slab, 7,992 outer boundary edges produce 3,162 reciprocal ranked
pairs and 141 dense monotone corridors. Native CT retains 59 of the 128 screened
corridors; 50 complete re-pairings reach trial replay, while exact arc closure
rejects 42 and the density audit rejects one more. The seven surviving repairs
add four net ribbons and 19 supported triangles, reduce triangle regions from
767 to 760, and retain all 4,307 physical-sheet components with zero interface
collision, profile crossing, or prior-component fusion. Flattened actual-CT
views of every accepted component are written alongside the corridor evidence
montage.

`analyze-physical-ribbon-corridor-variants` removes one remaining local-choice
assumption from that result. Rather than sending only the highest-factor-score
matching to replay, it retains four diverse, complete, conflict-free matchings
for every CT-supported corridor. Every matching must cover the patch and anchor
both boundary arcs. It is then reconstructed inside the complete source sheet;
exact edge-connected closure, no sheet split, non-increasing triangle-region
count, and at least 98% supported-area retention are hard requirements. Surface
region reduction and retained area outrank the original local factor score.
Enumeration and exact screening are immutable cached stages, so later replay or
visualization iterations do not repeat the expensive reconstructions.

On the current slab, 775,621 beam states yield 236 complete variants for the 59
CT-supported corridors. Sixty variants exactly connect their boundary arcs and
48 pass the density-preserving surface test, exposing valid closures for 20
corridors. Only two of the 20 selected variants were locally rank zero. The
seven original repairs are all retained, while 13 additional repairs were
hidden by the single-best local matching. Global replay accepts all 20 at once:
selected ribbons change from 37,889 to 37,887, supported triangles from 28,116
to 28,175, and triangle regions from 767 to 737. All 4,307 sheet components are
preserved with zero interface collision, profile crossing, component split, or
prior-component fusion. Exact screening takes about 135 seconds and the global
replay about 36 seconds on this block.

The reproducible commands are:

```bash
python -m backend.cubical build-physical-ribbon-bank \
  --interfaces work/multiseam-2x2-b00c03c/one-sided-interface-bank-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-bank-v1

python -m backend.cubical solve-physical-ribbon-continuity \
  --ribbons work/multiseam-2x2-b00c03c/physical-ribbon-bank-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-continuity-broad-v1 \
  --settings-json examples/physical-ribbon-continuity-broad.json

python -m backend.cubical optimize-physical-ribbon-configuration \
  --continuity work/multiseam-2x2-b00c03c/physical-ribbon-continuity-bridge-search-v1 \
  --topology-continuity work/multiseam-2x2-b00c03c/physical-ribbon-continuity-broad-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1

python -m backend.cubical analyze-physical-ribbon-patch-holes \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-v1 \
  --settings-json examples/physical-ribbon-patch-holes.json

python -m backend.cubical analyze-physical-ribbon-patch-corridors \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-v1 \
  --settings-json examples/physical-ribbon-patch-corridors.json

python -m backend.cubical analyze-physical-ribbon-corridor-variants \
  --corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-v1 \
  --settings-json examples/physical-ribbon-corridor-variants.json
```
