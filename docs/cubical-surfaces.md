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
matching to replay, it retains eight diverse, complete, conflict-free matchings
for every CT-supported corridor. Every matching must cover the patch and anchor
both boundary arcs. It is then reconstructed inside the complete source sheet;
exact edge-connected closure, no sheet split, non-increasing triangle-region
count, and at least 98% supported-area retention are hard requirements. Surface
region reduction and retained area outrank the original local factor score.
Enumeration and exact screening are immutable cached stages, so later replay or
visualization iterations do not repeat the expensive reconstructions.

On the current slab, 775,621 beam states yield 472 complete variants for the 59
CT-supported corridors. Of those, 120 exactly connect their boundary arcs and
98 pass the density-preserving surface test, exposing valid closures for 24
corridors. The locally preferred variants are distributed throughout ranks
zero through seven rather than concentrating at rank zero. Choosing one variant
per corridor locally and then replaying greedily accepts only 23: two choices in
one physical sheet compete even though a different pair of exact variants is
compatible. Exact screening takes about 266 seconds; its result is cached.

`optimize-physical-ribbon-corridor-sets` resolves that interaction before
replay. Exact variants are grouped by physical-sheet component. Complete
multi-corridor assignments are reconstructed as a single surface, scored by
actual region reduction and retained supported area, and retained as component
states. A block-wide beam then chooses one state per component under shared
interface and exact profile-crossing constraints. No individual cell or edge
can drive the decision.

The current block has 22 sheet components with exact corridor alternatives.
The optimizer retains 129 exact component states, reconstructs 24 multi-corridor
assignments, and chooses 24 mutually compatible repairs. All 24 survive the
ordinary counterfactual replay. Selected ribbons change from 37,889 to 37,883,
supported triangles from 28,116 to 28,186, and the complete edge-connected
triangle-region count from 832 to 801. All 4,307 sheet components are preserved
with zero interface collision, profile crossing, component split, or
prior-component fusion. Component/global optimization takes about 14 seconds
and final replay about 37 seconds. Flattened actual-CT previews have zero
nonadjacent chart overlap.

`extend-physical-ribbon-corridor-variants` measures search saturation without
discarding that expensive evidence. It re-enumerates a deeper complete-state
prefix, verifies every prior variant by its corridor/add/remove signature, and
copies prior exact results to their new indices. Only new ranks belonging to
previously unresolved corridors are reconstructed. The exact delta is a cached
artifact, so component and block objectives can be changed without repeating
the extension screen.

Extending the current bank from eight to 16 variants preserves all 472 prior
states and adds 276 targeted states across the 35 unresolved CT-supported
corridors. Twelve new states connect their arcs exactly, seven preserve sheet
density, and four corridors become resolvable (rows 45, 77, 111, and 119). The
targeted screen takes about 233 seconds. Joint block optimization accepts all
28 available repairs at once: selected ribbons change from 37,889 to 37,881,
supported triangles from 28,116 to 28,195, and complete edge-connected triangle
regions from 832 to 797. All component, collision, crossing, fusion, and chart
overlap invariants remain clean. The four new flattened fragments have native-CT
profile correlations of 0.918--0.988, competing-layer margins of 0.428--1.009,
and boundary-texture correlations of 0.589--0.867.

This also establishes the next bottleneck. Thirty of the remaining 31
corridors have no exact connecting state anywhere in ranks zero through 15;
one has connections that fail density preservation. Deeper permutations now
have sharply diminishing yield. The residual problem is missing physical
ribbon support, not local matching order.

`analyze-physical-ribbon-dormant-corridors` tests that missing-support claim
without permitting cell-at-a-time growth. It embeds the unchanged selected
configuration in a deeper strict continuity frontier and proves that the full
selected-node component partition is unchanged. Only complete matchings for a
previously unresolved CT corridor are considered, and each matching must add
at least one formerly unavailable bidirectional ribbon. Every state is then
reconstructed in its complete source sheet. This makes the expansion frontier
the whole multi-edge CT-supported corridor, rather than one locally plausible
cell.

The rank-15 frontier contains 218,698 candidates and 5,321,794 strict
continuation edges, adding 84,052 candidates and 3,663,205 edges without
changing any of the 4,307 selected base components. It yields 155 dormant-
supported states across 25 of the 31 residual corridors. Twelve reconstruct as
exact density-preserving connections and three corridors become resolvable
(rows 46, 67, and 88), using six previously unavailable ribbons with ray ranks
as deep as eight.

The crossing screen uses the union of the 5,321,794 expanded strict edges and
the original 3,263,301 support edges. This is essential: continuation-adjacent
ribbons are not profile-crossing conflicts merely because the topology graph is
stricter than the factor-support graph. The 6,926,506-edge union preserves the
original zero-conflict baseline and still checks every new counterfactual pair.
Joint replay retains all 28 prior repairs plus all three new repairs. The
cumulative surface has 37,882 selected ribbons, 28,211 supported triangles,
and 792 edge-connected triangle regions, versus 28,118 triangles and 830
regions in the remapped baseline. It preserves all 4,307 components with zero
new profile crossing, interface collision, component split, cross-sheet fusion,
or flattened chart overlap. The complete conditioned screen and cumulative
replay take about 243 seconds.

`build-physical-ribbon-corridor-frontier` admits the next candidate class
without turning the solve into unconstrained one-cell growth. It takes the
cumulative 31-corridor replay as immutable context, finds the 28 remaining
native-CT-supported strips, and queries the complete 1,702,134-ribbon bank only
inside those strips. Every one-sided hypothesis must pass the same patch-height,
tangent, unsigned-normal, and physical-thickness gates as a bidirectional
candidate. The union is then rebuilt with the unchanged strict-continuation and
crossing contracts. A missing reverse-ray rank receives a conservative rank-16
penalty rather than the accidental bonus a negative sentinel would otherwise
produce.

The current targeted frontier adds 4,238 unique one-sided hypotheses to the
218,698-candidate rank-15 frontier. It contains 5,585,520 strict edges; 4,222
of the new hypotheses have at least three strict neighbors and their median
support degree is 88. The cumulative 37,882-ribbon selection maps into it with
the same 4,307 component partition and zero crossing or interface conflict.
Candidate collection, topology, crossings, conditioning, and compressed
artifact writing take 20.6 seconds. The other 1,474,156 one-sided hypotheses
remain dormant.

`analyze-physical-ribbon-one-sided-corridors` keeps the complete strip as the
decision unit. It enumerates only matchings that add at least one explicitly
targeted one-sided ribbon, reconstructs every matching in its complete source
sheet, and jointly optimizes exact states within components and across the
block before cumulative replay. Thus an individual ribbon can contribute to a
repair but cannot independently grow a tendril.

The pilot enumerates 193 such states across 26 of the 28 residual corridors.
Forty-three connect exactly, 42 retain sheet density, and 11 distinct corridors
survive component/global optimization and replay (rows 29, 35, 38, 40, 54, 80,
87, 100, 106, 107, and 124). Cumulative resolved corridors rise from 31 to 42.
The selection adds 75 ribbons and removes 35 alternatives for a net change from
37,882 to 37,922. Supported triangles rise from 28,211 to 28,323 and true
edge-connected triangle regions fall from 792 to 776. Two prior components
disappear, but they contain only three ribbons and one ribbon; neither reaches
the 32-ribbon minimum for a reconstructable surface. No surface component is
lost or split, and the replay has zero interface collision, profile crossing,
cross-sheet fusion, or flattened chart overlap.

The accepted strips have native-CT profile correlations of 0.901--0.987,
competing-layer margins of 0.427--1.014, and boundary-texture correlations of
0.367--0.716. The selected ruled/Hermite models include minimum local curvature
radii down to 0.105 sheet thicknesses, so the candidate expansion materially
improves support for the observed hairpin geometry rather than merely filling
flat holes. Exact reconstruction dominates the 382-second solve; the exact
bank is checkpointed separately from later objective and preview iterations.

`materialize-physical-ribbon-replay-configuration` turns a cumulative exact
replay into the ordinary continuity/configuration contract consumed by the
rest of the pipeline. It preserves broad support continuity separately from
strict component topology: in the 42-event snapshot, 264,084 selected broad
support edges prevent legitimate neighbors from being misclassified as
profile crossings, while 139,520 strict edges define the 4,305 sheet
components. Materialization preserves all 37,922 selected ribbons and reports
zero interface or crossing conflict. It takes about four seconds and does not
reoptimize the selection.

Running the unchanged patch-corridor census on that materialized state changes
the residual problem from the stale 128-corridor catalog to 95 physical
corridors, of which 22 have native-CT evidence. Seventeen ordinary frontier
trials connect none of them. A new targeted frontier sees 3,292 unique
one-sided hypotheses across those complete strips; 2,504 were already present
and 788 are genuinely new. The resulting 223,724-node frontier has 5,639,708
strict continuation edges, preserves the exact baseline component partition,
and has zero inherited crossing debt.

Exact screening evaluates 137 one-sided states across 19 of the 22 CT strips.
Only row 93 connects and remains density eligible. It adds seven ribbons,
removes four, and therefore moves the cumulative state from 37,922 to 37,925
ribbons. Supported triangles rise from 28,323 to 28,332, edge-connected
triangle regions fall by two, and all 4,305 components remain. The accepted
strip has 0.884 context-profile correlation, 0.661 competing-layer margin,
0.363 boundary-trace correlation, 0.944 patch coverage, and a minimum modeled
curvature radius of 0.218 sheet thicknesses. Its 166-ribbon/256-triangle
flattened CT chart has no nonadjacent overlap. The exact pass takes 312 seconds.

The next recensus produces 92 corridors and 21 CT-eligible strips. Its target
frontier adds zero candidates. `assess-physical-ribbon-corridor-saturation`
then hashes exact CT patch positions, normals, and thicknesses across
iterations. All 21 residual patches match prior exact failures; their candidate
banks and the complete strict edge graph are identical. The accepted change
touches only topology component 24, and no residual corridor belongs to that
component. The audit therefore proves a fixed point for this candidate class
and skips a redundant exact pass. Across the original and refreshed catalogs,
the cumulative state contains 43 accepted corridor events. Further progress
requires a different missing-geometry representation rather than deeper
enumeration of the same one-sided ribbons.

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

python -m backend.cubical optimize-physical-ribbon-corridor-sets \
  --variants work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-sets-v1 \
  --settings-json examples/physical-ribbon-corridor-sets.json

python -m backend.cubical extend-physical-ribbon-corridor-variants \
  --variants work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-16-v1 \
  --settings-json examples/physical-ribbon-corridor-extension.json

python -m backend.cubical optimize-physical-ribbon-corridor-sets \
  --variants work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-16-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-sets-16-v1 \
  --settings-json examples/physical-ribbon-corridor-sets.json

python -m backend.cubical solve-physical-ribbon-continuity \
  --ribbons work/multiseam-2x2-b00c03c/physical-ribbon-bank-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-continuity-rank15-v1 \
  --settings-json examples/physical-ribbon-continuity-rank15.json

python -m backend.cubical analyze-physical-ribbon-dormant-corridors \
  --corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-v1 \
  --variants work/multiseam-2x2-b00c03c/physical-ribbon-corridor-variants-16-v1 \
  --corridor-sets work/multiseam-2x2-b00c03c/physical-ribbon-corridor-sets-16-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --expanded-continuity work/multiseam-2x2-b00c03c/physical-ribbon-continuity-rank15-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-dormant-corridors-v1 \
  --settings-json examples/physical-ribbon-dormant-corridors.json

python -m backend.cubical build-physical-ribbon-corridor-frontier \
  --corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-v1 \
  --prior-replay work/multiseam-2x2-b00c03c/physical-ribbon-dormant-corridors-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-configuration-multiscale-v1 \
  --bidirectional-continuity work/multiseam-2x2-b00c03c/physical-ribbon-continuity-rank15-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-v1 \
  --settings-json examples/physical-ribbon-corridor-frontier.json

python -m backend.cubical analyze-physical-ribbon-one-sided-corridors \
  --frontier work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-v1 \
  --settings-json examples/physical-ribbon-one-sided-corridors.json

python -m backend.cubical materialize-physical-ribbon-replay-configuration \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-one-sided-v1

python -m backend.cubical analyze-physical-ribbon-patch-corridors \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-one-sided-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration1-v1 \
  --settings-json examples/physical-ribbon-patch-corridors.json

python -m backend.cubical build-physical-ribbon-corridor-frontier \
  --corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration1-v1 \
  --prior-replay work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration1-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-one-sided-v1 \
  --bidirectional-continuity work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-one-sided-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-iteration1-v1 \
  --settings-json examples/physical-ribbon-corridor-frontier.json

python -m backend.cubical analyze-physical-ribbon-one-sided-corridors \
  --frontier work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-iteration1-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-iteration1-v1 \
  --settings-json examples/physical-ribbon-one-sided-corridors.json

python -m backend.cubical materialize-physical-ribbon-replay-configuration \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-iteration1-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-iteration2-v1

python -m backend.cubical analyze-physical-ribbon-patch-corridors \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-iteration2-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration2-v1 \
  --settings-json examples/physical-ribbon-patch-corridors.json

python -m backend.cubical build-physical-ribbon-corridor-frontier \
  --corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration2-v1 \
  --prior-replay work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration2-v1 \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-iteration2-v1 \
  --bidirectional-continuity work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-iteration2-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-iteration2-v1 \
  --settings-json examples/physical-ribbon-corridor-frontier.json

python -m backend.cubical assess-physical-ribbon-corridor-saturation \
  --prior-corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration1-v1 \
  --prior-frontier work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-iteration1-v1 \
  --prior-replay work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-iteration1-v1 \
  --current-corridors work/multiseam-2x2-b00c03c/physical-ribbon-patch-corridors-iteration2-v1 \
  --current-frontier work/multiseam-2x2-b00c03c/physical-ribbon-corridor-frontier-iteration2-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-saturation-v1
```
