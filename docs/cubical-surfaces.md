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

`analyze-physical-ribbon-corridor-deficits` reconstructs the strongest failed
complete matching in each residual CT strip and measures the selected nodes and
triangles against the fitted native-CT patch. Sixteen of the 21 strips have a
non-conflicting, component-preserving state that can be reconstructed. Their
median triangle coverage within half a local papyrus thickness is 1.0, median
central unsupported fraction is zero, and median separation between the two
dominant triangle islands is only 0.372 sheet thicknesses. Only row 90 is a
clear sparse-support outlier, at 0.787 coverage and 1.425 thicknesses of
separation. The residual class is therefore mostly not missing Acus samples.

The surface contract was the actual bottleneck. Sheet identity and mesh-face
support had shared one strict graph: a chart-Delaunay triangle was retained only
when all three of its edges were already strict continuation edges. That is a
valid conservative identity graph but an unnecessarily incomplete mesh graph.
`analyze-physical-ribbon-corridor-faces` separates the two roles. Strict ribbon
edges still define components. A supplemental face may only come from the
existing component's chart-Delaunay tessellation, and its center must agree
with the complete native-CT corridor patch in thickness-normalized distance,
height, tangent displacement, normal, and edge length. The solver selects a
minimum dual-face path between the two boundary-arc triangle regions; those
faces are never promoted to topology edges.

This representation repairs 14 of the 16 reconstructable failures. Each path
uses one to three faces. The accepted face centers lie at most 0.083 sheet
thicknesses from the CT model; maximum edge length is 1.204 local thicknesses.
The tight-bend cases explain the old false negatives directly: one valid face
has a 47.7-degree residual against its Acus node normals, just beyond the old
45-degree fixed gate, while its center is 0.012 thicknesses from CT. The sparse
row 90 and one other corridor remain rejected. Candidate tessellation is
restricted to the strict component containing the two arcs, so the operation
scales with the affected sheet rather than all 4,305 block components.

`replay-physical-ribbon-corridor-faces` jointly selects the resulting ribbon
assignments under the original interface and crossing conflicts, rebuilds the
chosen strict sheets once, and then recomputes all face paths in their shared
final charts. It starts from the cumulative 37,925-ribbon exact replay, thereby
preserving the previously accepted row-93 repair. Thirteen of 14 face repairs
coexist; rows 3 and 72 compete for one interface assignment, and the global
objective keeps the lower-face-debt row-3 state. The cumulative selection adds
80 ribbons and removes 34 for 37,971 total. Strict triangles rise from 28,332
to 28,412. Nineteen separately marked faces are sufficient to prove all 13
corridor connections. Growing the attached, physically eligible Delaunay
closure adds 112 marked CT faces in total, producing 28,524 output triangles
and reducing edge-connected triangle regions from 769 to 755.

The cumulative artifact retains all 4,305 strict components with zero
interface conflict, profile crossing, prior-component split, or cross-sheet
fusion. All mesh edges have at most two incident triangles. The minimum strict
surface-area retention over the 11 affected components is 0.999; the remaining
components gain area. The attached closure reduces interior boundary holes
from 30 to 27 and macro holes from three to two. Flattened native-CT previews
contain no nonadjacent chart overlap. Deficit analysis takes 59 seconds,
independent face screening about 85 seconds, and the cumulative replay plus
eight flattened previews about 123 seconds on the current CPU implementation.

The next residual audit removes candidate provenance from the decision. The
one-sided experiment had only exact-screened complete matchings that contained
a newly admitted one-sided ribbon, even when a complete bidirectional matching
already existed in the same native-CT strip. That filter is useful for testing
a frontier mechanism but is not a physical property of papyrus.
`analyze-physical-ribbon-complete-strips` starts from the cumulative
37,971-ribbon face replay, enumerates up to 16 complete both-arc assignments for
each of its eight unresolved CT strips, and screens all of them against the
whole strip. Interface conflicts, crossing conflicts, and inherited-component
splits are measured before any mesh completion.

The audit evaluates 128 assignments. Sixty-seven preserve their inherited
strict component and 16 also admit a physical CT-face path. Four alternatives
repair row 72 and twelve repair row 81. Row 81 is a useful tight-bend case: its
selected CT model reaches a minimum curvature radius of 0.490 local sheet
thicknesses while retaining 0.883 context-profile correlation, 0.352 boundary
trace correlation, and a 0.645 competing-layer margin. Rows 25, 83, and 84
instead split every complete assignment, detaching respectively 20--21, nine,
and six--seven inherited ribbons. Row 51 has only three component-preserving
states and no physical face path. Row 87 has physical paths but loses at least
2.2 percent of strict supported area, while sparse row 90 still has no path.
These are now explicit structural classes rather than missing-candidate counts.

`replay-physical-ribbon-complete-strips` groups alternatives by CT strip,
chooses at most one state per strip under the exact interface and crossing
constraints, and then rebuilds the cumulative charts. Every earlier face path
and prior exact corridor is recomputed jointly before the state can commit.
Rows 72 and 81 coexist on the first exact state: ten ribbons replace ten,
all 4,305 strict components remain, and all 14 previous corridor repairs are
preserved. Strict triangles rise from 28,412 to 28,414; the attached physical
closure grows from 112 to 130 faces, yielding 28,544 triangles and reducing
edge-connected triangle regions from 755 to 753. The minimum affected strict
area retention is 0.999985, and the row-81 component gains 5.9 percent area.
The mesh remains edge-manifold with no interface conflict, crossing, inherited
split, cross-sheet fusion, or flattened chart overlap. Macro holes remain at
two; the boundary audit exposes one additional small interior loop (27 to 28),
which remains a target rather than being hidden by the corridor success count.
Complete-strip screening takes 274 seconds and cumulative replay with both the
global and new-only flattened native-CT montages takes 100 seconds on CPU.

The apparent failure of rows 25, 51, 83, and 84 was an objective failure, not
a missing-Acus-mode failure.  Their highest-ranked complete assignments remove
articulation ribbons from an inherited component; the surface screen therefore
sees plausible local strips that detach between six and 31 existing ribbons.
`analyze-physical-ribbon-lineage-strips` scans the retained 16,384-state factor
beam with connectivity as a hard constraint.  Every inherited component touched
by an assignment must remain nonempty and connected, including components that
are not the target sheet.  The latter clause matters: 2,659 otherwise competitive
row-25 states delete a neighboring one-ribbon component.  The first 16 valid
states occur by beam rank 3,962 for row 25, 85 for row 51, 779 for row 83, and
2,570 for row 84.

All four strips remain recoverable under that stronger contract.  The audit
retains 64 lineage-safe complete assignments, of which 44 pass the native-CT
surface screen: 16 for row 25, one for row 51, 16 for row 83, and 11 for row 84.
Row 51 is already connected by strict triangles and therefore needs zero
supplemental faces.  Cumulative replay explicitly accepts that zero-face success
instead of requiring an unnecessary Delaunay closure.

The replay also distinguishes provisional fragments from established sheet
identity.  A CT-backed strip may absorb a component below the existing
`minimum_component_ribbon_count` chart threshold, but it may not fuse two
components that independently meet that threshold.  This is a scale-derived
rule rather than a row-specific exception.  In the selected row-25 state the
300-ribbon sheet absorbs one isolated ribbon; no inherited component is deleted
or split, no orphan component is created, and no two substantial sheets fuse.

`replay-physical-ribbon-lineage-strips` commits all four repairs in its first
exact state.  Thirty-two ribbons replace 18, strict triangles rise from 28,414
to 28,451, and 169 cumulative CT faces produce 28,620 total triangles.  All 20
required prior and new corridor connections survive.  Edge-connected triangle
regions fall from 753 to 746 and interior holes from 28 to 27, while the two
macro holes remain explicit.  The mesh has no interface conflict, profile
crossing, inherited split, deleted component, orphan component, substantial
sheet fusion, non-manifold edge, or nonadjacent flattened-chart overlap.  The
minimum affected strict area retention is 0.994591.  Lineage enumeration and
physical screening take 88 seconds; cumulative exact replay and four new native-
CT flattenings take 102 seconds on the current CPU implementation.

The two remaining CT-supported strips exposed a different search bias.  The
lineage scan had stopped as soon as it found 16 valid states in factor-objective
order.  Row 90 actually has 853 lineage-safe states in the retained 16,384-state
beam, but the early states all cover only about 45 percent of the complete CT
strip and cannot form a face path.  The lineage enumerator now scans the whole
beam and reserves half of its unchanged 16-state exact-screen budget for
whole-strip-coverage-priority states.  This is still a complete multi-cell strip
decision: no ribbon, cell, or frontier endpoint can grow independently.  The
best row-90 states cover 55.9 percent of the dense patch; three admit an exact
three-face CT path.  Cumulative replay selects five added and three removed
ribbons, preserves every earlier repair, and moves the global state to 37,987
ribbons, 28,453 strict triangles, 173 supplemental CT faces, and 745
edge-connected triangle regions.

Row 87 is the observed hairpin with a modeled minimum curvature radius of only
0.182 local sheet thicknesses.  Its factor beam contains 14,151 lineage-safe
states and the coverage-priority states reach the whole strip, but a strict-only
area test rejects every one: the best retains 97.56 percent before physical face
completion.  That test was accounting for the conservative identity mesh while
ignoring the native-CT-gated closure that is actually written to the surface.
The screen now distinguishes those roles explicitly.  At least 95 percent of
the strict preclosure area must survive, and the completed CT-supported surface
must retain at least 98 percent.  Final replay repeats the same audit against the
complete prior and final augmented meshes, so an overlapping, duplicated, or
lost earlier CT face cannot inflate the result.

The selected row-87 state covers 100 percent of its fitted strip, adds six
ribbons, removes four, retains 97.563 percent of strict area, and retains 108.557
percent of the prior augmented area after five attached CT faces are rebuilt.
It has no interface conflict, profile crossing, inherited split, deleted or
orphan component, substantial-component fusion, non-manifold edge, or
nonadjacent flattened-chart overlap.  The final cumulative state has 37,989
ribbons, 28,455 strict triangles, 178 CT faces, 28,633 total triangles, and 744
edge-connected triangle regions.  All 22 native-CT-supported corridors in this
census are connected.  Interior holes remain at 27 and macro holes at two;
those are preserved as the next refreshed-census targets rather than hidden by
the completed-strip count.

The cumulative replay loader accepts both complete-strip and lineage-strip
replays, and `--corridor-row` may be repeated to target explicit rows.  This
makes the same immutable factor graph, CT evidence, lineage audit, and exact
replay usable for successive repairs rather than requiring a one-off script.
The two coverage-aware audits take 55 and 44 seconds respectively; each exact
cumulative replay takes about 103 seconds on the current CPU implementation.

The next repair generation removes the last implicit single-boundary-cycle
assumption.  A boundary graph can contain several triangle fans that touch at
one ribbon vertex.  Treating that pinched graph as one failed cycle had silently
discarded 137 complexes.  The boundary tracer now pairs incident boundary edges
inside each triangle-link fan before tracing loops.  All 137 complexes resolve:
the refreshed surface has no unresolved boundary fan and exposes 122 genuine
interior loops, including 48 macro holes, rather than deleting difficult
geometry from the census.

`replay-physical-ribbon-cumulative-holes` makes a closed loop the smallest
admissible mutation.  It optimizes complete patch-covering re-pairings, then
rebuilds every inherited CT connection, chart, triangle, boundary loop, and
manifold edge in one exact state.  When a dense multi-hole proposal leaves some
of its target loops open, the next state removes all observed counterexamples
at once instead of trying one ribbon or one cell.  Two saturation passes close
seven macro holes; two later refreshed passes close one each, first by
exchanging five ribbons for five and then by replacing three with seven.  Every
accepted pass preserves inherited components and CT connections with zero
interface conflict, profile crossing, or non-manifold edge.

`replay-physical-ribbon-cumulative-corridors` applies the same contract to open
multi-edge strips.  The assignment beam may select only one complete matching
per CT corridor and may not remove the final selected anchor of any inherited
component.  Exact replay then rejects component splits, substantial-component
fusion, incomplete provisional-fragment replacement, deleted components,
failed inherited connections, and surface-area regression.  This prevents a
high-scoring strip from erasing a singleton fragment or creating a thin local
tendril while still allowing a fully supported provisional fragment to be
absorbed.

The first fan-aware strip pass exact-screens 196 complete states and commits 15
corridors.  The next refreshed pass exact-screens 159 complete states and finds
eight locally valid corridors; the inherited-component anchor constraint keeps
the strongest six.  Across the final pass, 65 ribbons replace 24 alternatives,
strict triangles rise from 28,748 to 28,857, augmented triangles from 28,926 to
29,035, and edge-connected triangle regions fall from 727 to 718.  All 4,303
inherited components and 38 previous CT connections remain; six new corridor
connections are added.  Minimum affected strict and augmented area retention
are both 1.0, with no deletion, split, fusion, crossing, interface conflict, or
non-manifold edge.  The expensive one-sided exact checkpoint is fingerprinted
from its numerical surface and variant arrays plus exact-solver dependencies,
so report or preview edits no longer invalidate a several-minute solve.
The following whole-hole pass brings the canonical state to 38,134 ribbons,
28,870 strict triangles, and 29,048 augmented triangles while keeping 718
triangle regions and all 4,303 components.  It preserves all 44 cumulative CT
connections and leaves 40 macro holes explicit for later collective passes.

The remaining free frontier exposes a configuration-energy barrier rather
than a lack of local evidence.  Of 125,574 candidate interfaces, the canonical
state initially uses 76,272; 14,781 noncrossing ribbons remain physically free.
Many residual continuation regions have negative one-ribbon marginals but a
positive complete-patch objective.  `optimize-physical-ribbon-collective-patches`
therefore optimizes each connected residual region from several dense starts,
then performs exact add/remove/swap ascent.  A proposal must have
two-dimensional continuation support, attach to one prior component, become
real triangle area, and not split an existing triangle region.  The first pass
adds 140 ribbons and 361 strict triangles, closes six macro holes, and reduces
triangle regions by one.  A second identical pass accepts no proposal, making
the additive move class an explicit fixed point rather than an indefinitely
repeated cell-growth heuristic.

Additive saturation does not imply assignment saturation.  A missing patch can
require several existing interface pairings to be replaced before its better
surface state is reachable.  `optimize-physical-ribbon-patch-states` consumes
the ordinary native-CT hole artifact and groups every scored hole frontier on
one reconstructed surface component into a single factor state.  Candidate
ribbons and the same-component incumbents they physically exclude are mutable;
every other selected ribbon is a fixed halo.  Other components are immutable
blockers.  The solver crosses negative unary barriers collectively, but an
incumbent is removed only when a chosen alternative shares an interface or an
exact profile-crossing conflict with it.

Proposal gates use the already declared whole-patch air-material-air profile,
normal-offset competing layers, two-dimensional continuation, and patch
coverage.  They are intentionally broad: all component states are rebuilt in
one exact surface pass, so screening more hypotheses does not add one expensive
reconstruction per hole.  Exact acceptance then requires complete surface
realization, unchanged component lineage, no interface or profile conflict, no
new triangle region or closed hole, and at least 99.5 percent prior triangle
area.  Low-anchor states must actually close a hole or join a triangle region;
mere area growth still requires a well-retained boundary.

On the 384 x 384 x 320 pilot block, 33 macro holes form 19 component-level
states over 3,225 alternatives.  Fourteen reach exact replay and 12 survive.
They replace 182 incumbents with 193 alternatives, raise strict triangles from
29,237 to 29,346, reduce triangle regions from 735 to 734, reduce interior
holes from 120 to 108, and reduce macro holes from 33 to 17.  The largest
accepted state jointly re-pairs 206 ribbon assignments on one component and
closes four of its six macro holes.  All 4,303 physical components remain,
with zero deletion, split, fusion, interface reuse, or profile crossing.  The
collective optimization itself takes 4.5 seconds; the two complete exact
surface reconstructions dominate the 73-second run.

`audit-physical-ribbon-flat-texture` samples the exact intrinsic charts from
native CT at fixed ply depths and compares axial texture disagreement on new
mesh edges against old edges from the same surface.  The 12 changed components
contain no nonadjacent chart-overlap pixels.  In the eight largest, five new
boundaries match or improve their own baseline at at least one depth; the
101-ribbon state is 3--4 degrees above a roughly 17-degree baseline, while the
weakest persistent diagnostic is about 9 degrees above baseline.  These are
reported diagnostics, not post-hoc labels or slice-specific acceptance tuning.

The audit also publishes a label-free compatibility decision.  At each fixed
depth, the new-boundary median may exceed the same-surface control median by
the larger of five degrees or one quarter of the control median-to-p90 spread.
A component passes when any measured depth is compatible.  This adapts to the
texture noise of each reconstructed surface rather than imposing a truth label
or optimizing the depth after the fact.  All 12 first-wave components pass.

`gate-physical-ribbon-patch-texture` compiles that decision back into an exact,
materializable surface state.  A refreshed geometry pass proposed one further
7-for-7 re-pairing that would add seven triangles and close two macro holes,
but its flattened boundary was 15--31 degrees worse than its control at all
three depths.  The gate rejects it and reproduces the 29,346-triangle,
17-macro-hole first-wave state exactly.  Thus the current alternating move
class is saturated under both exact geometry and actual-CT fiber continuity;
the attractive geometry-only second wave is retained as a counterexample, not
silently promoted to the canonical configuration.

### Dense normal-depth fields and coverage-aware matching

The 17-hole fixed point above was a fixed point of the *matching objective*,
not of the CT.  `analyze-physical-ribbon-depth-fields` now samples every pixel
of every complete missing-patch raster at 25 normal shifts and seven
air-material-air profile depths.  It solves the whole ordered-label raster with
multi-start alternating exact row and column Viterbi passes.  No pixel, cell,
or occupied voxel is ever an expansion decision.  The boundary is a soft
zero-shift condition, and spatial depth-jump costs are truncated so a coherent
delamination step can survive instead of being smoothed onto a neighboring
layer.

The artifact then audits the ribbon bank *against* this independently solved CT
field.  It records raw normal profiles, physical contrast, context-profile
correlation, independent and collective depth labels, far-layer margins,
candidate depth compatibility, and both direct and mesh-radius candidate
coverage.  Its PNG shows, for each complete hole, the actual CT on the solved
surface, collective depth shift, physical score, and CT-versus-bank coverage.
The field is reusable and immutable; it does not select a ribbon.

On the first residual census, 619 of 620 patch pixels are CT-supported.  Every
hole has one coherent supported field.  Median shifts are zero for 15 holes;
the other two drift only 0.125 and 0.25 local thicknesses.  Depth-compatible
ribbon candidates cover 93--100 percent of every patch within one existing
mesh edge, although they land directly on only 27.5 and 44.7 percent of the
two largest component-0 holes.  All 17 failures are therefore classified as
assignment/topology-limited, rather than as absent CT structure or a missing
ribbon bank.

`optimize-physical-ribbon-patch-states --depth-field ...` uses that result as a
whole-patch objective.  Each CT-supported raster pixel is a saturated coverage
factor: the objective pays once when any depth-compatible candidate covers it,
so five competing ribbons at one location cannot outvote a candidate that
fills an uncovered part of the sheet.  Deterministic global starts construct
complete interface matchings before exact add/remove/swap ascent.  The mutable
unit remains every hole on one surface component, all other selected ribbons
remain a fixed halo, and interface reuse plus exact profile crossings remain
hard constraints.

Proposal prefilters are deliberately permissive because exact surface and
flattened-texture stages are the decision gates.  Two-ribbon complete repairs
are allowed, the proposal-stage profile floor is 0.50 correlation with a 0.10
competing-layer margin, and a hole-closing state may retain 98 percent of prior
triangle area.  This recovered a valid two-ribbon repair on a small component
that the old minimum-size and 99.5-percent-area prefilters never screened.  The
flattened audit now accepts six boundary measurements as the smallest
reportable median; the recovered small repair has nine and improves on its
same-surface control at two depths.

Two texture-gated waves reduce the canonical residual from 17 to 10 macro
holes.  The first broad exact screen retains components 16 and 222 and rejects
the known component-7 shear jump.  The dense-coverage wave then retains three
different complete matchings on components 4, 7, and 12.  They close five more
macro holes; all three pass flattened actual-CT texture at at least one fixed
ply depth.  Across both waves, strict triangles rise from 29,346 to 29,368,
interior holes fall from 108 to 104, triangle regions remain 734, and all 4,303
component lineages remain unchanged with zero interface reuse or profile
crossing.  An independent materialized restart reproduces exactly 10 macro
holes.  A third coverage solve returns only component states already rejected
by exact geometry, so this objective is at a measured fixed point.  The next
search layer should retain several diverse complete matchings per component
for exact screening; it should not resume local frontier growth or weaken the
final gates.

### Diverse whole-patch state ensembles

A single proxy-optimal assignment is not enough to establish saturation.  Two
complete matchings can have nearly identical unary, continuity, and dense-CT
coverage scores while producing different strict-mesh topology.  The patch
solver now declares six fixed physical objective profiles, constructs complete
global matchings from multiple deterministic starts, and adds forced-exclusion
branches around choices in the canonical-best ensemble state.  It retains up
to 16 distinct whole-component states after normalized-Hamming deduplication.
These are alternative assignments for the entire mutable interface set; no
individual cell or ribbon is used as an expansion decision.

Every qualified state is rebuilt on the exact induced graph of its inherited
surface component.  The local rebuild includes all inherited nodes and every
candidate used by any variant, rejects an added node that joins another
selected component, and requires all surviving inherited and added nodes to
remain in one component.  It then recomputes the strict chart, triangles,
triangle regions, and boundary loops.  A final whole-volume reconstruction
proves the combined choices against interface reuse, profile crossings,
deletion, split, fusion, and area loss.  On the 50-state benchmark this
component-local screen reproduced the prior whole-volume NPZ byte-for-byte
while reducing total runtime from 313.5 to 124.7 seconds.

Starting from the independently materialized 10-hole checkpoint, the first
ensemble produces 72 variants and exact-screens all 50 qualified states.
Eighteen are exact-valid and the best six compatible components replace 40
incumbents with 41 alternatives.  They close six macro holes and three
interior holes, add 12 strict triangles, and leave all 734 triangle regions
and 4,303 component lineages unchanged.  Flattened native-CT audits accept all
six components; five are compatible at every fixed depth and the remaining
component is compatible at one depth, preserving a possible delamination
rather than forcing agreement across normal offsets.

A refreshed 16-state wave at four holes finds a lower-proxy rank-15 state that
replaces four ribbons with five, adds two triangles, and passes actual-CT
texture at all three depths without changing topology counts.  Recomputing the
dense depth field after that assignment exposes a rank-2 forced-exclusion state
on another component.  It replaces eight ribbons with seven and closes one
more macro and interior hole while retaining 98.14 percent of affected area;
all three flattened CT depths are compatible.  An independent final census is
therefore at 3 macro holes, 100 interior holes, 734 triangle regions, 29,379
strict triangles, 38,291 selected ribbons, and 4,303 components, with no
unresolved boundary fan or non-cycle boundary component.  Relative to the
17-hole checkpoint, 14 macro holes have now been closed without weakening the
physical or exact-topology gates.

All 347 pixels in the three remaining macro holes are CT-supported.  Two holes
belong to component 0: its enumerated states regress exact interior-hole
topology.  The last belongs to component 12: its enumerated states do not pass
the declared physical proxy gates.  This is a measured fixed point of the
current 16-state generator, not evidence that the scan lacks a sheet.  The
next solver work should target complete correspondences that explicitly
reduce a named boundary loop, rather than widening blind enumeration or
returning to one-cell frontier growth.

That targeted solve is now implemented.  A component still receives one
complete matching scope, but each of its largest named loops also receives a
complete isolated scope with the other holes frozen into the halo.  Proxy
objective sign is no longer a truth gate when at least 90 percent of the full
patch raster has independent CT support.  Every such counterexample is sent to
the same exact component reconstruction.  Finally, flattened native CT is run
for every exact-valid alternative, not just the first geometry-ranked state;
the best texture-compatible complete state is then selected per component.

On the three-hole checkpoint this exposes seven exact-valid states.  The
locally negative component-12 state closes its hole and passes all three CT
depths.  The first two and fourth component-0 alternatives cut across fiber
texture, but its third complete matching passes at one fixed ply depth.  The
two compatible states replace 20 incumbents with 18 alternatives.  An
independent materialized restart verifies 3 to 1 macro holes, exactly 100
interior holes, 734 triangle regions, 29,378 strict triangles, 38,289 selected
ribbons, and all 4,303 component lineages, with no interface reuse, profile
crossing, split, deletion, or fusion.

The final hole then provides a clean representation audit.  All 207 raster
pixels have coherent air-material-air CT support, with 0.951 median profile
correlation and 0.552 median displaced-layer margin.  Its 154 alternatives
form one continuation component; all attach to the inherited sheet.  The
local matching has 209 binary variables, 3,014 explicit continuation products,
730 hard assignment/crossing conflicts, and 207 saturated CT coverage
variables.  HiGHS now solves both the canonical physical objective and a
coverage-lexicographic objective under a declared time limit.  Binary states
are reserved for exact reconstruction even when their proxy rank is below the
heuristic top 16.

The coverage state reaches 91.787 percent of the patch against a 92.754 percent
candidate-bank ceiling, with a reported relative MIP gap of 0.000037.  Exact
reconstruction still leaves the macro hole open and increases its component's
triangle-region count from 12 to 13.  The canonical binary state adds five
triangles and 0.31 percent area but likewise moves the hole and creates an
interior loop.  This is materially different from a failed frontier heuristic:
the present discrete ribbon hypotheses nearly saturate the observed CT yet
cannot realize it as one attached surface.  Further score tuning is therefore
off target.  The next representation layer should promote the coherent dense
CT field to adaptive surface elements, stitch the complete patch boundary at
once, and retain the ribbon bank as supporting/collision evidence rather than
requiring one existing ribbon at every mesh vertex.

The accepted artifacts are rooted at:

```text
work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-ensemble-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-ensemble-iteration2-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-ensemble-iteration3-v1
work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-ensemble-iteration3-v1
work/multiseam-2x2-b00c03c/physical-ribbon-texture-gate-ensemble-iteration3-v1
work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-ensemble-iteration3-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-ensemble-iteration3-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-topology-scopes-v1
work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-topology-variants-v1
work/multiseam-2x2-b00c03c/physical-ribbon-texture-gate-topology-variants-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-topology-variants-v1
work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-binary-topology-v2
```

The reproducible dense-field sequence is:

```bash
python -m backend.cubical analyze-physical-ribbon-depth-fields \
  --holes work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-depth-field-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-depth-fields-iteration2-v1 \
  --settings-json examples/physical-ribbon-depth-fields.json

python -m backend.cubical optimize-physical-ribbon-patch-states \
  --holes work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-depth-field-v1 \
  --depth-field work/multiseam-2x2-b00c03c/physical-ribbon-depth-fields-iteration2-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-depth-coverage-v1 \
  --settings-json examples/physical-ribbon-patch-states.json

python -m backend.cubical audit-physical-ribbon-flat-texture \
  --surface work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-depth-coverage-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-depth-coverage-v1 \
  --settings-json examples/physical-ribbon-flattened-audit.json

python -m backend.cubical gate-physical-ribbon-patch-texture \
  --patch-state work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-depth-coverage-v1 \
  --texture-audit work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-depth-coverage-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-texture-gate-depth-coverage-v1

python -m backend.cubical materialize-physical-ribbon-replay-configuration \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-texture-gate-depth-coverage-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-depth-coverage-v1
```

The final two iterative commands are:

```bash
python -m backend.cubical analyze-physical-ribbon-lineage-strips \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-lineage-strip-replay-v1 \
  --corridor-row 87 --corridor-row 90 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-coverage-strips-v1

python -m backend.cubical replay-physical-ribbon-lineage-strips \
  --lineage-strips work/multiseam-2x2-b00c03c/physical-ribbon-lineage-coverage-strips-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-coverage-replay-v1

python -m backend.cubical analyze-physical-ribbon-lineage-strips \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-lineage-coverage-replay-v1 \
  --corridor-row 87 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-bend-strips-v1

python -m backend.cubical replay-physical-ribbon-lineage-strips \
  --lineage-strips work/multiseam-2x2-b00c03c/physical-ribbon-lineage-bend-strips-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-bend-replay-v1
```

The reproducible commands are:

```bash
python -m backend.cubical optimize-physical-ribbon-collective-patches \
  --configuration work/multiseam-2x2-b00c03c/physical-ribbon-replay-configuration-cumulative-v10 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-collective-v1

python -m backend.cubical optimize-physical-ribbon-patch-states \
  --holes work/multiseam-2x2-b00c03c/physical-ribbon-patch-holes-collective-v5 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-v2

python -m backend.cubical audit-physical-ribbon-flat-texture \
  --surface work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-v2 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-patch-state-v2 \
  --settings-json examples/physical-ribbon-flattened-audit.json

python -m backend.cubical gate-physical-ribbon-patch-texture \
  --patch-state work/multiseam-2x2-b00c03c/physical-ribbon-patch-state-v2 \
  --texture-audit work/multiseam-2x2-b00c03c/physical-ribbon-flat-texture-patch-state-v2 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-texture-gate-patch-state-v2

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

python -m backend.cubical analyze-physical-ribbon-corridor-deficits \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-iteration1-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-deficits-v1

python -m backend.cubical analyze-physical-ribbon-corridor-faces \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-one-sided-corridors-iteration1-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-faces-v1

python -m backend.cubical replay-physical-ribbon-corridor-faces \
  --faces work/multiseam-2x2-b00c03c/physical-ribbon-corridor-faces-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-corridor-face-replay-v1

python -m backend.cubical analyze-physical-ribbon-complete-strips \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-corridor-face-replay-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-complete-strips-v1

python -m backend.cubical replay-physical-ribbon-complete-strips \
  --strips work/multiseam-2x2-b00c03c/physical-ribbon-complete-strips-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-complete-strip-replay-v1

python -m backend.cubical analyze-physical-ribbon-lineage-strips \
  --replay work/multiseam-2x2-b00c03c/physical-ribbon-complete-strip-replay-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-strips-v1

python -m backend.cubical replay-physical-ribbon-lineage-strips \
  --lineage-strips work/multiseam-2x2-b00c03c/physical-ribbon-lineage-strips-v1 \
  --output work/multiseam-2x2-b00c03c/physical-ribbon-lineage-strip-replay-v1
```
