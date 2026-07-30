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
  distributions support that corner.
- Up to four incident cells can observe one physical crossing of a global grid
  edge. Accepted face joins weld those observations into one latent crossing.
- Pairwise-compatible corner transitions are processed in descending evidence
  order. A join is deferred if its transitive crossing group has no common edge
  or vertex, or if it would put two locally planar patches from one cell in one
  surface component.
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
