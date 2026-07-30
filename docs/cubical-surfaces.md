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
6. Neighboring blocks are composed using only their common boundary traces,
   followed by a narrow-band continuous relaxation.
7. Papyrus-specific sheet packets add thickness, paired-ply, fiber, and air-gap
   factors without changing the geometric representation.

The first packet layer is now implemented as a separate dual-axis connectivity
artifact. It fixes every retained single-ply join, admits only well-supported
parallel-to-orthogonal fiber-frame continuations, and reruns the same global
collision, crossing, and orientability checks for additions. The underlying
patch shard and strict graph remain immutable. Thickness and air-gap factors
remain later packet refinements rather than implicit properties of this graph.

The geometry stages are validated independently on analytic surfaces. The
native-CT implementation and its measured pilot are documented in
[`raw-acus-cubical-pipeline.md`](raw-acus-cubical-pipeline.md).
