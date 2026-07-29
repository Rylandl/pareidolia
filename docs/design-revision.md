# Evidence-to-layer design revision

This revision separates observations, local hypotheses, sparse associations,
and physical interpretation. The pipeline may build globally consistent sparse
graphs, but it does not perform a global dense-geometry solve and does not call
a graph component a physical papyrus sheet.

## Layered contracts

1. **Native evidence**
   - CT intensity and finite-support Acus needles are observations.
   - Normal families are local, unsigned orientation hypotheses with disjoint
     needle ownership.
   - Replication measurements estimate the probability that a local flake is
     rediscovered. They are not probabilities of physical sheet identity.

2. **Hypothesis-independent material accountability**
   - Every accepted local normal has a native-CT profile from -32 through +32
     voxels at one-voxel spacing.
   - The declared analysis air cutoff produces a material mask before any flake or
     carrier is consulted.
   - A separate claim overlay labels samples as air, unassigned material,
     singly claimed material, or contested material.
   - Contested means that multiple separated local flake-depth clusters occupy
     one unresolved CT material run. It is an ambiguity/contact diagnostic,
     not evidence of two particular sheets.
   - Apparent CT thickness is permitted only for singly claimed intervals that
     do not touch either profile boundary. Fitted flake thickness is never
     substituted for physical papyrus thickness.

3. **Local surface branches**
   - Flakes are linked by finite-patch edge agreement and transported fiber
     direction. A changing normal is curvature, not an identity failure.
   - Relative normal signs may be synchronized for ordering, but each connected
     region retains an arbitrary sign. Inward/outward and recto/verso are later
     physical interpretations.
   - Neighboring flake sequences use order-preserving partial matching.
     Missing modes are explicit births/deaths; equal ordinal numbers are not
     required, and crossings are forbidden.
   - Collision-safe graph components are local surface branches only.

4. **Sparse association**
   - Branch joins use compatible facing edges, fiber continuation, order
     feasibility, and material accountability.
   - A join must be accepted in the main chunk and every overlapping subwindow
     that observes it. It is then rebuilt with the active MLS carrier; this is a
     same-sample construction gate, not independent validation.
   - Pairwise-incoherent joins are deferred. If individually coherent joins make
     an incoherent transitive association, the weakest retained construction
     edge is removed until every output association passes the carrier gate.
   - A completeness prior applies only to flakes with a calibrated reliability
     estimate. Every solve retains an explicit outlier/unassigned state.
   - Secondary-normal fragments remain separate branches until the association
     model can represent family ambiguity rather than absorbing them as extreme
     curvature.

5. **Physical interpretation**
   - Page identity, winding order, side, and recto/verso are downstream
   interpretations. None is encoded in voxel labels, flake IDs, or branch
   component IDs.

6. **Cross-association integrity**
   - Every accepted merged association is reconstructed as the active MLS
     carrier and triangulated for an explicit non-intersection audit.
   - Intersections within 24 voxels of supporting flakes on both surfaces are
     separated from intersections in the wider carrier support skirt.
   - A bidirectional sampled-clearance sweep is reported descriptively. It is
     not called papyrus thickness, and shared-cell depth order distinguishes
     an ordered near-contact from an unexplained boundary approach.

## Current z512 findings

The active multi-normal benchmark freezes 37 artifacts (781,614,775 bytes) and
nine preservation/construction guards. It deliberately excludes stale
single-normal carrier descendants.

The full material census samples 261,302 profiles and 16,984,630 native CT
positions in 26 seconds. At the declared analysis air threshold:

- 96.32% of positions are material;
- 243,069 profiles contain material across the entire 65-voxel window;
- 234,496 profiles contain contested material;
- 695,183 of 724,672 local flake claims are supported by material; and
- only 72 intervals qualify for the deliberately narrow apparent-thickness
  statistic.

A deterministic 20,000-profile threshold sweep confirms that higher arbitrary
cutoffs create apparent separators by discarding evidence. At raw threshold 96,
for example, claims are separated across CT runs in 50.87% of profiles, but
claim support has already fallen to 78.27%. Air/material thresholding is
therefore useful for accountability and outlier handling, not as the layer
solver.

The first monotone prototype uses the densest 32 x 32 x 14-cell primary-family
window. Relative signs form one connected region with 0.83% weighted sign
frustration. Compared with the active graph it:

- preserves 49,549 of 49,745 retained links;
- removes all 149 pairwise order crossings;
- adds 112 compatible non-crossing links;
- leaves median edge residual and fiber disagreement unchanged at about 0.73
  voxels and 1.08 degrees; and
- changes 3,361 local branches into 3,349, with the largest branch changing
  from 372 to 361 flakes.

This is a useful negative result: the current pairwise geometry is already
almost order-consistent, so ordinal mistakes are not the main source of
fragmentation. The next useful solve is branch-level sparse association across
missing continuation edges, with order feasibility and material accountability
as constraints. It should not be another local ordinal rematcher or a dense
global surface fit.

The first branch-association solve implements that next step in the same window.
It evaluates 75,118 spatial endpoint pairs, retains 1,775 geometry hits, and
aggregates them into 1,499 branch-pair candidates in 12.5 seconds. At the
selected 0.45 score threshold, 56 joins survive the main-window solve and 45
are unanimous wherever observed by the four overlapping subwindows. Exact MLS
reconstruction defers 11 of those joins: one fails only the 3-voxel median
height gate, four fail only the 6-degree median normal gate, and six fail both.
The high-level endpoint score alone is not sufficient—the strongest rejected
join scores 0.843—so this reconstruction gate is materially useful.

The remaining 34 joins form 33 associations over 67 original branches. Every
association passes exact reconstruction without requiring transitive pruning.
The largest joins three branches and 219 flakes with 2.33-voxel median height
residual and 4.26-degree median normal residual. These are deliberately modest,
chunk-local surface associations; 3,282 linked branches remain explicitly
unassociated, and no output identity is promoted to a physical sheet.

The first joint integrity audit triangulates all 33 accepted association
carriers into 84,272 finite surface triangles and checks 29,844 sampled points
within 24 voxels of their supporting flakes. It finds zero surface
intersections, zero within-association cell collisions, and only one pair of
carriers within 12 voxels. That pair approaches to 5.67 voxels in the sampled
grids, but the closest displacement is primarily normal (5.25 voxels), the
normals and fibers agree within 4.53 and 3.55 degrees, and their one shared cell
preserves an 8.68-voxel depth order. It is therefore retained as an explicit
ordered near-contact rather than treated as a crossing or an automatic merge.

## Reproducible commands

```bash
.venv/bin/python scripts/science-ci.py \
  --root work/cross-scroll-analysis-z512 --verify-artifacts

.venv/bin/python scripts/build-material-intervals.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/prototype-monotone-layers.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-monotone-branches.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/audit-branch-association-integrity.py \
  --root work/cross-scroll-analysis-z512
```

The large NumPy products remain ignored under `work/`. Their summaries contain
content hashes for all generated arrays and input identities. The compact
science baseline is committed under `benchmarks/`.
